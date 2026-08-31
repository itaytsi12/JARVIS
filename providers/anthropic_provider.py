"""Anthropic (Claude) model provider.

This is the ONLY module in JARVIS that imports the `anthropic` SDK. It
translates the provider-independent types in `providers/base.py` to and
from the Messages API, and translates the SDK's exception hierarchy into
JARVIS's own.

Deliberately single-turn: the SDK ships a `tool_runner` helper that owns
the agentic loop, and this provider does NOT use it. JARVIS's agent loop
(`brain/agent_loop.py`) is the architecture; Claude is a model inside it.
Using the vendor's loop would move plan/observe/retry/cancel semantics
into the vendor SDK and make a second provider impossible to add.

No `thinking` parameter is sent: the current Opus/Sonnet models run
adaptive thinking by default, `budget_tokens` was removed from the API,
and pinning a vendor-specific reasoning configuration here is exactly the
detail this layer exists to keep out of the rest of JARVIS.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from config import estimate_cost, get_config
from config import events
from providers.base import (
    Message,
    ModelResponse,
    ProviderAuthError,
    ProviderRateLimited,
    ProviderRequestError,
    ProviderServerError,
    ProviderTimeout,
    ProviderUnavailable,
    ToolCall,
    ToolSpec,
    Usage,
)

log = logging.getLogger("jarvis.providers.anthropic")


CACHE_CONTROL: dict[str, Any] = {"type": "ephemeral"}


def _mark_cache_breakpoint(payload: list[dict[str, Any]]) -> None:
    """Put a rolling cache breakpoint on the last content block.

    Caching is a PREFIX match rendered in the order `tools` -> `system` ->
    `messages`, so a breakpoint here caches the entire conversation so far.
    In an agent loop that is exactly the part that keeps being re-sent: every
    step resends every previous observation, so by step four the same tokens
    have been billed four times.

    Only the LAST block is marked (one breakpoint, re-placed each turn),
    which stays well inside the API's 4-breakpoint limit however long the run
    gets. A block that is not a dict -- a plain string message -- cannot
    carry `cache_control`, so it is left alone rather than restructured.
    """
    if not payload:
        return
    content = payload[-1].get("content")
    if isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1] = {**content[-1], "cache_control": CACHE_CONTROL}


def _to_api_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate JARVIS messages into Anthropic Messages API content blocks."""
    payload: list[dict[str, Any]] = []
    for message in messages:
        if message.tool_outcomes:
            blocks: list[dict[str, Any]] = [
                {
                    "type": "tool_result",
                    "tool_use_id": outcome.call_id,
                    "content": outcome.content or "(no output)",
                    "is_error": bool(outcome.is_error),
                }
                for outcome in message.tool_outcomes
            ]
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            payload.append({"role": "user", "content": blocks})
            continue
        if message.tool_calls:
            blocks = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                for call in message.tool_calls
            )
            payload.append({"role": message.role, "content": blocks})
            continue
        payload.append({"role": message.role, "content": message.content or "(no content)"})
    return payload


class AnthropicProvider:
    """A `providers.base.ModelProvider` backed by the Claude Messages API."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        max_retries: int | None = None,
    ):
        config = get_config()
        self._api_key = api_key if api_key is not None else config.anthropic_api_key
        self.model = model or config.agent_model
        self._max_retries = config.agent_max_provider_retries if max_retries is None else max_retries
        self._client = client
        self._client_error: str | None = None

    # -- availability -------------------------------------------------
    def is_available(self) -> bool:
        if self._client is not None:
            return True
        if not self._api_key:
            self._client_error = "missing_api_key"
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            self._client_error = "anthropic_sdk_not_installed"
            return False
        self._client_error = None
        return True

    def unavailable_reason(self) -> str | None:
        if self.is_available():
            return None
        return self._client_error or "unavailable"

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "credentials": "<set>" if self._api_key else "<unset>",
            "available": self.is_available(),
            "unavailable_reason": self.unavailable_reason(),
            "max_retries": self._max_retries,
        }

    # -- client -------------------------------------------------------
    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise ProviderUnavailable(
                "ANTHROPIC_API_KEY is not set; Claude is disabled and JARVIS is running locally only."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - only without the SDK
            raise ProviderUnavailable("The `anthropic` package is not installed.") from exc
        self._client = anthropic.Anthropic(
            api_key=self._api_key,
            max_retries=self._max_retries,
            timeout=get_config().agent_request_timeout,
        )
        return self._client

    # -- completion ---------------------------------------------------
    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        model: str | None = None,
        effort: str | None = None,
        cache: bool = True,
        on_text: Any = None,
    ) -> ModelResponse:
        config = get_config()
        client = self._get_client()
        model_id = model or self.model
        api_messages = _to_api_messages(messages)
        request: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens or config.agent_max_tokens,
            "messages": api_messages,
        }
        if tools:
            request["tools"] = [spec.to_dict() for spec in tools]
        if system:
            # Prompt caching. The prefix is rendered `tools` -> `system` ->
            # `messages`, so ONE breakpoint at the end of `system` caches the
            # tool schemas AND the system prompt together -- the two parts
            # that are byte-identical on every step of a run (and across
            # runs), and that were previously re-billed at full price every
            # time. The system prompt must become a content-block list for
            # `cache_control` to have anywhere to live.
            if cache:
                request["system"] = [{"type": "text", "text": system, "cache_control": CACHE_CONTROL}]
            else:
                request["system"] = system
        if cache:
            _mark_cache_breakpoint(api_messages)
        if effort:
            # `output_config.effort` (GA, not a top-level parameter) controls
            # how much thinking and total token spend a turn gets. The API
            # default is "high"; a routine read-only step does not need it,
            # and a genuinely hard one still asks for it -- see
            # `brain/agent_service.py::select_effort`.
            request["output_config"] = {"effort": effort}
        if timeout is not None and hasattr(client, "with_options"):
            client = client.with_options(timeout=timeout)

        started = time.perf_counter()
        first_event_ms: float | None = None
        # UI/status hook. Emitted here, in the vendor module itself, rather
        # than in `providers/usage.py::TrackedProvider` -- that wrapper is
        # only applied by `brain/agent_service.py`, so a direct provider
        # call (a benchmark, a script, a future call site) would silently
        # never light the node up. `publish` never raises.
        events.publish(events.MODEL_REQUEST_STARTED, model="anthropic")
        try:
            if on_text is None:
                response = client.messages.create(**request)
            else:
                # Streaming, used ONLY when the caller wants text as it is
                # produced (the final spoken answer). `get_final_message()`
                # returns the same Message object the non-streaming call
                # would have, so everything below is identical -- streaming
                # changes when text arrives, never what the run concludes.
                #
                # `on_text` receives PUBLIC assistant text only. Thinking
                # blocks and tool-call payloads are deliberately not
                # forwarded: `text` events fire for `text` content blocks,
                # and `input_json` / `thinking` events are simply not
                # subscribed to here.
                tool_use_started = False
                with client.messages.stream(**request) as stream:
                    for event in stream:
                        event_type = getattr(event, "type", None)
                        if event_type == "content_block_start":
                            block = getattr(event, "content_block", None)
                            if getattr(block, "type", None) == "tool_use":
                                # This turn is an ACTION, not an answer.
                                # Stop forwarding text: everything after
                                # this point is tool-call payload, and
                                # anything before it was a preamble the
                                # caller must not treat as the answer.
                                tool_use_started = True
                            continue
                        if event_type != "text" or tool_use_started:
                            continue
                        if first_event_ms is None:
                            first_event_ms = (time.perf_counter() - started) * 1000
                        chunk = getattr(event, "text", "") or ""
                        if chunk:
                            on_text(chunk)
                    response = stream.get_final_message()
        except Exception as exc:  # translated below; never leaked raw
            # Translate FIRST, then report the translated type. The UI
            # distinguishes a rate limit (amber, "wait") from a genuine
            # failure (red, "broken"), and `ProviderRateLimited` is the
            # neutral name that decision is keyed on -- publishing the raw
            # vendor class name would make that mapping depend on which
            # SDK happened to raise. The exception raised is unchanged.
            translated = _translate_error(exc)
            # The event carries the EXCEPTION TYPE only, never its message:
            # a provider error string is the one place a credential could
            # plausibly appear, and this payload reaches the UI.
            events.publish(events.MODEL_REQUEST_FAILED, model="anthropic", error=type(translated).__name__)
            _log_provider_failure(exc, model_id)
            raise translated from exc
        events.publish(events.MODEL_REQUEST_SUCCEEDED, model="anthropic")
        latency_ms = (time.perf_counter() - started) * 1000

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=str(getattr(block, "id", "")),
                        name=str(getattr(block, "name", "")),
                        arguments=dict(getattr(block, "input", {}) or {}),
                    )
                )

        usage = _extract_usage(getattr(response, "usage", None))
        cost = (
            estimate_cost(
                model_id,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_creation_tokens,
                usage.cache_read_tokens,
            )
            if config.cost_tracking_enabled
            else None
        )

        stop_details = getattr(response, "stop_details", None)
        return ModelResponse(
            text="\n".join(part for part in text_parts if part).strip(),
            tool_calls=tool_calls,
            stop_reason=getattr(response, "stop_reason", None),
            model=str(getattr(response, "model", model_id) or model_id),
            provider=self.name,
            usage=usage,
            latency_ms=round(latency_ms, 3),
            first_event_ms=round(first_event_ms, 3) if first_event_ms is not None else None,
            estimated_cost_usd=cost,
            raw_stop_details=(
                {
                    "type": getattr(stop_details, "type", None),
                    "category": getattr(stop_details, "category", None),
                    "explanation": getattr(stop_details, "explanation", None),
                }
                if stop_details is not None
                else None
            ),
        )


_SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")


def _redact(text: str) -> str:
    """Strip anything shaped like an API key before it reaches a log.

    The SDK's own error text does not echo the key, but this layer is the
    one place a credential is in scope, so redaction is unconditional
    rather than dependent on the vendor never changing that.
    """
    return _SECRET_PATTERN.sub("sk-<redacted>", text)


def _log_provider_failure(exc: Exception, model_id: str) -> None:
    """Log the REAL SDK exception before it is translated.

    Without this, a genuine 401/404/400 from the API reaches the agent
    loop as a generic `provider_error` with no way to tell an invalid key
    from an unknown model. Logs the exception class, HTTP status, the
    API's own error type/message, and the request id -- never the key.
    """
    status = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    api_error_type = None
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict):
            api_error_type = inner.get("type")
    log.error(
        "Anthropic call failed: exception=%s status=%s api_error_type=%s model=%s request_id=%s message=%s",
        type(exc).__name__,
        status,
        api_error_type,
        model_id,
        request_id,
        _redact(str(getattr(exc, "message", None) or exc))[:500],
    )


def _extract_usage(raw: Any) -> Usage:
    if raw is None:
        return Usage(reported=False)

    def read(name: str) -> int:
        value = getattr(raw, name, None)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    return Usage(
        input_tokens=read("input_tokens"),
        output_tokens=read("output_tokens"),
        cache_creation_tokens=read("cache_creation_input_tokens"),
        cache_read_tokens=read("cache_read_input_tokens"),
        reported=True,
    )


def _translate_error(exc: Exception):
    """Map SDK exceptions to JARVIS's provider errors.

    Matched by class NAME as well as isinstance, so a test double (or a
    future SDK reshuffle) still classifies correctly instead of collapsing
    every failure into one generic error.
    """
    name = type(exc).__name__
    try:
        import anthropic
    except ImportError:
        anthropic = None  # type: ignore[assignment]

    def isa(attr: str) -> bool:
        cls = getattr(anthropic, attr, None) if anthropic is not None else None
        return bool(cls and isinstance(exc, cls))

    message = str(getattr(exc, "message", None) or exc)
    if isa("AuthenticationError") or isa("PermissionDeniedError") or name in {
        "AuthenticationError",
        "PermissionDeniedError",
    }:
        return ProviderAuthError(f"Claude rejected the credentials: {message}")
    if isa("APITimeoutError") or name == "APITimeoutError":
        return ProviderTimeout(f"Claude request timed out: {message}")
    if isa("RateLimitError") or name == "RateLimitError":
        retry_after = None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                retry_after = float(headers.get("retry-after"))
            except (TypeError, ValueError, AttributeError):
                retry_after = None
        return ProviderRateLimited(f"Claude rate limit reached: {message}", retry_after)
    if isa("APIConnectionError") or name == "APIConnectionError":
        return ProviderServerError(f"Could not reach Claude: {message}")
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status >= 500:
            return ProviderServerError(f"Claude server error {status}: {message}")
        if status == 429:
            return ProviderRateLimited(f"Claude rate limit reached: {message}")
        return ProviderRequestError(f"Claude rejected the request ({status}): {message}")
    if isa("BadRequestError") or name in {"BadRequestError", "NotFoundError", "UnprocessableEntityError"}:
        return ProviderRequestError(f"Claude rejected the request: {message}")
    return ProviderServerError(f"Claude call failed ({name}): {message}")
