"""Reusable adapter for OpenAI-compatible cloud and local endpoints."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import requests

from providers.base import (
    Message, ModelResponse, ProviderAuthError, ProviderRateLimited,
    ProviderRequestError, ProviderServerError, ProviderTimeout,
    ProviderUnavailable, ToolCall, ToolSpec, Usage,
)
from providers.model_registry import ModelRoute, infer_capabilities


def _messages(messages: list[Message]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for message in messages:
        if message.tool_outcomes:
            for outcome in message.tool_outcomes:
                output.append({"role": "tool", "tool_call_id": outcome.call_id, "content": outcome.content})
            continue
        item: dict[str, Any] = {"role": message.role, "content": message.content or None}
        if message.tool_calls:
            item["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in message.tool_calls]
        output.append(item)
    return output


def _tools(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}} for t in tools]


class OpenAICompatibleProvider:
    def __init__(self, *, name: str, base_url: str, api_key: str | None, model: str = "", headers: dict[str, str] | None = None, credential_required: bool = True, timeout: float = 60.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.extra_headers = dict(headers or {})
        self.credential_required = credential_required
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.base_url and (self._api_key or not self.credential_required))

    def unavailable_reason(self) -> str | None:
        return None if self.is_available() else ("missing_api_key" if self.credential_required else "missing_base_url")

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "base_url": self.base_url, "credentials": "<set>" if self._api_key else "<not-required>" if not self.credential_required else "<unset>", "available": self.is_available(), "unavailable_reason": self.unavailable_reason()}

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def complete(self, messages: list[Message], *, system: str | None = None, tools: list[ToolSpec] | None = None, max_tokens: int | None = None, temperature: float | None = None, timeout: float | None = None, model: str | None = None, effort: str | None = None, cache: bool = True, on_text: Any = None) -> ModelResponse:
        if not self.is_available():
            raise ProviderUnavailable(f"{self.name} is unavailable: {self.unavailable_reason()}")
        selected = model or self.model
        if not selected:
            raise ProviderRequestError(f"No model selected for {self.name}")
        wire_messages = _messages(messages)
        if system:
            wire_messages.insert(0, {"role": "system", "content": system})
        payload: dict[str, Any] = {"model": selected, "messages": wire_messages, "stream": False}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        wire_tools = _tools(tools)
        if wire_tools:
            payload["tools"] = wire_tools
        started = time.perf_counter()
        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload, timeout=timeout or self.timeout)
        except requests.Timeout as exc:
            raise ProviderTimeout(f"{self.name} timed out") from exc
        except requests.RequestException as exc:
            raise ProviderServerError(f"Could not reach {self.name}: {type(exc).__name__}") from exc
        self._raise_http(response)
        try:
            raw = response.json()
            choice = raw["choices"][0]
            message = choice.get("message", {})
            calls = []
            for item in message.get("tool_calls") or []:
                function = item.get("function") or {}
                arguments = function.get("arguments") or "{}"
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict) or not function.get("name"):
                    raise ValueError("invalid tool call")
                calls.append(ToolCall(item.get("id") or uuid.uuid4().hex, function["name"], arguments))
            usage = raw.get("usage") or {}
            result = ModelResponse(text=message.get("content") or "", tool_calls=calls, stop_reason=choice.get("finish_reason"), model=raw.get("model") or selected, provider=self.name, usage=Usage(int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0), reported=bool(usage)), latency_ms=(time.perf_counter() - started) * 1000)
            if on_text and result.text:
                on_text(result.text)
            return result
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderRequestError(f"{self.name} returned malformed structured output") from exc

    def _raise_http(self, response: requests.Response) -> None:
        if response.status_code < 400:
            return
        message = (response.text or response.reason or "request failed")[:500]
        if response.status_code in (401, 403):
            raise ProviderAuthError(f"{self.name} rejected credentials")
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            try: retry_after_value = float(retry_after) if retry_after else None
            except ValueError: retry_after_value = None
            raise ProviderRateLimited(f"{self.name} rate/quota limited: {message}", retry_after_value)
        if response.status_code in (408, 500, 502, 503, 504):
            raise ProviderServerError(f"{self.name} temporary error {response.status_code}: {message}")
        if response.status_code == 404:
            error = ProviderServerError(f"{self.name} model/route unavailable: {message}")
            error.status_code = 404
            raise error
        raise ProviderRequestError(f"{self.name} rejected request ({response.status_code}): {message}")

    def discover_models(self, *, timeout: float = 3.0, priority: int = 100, free_only: bool = False) -> list[ModelRoute]:
        if not self.is_available():
            return []
        try:
            response = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=timeout)
            self._raise_http(response)
            items = response.json().get("data", [])
        except (requests.RequestException, ProviderRequestError, ProviderServerError, ProviderRateLimited, ProviderAuthError, ValueError, TypeError):
            return []
        routes = []
        for item in items:
            name = item.get("id") if isinstance(item, dict) else None
            if not name:
                continue
            pricing = item.get("pricing") if isinstance(item, dict) else None
            is_free = bool(isinstance(pricing, dict) and pricing.get("prompt") in ("0", 0) and pricing.get("completion") in ("0", 0))
            if free_only and not is_free and not str(name).endswith(":free"):
                continue
            caps = infer_capabilities(str(name))
            supported = item.get("supported_parameters") if isinstance(item, dict) else None
            if isinstance(supported, list):
                if "tools" in supported or "tool_choice" in supported:
                    caps = caps | {"tool_use"}
                if "response_format" in supported or "structured_outputs" in supported:
                    caps = caps | {"structured_output"}
            architecture = item.get("architecture") if isinstance(item, dict) else None
            modality = str((architecture or {}).get("modality") or "").lower() if isinstance(architecture, dict) else ""
            if "image" in modality:
                caps = caps | {"vision"}
            context_window = item.get("context_length") if isinstance(item, dict) else None
            if not self.credential_required:
                caps = caps | {"local"}
            routes.append(ModelRoute(f"{self.name}:{name}", str(name), self.name, str(name), priority, free_tier=is_free or not self.credential_required, timeout=self.timeout, base_url=self.base_url, capabilities=caps, metadata={"discovered": True, "context_window": context_window, "supported_parameters": supported or []}))
        return routes


__all__ = ["OpenAICompatibleProvider"]
