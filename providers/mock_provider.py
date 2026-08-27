"""Deterministic providers for tests, demos and offline development.

`ScriptedProvider` replays a fixed list of `ModelResponse`s so the whole
agent loop -- planning, tool execution, observation, retry, verification,
episode capture -- can be exercised end to end without an API key and
without spending money. It is a TEST/DEMO backend and is never selected
automatically by `providers.registry`: production selection only ever
returns a real provider.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

from providers.base import (
    Message,
    ModelResponse,
    ProviderUnavailable,
    ToolCall,
    ToolSpec,
    Usage,
)


class ScriptedProvider:
    """Replays pre-built responses, recording every request it received."""

    name = "scripted"

    def __init__(self, responses: Iterable[ModelResponse] | None = None, model: str = "scripted-model"):
        self._responses = list(responses or [])
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str | None:
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "available": True,
            "queued_responses": len(self._responses),
        }

    def queue(self, response: ModelResponse) -> None:
        self._responses.append(response)

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
        **kwargs: Any,
    ) -> ModelResponse:
        # `effort`, `cache` and `on_text` are HINTS in the
        # `providers.base.ModelProvider` contract: a provider whose backend
        # has no equivalent ignores them and behaves exactly as before.
        # Accepting them here is what lets this scripted provider stand in
        # for a real one in the agent-loop tests.
        self.calls.append(
            {
                "messages": list(messages),
                "system": system,
                "tools": [spec.name for spec in tools or []],
                "max_tokens": max_tokens,
                "model": model or self.model,
            }
        )
        if not self._responses:
            raise ProviderUnavailable("ScriptedProvider ran out of scripted responses.")
        response = self._responses.pop(0)
        response.provider = self.name
        if not response.model:
            response.model = model or self.model
        return response


class CallableProvider:
    """Wraps a plain function as a provider.

    Useful when a test wants to react to what the loop actually sent
    rather than replaying a fixed script.
    """

    name = "callable"

    def __init__(self, handler: Callable[..., ModelResponse], model: str = "callable-model"):
        self._handler = handler
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str | None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "available": True}

    def complete(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        self.calls.append({"messages": list(messages), **kwargs})
        response = self._handler(messages, **kwargs)
        response.provider = self.name
        if not response.model:
            response.model = self.model
        return response


def text_response(text: str, *, stop_reason: str = "end_turn", input_tokens: int = 10, output_tokens: int = 5) -> ModelResponse:
    return ModelResponse(
        text=text,
        stop_reason=stop_reason,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def tool_response(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    call_id: str | None = None,
    text: str = "",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=[ToolCall(id=call_id or f"call_{name}", name=name, arguments=dict(arguments or {}))],
        stop_reason="tool_use",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )
