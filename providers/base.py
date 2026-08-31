"""Provider-independent model interface.

Nothing outside `providers/` may import `anthropic` (or any other vendor
SDK). Everything the rest of JARVIS needs is expressed with the plain
dataclasses in this module, so a second provider -- a local model, a
different vendor -- can be added by writing one new class here and
registering it, with no change to the agent loop, the task manager, or
the tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class ProviderError(RuntimeError):
    """Base class for every model-provider failure."""

    retryable = False


class ProviderUnavailable(ProviderError):
    """The provider cannot be used at all: no SDK installed, no API key,
    or an unknown provider name. Callers treat this as "fall back to local
    behavior", never as "the request failed"."""


class ProviderAuthError(ProviderError):
    """Credentials were present but rejected."""


class ProviderTimeout(ProviderError):
    retryable = True


class ProviderRateLimited(ProviderError):
    retryable = True

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderRequestError(ProviderError):
    """A malformed request -- retrying the identical request will not help."""


class ProviderServerError(ProviderError):
    retryable = True


# --------------------------------------------------------------------------
# Wire types
# --------------------------------------------------------------------------
@dataclass
class ToolSpec:
    """A tool as described TO a model."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@dataclass
class ToolCall:
    """A model's request to run one tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolOutcome:
    """The result of running one tool, fed back to the model."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One conversation turn.

    `role` is "user" or "assistant". `content` is the plain text of the
    turn; `tool_calls` / `tool_outcomes` carry the structured parts. A
    provider is responsible for translating this into its own wire format.
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_outcomes: list[ToolOutcome] = field(default_factory=list)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: Iterable[ToolCall] = ()) -> "Message":
        return cls(role="assistant", content=content, tool_calls=list(tool_calls))

    @classmethod
    def tool_results(cls, outcomes: Iterable[ToolOutcome]) -> "Message":
        return cls(role="user", tool_outcomes=list(outcomes))


@dataclass
class Usage:
    """Token accounting for one model call.

    Every field is what the provider actually reported. A provider that
    reports nothing leaves these at zero and sets `reported=False`, so a
    consumer can tell "no tokens used" apart from "usage unknown".
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    reported: bool = True

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "reported": self.reported,
        }


@dataclass
class ModelResponse:
    """One completed model turn."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    model: str = ""
    provider: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    #: Milliseconds until the FIRST text token arrived. Only a streaming
    #: call can know this; None means the response was not streamed, which
    #: is deliberately distinct from "arrived instantly".
    first_event_ms: float | None = None
    estimated_cost_usd: float | None = None
    raw_stop_details: dict[str, Any] | None = None
    request_id: str = ""
    route_id: str = ""
    capability: str = "chat"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    def as_message(self) -> Message:
        return Message.assistant(self.text, self.tool_calls)


@runtime_checkable
class ModelProvider(Protocol):
    """The one interface every reasoning backend implements."""

    name: str

    def is_available(self) -> bool:
        """True when this provider can actually serve a request right now
        (SDK importable, credentials present). Never raises."""

    def describe(self) -> dict[str, Any]:
        """Log-safe description -- never includes credentials."""

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
        """Run exactly ONE model turn and return it.

        `effort` and `cache` are hints, not requirements: a provider whose
        API has no equivalent ignores them, and behaviour is unchanged.
        `on_text`, when given, is called with each chunk of PUBLIC assistant
        text as it is produced, so a caller can start speaking before
        generation finishes; a provider without streaming may ignore it and
        simply return the finished response.

        Deliberately single-turn: the agent loop lives in JARVIS
        (`brain/agent_loop.py`), not inside a provider or a vendor SDK
        helper, so swapping providers never changes agent behavior.
        """
