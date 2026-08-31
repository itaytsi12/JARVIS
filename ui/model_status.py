"""Which model modules this JARVIS install actually has.

The UI shows a ring of AI nodes. It must never claim a provider is active
that is not configured, so this module answers the question honestly, from
the same sources the runtime itself uses:

- **anthropic** -- the agent runtime's provider. Asked via
  `providers/registry.py`, the single answer to "is the agent reachable"
  (`agent_escalation_available` / `agent_unavailable_reason`), so the UI
  and the router can never disagree.
- **openai** -- the cloud planner (`brain/planner.py`), the gpt-5-mini
  intent classifier (`brain/intent_router.py`) and the web-answer service
  (`brain/web_answer.py`). Configured when `OPENAI_API_KEY` is present and
  the `openai` SDK imports.
- **vision** -- `vision/screen_analyzer.py`. A distinct module with its
  own model (`JARVIS_VISION_MODEL`), but it rides on the same OpenAI
  credential, so it is reported as its own node with that dependency
  stated rather than pretended away.
- **local** -- the local MiniLM intent classifier
  (`brain/local_intent_model.py`, an optional HTTP service on
  127.0.0.1:5050) and, once one has been trained, the promoted local
  coding model in `brain/learning_training.py`'s registry.
- **gemini** -- **not implemented anywhere in this repository.** It is
  listed so the node exists in the UI, and it reports `offline` with the
  reason `not_implemented`. It is deliberately never marked available.

Every probe here is cheap, offline and non-raising: this runs at UI
startup and on a slow refresh timer, never in a request path, and a
failing probe reports "unknown/offline", never an exception.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# `.env` is loaded exactly once, from the project root, by
# `config/settings.py`. Importing `config` HERE, at module scope, is what
# guarantees that has happened before any probe below reads the
# environment -- the same convention `brain/agent.py` and
# `vision/screen_analyzer.py` follow, and never a second `load_dotenv()`.
#
# Confirmed live that omitting it silently produced a WRONG report rather
# than an error: `MODEL_IDS` probes `openai` first, before anything else
# had imported `config`, so it read an unset `OPENAI_API_KEY` and reported
# `missing_api_key` -- while `vision`, probed after `_anthropic_status()`
# had imported `config` as a side effect, saw the very same key and
# reported available. Two nodes disagreeing about one credential.
import config  # noqa: F401  -- imported for its .env loading side effect

log = logging.getLogger("jarvis.ui")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Node ids, in the order the UI lays them out.
MODEL_IDS = ("openai", "gemini", "anthropic", "local", "vision")

STATE_OFFLINE = "offline"
STATE_IDLE = "idle"
STATE_THINKING = "thinking"
STATE_ACTIVE = "active"
STATE_ERROR = "error"
#: Distinct from `error` on purpose. A rate limit is not a broken module:
#: the credential is good, the request was well-formed, and the same call
#: will succeed shortly. Showing it as a red failure would train you to
#: ignore the red. It renders amber (`ui/qml/components/ModelNode.qml`).
STATE_RATE_LIMITED = "rate_limited"

VALID_STATES = (
    STATE_OFFLINE,
    STATE_IDLE,
    STATE_THINKING,
    STATE_ACTIVE,
    STATE_ERROR,
    STATE_RATE_LIMITED,
)

#: Exception class names that mean "slow down", not "broken". Both the
#: vendor SDK names (published by `config/events.py::model_activity`, which
#: sees the raw exception) and JARVIS's own neutral translation
#: (`providers/base.py::ProviderRateLimited`) appear here, because the
#: OpenAI call sites are bracketed by the generic context manager while
#: `providers/anthropic_provider.py` publishes its translated type.
RATE_LIMIT_ERROR_NAMES = frozenset(
    {"ProviderRateLimited", "RateLimitError", "TooManyRequests", "RateLimitExceeded"}
)


def state_for_error(error_name: str) -> str:
    """`error` or `rate_limited`, from the exception TYPE name alone.

    Never the message: an exception's text is the one place a credential
    could plausibly appear, and this value is displayed.
    """
    return STATE_RATE_LIMITED if error_name in RATE_LIMIT_ERROR_NAMES else STATE_ERROR


@dataclass
class ModelStatus:
    """One node's real, log-safe status. Never carries a credential."""

    model_id: str
    label: str
    available: bool
    #: A short machine-readable reason when `available` is False.
    reason: str | None = None
    #: The concrete model identifier in use, when there is one.
    model_name: str = ""
    #: What this module is used for, shown as the node's tooltip.
    role: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "label": self.label,
            "available": self.available,
            "reason": self.reason or "",
            "modelName": self.model_name,
            "role": self.role,
        }


def _openai_reason() -> str | None:
    """None when the OpenAI path is usable, else why it is not."""
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        return "missing_api_key"
    try:
        import openai  # noqa: F401
    except Exception:
        return "openai_sdk_not_installed"
    return None


def _anthropic_status() -> ModelStatus:
    model_name = ""
    reason: str | None = "unavailable"
    try:
        from config import get_config
        from providers.registry import agent_unavailable_reason, get_agent_provider

        model_name = get_config().agent_model
        provider = get_agent_provider()
        if provider is not None:
            return ModelStatus(
                "anthropic",
                "Anthropic",
                True,
                None,
                getattr(provider, "model", model_name) or model_name,
                "Agent runtime -- plan/act/observe loop",
            )
        reason = agent_unavailable_reason() or "unavailable"
    except Exception as exc:
        log.debug("Anthropic status probe failed: %s", type(exc).__name__)
        reason = f"probe_failed:{type(exc).__name__}"
    return ModelStatus("anthropic", "Anthropic", False, reason, model_name, "Agent runtime -- plan/act/observe loop")


def _openai_status() -> ModelStatus:
    reason = _openai_reason()
    return ModelStatus(
        "openai",
        "OpenAI",
        reason is None,
        reason,
        os.getenv("JARVIS_WEB_ANSWER_MODEL", "gpt-5-mini"),
        "Cloud planner, intent classifier, web answers",
    )


def _vision_status() -> ModelStatus:
    reason = _openai_reason()
    if reason == "missing_api_key":
        reason = "missing_openai_api_key"
    return ModelStatus(
        "vision",
        "Vision",
        reason is None,
        reason,
        os.getenv("JARVIS_VISION_MODEL", "gpt-5-mini"),
        "Screen analysis (vision/screen_analyzer.py)",
    )


def _intent_service_reachable(timeout: float = 0.35) -> bool:
    """Is the optional local MiniLM classifier service answering?

    Uses its real `/health` endpoint (`training/intent_service.py`). A
    refused connection is the normal state when the service is not
    running and is not an error.
    """
    from urllib.request import urlopen

    try:
        with urlopen("http://127.0.0.1:5050/health", timeout=timeout) as response:
            return 200 <= getattr(response, "status", 200) < 300
    except Exception:
        return False


def _local_status() -> ModelStatus:
    """The local model node: the MiniLM intent service, or a promoted
    locally-trained coding model, whichever is genuinely present."""
    if _intent_service_reachable():
        return ModelStatus(
            "local",
            "Local LLM",
            True,
            None,
            "minilm-intent-classifier",
            "Local intent classification (127.0.0.1:5050)",
        )

    # No live service. A locally TRAINED and promoted model still counts as
    # a real local model module -- ask the registry that owns that fact.
    try:
        from brain.learning_training import get_model_registry

        active = get_model_registry().get_active()
        if active is not None:
            name = getattr(active, "model_version", "") or "local-model"
            base = getattr(active, "base_model", None) or ""
            return ModelStatus(
                "local",
                "Local LLM",
                True,
                None,
                str(name),
                f"Locally trained coding model{f' ({base})' if base else ''}",
            )
    except Exception as exc:
        log.debug("Local model registry probe failed: %s", type(exc).__name__)

    classifier = PROJECT_ROOT / "models" / "intent_classifier" / "classifier.joblib"
    reason = "service_not_running" if classifier.is_file() else "not_installed"
    return ModelStatus("local", "Local LLM", False, reason, "", "Local intent classification (127.0.0.1:5050)")


def _gemini_status() -> ModelStatus:
    """Gemini is not implemented in this repository.

    The node exists so the layout the UI was designed around is complete,
    but it is never reported as available -- claiming otherwise would be
    exactly the "pretend a provider is active" failure this module exists
    to prevent. Wiring a real Gemini provider is a registration in
    `providers/registry.py`; this function would then ask the registry the
    same way `_anthropic_status` does.
    """
    return ModelStatus("gemini", "Gemini", False, "not_implemented", "", "No Gemini provider is implemented")


_PROBES = {
    "anthropic": _anthropic_status,
    "openai": _openai_status,
    "vision": _vision_status,
    "local": _local_status,
    "gemini": _gemini_status,
}


def discover_models() -> list[ModelStatus]:
    """Every model node, in display order, with its real availability."""
    found: list[ModelStatus] = []
    for model_id in MODEL_IDS:
        probe = _PROBES[model_id]
        try:
            found.append(probe())
        except Exception as exc:
            log.warning("Model status probe %r failed: %s", model_id, type(exc).__name__)
            found.append(ModelStatus(model_id, model_id.title(), False, f"probe_failed:{type(exc).__name__}"))
    return found


def online_count(statuses: list[ModelStatus] | None = None) -> int:
    return sum(1 for item in (statuses if statuses is not None else discover_models()) if item.available)


def online_caption(count: int) -> str:
    """"1 MODEL ONLINE" / "3 MODELS ONLINE" -- never a hard-coded 5."""
    return f"{count} MODEL{'' if count == 1 else 'S'} ONLINE"


__all__ = [
    "MODEL_IDS",
    "VALID_STATES",
    "STATE_OFFLINE",
    "STATE_IDLE",
    "STATE_THINKING",
    "STATE_ACTIVE",
    "STATE_ERROR",
    "STATE_RATE_LIMITED",
    "RATE_LIMIT_ERROR_NAMES",
    "state_for_error",
    "ModelStatus",
    "discover_models",
    "online_count",
    "online_caption",
]
