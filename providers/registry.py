"""Provider selection and the escalation ladder.

The ladder JARVIS follows for any single request is:

    deterministic local route      (brain/router.py -- no model at all)
      -> local reasoning model     (not implemented yet; slot reserved)
      -> a real cloud provider     (Anthropic today)

Only the last rung lives here. The important property is that adding a
local reasoning model later is a REGISTRATION, not a rewrite: implement
`providers.base.ModelProvider`, call `register_provider(...)`, and put
its name earlier in `JARVIS_PROVIDER_ORDER`.

There is deliberately no fake/stub entry in the production ladder. When
no provider is available, `get_agent_provider()` returns None and the
caller keeps working locally -- it never pretends a model answered.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from config import get_config
from providers.base import ModelProvider

log = logging.getLogger("jarvis.providers")

ProviderFactory = Callable[[], Any]

_FACTORIES: dict[str, ProviderFactory] = {}
_INSTANCES: dict[str, Any] = {}
_LOCK = threading.RLock()


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register (or replace) a provider factory under `name`."""
    with _LOCK:
        _FACTORIES[name.lower()] = factory
        _INSTANCES.pop(name.lower(), None)


def _default_factories() -> dict[str, ProviderFactory]:
    from providers.anthropic_provider import AnthropicProvider

    return {"anthropic": AnthropicProvider}


def _ensure_defaults() -> None:
    with _LOCK:
        for name, factory in _default_factories().items():
            _FACTORIES.setdefault(name, factory)


def get_provider(name: str) -> Any | None:
    """Return the cached instance of one named provider, or None when the
    name is unknown or the provider cannot be constructed."""
    _ensure_defaults()
    key = name.lower()
    with _LOCK:
        if key in _INSTANCES:
            return _INSTANCES[key]
        factory = _FACTORIES.get(key)
        if factory is None:
            return None
        try:
            instance = factory()
        except Exception:
            log.exception("Provider %r could not be constructed", key)
            return None
        _INSTANCES[key] = instance
        return instance


def available_providers() -> list[Any]:
    _ensure_defaults()
    found = []
    for name in get_config().provider_order:
        provider = get_provider(name)
        if provider is not None and provider.is_available():
            found.append(provider)
    return found


def get_agent_provider() -> Any | None:
    """The best available provider for agentic reasoning, or None.

    `None` is a normal, expected state -- it means "no API key
    configured", and every caller must degrade to local behavior rather
    than fail.
    """
    config = get_config()
    if not config.agent_enabled:
        return None
    for provider in available_providers():
        return provider
    return None


def agent_escalation_available() -> bool:
    """Can a request that the deterministic layer cannot resolve be handed
    to a real agent runtime right now?

    False whenever escalation is disabled or no provider is configured --
    the normal state until an API key is added -- so every deterministic
    local route behaves exactly as it always has, with or without Claude
    ("Claude is optional", docs/AGENT_ARCHITECTURE.md).

    Lives here rather than in `brain/agent.py` because `brain/router.py`
    needs the same answer and must not import the agent module (circular),
    and because two independently-maintained copies of this predicate is
    exactly how the router and the runtime would start to disagree about
    where a request is going.

    Never raises: a broken availability check must degrade to "no agent",
    never take down routing.
    """
    try:
        if not get_config().agent_escalation_enabled:
            return False
        return get_agent_provider() is not None
    except Exception:
        log.exception("Agent availability check failed; treating the agent as unavailable")
        return False


def agent_unavailable_reason() -> str | None:
    """Why the agent cannot be used right now, or None when it can.

    The reason already existed on the provider
    (`unavailable_reason() == "anthropic_sdk_not_installed"`) and was simply
    never surfaced anywhere a person would see it -- the live tray reported
    only "no agent provider is configured" while a valid key was loaded.
    Callers that degrade to a lesser path use this to say WHY they did.

    Log-safe: reason codes only, never the key.
    """
    try:
        config = get_config()
        if not config.agent_enabled:
            return "agent_disabled"
        if not config.agent_escalation_enabled:
            return "escalation_disabled"
        if get_agent_provider() is not None:
            return None
        reasons = []
        for name in config.provider_order:
            provider = get_provider(name)
            if provider is None:
                reasons.append(f"{name}=construction_failed")
                continue
            if not provider.is_available():
                reasons.append(f"{name}={provider.unavailable_reason()}")
        return ", ".join(reasons) or "no_providers_registered"
    except Exception as exc:
        log.exception("Could not determine why the agent is unavailable")
        return f"availability_check_failed:{type(exc).__name__}"


def provider_status() -> dict[str, Any]:
    """A log-safe report of every registered provider -- what is
    configured, what is usable, and why anything unusable is unusable."""
    _ensure_defaults()
    config = get_config()
    with _LOCK:
        names = sorted(_FACTORIES)
    entries = []
    for name in names:
        provider = get_provider(name)
        if provider is None:
            entries.append({"provider": name, "available": False, "unavailable_reason": "construction_failed"})
            continue
        entries.append(provider.describe())
    active = get_agent_provider()
    return {
        "agent_enabled": config.agent_enabled,
        "provider_order": list(config.provider_order),
        "active_provider": getattr(active, "name", None),
        "active_model": getattr(active, "model", None),
        "providers": entries,
    }


def reset_providers_for_tests() -> None:
    with _LOCK:
        _INSTANCES.clear()
        _FACTORIES.clear()


__all__ = [
    "ModelProvider",
    "register_provider",
    "get_provider",
    "get_agent_provider",
    "agent_escalation_available",
    "agent_unavailable_reason",
    "available_providers",
    "provider_status",
    "reset_providers_for_tests",
]
