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
    "available_providers",
    "provider_status",
    "reset_providers_for_tests",
]
