"""Per-process health tracking for voice providers (ElevenLabs STT/TTS
today; any future paid provider can reuse this the same way).

Before this module existed, a quota/insufficient-funds failure was
retried on every single command: each new interaction paid a fresh
connect/request attempt, got `quota_exceeded` back, and only THEN fell
back to Whisper/pyttsx3 -- repeated, logged, and slow on every command for
the rest of the session even though the outcome was already known.

The fix is a small, explicit health flag per provider, scoped to this
process/session (never persisted -- a fresh process always tries again):
a DEFINITE non-transient failure (quota exhausted, insufficient funds)
marks the provider unavailable immediately, logs the reason exactly once,
and every later caller's own `is_available()` check sees that and skips
straight to the configured fallback with no network attempt at all. A
transient failure (a dropped connection, a timeout, a 5xx) never marks
anything -- only the specific, known-non-transient error signatures below
do, so a real but temporary outage can still recover on its own the next
time it's tried.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("jarvis.provider_health")

#: Substrings that mean "this provider is out of quota/funds for the rest
#: of this session", drawn from ElevenLabs' own documented error
#: vocabulary (`quota_exceeded` is a real message_type/detail.status value
#: on both their realtime STT and TTS APIs) plus generic HTTP billing
#: signals. Matched against the STRINGIFIED exception, so it is provider-
#: agnostic and never needs to parse a specific response shape.
_NON_TRANSIENT_MARKERS = (
    "quota_exceeded",
    "quota exceeded",
    "insufficient_quota",
    "insufficient quota",
    "insufficient_funds",
    "insufficient funds",
    "payment required",
    "payment_required",
)


def is_non_transient_error(exc: BaseException) -> bool:
    """Is `exc` one of the known non-transient (quota/funds) failures, as
    opposed to an ordinary transient network/connect error?"""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _NON_TRANSIENT_MARKERS)


class ProviderHealth:
    """Tracks whether one named provider is known-dead for the rest of
    this process."""

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
        self._unavailable_reason: str | None = None
        self._marked_at: float | None = None

    @property
    def available(self) -> bool:
        with self._lock:
            return self._unavailable_reason is None

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._unavailable_reason

    def mark_unavailable(self, reason: str) -> None:
        with self._lock:
            already_marked = self._unavailable_reason is not None
            self._unavailable_reason = reason
            self._marked_at = time.monotonic()
        if not already_marked:
            log.warning(
                "[provider-health] %s marked unavailable for the rest of this session: %s "
                "(future requests use the configured fallback immediately; call "
                "voice.provider_health.reset(%r) to retry after this is resolved)",
                self.name, reason, self.name,
            )

    def note_result(self, exc: BaseException | None) -> None:
        """Record the outcome of one real attempt. Only a known
        non-transient error (see `is_non_transient_error`) marks the
        provider unavailable -- success or a transient failure never does,
        so a genuine temporary outage can still recover on its own."""
        if exc is not None and is_non_transient_error(exc):
            self.mark_unavailable(f"{type(exc).__name__}: {exc}"[:300])

    def reset(self) -> None:
        with self._lock:
            self._unavailable_reason = None
            self._marked_at = None


_REGISTRY: dict[str, ProviderHealth] = {}
_REGISTRY_LOCK = threading.Lock()


def get_provider_health(name: str) -> ProviderHealth:
    with _REGISTRY_LOCK:
        health = _REGISTRY.get(name)
        if health is None:
            health = ProviderHealth(name)
            _REGISTRY[name] = health
        return health


def reset(name: str) -> None:
    """Manual refresh for one provider (e.g. after topping up credits) --
    the next request tries it again instead of waiting out the session."""
    get_provider_health(name).reset()


def reset_all() -> None:
    with _REGISTRY_LOCK:
        healths = list(_REGISTRY.values())
    for health in healths:
        health.reset()


__all__ = [
    "ProviderHealth",
    "get_provider_health",
    "is_non_transient_error",
    "reset",
    "reset_all",
]
