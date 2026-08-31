"""Process-wide runtime event bus.

Why here, and why it imports nothing
------------------------------------
Several layers need to *report* what they are doing so an observer (today:
the Qt UI in `ui/`; tomorrow: anything else) can display it:

- `providers/anthropic_provider.py` -- a real Claude request started/ended.
- `brain/planner.py`, `brain/intent_router.py`, `brain/web_answer.py` --
  the OpenAI cloud calls.
- `vision/screen_analyzer.py` -- the vision model call.
- `voice/background_assistant.py` -- assistant state, transcripts, replies.

Those layers sit at very different heights in this codebase's dependency
graph (`brain` imports `providers`, everything imports `config`), so the
bus has to live *below* all of them or publishing would create an import
cycle. `config` is the one package every layer already depends on, and
this module deliberately imports nothing from the project at all -- the
same "one narrow, one-directional channel" reasoning that produced
`brain/activity_state.py`, generalized from a single boolean to named
events.

Rules this module follows
-------------------------
- **A subscriber must never be able to break a publisher.** Every
  callback is invoked inside its own try/except; an exception is logged
  and swallowed. A UI that crashes must not take down a model call.
- **Publishing is cheap and never blocks.** With no subscribers (the
  normal state for the CLI, the tests, and `--no-ui` runs) `publish` does
  a dict lookup on an empty list and returns.
- **Payloads carry no secrets.** Callers pass identifiers, states and
  short user-facing text only -- never keys, tokens or cookies. Nothing
  here logs the payload.
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable

log = logging.getLogger("jarvis.events")

Subscriber = Callable[[str, dict], None]

# -- event names ----------------------------------------------------------
#: A model module began a request. payload: {"model": <id>}
MODEL_REQUEST_STARTED = "model.request.started"
#: A model module finished successfully. payload: {"model": <id>}
MODEL_REQUEST_SUCCEEDED = "model.request.succeeded"
#: A model module failed. payload: {"model": <id>, "error": <type name>}
MODEL_REQUEST_FAILED = "model.request.failed"
#: The set of configured model modules changed / was (re)computed.
MODEL_AVAILABILITY = "model.availability"

#: The voice assistant changed state. payload: {"state": str, "detail": str}
ASSISTANT_STATE = "assistant.state"
#: A recognized user utterance. payload: {"text": str}
USER_TEXT = "assistant.user_text"
#: Something JARVIS is about to say. payload: {"text": str}
JARVIS_TEXT = "assistant.jarvis_text"
#: A free-form startup/status line for the UI. payload: {"text": str}
STATUS_TEXT = "runtime.status"

# Vendor-neutral multi-model lifecycle events. Payloads are request-scoped
# and safe for a UI subscriber; no credentials or message contents.
REQUEST_STARTED = "request_started"
ROUTE_SELECTED = "route_selected"
MODEL_THINKING = "model_thinking"
FALLBACK_STARTED = "fallback_started"
MODEL_ACTIVE = "model_active"
MODEL_ERROR = "model_error"
MODEL_RATE_LIMITED = "model_rate_limited"
REQUEST_COMPLETED = "request_completed"


_LOCK = threading.RLock()
_SUBSCRIBERS: dict[str, list[Subscriber]] = {}
#: Subscribers that receive every event, whatever its name.
_WILDCARD: list[Subscriber] = []


def subscribe(event: str | None, callback: Subscriber) -> Callable[[], None]:
    """Register `callback` for `event` (or for every event when `event` is
    None). Returns a callable that unsubscribes it again."""
    with _LOCK:
        target = _WILDCARD if event is None else _SUBSCRIBERS.setdefault(event, [])
        target.append(callback)

    def unsubscribe() -> None:
        with _LOCK:
            listeners = _WILDCARD if event is None else _SUBSCRIBERS.get(event, [])
            if callback in listeners:
                listeners.remove(callback)

    return unsubscribe


def publish(event: str, **payload: Any) -> None:
    """Deliver `event` to every subscriber. Never raises."""
    with _LOCK:
        listeners = list(_SUBSCRIBERS.get(event, ())) + list(_WILDCARD)
    if not listeners:
        return
    for callback in listeners:
        try:
            callback(event, payload)
        except Exception:
            # A display failure must never propagate into the model call,
            # the voice loop or the agent runtime that published this.
            log.exception("Runtime event subscriber failed for %s", event)


@contextmanager
def model_activity(model: str):
    """Bracket one model call with started/succeeded/failed events.

    Used at the real call sites (`brain/planner.py`, `brain/web_answer.py`,
    `vision/screen_analyzer.py`, ...) so the UI reflects genuine work
    rather than a second, parallel notion of what the model is doing. The
    request logic itself is untouched -- this only observes it.
    """
    publish(MODEL_REQUEST_STARTED, model=model)
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 -- re-raised untouched below
        publish(MODEL_REQUEST_FAILED, model=model, error=type(exc).__name__)
        raise
    publish(MODEL_REQUEST_SUCCEEDED, model=model)


def reset_for_tests() -> None:
    with _LOCK:
        _SUBSCRIBERS.clear()
        _WILDCARD.clear()


__all__ = [
    "MODEL_REQUEST_STARTED",
    "MODEL_REQUEST_SUCCEEDED",
    "MODEL_REQUEST_FAILED",
    "MODEL_AVAILABILITY",
    "ASSISTANT_STATE",
    "USER_TEXT",
    "JARVIS_TEXT",
    "STATUS_TEXT",
    "REQUEST_STARTED", "ROUTE_SELECTED", "MODEL_THINKING",
    "FALLBACK_STARTED", "MODEL_ACTIVE", "MODEL_ERROR",
    "MODEL_RATE_LIMITED", "REQUEST_COMPLETED",
    "subscribe",
    "publish",
    "model_activity",
    "reset_for_tests",
]
