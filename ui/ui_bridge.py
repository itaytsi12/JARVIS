"""The one object QML talks to, and the only place backend state becomes
UI state.

Threading contract
------------------
Everything that publishes into this bridge runs on some other thread: the
audio thread (`voice/background_assistant.py`), an agent worker
(`tasks/manager.py`), a provider call inside `brain/agent_loop.py`. Qt
requires every property write and every signal emission that QML is bound
to happen on the GUI thread.

So every public setter does exactly one thing off-thread: emit
`_invoke`, a `Signal(object)` carrying a plain callable, which is
connected to `_apply` with `Qt.QueuedConnection`. Qt then runs that
callable on the GUI thread's event loop. Nothing here blocks a caller,
nothing sleeps, and no backend thread ever touches a QML-visible property
directly.

The bridge never calls into the backend. It subscribes to
`config/events.py` and renders what it is told, so it cannot duplicate,
re-order or interfere with real request logic.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from PySide6.QtCore import Property, QObject, Qt, QTimer, Signal, Slot

from config import events
from ui.model_status import (
    MODEL_IDS,
    STATE_ACTIVE,
    STATE_ERROR,
    STATE_IDLE,
    STATE_OFFLINE,
    STATE_RATE_LIMITED,
    STATE_THINKING,
    VALID_STATES,
    ModelStatus,
    discover_models,
    online_caption,
    state_for_error,
)

log = logging.getLogger("jarvis.ui")

#: How long an "active" flash lasts before the node settles back to idle.
ACTIVE_FLASH_MS = 900
#: How long an error stays visible on a node before it settles back.
ERROR_HOLD_MS = 4000
#: A rate limit is held longer than an ordinary error: it describes a
#: condition that is still true for a while (the window has to pass), not
#: a single failed call.
RATE_LIMIT_HOLD_MS = 20_000
#: Availability re-probe interval. This is genuine polling, and it is here
#: because availability can change with no event to observe: the optional
#: local intent service on 127.0.0.1:5050 may be started or stopped by
#: hand at any time. The probe itself runs on a worker thread (it does a
#: short localhost request) and only its RESULT is marshalled back, so the
#: GUI thread is never blocked by it.
AVAILABILITY_REFRESH_MS = 15_000

#: The vendor-neutral lifecycle events (`config/events.py`) this window
#: knows how to draw, mapped to node states. Resolved with `getattr` at
#: import: that family is owned by the provider/router work, and this UI
#: must keep working whether or not a given name exists in it yet.
NEUTRAL_EVENT_STATES = {
    name: state
    for name, state in (
        (getattr(events, "MODEL_THINKING", None), STATE_THINKING),
        (getattr(events, "MODEL_ACTIVE", None), STATE_ACTIVE),
        (getattr(events, "MODEL_ERROR", None), STATE_ERROR),
        (getattr(events, "MODEL_RATE_LIMITED", None), STATE_RATE_LIMITED),
    )
    if name
}

#: The states the voice layer maps onto the three UI flags.
_LISTENING_STATES = {"LISTENING", "WAKE_DETECTED", "INTERRUPTED_LISTENING", "WAITING_FOR_LEARNING_APPROVAL"}
_PROCESSING_STATES = {"PROCESSING", "EXECUTING"}
_SPEAKING_STATES = {"SPEAKING"}


class UiBridge(QObject):
    """Backend state, exposed to QML as properties and signals."""

    modelsChanged = Signal()
    statusTextChanged = Signal()
    listeningChanged = Signal()
    speakingChanged = Signal()
    processingChanged = Signal()
    userTextChanged = Signal()
    jarvisTextChanged = Signal()
    readyChanged = Signal()
    subtitleChanged = Signal()

    #: Internal GUI-thread marshalling channel. Not for QML.
    _invoke = Signal(object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._status_text = "INITIALIZING"
        # NOT `online_caption(0)`: the first probe runs on a worker
        # thread and takes a few seconds in a cold process (the vendor
        # SDK imports dominate it). Showing "0 MODELS ONLINE" in the
        # meantime would be a wrong statement rather than an absent one.
        self._subtitle = "DETECTING MODULES"
        self._listening = False
        self._speaking = False
        self._processing = False
        self._user_text = ""
        self._jarvis_text = ""
        self._ready = False
        self._models: dict[str, dict[str, Any]] = {
            model_id: {
                "id": model_id,
                "label": model_id.title(),
                "available": False,
                "state": STATE_OFFLINE,
                "reason": "",
                "modelName": "",
                "role": "",
            }
            for model_id in MODEL_IDS
        }
        #: Model ids with a request in flight, and whether the voice layer
        #: says it is busy. `processing` is true when EITHER is -- tracked
        #: separately so a finished model call cannot clear a busy state the
        #: assistant set, or vice versa.
        self._in_flight: set[str] = set()
        self._assistant_busy = False
        self._unsubscribe: Callable[[], None] | None = None
        self._refresh_timer: QTimer | None = None
        self._refresh_running = threading.Event()

        self._invoke.connect(self._apply, Qt.QueuedConnection)

    # -- GUI-thread marshalling ------------------------------------------
    @Slot(object)
    def _apply(self, action) -> None:
        """Run one marshalled callable on the GUI thread. Never raises:
        a display bug must not tear down the event loop."""
        try:
            action()
        except Exception:
            log.exception("UI update failed")

    def _on_gui_thread(self, action: Callable[[], None]) -> None:
        self._invoke.emit(action)

    def run_on_gui_thread(self, action: Callable[[], None]) -> None:
        """Public form of the marshalling above, for a caller outside the
        bridge that must touch Qt from another thread -- today
        `startup/launcher.py`, which closes the window when the tray's
        Exit item is chosen on the tray's own thread."""
        self._on_gui_thread(action)

    # -- properties -------------------------------------------------------
    @Property("QVariantList", notify=modelsChanged)
    def models(self) -> list:
        return [dict(entry) for entry in self._models.values()]

    @Property(str, notify=statusTextChanged)
    def statusText(self) -> str:
        return self._status_text

    @Property(str, notify=subtitleChanged)
    def subtitle(self) -> str:
        return self._subtitle

    @Property(bool, notify=listeningChanged)
    def listening(self) -> bool:
        return self._listening

    @Property(bool, notify=speakingChanged)
    def speaking(self) -> bool:
        return self._speaking

    @Property(bool, notify=processingChanged)
    def processing(self) -> bool:
        return self._processing

    @Property(str, notify=userTextChanged)
    def userText(self) -> str:
        return self._user_text

    @Property(str, notify=jarvisTextChanged)
    def jarvisText(self) -> str:
        return self._jarvis_text

    @Property(bool, notify=readyChanged)
    def ready(self) -> bool:
        return self._ready

    # -- public API (safe to call from ANY thread) ------------------------
    def set_model_state(self, model_id: str, state: str) -> None:
        """`set_model_state("anthropic", "thinking")`.

        An unknown model id or state is logged and ignored rather than
        raising into whatever backend thread called it.
        """
        if model_id not in self._models:
            log.warning("Unknown model id for UI state update: %r", model_id)
            return
        if state not in VALID_STATES:
            log.warning("Unknown UI model state %r for %r", state, model_id)
            return
        self._on_gui_thread(lambda: self._set_model_state_now(model_id, state))

    def set_model_enabled(self, model_id: str, enabled: bool, reason: str = "") -> None:
        """Mark a node available or not. An unavailable node is forced to
        `offline`, so it can never sit lit while nothing backs it."""
        if model_id not in self._models:
            log.warning("Unknown model id for UI enablement: %r", model_id)
            return
        self._on_gui_thread(lambda: self._set_model_enabled_now(model_id, bool(enabled), reason))

    def set_model_label(self, model_id: str, label: str) -> None:
        if model_id not in self._models:
            log.warning("Unknown model id for UI label: %r", model_id)
            return
        self._on_gui_thread(lambda: self._update_model(model_id, label=str(label)))

    def set_status_text(self, text: str) -> None:
        self._on_gui_thread(lambda: self._set_scalar("_status_text", str(text), self.statusTextChanged))

    def set_subtitle(self, text: str) -> None:
        self._on_gui_thread(lambda: self._set_scalar("_subtitle", str(text), self.subtitleChanged))

    def set_listening(self, value: bool) -> None:
        self._on_gui_thread(lambda: self._set_scalar("_listening", bool(value), self.listeningChanged))

    def set_speaking(self, value: bool) -> None:
        self._on_gui_thread(lambda: self._set_scalar("_speaking", bool(value), self.speakingChanged))

    def set_processing(self, value: bool) -> None:
        self._on_gui_thread(lambda: self._set_scalar("_processing", bool(value), self.processingChanged))

    def set_user_text(self, text: str) -> None:
        self._on_gui_thread(lambda: self._set_scalar("_user_text", str(text), self.userTextChanged))

    def set_jarvis_text(self, text: str) -> None:
        self._on_gui_thread(lambda: self._set_scalar("_jarvis_text", str(text), self.jarvisTextChanged))

    def set_ready(self, value: bool) -> None:
        self._on_gui_thread(lambda: self._set_scalar("_ready", bool(value), self.readyChanged))

    # -- GUI-thread mutators ----------------------------------------------
    def _set_scalar(self, attribute: str, value: Any, signal: Signal) -> None:
        if getattr(self, attribute) == value:
            return
        setattr(self, attribute, value)
        signal.emit()

    def _update_model(self, model_id: str, **changes: Any) -> None:
        entry = self._models[model_id]
        if all(entry.get(key) == value for key, value in changes.items()):
            return
        entry.update(changes)
        self.modelsChanged.emit()

    def _note_in_flight(self, model_id: str, active: bool) -> None:
        if active:
            self._in_flight.add(model_id)
        else:
            self._in_flight.discard(model_id)
        self._set_scalar("_processing", bool(self._in_flight or self._assistant_busy), self.processingChanged)

    def _set_model_state_now(self, model_id: str, state: str) -> None:
        entry = self._models[model_id]
        if not entry["available"]:
            # An unconfigured module never lights up -- not even red. A
            # module that is not installed failing to answer is its normal
            # state, not an error worth alarming about; the optional local
            # intent service on 127.0.0.1:5050 reports exactly that on
            # every command when it is not running. A module that IS
            # configured and then fails still shows the error.
            state = STATE_OFFLINE
        self._update_model(model_id, state=state)
        if state == STATE_ACTIVE:
            QTimer.singleShot(ACTIVE_FLASH_MS, lambda: self._settle(model_id, STATE_ACTIVE))
        elif state == STATE_ERROR:
            QTimer.singleShot(ERROR_HOLD_MS, lambda: self._settle(model_id, STATE_ERROR))
        elif state == STATE_RATE_LIMITED:
            QTimer.singleShot(RATE_LIMIT_HOLD_MS, lambda: self._settle(model_id, STATE_RATE_LIMITED))

    def _settle(self, model_id: str, only_if: str) -> None:
        """Return a node to its resting state, but only if it is still in
        the transient state that scheduled this -- a request that started
        in the meantime must not be cancelled by an older timer."""
        entry = self._models.get(model_id)
        if entry is None or entry["state"] != only_if:
            return
        self._update_model(model_id, state=STATE_IDLE if entry["available"] else STATE_OFFLINE)

    def _set_model_enabled_now(self, model_id: str, enabled: bool, reason: str) -> None:
        entry = self._models[model_id]
        state = entry["state"]
        if not enabled:
            state = STATE_OFFLINE
        elif state == STATE_OFFLINE:
            state = STATE_IDLE
        self._update_model(model_id, available=enabled, state=state, reason=reason)
        self._refresh_subtitle()

    def _refresh_subtitle(self) -> None:
        count = sum(1 for entry in self._models.values() if entry["available"])
        self._set_scalar("_subtitle", online_caption(count), self.subtitleChanged)

    # -- backend wiring ----------------------------------------------------
    def apply_statuses(self, statuses: list[ModelStatus]) -> None:
        """Adopt a freshly discovered set of model statuses."""

        def work() -> None:
            for status in statuses:
                if status.model_id not in self._models:
                    continue
                entry = self._models[status.model_id]
                state = entry["state"]
                if not status.available:
                    state = STATE_OFFLINE
                elif state == STATE_OFFLINE:
                    state = STATE_IDLE
                self._update_model(
                    status.model_id,
                    label=status.label,
                    available=status.available,
                    reason=status.reason or "",
                    modelName=status.model_name,
                    role=status.role,
                    state=state,
                )
            self._refresh_subtitle()

        self._on_gui_thread(work)

    def start(self) -> None:
        """Subscribe to the runtime event bus and begin refreshing
        availability. Call once, from the GUI thread."""
        if self._unsubscribe is None:
            self._unsubscribe = events.subscribe(None, self._on_event)
        self.refresh_availability()
        if self._refresh_timer is None:
            self._refresh_timer = QTimer(self)
            self._refresh_timer.setInterval(AVAILABILITY_REFRESH_MS)
            self._refresh_timer.timeout.connect(self.refresh_availability)
            self._refresh_timer.start()

    def stop(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
            self._refresh_timer = None
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    @Slot()
    def refresh_availability(self) -> None:
        """Re-probe every model module on a worker thread.

        `discover_models()` makes a short localhost request, so it must
        not run on the GUI thread. Only one refresh is ever in flight.
        """
        if self._refresh_running.is_set():
            return
        self._refresh_running.set()

        def work() -> None:
            try:
                self.apply_statuses(discover_models())
            except Exception:
                log.exception("Model availability refresh failed")
            finally:
                self._refresh_running.clear()

        threading.Thread(target=work, name="jarvis-ui-model-probe", daemon=True).start()

    # -- event bus ---------------------------------------------------------
    def _model_id_from(self, payload: dict) -> str:
        """The node id a payload refers to, or "" if it names none of ours.

        `config/events.py` also declares a vendor-neutral, multi-provider
        lifecycle event family (`model_thinking`, `model_active`,
        `model_rate_limited`, ...) owned by the provider/router work. Its
        payload key is not fixed here, so several plausible ones are
        accepted and anything that does not name a node this UI actually
        draws is ignored quietly -- a new provider or capability appearing
        on that bus must never make the window log a warning per request.
        """
        for key in ("model", "model_id", "provider", "capability", "node"):
            candidate = str(payload.get(key) or "")
            if candidate in self._models:
                return candidate
        return ""

    def _on_neutral_event(self, event: str, payload: dict) -> bool:
        """Handle the vendor-neutral lifecycle family. Returns whether the
        event belonged to it (whether or not a node matched)."""
        state = NEUTRAL_EVENT_STATES.get(event)
        if state is None:
            return False
        model = self._model_id_from(payload)
        if not model:
            log.debug("Ignoring %s for an unknown node: %r", event, payload)
            return True
        self.set_model_state(model, state)
        self._on_gui_thread(lambda: self._note_in_flight(model, state == STATE_THINKING))
        return True

    def _on_event(self, event: str, payload: dict) -> None:
        """Called from whatever thread published. Only ever schedules."""
        if self._on_neutral_event(event, payload):
            return
        if event == events.MODEL_REQUEST_STARTED:
            model = str(payload.get("model") or "")
            self.set_model_state(model, STATE_THINKING)
            self._on_gui_thread(lambda: self._note_in_flight(model, True))
        elif event == events.MODEL_REQUEST_SUCCEEDED:
            model = str(payload.get("model") or "")
            self.set_model_state(model, STATE_ACTIVE)
            self._on_gui_thread(lambda: self._note_in_flight(model, False))
        elif event == events.MODEL_REQUEST_FAILED:
            model = str(payload.get("model") or "")
            # A rate limit gets its own amber state rather than the red
            # one: the module is configured and working, it is being
            # throttled. Decided from the exception TYPE name only --
            # `state_for_error` never looks at a message.
            self.set_model_state(model, state_for_error(str(payload.get("error") or "")))
            self._on_gui_thread(lambda: self._note_in_flight(model, False))
        elif event == events.ASSISTANT_STATE:
            self._on_assistant_state(str(payload.get("state") or ""), str(payload.get("detail") or ""))
        elif event == events.USER_TEXT:
            self.set_user_text(str(payload.get("text") or ""))
        elif event == events.JARVIS_TEXT:
            self.set_jarvis_text(str(payload.get("text") or ""))
        elif event == events.STATUS_TEXT:
            self.set_status_text(str(payload.get("text") or ""))

    def _on_assistant_state(self, state: str, detail: str) -> None:
        self.set_listening(state in _LISTENING_STATES)
        self.set_speaking(state in _SPEAKING_STATES)

        def note_busy() -> None:
            self._assistant_busy = state in _PROCESSING_STATES
            self._set_scalar("_processing", bool(self._in_flight or self._assistant_busy), self.processingChanged)

        self._on_gui_thread(note_busy)
        self.set_status_text(detail or state or "IDLE")
        if state:
            self.set_ready(True)


__all__ = [
    "UiBridge",
    "ACTIVE_FLASH_MS",
    "ERROR_HOLD_MS",
    "RATE_LIMIT_HOLD_MS",
    "AVAILABILITY_REFRESH_MS",
]
