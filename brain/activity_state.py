"""Process-wide "is JARVIS currently speaking" signal.

`brain/router.py`'s task-priority guard (stop/cancel/pause must outrank an
ambiguous media command while JARVIS is actively working OR speaking) needs
to know whether speech is in flight right now. That is voice-layer state,
and the brain layer must never import `voice/*` (the dependency runs the
other way throughout this codebase). This tiny module is the one narrow,
one-directional channel: `voice/background_assistant.py` writes it whenever
it enters/leaves the SPEAKING state, and `brain/task_supervisor.py` reads
it -- no import cycle, no new coupling beyond this one flag.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_speaking = False


def set_speaking(value: bool) -> None:
    global _speaking
    with _lock:
        _speaking = bool(value)


def is_speaking() -> bool:
    with _lock:
        return _speaking


__all__ = ["set_speaking", "is_speaking"]
