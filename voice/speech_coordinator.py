"""Prioritizes and preempts every call into `voice.text_to_speech`.

JARVIS speaks from several independent threads: an immediate command
acknowledgement, agent progress narration, the final streamed answer, and
the read-only question path. They already serialize on
`voice.text_to_speech`'s `speak_response` resource lock
(`brain/resource_locks.py`), which guarantees exactly one of them is ever
actually producing audio -- but nothing made a HIGHER-priority utterance
(the final answer) interrupt an already-in-flight LOWER-priority one (a
stale progress phrase). A live follow-up could sit behind a slow or
network-stuck ElevenLabs call all the way out to that lock's own 30-second
timeout (`TimeoutError: resource_timeout:speaker`) for no reason -- the
progress phrase was already stale, and the answer should have simply
interrupted it.

`SpeechCoordinator.speak()` fixes that with one rule: before it blocks on
`voice.text_to_speech.speak`, it stops whatever equal-or-lower priority
utterance is CURRENTLY in flight, so the resource lock is released almost
immediately (barge-in already relies on `stop()` doing exactly this)
instead of being waited out. It never changes the "exactly one owner plays
audio at a time" guarantee -- that is still the resource lock's job -- it
only shortens how long a stale lower-priority utterance is allowed to hold
it once something more important wants to speak.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

log = logging.getLogger("jarvis.speech_coordinator")

#: Acknowledgements, agent progress narration, heartbeats, and other
#: non-final status speech.
PRIORITY_STATUS = 0

#: The final answer for a command -- a narrated agent answer, a
#: non-narrated command result, or a read-only question's answer. Always
#: preempts an in-flight PRIORITY_STATUS utterance.
PRIORITY_FINAL = 1


class SpeechCoordinator:
    """`speak_module`/`stop_module` are typically the SAME module object
    (`voice.text_to_speech`); kept as two parameters only so tests can
    inject fakes independently. Deliberately NOT captured as bound
    `speak`/`stop` callables at construction time: this coordinator is a
    long-lived singleton (`get_speech_coordinator()`), and every other
    call site in this codebase re-resolves `voice.text_to_speech.speak`/
    `.stop` fresh on every call (`from .text_to_speech import speak`
    inside the function that uses it, not at module import time) so that
    `unittest.mock.patch.object(text_to_speech, "speak", ...)` -- which
    replaces the ATTRIBUTE on the module -- is honored no matter when the
    patch is applied relative to when this object was built. Binding fixed
    function references here instead would silently keep calling whatever
    was live the FIRST time this singleton's `speak()` ran, for the rest
    of the process -- including, in tests, a `Mock` from a completely
    unrelated, already-finished test.
    """

    def __init__(self, speak_module, stop_module=None):
        self._speak_module = speak_module
        self._stop_module = stop_module if stop_module is not None else speak_module
        self._lock = threading.Lock()
        #: Priorities of every call currently in flight. A list (not a
        #: single value) because more than one caller can be mid-call at
        #: once -- e.g. one just preempted and is waiting for the resource
        #: lock while the one it preempted is still unwinding.
        self._active: list[int] = []

    def speak(self, text: str, lang: str | None = None, priority: int = PRIORITY_STATUS, **kwargs) -> dict:
        with self._lock:
            current_max = max(self._active) if self._active else None
            # Strictly greater, not >=: two equal-priority utterances (two
            # ordinary acks, an ack followed by its own ordinary result,
            # two progress phrases) are meant to simply queue for the
            # resource lock and both play in full -- preemption is only for
            # a genuinely HIGHER-priority utterance (the narrated final
            # answer over a stale progress phrase) cutting off something
            # less important, never same-tier utterances cutting off each
            # other's siblings.
            preempt = current_max is not None and priority > current_max
            self._active.append(priority)
        if preempt:
            try:
                self._stop_module.stop()
            except Exception:
                log.exception("Failed to preempt in-flight speech for a higher-priority utterance")
        try:
            return self._speak_module.speak(text, lang=lang, **kwargs)
        finally:
            with self._lock:
                self._active.remove(priority)


_coordinator: SpeechCoordinator | None = None
_coordinator_lock = threading.Lock()


def get_speech_coordinator() -> SpeechCoordinator:
    global _coordinator
    if _coordinator is None:
        with _coordinator_lock:
            if _coordinator is None:
                from voice import text_to_speech
                _coordinator = SpeechCoordinator(text_to_speech)
    return _coordinator


__all__ = ["SpeechCoordinator", "get_speech_coordinator", "PRIORITY_STATUS", "PRIORITY_FINAL"]
