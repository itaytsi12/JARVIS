"""Speak while the agent works, without ever inventing progress.

A long agent task used to be silent from the acknowledgement until the
finished answer -- a minute or more of nothing, which reads as "it broke".
This module fills that silence, under strict rules:

- **Everything spoken is derived from real state.** Progress lines come from
  `brain/agent_loop.py`'s `tool_started` / `tool_result` callbacks, so a line
  is only ever said because a tool actually ran. The heartbeat that fills a
  long gap is timed, but WHAT it says is not: while a tool is in flight it
  says the tool is still running, and once every tool has returned it says
  the model is working from what they found. Both are true at the moment
  they are said, and neither describes reasoning.
- **Never internal reasoning.** The only two things that reach speech are a
  fixed status phrase chosen by tool CATEGORY, and the model's public final
  answer. Thinking blocks are not subscribed to anywhere in the stack, and
  `providers/anthropic_provider.py` stops forwarding text the moment a
  `tool_use` block starts, so a tool payload can never arrive here either.
- **Rate limited and deduplicated.** At most one line per
  `MIN_INTERVAL_SECONDS`, and never the same line twice in a row -- narrating
  every tool call would be worse than silence.
- **The answer always wins.** As soon as final text starts arriving, queued
  progress is dropped: "I'm still checking" must never be spoken after the
  answer has begun.

Speech is dispatched through an injected callable and runs on its own
thread, so the agent never waits for the speaker.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from voice.sentence_stream import SentenceStream

log = logging.getLogger("jarvis.narration")

#: Roughly one meaningful update per this many seconds. Below this the
#: assistant sounds like it is narrating itself; far above it and the silence
#: is back.
MIN_INTERVAL_SECONDS = 5.0

#: How long a task must already have been running before the FIRST progress
#: line. A task that finishes in three seconds should just answer.
FIRST_UPDATE_AFTER_SECONDS = 4.0

#: Priorities. A higher number wins; when the final answer starts, anything
#: queued below it is discarded.
PRIORITY_STATUS = 0
PRIORITY_PROGRESS = 1
PRIORITY_ANSWER = 2

#: Tool -> what is truthfully happening. Keyed by tool name so a phrase can
#: never describe something the agent did not do. Anything absent produces NO
#: speech at all rather than a vague guess.
_PHRASES: dict[str, str] = {
    "list_files": "I'm looking through the files, sir.",
    "find_file": "I'm looking through the files, sir.",
    "exists": "I'm checking the files, sir.",
    "read_text_file": "I'm reading the relevant files, sir.",
    "read_code": "I'm reading the relevant code, sir.",
    "search_text": "I'm searching through the files, sir.",
    "search_code": "I'm searching the code, sir.",
    "inspect_project": "I'm inspecting the project structure, sir.",
    "check_syntax": "I'm checking the code, sir.",
    "verify_file": "I'm verifying that, sir.",
    "edit_code": "I'm making the change now, sir.",
    "write_text_file": "I'm writing that file now, sir.",
    "create_text_file": "I'm writing that file now, sir.",
    "run_command": "I'm running that now, sir.",
    "browser_open_url": "I'm checking the sources, sir.",
    "browser_click_first_result": "I'm checking the sources, sir.",
    "take_screenshot": "I'm taking a look at the screen, sir.",
    "analyze_screen": "I'm taking a look at the screen, sir.",
    "inspect_window": "I'm looking at the window, sir.",
    "recall_memory": "I'm checking what I know, sir.",
}

#: Said when a tool has been running for a long time. The honest answer to
#: "what is taking so long" is the tool, not the model -- and this is only
#: ever said while a tool is genuinely still in flight.
STILL_RUNNING = "That's still running, sir."

#: Said when the last tool has RETURNED and the model has not answered yet.
#: At that moment the model really is the thing taking the time, and it
#: really is working from what the tools found -- so this describes actual
#: state rather than narrating reasoning.
STILL_THINKING = "I'm working through what I found, sir."

#: How long a gap in speech may last before the heartbeat fills it. Long
#: enough that a normally-paced task never triggers it, short enough that a
#: minute of model reasoning is not a minute of silence.
MAX_SILENCE_SECONDS = 12.0


@dataclass
class AgentNarrator:
    """Turns real agent events into rate-limited, truthful speech.

    `speak` is injected (the assistant's speech task in production, a list
    append in tests), so nothing here needs audio hardware.
    """

    speak: Callable[[str], None]
    #: Used ONLY for the final streamed answer (`answer_stream()`'s
    #: `emit`), so a caller wiring this up can give the final answer a
    #: higher speech priority than progress lines without the narrator
    #: itself needing to know anything about priorities
    #: (`voice/speech_coordinator.py` owns that policy). Defaults to
    #: `speak` when not given, so existing callers/tests are unaffected.
    speak_final: Callable[[str], None] | None = None
    clock: Callable[[], float] = time.monotonic
    min_interval: float = MIN_INTERVAL_SECONDS
    first_update_after: float = FIRST_UPDATE_AFTER_SECONDS
    max_silence: float = MAX_SILENCE_SECONDS
    started_at: float = field(default=0.0)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _stopped: threading.Event = field(default_factory=threading.Event)
    _heartbeat_thread: threading.Thread | None = None
    _tool_in_flight: str | None = None
    _last_spoken_at: float = 0.0
    _last_phrase: str | None = None
    _answer_started: bool = False
    spoken: list[str] = field(default_factory=list)
    suppressed: int = 0

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = self.clock()

    # -- progress ------------------------------------------------------
    def on_event(self, stage: str, payload: dict) -> None:
        """`brain/agent_loop.py`'s progress callback. Never raises: a broken
        narrator must not be able to fail the task it is describing."""
        try:
            if stage == "tool_started":
                with self._lock:
                    self._tool_in_flight = str(payload.get("tool") or "") or None
                return
            if stage != "tool_result":
                return
            with self._lock:
                self._tool_in_flight = None
            phrase = _PHRASES.get(str(payload.get("tool") or ""))
            if phrase:
                self.announce(phrase)
        except Exception:
            log.exception("Progress narration failed; continuing silently")

    # -- filling a long silence ----------------------------------------
    def heartbeat(self) -> bool:
        """Say something true if the task has gone quiet for too long.

        What it says is decided by REAL state, never by a timer alone:
        while a tool is still in flight the tool is what is slow, and once
        every tool has returned the model is. If neither is true -- the
        answer has started, or the silence is not long yet -- it says
        nothing.
        """
        with self._lock:
            if self._answer_started:
                return False
            now = self.clock()
            reference = self._last_spoken_at or self.started_at
            if now - reference < self.max_silence:
                return False
            phrase = STILL_RUNNING if self._tool_in_flight else STILL_THINKING
            # A heartbeat may repeat where a progress line may not: after
            # twelve more seconds of silence, saying it again is useful.
            self._last_spoken_at = now
            self._last_phrase = phrase
            self.spoken.append(phrase)
        self.speak(phrase)
        return True

    def start_heartbeat(self) -> None:
        """Run `heartbeat` on its own daemon thread until `stop()`.

        A separate thread, so a long silence is filled while the agent is
        blocked inside a model call or a tool -- the agent is never waited on
        and never waits for this.
        """
        if self._heartbeat_thread is not None:
            return
        self._stopped.clear()

        def loop() -> None:
            while not self._stopped.wait(1.0):
                try:
                    if self.answer_started:
                        return
                    self.heartbeat()
                except Exception:
                    log.exception("Heartbeat failed; stopping it")
                    return

        self._heartbeat_thread = threading.Thread(target=loop, name="jarvis-narration", daemon=True)
        self._heartbeat_thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._heartbeat_thread = None

    def announce(self, phrase: str) -> bool:
        """Speak `phrase` if the rate limit, the dedupe rule and the
        answer-wins rule all allow it. Returns whether it was spoken."""
        with self._lock:
            now = self.clock()
            if self._answer_started:
                self.suppressed += 1
                return False
            if now - self.started_at < self.first_update_after:
                self.suppressed += 1
                return False
            if self._last_spoken_at and now - self._last_spoken_at < self.min_interval:
                self.suppressed += 1
                return False
            if phrase == self._last_phrase:
                self.suppressed += 1
                return False
            self._last_spoken_at = now
            self._last_phrase = phrase
            self.spoken.append(phrase)
        self.speak(phrase)
        return True

    # -- the final answer ----------------------------------------------
    def answer_stream(self) -> SentenceStream:
        """A sink for streamed final-answer text.

        The first released sentence flips `answer_started`, which
        permanently silences progress: nothing may be spoken after the
        answer has begun.
        """

        def emit(sentence: str) -> None:
            with self._lock:
                self._answer_started = True
            self._stopped.set()
            (self.speak_final or self.speak)(sentence)

        return SentenceStream(emit=emit)

    @property
    def answer_started(self) -> bool:
        with self._lock:
            return self._answer_started

    def mark_answer_started(self) -> None:
        with self._lock:
            self._answer_started = True
        self._stopped.set()


def phrase_for_tool(tool: str) -> str | None:
    """The truthful status line for one tool, or None when there is nothing
    honest and useful to say about it."""
    return _PHRASES.get(tool)


__all__ = [
    "AgentNarrator",
    "FIRST_UPDATE_AFTER_SECONDS",
    "MIN_INTERVAL_SECONDS",
    "PRIORITY_ANSWER",
    "PRIORITY_PROGRESS",
    "PRIORITY_STATUS",
    "MAX_SILENCE_SECONDS",
    "STILL_RUNNING",
    "STILL_THINKING",
    "phrase_for_tool",
]
