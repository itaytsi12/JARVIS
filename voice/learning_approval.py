"""Voice learning-approval state machine (Phase 3).

Pure, dependency-injected logic: `speak_fn`/`listen_fn`/`clock` are all
callables the caller supplies, exactly like `AlwaysOnAssistant`'s existing
`stream_factory`/`clock` injection pattern in
`voice/background_assistant.py`. That means this module's real behavior
(phrase matching, the 30-second window, what counts as a valid answer) is
fully testable with fakes -- no physical microphone, no real TTS -- while
`voice/background_assistant.py::AlwaysOnAssistant.request_learning_approval`
supplies the real `speak`/listen implementations for production use.

Only the exact phrases "yes jarvis" / "no jarvis" (case/punctuation
normalized) are ever accepted as a terminal answer, deliberately excluding
bare "yes"/"no"/"yeah"/"okay" -- the whole point of requiring the wake name
is reducing accidental approval of an expensive, training-data-affecting
action. Anything else heard during the window does not end it early; JARVIS
keeps listening until an explicit phrase is heard or the window elapses.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Protocol


class LearningApprovalOutcome(str, Enum):
    APPROVED = "APPROVED"
    DECLINED = "DECLINED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"


DEFAULT_LEARNING_QUESTION = "Do you want me to learn how to do that, sir?"
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 30.0

_APPROVE_PHRASES = {"yes jarvis"}
_DECLINE_PHRASES = {"no jarvis"}

_PUNCTUATION = re.compile(r"[.?!,;:]+")
_WHITESPACE = re.compile(r"\s+")


def normalize_approval_text(text: str | None) -> str:
    """Lowercase, drop punctuation anywhere in the utterance (STT commonly
    inserts a comma after "yes"), collapse whitespace. Never strips the wake
    name itself; that's the whole safety property this module provides."""
    value = (text or "").strip().lower()
    value = _PUNCTUATION.sub(" ", value)
    value = _WHITESPACE.sub(" ", value).strip()
    return value


def classify_approval_response(text: str | None) -> Optional[LearningApprovalOutcome]:
    """Returns APPROVED/DECLINED only for an exact recognized phrase, or
    None if the utterance isn't a recognized answer (including bare
    "yes"/"no" -- those are deliberately NOT sufficient)."""
    normalized = normalize_approval_text(text)
    if normalized in _APPROVE_PHRASES:
        return LearningApprovalOutcome.APPROVED
    if normalized in _DECLINE_PHRASES:
        return LearningApprovalOutcome.DECLINED
    return None


class SupportsCancelled(Protocol):
    cancelled: bool


@dataclass
class LearningApprovalResult:
    outcome: LearningApprovalOutcome
    transcript: str | None = None
    elapsed_seconds: float = 0.0


def request_learning_approval(
    *,
    speak_fn: Callable[[str], None],
    listen_fn: Callable[[float], Optional[str]],
    clock: Callable[[], float] = time.monotonic,
    timeout_seconds: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    question: str = DEFAULT_LEARNING_QUESTION,
    cancellation_token: SupportsCancelled | None = None,
) -> LearningApprovalResult:
    """Speak `question`, then listen for up to `timeout_seconds` for an
    explicit "yes jarvis"/"no jarvis". `listen_fn(remaining_seconds)` must be
    a BLOCKING call that returns the transcript of one captured utterance
    (or None/"" if nothing was heard in that sub-window) -- the real
    integration in `voice/background_assistant.py` hands the assistant's
    existing single audio-thread capture path in as `listen_fn` so this
    never opens a second, conflicting microphone consumer (Phase 4).

    A no-valid-answer outcome (unrecognized speech, silence, or elapsed
    time) always ends in TIMED_OUT, never a silent hang -- Phase 3's "timeout
    = NO" requirement is enforced here, once, for every caller.
    """
    started = clock()
    speak_fn(question)
    deadline = started + max(0.0, timeout_seconds)

    while True:
        if cancellation_token is not None and getattr(cancellation_token, "cancelled", False):
            return LearningApprovalResult(LearningApprovalOutcome.CANCELLED, None, clock() - started)
        remaining = deadline - clock()
        if remaining <= 0:
            break
        transcript = listen_fn(remaining)
        if cancellation_token is not None and getattr(cancellation_token, "cancelled", False):
            return LearningApprovalResult(LearningApprovalOutcome.CANCELLED, None, clock() - started)
        if transcript:
            outcome = classify_approval_response(transcript)
            if outcome is not None:
                return LearningApprovalResult(outcome, transcript, clock() - started)
            # Heard something, but not a recognized phrase -- keep the
            # window open for the remaining time rather than ending early.

    return LearningApprovalResult(LearningApprovalOutcome.TIMED_OUT, None, clock() - started)
