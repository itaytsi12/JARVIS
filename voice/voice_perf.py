"""Per-interaction voice latency instrumentation (Part N).

One `VoiceInteractionTimer` per wake/listen/respond cycle. Every stage is a
real `clock()` timestamp recorded at the moment the caller reports it --
nothing here is estimated, interpolated, or fabricated. Stages that a given
interaction never reaches (e.g. `planner_started` for a fast-path command
that skips the planner entirely) are simply absent from the summary rather
than printed as a fake zero.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

log = logging.getLogger("jarvis.voice_perf")

# Canonical stage order for the printed summary; unlisted stages (if a
# caller ever records a custom one) are appended after these, in the order
# first observed.
STAGE_ORDER = [
    "wake_detected",
    "cloud_stt_connect_start",
    "cloud_stt_connected",
    "first_audio_chunk_sent",
    "first_partial_transcript",
    "first_stable_intent",
    "first_action_started",
    "acknowledgement_tts_request",
    "first_ack_audio_chunk",
    "first_ack_audio_played",
    "committed_transcript",
    "planner_started",
    "planner_finished",
    "final_action_finished",
    "final_tts_started",
    "first_final_tts_audio",
    "interaction_finished",
]

# (from_stage, to_stage, label) for the compact "VOICE PERF" summary.
_SUMMARY_DELTAS = [
    ("wake_detected", "first_partial_transcript", "wake->partial"),
    ("wake_detected", "first_stable_intent", "wake->intent"),
    ("wake_detected", "first_action_started", "wake->first action"),
    ("wake_detected", "first_ack_audio_played", "wake->first speech"),
    ("committed_transcript", "planner_started", "commit->plan start"),
    ("planner_started", "planner_finished", "plan duration"),
    ("wake_detected", "interaction_finished", "total task"),
]


class VoiceInteractionTimer:
    def __init__(self, clock: Callable[[], float] = time.monotonic, enabled: bool = True):
        self._clock = clock
        self.enabled = enabled
        self._stamps: dict[str, float] = {}
        self._order: list[str] = []

    def mark(self, stage: str, when: float | None = None) -> float:
        """Record `stage` at `when` (default: now). Idempotent per stage --
        only the FIRST occurrence is kept and returned on every subsequent
        call, since "first partial transcript" etc. are meaningful exactly
        once per interaction."""
        if stage in self._stamps:
            return self._stamps[stage]
        timestamp = self._clock() if when is None else when
        self._stamps[stage] = timestamp
        self._order.append(stage)
        return timestamp

    def has(self, stage: str) -> bool:
        return stage in self._stamps

    def elapsed_ms(self, from_stage: str, to_stage: str) -> float | None:
        if from_stage not in self._stamps or to_stage not in self._stamps:
            return None
        return (self._stamps[to_stage] - self._stamps[from_stage]) * 1000.0

    def summary_lines(self) -> list[str]:
        lines = []
        for from_stage, to_stage, label in _SUMMARY_DELTAS:
            ms = self.elapsed_ms(from_stage, to_stage)
            if ms is not None:
                lines.append(f"{label}: {ms:.0f} ms")
        return lines

    def log_summary(self) -> None:
        if not self.enabled:
            return
        lines = self.summary_lines()
        if not lines:
            return
        log.info("VOICE PERF: %s", "; ".join(lines))

    def all_stamps(self) -> dict[str, float]:
        return dict(self._stamps)
