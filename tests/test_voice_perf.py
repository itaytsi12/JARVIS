"""Focused tests for voice/voice_perf.py (Part N latency instrumentation)."""
from __future__ import annotations

import unittest

from voice.voice_perf import VoiceInteractionTimer


class StepClock:
    def __init__(self, step=0.1):
        self.value = 0.0
        self.step = step

    def __call__(self):
        self.value += self.step
        return self.value


class VoiceInteractionTimerTests(unittest.TestCase):
    def test_mark_records_first_occurrence_only(self):
        clock = StepClock()
        timer = VoiceInteractionTimer(clock=clock)
        first = timer.mark("wake_detected")
        second = timer.mark("wake_detected")
        self.assertEqual(first, second)
        self.assertEqual(timer.all_stamps()["wake_detected"], first)

    def test_elapsed_ms_none_when_a_stage_never_happened(self):
        timer = VoiceInteractionTimer(clock=StepClock())
        timer.mark("wake_detected")
        self.assertIsNone(timer.elapsed_ms("wake_detected", "first_partial_transcript"))

    def test_elapsed_ms_is_a_real_measured_delta_not_fabricated(self):
        clock = StepClock(step=0.05)
        timer = VoiceInteractionTimer(clock=clock)
        timer.mark("wake_detected")  # t=0.05
        timer.mark("first_partial_transcript")  # t=0.10
        timer.mark("first_partial_transcript")  # idempotent, t=0.15 but ignored
        ms = timer.elapsed_ms("wake_detected", "first_partial_transcript")
        self.assertAlmostEqual(ms, 50.0, places=3)

    def test_summary_only_includes_stages_actually_reached(self):
        clock = StepClock()
        timer = VoiceInteractionTimer(clock=clock)
        timer.mark("wake_detected")
        timer.mark("first_partial_transcript")
        lines = timer.summary_lines()
        self.assertTrue(any("wake->partial" in line for line in lines))
        self.assertFalse(any("commit->plan" in line for line in lines), "no committed_transcript/planner marks were made")

    def test_empty_timer_produces_no_summary_lines(self):
        timer = VoiceInteractionTimer(clock=StepClock())
        self.assertEqual(timer.summary_lines(), [])

    def test_explicit_when_argument_is_honored_over_clock(self):
        timer = VoiceInteractionTimer(clock=StepClock())
        timer.mark("wake_detected", when=123.0)
        self.assertEqual(timer.all_stamps()["wake_detected"], 123.0)

    def test_has_reflects_whether_a_stage_was_marked(self):
        timer = VoiceInteractionTimer(clock=StepClock())
        self.assertFalse(timer.has("wake_detected"))
        timer.mark("wake_detected")
        self.assertTrue(timer.has("wake_detected"))


if __name__ == "__main__":
    unittest.main()
