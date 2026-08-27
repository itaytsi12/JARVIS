"""Regression tests for bug report 3: task stop/pause/cancel must outrank
media pause, and a small set of high-confidence multilingual control-word
equivalents must be recognized without any model call."""
import unittest
from unittest.mock import patch

from brain.control_words import normalize_control_word
from brain.router import route_command
from brain import task_supervisor
from brain.task_supervisor import (
    CancellationToken,
    any_active_interactive_work,
    register_interactive_task,
    unregister_interactive_task,
)
import brain.activity_state as activity_state


class NormalizeControlWordTests(unittest.TestCase):
    def test_russian_stop_normalizes_to_stop(self):
        self.assertEqual(normalize_control_word("Стоп!"), "stop")

    def test_russian_cancel_variants_normalize_to_cancel(self):
        self.assertEqual(normalize_control_word("отмена"), "cancel")
        self.assertEqual(normalize_control_word("отменить"), "cancel")

    def test_hebrew_stop_and_cancel_normalize(self):
        self.assertEqual(normalize_control_word("עצור"), "stop")
        self.assertEqual(normalize_control_word("בטל"), "cancel")

    def test_unrecognized_text_is_returned_lowercased_and_trimmed(self):
        self.assertEqual(normalize_control_word("Open Notepad."), "open notepad")

    def test_ordinary_english_control_words_pass_through(self):
        self.assertEqual(normalize_control_word("stop"), "stop")
        self.assertEqual(normalize_control_word("Cancel that."), "cancel that")


class AnyActiveInteractiveWorkTests(unittest.TestCase):
    def setUp(self):
        activity_state.set_speaking(False)

    def tearDown(self):
        activity_state.set_speaking(False)

    def test_false_when_nothing_is_happening(self):
        self.assertFalse(any_active_interactive_work())

    def test_true_while_an_interactive_task_is_registered(self):
        token = CancellationToken()
        task_id = register_interactive_task(token)
        try:
            self.assertTrue(any_active_interactive_work())
        finally:
            unregister_interactive_task(task_id)
        self.assertFalse(any_active_interactive_work())

    def test_true_while_jarvis_is_speaking(self):
        activity_state.set_speaking(True)
        self.assertTrue(any_active_interactive_work())
        activity_state.set_speaking(False)
        self.assertFalse(any_active_interactive_work())

    def test_never_raises_even_if_a_signal_source_is_broken(self):
        with patch.object(task_supervisor, "active_interactive_task_summary", side_effect=RuntimeError("boom")):
            self.assertFalse(any_active_interactive_work())


class StopCancelAlwaysWinTests(unittest.TestCase):
    """"stop"/"cancel"/"never mind"/"forget it" are unambiguous task-control
    vocabulary (no music-intent collision), so they route to the task
    exactly as before -- unconditionally, not gated on task state."""

    def test_english_control_words(self):
        for text in ["stop", "cancel", "cancel that", "stop that", "never mind", "forget it"]:
            with self.subTest(text=text):
                self.assertEqual(route_command(text)["type"], "cancel_read_only_task")

    def test_russian_mis_detected_stop_routes_to_cancel_not_agent(self):
        # The reported live bug: STT committed English "Stop!" as Russian
        # "Стоп!", which used to fall through to the agent runtime instead
        # of cancelling the active task.
        route = route_command("Стоп!")
        self.assertEqual(route["type"], "cancel_read_only_task")

    def test_hebrew_cancel_routes_to_cancel(self):
        route = route_command("בטל")
        self.assertEqual(route["type"], "cancel_read_only_task")


class PauseTaskPriorityTests(unittest.TestCase):
    """Sequences C/D/E from the bug report: bare "pause"/"pause that" must
    prefer the active JARVIS task over music_pause, but an explicit media
    phrase always reaches the music route regardless of task state."""

    def setUp(self):
        activity_state.set_speaking(False)

    def tearDown(self):
        activity_state.set_speaking(False)

    def test_bare_pause_goes_to_music_when_no_task_is_active(self):
        route = route_command("pause")
        self.assertEqual(route.get("tool"), "music_pause")

    def test_bare_pause_goes_to_task_when_a_task_is_active(self):
        token = CancellationToken()
        task_id = register_interactive_task(token)
        try:
            route = route_command("pause")
            self.assertEqual(route["type"], "cancel_read_only_task")
            self.assertEqual(route.get("route_source"), "task_priority_over_media")
        finally:
            unregister_interactive_task(task_id)

    def test_pause_that_goes_to_task_when_jarvis_is_speaking(self):
        activity_state.set_speaking(True)
        try:
            route = route_command("pause that")
            self.assertEqual(route["type"], "cancel_read_only_task")
        finally:
            activity_state.set_speaking(False)

    def test_explicit_pause_the_music_always_wins_even_with_active_task(self):
        # Sequence D: explicit media phrase must win regardless of task
        # state -- checked here WITH a task active, the harder case.
        token = CancellationToken()
        task_id = register_interactive_task(token)
        try:
            route = route_command("pause the music")
            self.assertEqual(route.get("tool"), "music_pause")
        finally:
            unregister_interactive_task(task_id)

    def test_explicit_pause_spotify_always_wins(self):
        route = route_command("pause Spotify")
        # "pause Spotify" isn't literally in music_intent's fixed PAUSE
        # phrase set, but it must still never be captured by the bare
        # "pause"/"pause that" task-priority gate.
        self.assertNotEqual(route.get("route_source"), "task_priority_over_media")


if __name__ == "__main__":
    unittest.main()
