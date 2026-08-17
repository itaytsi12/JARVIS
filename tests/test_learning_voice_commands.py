"""Phase 11/19/20: the deterministic "start learning"/"stop learning" voice
commands. Router-level matching (never the cloud planner/intent model) and
the AlwaysOnAssistant dispatch that wires them to
brain/learning_orchestrator.py, using fakes throughout -- no real Claude,
no real training, no physical microphone.
"""
import threading
import time
import unittest
from unittest.mock import patch

from brain.router import route_command
from brain.learning_models import LearningJob, LearningJobStatus
from brain.learning_store import LearningJobStore
from voice.background_assistant import AlwaysOnAssistant


class RouteCommandLearningTests(unittest.TestCase):
    def test_start_learning_is_deterministic(self):
        for phrase in ("start learning", "Start Learning", "start learning.", "start the learning", "begin learning"):
            with self.subTest(phrase=phrase):
                self.assertEqual(route_command(phrase), {"type": "start_learning"})

    def test_stop_learning_is_deterministic(self):
        for phrase in ("stop learning", "Stop Learning!", "cancel learning", "stop the learning"):
            with self.subTest(phrase=phrase):
                self.assertEqual(route_command(phrase), {"type": "stop_learning"})

    def test_start_learning_never_falls_through_to_cloud_fallback(self):
        # If this ever accidentally routed through classify_intent (the
        # cloud fallback), it would require network/API mocking to not
        # raise. Getting a plain dict back with no cloud call proves the
        # deterministic branch was taken.
        result = route_command("start learning")
        self.assertEqual(result, {"type": "start_learning"})

    def test_unrelated_learning_word_usage_is_not_captured(self):
        result = route_command("I want to learn python")
        self.assertNotEqual(result.get("type"), "start_learning")


class StartLearningDispatchTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.temp = tempfile.TemporaryDirectory()
        self.job_store = LearningJobStore(Path(self.temp.name) / "jobs.sqlite3")

    def tearDown(self):
        self.job_store.close()
        self.temp.cleanup()

    def test_reports_zero_approved_jobs(self):
        assistant = AlwaysOnAssistant()
        spoken = threading.Event()
        messages = []

        def fake_speak(text, **_):
            messages.append(text)
            spoken.set()

        with patch("brain.learning_store.get_learning_job_store", return_value=self.job_store), \
             patch("voice.text_to_speech.speak", side_effect=fake_speak):
            assistant._start_learning_task()
            self.assertTrue(spoken.wait(2))

        self.assertIn("don't have any approved", messages[0])

    def test_reports_job_count_and_singular_plural_phrasing(self):
        self.job_store.create(LearningJob(
            learning_job_id="j1", created_at="t", updated_at="t", candidate_id="c1", improvement_attempt_id="a1",
            fingerprint="fp1", learning_status=LearningJobStatus.APPROVED.value,
        ))
        assistant = AlwaysOnAssistant()
        spoken = []
        event = threading.Event()

        def fake_speak(text, **_):
            spoken.append(text)
            event.set()

        with patch("brain.learning_store.get_learning_job_store", return_value=self.job_store), \
             patch("voice.text_to_speech.speak", side_effect=fake_speak):
            assistant._start_learning_task()
            self.assertTrue(event.wait(2))

        self.assertIn("I have 1 approved learning task.", spoken[0])

    def test_stop_learning_without_an_active_run_says_so(self):
        assistant = AlwaysOnAssistant()
        spoken = []
        event = threading.Event()

        def fake_speak(text, **_):
            spoken.append(text)
            event.set()

        with patch("voice.text_to_speech.speak", side_effect=fake_speak):
            assistant._stop_learning_task()
            self.assertTrue(event.wait(2))
        self.assertIn("not learning", spoken[0])

    def test_stop_learning_cancels_the_active_token(self):
        from brain.task_supervisor import CancellationToken
        assistant = AlwaysOnAssistant()
        token = CancellationToken()
        assistant._learning_token = token
        spoken = []
        event = threading.Event()

        def fake_speak(text, **_):
            spoken.append(text)
            event.set()

        with patch("voice.text_to_speech.speak", side_effect=fake_speak):
            assistant._stop_learning_task()
            self.assertTrue(event.wait(2))
        self.assertTrue(token.cancelled)
        self.assertIn("Stopping learning", spoken[0])

    def test_start_learning_refuses_a_second_concurrent_run(self):
        from brain.task_supervisor import CancellationToken
        assistant = AlwaysOnAssistant()
        assistant._learning_token = CancellationToken()  # already "running"
        spoken = []
        event = threading.Event()

        def fake_speak(text, **_):
            spoken.append(text)
            event.set()

        with patch("voice.text_to_speech.speak", side_effect=fake_speak):
            assistant._start_learning_task()
            self.assertTrue(event.wait(2))
        self.assertIn("already learning", spoken[0])


if __name__ == "__main__":
    unittest.main()
