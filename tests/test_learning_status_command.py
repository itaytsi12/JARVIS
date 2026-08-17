"""Phase 21: "Hey Jarvis, learning status"."""
import threading
import unittest
from unittest.mock import patch

from brain.router import route_command
from brain.learning_models import LearningJob, LearningJobStatus
from brain.learning_store import LearningJobStore
from voice.background_assistant import AlwaysOnAssistant


class RouteCommandLearningStatusTests(unittest.TestCase):
    def test_learning_status_is_deterministic(self):
        for phrase in ("learning status", "Learning Status?", "what's the learning status", "status of learning"):
            with self.subTest(phrase=phrase):
                self.assertEqual(route_command(phrase), {"type": "learning_status"})


class LearningStatusDispatchTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.temp = tempfile.TemporaryDirectory()
        self.job_store = LearningJobStore(Path(self.temp.name) / "jobs.sqlite3")

    def tearDown(self):
        self.job_store.close()
        self.temp.cleanup()

    def _speak_and_capture(self, assistant):
        spoken = []
        event = threading.Event()

        def fake_speak(text, **_):
            spoken.append(text)
            event.set()

        with patch("brain.learning_store.get_learning_job_store", return_value=self.job_store), \
             patch("voice.text_to_speech.speak", side_effect=fake_speak):
            assistant._learning_status_task()
            self.assertTrue(event.wait(2))
        return spoken

    def test_reports_no_active_run_and_zero_jobs(self):
        assistant = AlwaysOnAssistant()
        spoken = self._speak_and_capture(assistant)
        self.assertIn("No learning run is active", spoken[0])
        self.assertIn("0 approved learning task", spoken[0])

    def test_reports_pending_job_count(self):
        self.job_store.create(LearningJob(
            learning_job_id="j1", created_at="t", updated_at="t", candidate_id="c1", improvement_attempt_id="a1",
            fingerprint="fp1", learning_status=LearningJobStatus.READY_FOR_TRAINING.value,
        ))
        assistant = AlwaysOnAssistant()
        spoken = self._speak_and_capture(assistant)
        self.assertIn("1 approved learning task waiting.", spoken[0])

    def test_reports_active_run_and_last_progress_stage(self):
        from brain.task_supervisor import CancellationToken
        assistant = AlwaysOnAssistant()
        assistant._learning_token = CancellationToken()
        assistant._learning_last_status = ("TRAINING", "some detail")
        spoken = self._speak_and_capture(assistant)
        self.assertIn("training stage", spoken[0].lower())


if __name__ == "__main__":
    unittest.main()
