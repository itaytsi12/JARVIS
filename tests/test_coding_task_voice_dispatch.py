"""Part A production wiring: the "coding_task" route type and
`AlwaysOnAssistant._start_coding_task` dispatch. Non-coding ordinary
commands must never produce a "coding_task" route (test #13 of Phase A13).
"""
import threading
import unittest
from unittest.mock import patch

from brain.router import route_command
from brain.improvement_student_teacher import CodingTaskResult
from voice.background_assistant import AlwaysOnAssistant


class RouteCommandCodingTaskTests(unittest.TestCase):
    def test_coding_phrase_routes_to_coding_task(self):
        route = route_command("fix this bug")
        self.assertEqual(route["type"], "coding_task")
        self.assertEqual(route["task"], "fix this bug")

    def test_longer_coding_sentence_routes_to_coding_task(self):
        route = route_command("inspect this repository and fix the failing test")
        self.assertEqual(route["type"], "coding_task")

    def test_ordinary_commands_never_produce_coding_task_route(self):
        ordinary = ["open chrome", "volume up", "play music", "what time is it", "mute", "take a screenshot"]
        for command in ordinary:
            with self.subTest(command=command):
                route = route_command(command)
                self.assertNotEqual(route.get("type"), "coding_task")


class CodingTaskDispatchTests(unittest.TestCase):
    def _run_and_capture_speech(self, result):
        """Mocks `_start_speech_task` directly (rather than racing on the
        real threaded `speak()` pipeline) and blocks until the background
        `work()` thread has made both its expected calls -- deterministic,
        no timing race."""
        assistant = AlwaysOnAssistant()
        calls = []
        done = threading.Event()

        def fake_start_speech_task(text, *args, **kwargs):
            calls.append(text)
            if len(calls) >= 2:
                done.set()

        with patch("brain.improvement_student_teacher.run_coding_task", return_value=result), \
             patch.object(assistant, "_start_speech_task", side_effect=fake_start_speech_task):
            assistant._start_coding_task("fix this bug")
            self.assertTrue(done.wait(2), f"expected 2 speech calls, got {calls}")
        return calls

    def test_student_solved_speech(self):
        result = CodingTaskResult(task="fix this bug", candidate_id="c1", started_at="t", solved_by="student")
        calls = self._run_and_capture_speech(result)
        self.assertEqual(calls[0], "Let me look into that, sir.")
        self.assertEqual(calls[1], "I fixed it myself, sir.")

    def test_no_active_student_still_reports_teacher_result(self):
        result = CodingTaskResult(task="fix this bug", candidate_id="c1", started_at="t", solved_by="teacher", learning_offer=None)
        calls = self._run_and_capture_speech(result)
        self.assertEqual(calls[1], "It's fixed and verified, sir.")

    def test_refuses_concurrent_coding_task_while_busy(self):
        from brain.task_supervisor import CancellationToken
        assistant = AlwaysOnAssistant()
        assistant._learning_token = CancellationToken()
        calls = []

        with patch.object(assistant, "_start_speech_task", side_effect=lambda text, *a, **k: calls.append(text)):
            assistant._start_coding_task("fix this bug")
        self.assertIn("already working", calls[0])

    def test_coding_task_result_speech_covers_all_outcomes(self):
        from brain.learning_orchestrator import LearningOfferOutcome

        solved_by_student = CodingTaskResult(task="t", candidate_id="c", started_at="t", solved_by="student")
        self.assertEqual(AlwaysOnAssistant._coding_task_result_speech(solved_by_student), "I fixed it myself, sir.")

        approved = CodingTaskResult(task="t", candidate_id="c", started_at="t", solved_by="teacher", learning_offer=LearningOfferOutcome(offered=True, approval_outcome="APPROVED"))
        self.assertEqual(AlwaysOnAssistant._coding_task_result_speech(approved), "I've added that to my learning queue, sir.")

        declined = CodingTaskResult(task="t", candidate_id="c", started_at="t", solved_by="teacher", learning_offer=LearningOfferOutcome(offered=True, approval_outcome="DECLINED"))
        self.assertEqual(AlwaysOnAssistant._coding_task_result_speech(declined), "Understood, sir.")

        timed_out = CodingTaskResult(task="t", candidate_id="c", started_at="t", solved_by="teacher", learning_offer=LearningOfferOutcome(offered=True, approval_outcome="TIMED_OUT"))
        self.assertIsNone(AlwaysOnAssistant._coding_task_result_speech(timed_out))

        not_offered = CodingTaskResult(task="t", candidate_id="c", started_at="t", solved_by="teacher", learning_offer=LearningOfferOutcome(offered=False, reason="dup"))
        self.assertEqual(AlwaysOnAssistant._coding_task_result_speech(not_offered), "It's fixed and verified, sir.")

        unsolved = CodingTaskResult(task="t", candidate_id="c", started_at="t", solved_by="none")
        self.assertEqual(AlwaysOnAssistant._coding_task_result_speech(unsolved), "I wasn't able to fix that, sir.")


if __name__ == "__main__":
    unittest.main()
