import threading
import unittest
from unittest.mock import patch

from brain import agent
from brain.router import route_command
from brain.task_supervisor import cancel_read_only_tasks
from brain.web_answer import WebAnswer
from voice.response_formatter import format_spoken_response


class VoiceCancellationCommandTests(unittest.TestCase):
    def tearDown(self):
        cancel_read_only_tasks()

    def test_phrases_are_highest_priority_and_make_no_llm_call(self):
        phrases = ("cancel", "cancel that", "stop that", "stop the current task")
        with patch("brain.router.classify_intent") as llm_router, patch.object(agent, "ask_ai") as llm_answer:
            for phrase in phrases:
                with self.subTest(phrase=phrase):
                    route = route_command(phrase)
                    self.assertEqual(route, {"type": "cancel_read_only_task"})
                    self.assertEqual(agent.run_agent(phrase, route=route), "There's nothing to cancel.")
            llm_router.assert_not_called()
            llm_answer.assert_not_called()

    def test_cancel_active_question_and_discard_late_web_result(self):
        started = threading.Event()
        release = threading.Event()
        answers = []

        class SlowService:
            model = "test-model"
            timeout = 3
            def answer(self, question, cancellation):
                started.set()
                release.wait(2)
                return WebAnswer("This late answer must be discarded.", True, model=self.model)

        question = "who created Minecraft"
        with patch.object(agent, "get_web_answer_service", return_value=SlowService()), patch.object(
            agent.executor, "execute_action"
        ) as desktop, patch.object(agent, "ask_ai") as llm_answer:
            worker = threading.Thread(
                target=lambda: answers.append(agent.run_agent(question, route=route_command(question))),
                daemon=True,
            )
            worker.start()
            self.assertTrue(started.wait(1))
            cancel_route = route_command("cancel that")
            confirmation = agent.run_agent("cancel that", route=cancel_route)
            release.set()
            worker.join(2)

        self.assertEqual(confirmation, "Cancelled.")
        self.assertEqual(answers, ["I stopped that question."])
        self.assertNotIn("late answer", answers[0].lower())
        desktop.assert_not_called()
        llm_answer.assert_not_called()
        self.assertEqual(format_spoken_response("cancel that", cancel_route, confirmation, "en"), "Cancelled, sir.")

    def test_nothing_running_has_truthful_spoken_response(self):
        cancel_read_only_tasks()
        route = route_command("stop the current task")
        response = agent.run_agent("stop the current task", route=route)
        self.assertEqual(response, "There's nothing to cancel.")
        self.assertEqual(format_spoken_response("stop the current task", route, response, "en"), "There's nothing to cancel, sir.")

    def test_task_status_is_local_truthful_and_does_not_call_llm(self):
        with patch("brain.router.classify_intent") as llm_router,patch.object(agent,"ask_ai") as llm_answer:
            route=route_command("what tasks are running")
            self.assertEqual(route,{"type":"task_status"})
            response=agent.run_agent("what tasks are running",route=route)
        self.assertEqual(response,"No interactive tasks are running.")
        self.assertEqual(format_spoken_response("what tasks are running",route,response,"en"),"No interactive tasks are running, sir.")
        llm_router.assert_not_called();llm_answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
