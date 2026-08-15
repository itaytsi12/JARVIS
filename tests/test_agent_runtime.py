import tempfile
import unittest
from pathlib import Path

from brain.agent_runtime import AgentRuntime
from brain.models import Action, ActionRisk, Plan, PlanStatus, ToolResult
from tools.browser_agent import HumanActionRequired


class FailingRuntime(AgentRuntime):
    def __init__(self):
        super().__init__(trace=False)
        self.calls = 0

    def _execute_action(self, action):
        self.calls += 1
        return ToolResult(False, action.tool, error="not_ready")


class HandoffRuntime(AgentRuntime):
    def __init__(self):
        super().__init__(trace=False)

    def _execute_action(self, action):
        raise HumanActionRequired("SMS verification is required.")


class RuntimeTests(unittest.TestCase):
    def test_retry_is_bounded_to_two_retries(self):
        runtime = FailingRuntime()
        plan = Plan("test", [Action("safe_action")])
        runtime.execute(plan)
        self.assertEqual(runtime.calls, 3)
        self.assertEqual(plan.retry_count, 2)
        self.assertEqual(plan.status, PlanStatus.FAILED)

    def test_direct_file_execution_and_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "hello.txt")
            plan = Plan("create", [Action("create_text_file", {"path": path, "contents": "hello"}), Action("verify_file", {"path": path}, depends_on=[0])])
            runtime = AgentRuntime(trace=False)
            results = runtime.execute(plan)
            self.assertTrue(all(item.success for item in results))
            self.assertEqual(Path(path).read_text(encoding="utf-8"), "hello")
            self.assertEqual(plan.status, PlanStatus.COMPLETED)

    def test_caution_action_is_not_retried(self):
        runtime = FailingRuntime()
        plan = Plan("test", [Action("submit_form", risk=ActionRisk.CAUTION)])
        runtime.execute(plan)
        self.assertEqual(runtime.calls, 1)

    def test_human_verification_pauses_resumably(self):
        runtime = HandoffRuntime()
        plan = Plan("verify", [Action("browser_click", {"target": "Continue"})])
        results = runtime.execute(plan)
        self.assertEqual(plan.status, PlanStatus.PAUSED)
        self.assertEqual(plan.current_action_index, 0)
        self.assertIn("SMS verification", results[0].error)


if __name__ == "__main__":
    unittest.main()
