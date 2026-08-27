"""Tests for what the new execution layer records: the plan's dependency
structure in the training dataset, and measured scheduling metrics.

The dataset assertions matter twice over -- they check that the structure a
future model would need is actually captured, AND that widening the recorded
payload did not widen what leaks. Secrets must still be redacted.
"""
from __future__ import annotations

import time
import unittest

from brain.agent import _plan_action_records, _plan_schedule_record
from brain.agent_runtime import AgentRuntime
from brain.models import Action, ActionRisk, Plan, ToolResult
from training_data.sanitizer import privacy_safe_event


class PlanRecordTests(unittest.TestCase):
    def setUp(self):
        self.actions = [
            Action("open_application", {"app_name": "chrome"}),
            Action("open_application", {"app_name": "spotify"}),
            Action("volume_down", {}, depends_on=[0, 1], optional=True),
        ]

    def test_dependency_structure_is_recorded_not_just_a_flat_tool_list(self):
        records = _plan_action_records(self.actions)
        self.assertEqual([r["index"] for r in records], [0, 1, 2])
        self.assertEqual(records[2]["depends_on"], [0, 1])
        self.assertTrue(records[2]["optional"])
        self.assertFalse(records[0]["optional"])

    def test_risk_is_recorded_as_a_plain_string(self):
        records = _plan_action_records([Action("open_application", {"app_name": "x"}, risk=ActionRisk.CAUTION)])
        self.assertEqual(records[0]["risk"], "caution")

    def test_the_schedule_shows_the_concurrency_that_was_available(self):
        schedule = _plan_schedule_record(self.actions)
        self.assertEqual(len(schedule), 2)
        self.assertEqual(len(schedule[0]["parallel"]), 2)

    def test_a_malformed_plan_is_still_recordable(self):
        # A cyclic plan is a planning bug; recording must not raise and lose
        # the very trace that would explain it.
        cyclic = [Action("a", {}, depends_on=[1]), Action("b", {}, depends_on=[0])]
        self.assertEqual(_plan_schedule_record(cyclic), [])


class PlanRecordSanitizationTests(unittest.TestCase):
    def test_a_secret_in_a_typed_payload_is_still_redacted(self):
        records = _plan_action_records([Action("type_text", {"text": "my password is hunter2"})])
        payload = privacy_safe_event("PLAN_CREATED", {"actions": records})
        arguments = payload["actions"][0]["arguments"]
        self.assertEqual(arguments["text"], "<REDACTED>")
        self.assertNotIn("hunter2", str(payload))

    def test_an_api_key_never_reaches_the_dataset(self):
        records = _plan_action_records([Action("run_command", {"command": "export API_KEY=sk-abcdefghijklmnop"})])
        payload = privacy_safe_event("PLAN_CREATED", {"actions": records})
        self.assertNotIn("sk-abcdefghijklmnop", str(payload))

    def test_the_new_structural_fields_survive_sanitization(self):
        records = _plan_action_records([Action("open_application", {"app_name": "chrome"}), Action("volume_down", {}, depends_on=[0])])
        payload = privacy_safe_event("PLAN_CREATED", {"actions": records, "schedule": _plan_schedule_record([Action("open_application", {"app_name": "chrome"}), Action("volume_down", {}, depends_on=[0])])})
        self.assertEqual(payload["actions"][1]["depends_on"], [0])
        self.assertEqual(len(payload["schedule"]), 2)


class _SlowRuntime(AgentRuntime):
    def __init__(self, delay=0.2):
        super().__init__(trace=False)
        self._delay = delay

    def _execute_action(self, action, cancellation_token=None, plan_lock_held=False):
        time.sleep(self._delay)
        return ToolResult(True, action.tool, "ok")


class ExecutionMetricsTests(unittest.TestCase):
    def test_metrics_report_waves_and_measured_parallel_saving(self):
        plan = Plan("g", [
            Action("open_application", {"app_name": "chrome"}),
            Action("open_application", {"app_name": "spotify"}),
            Action("volume_down", {}, depends_on=[0, 1]),
        ])
        _SlowRuntime(delay=0.2).execute(plan)
        metrics = plan.context["execution_metrics"]
        self.assertEqual(metrics["waves"], 2)
        self.assertEqual(metrics["parallel_actions"], 2)
        self.assertGreater(metrics["parallel_saved_ms"], 100, "overlapping two 200ms actions should show a real saving")
        self.assertGreater(metrics["scheduled_ms"], 0)

    def test_a_plan_with_nothing_to_overlap_reports_no_saving(self):
        plan = Plan("g", [
            Action("open_application", {"app_name": "notepad"}),
            Action("type_text", {"text": "hi"}, depends_on=[0]),
        ])
        _SlowRuntime(delay=0.05).execute(plan)
        # a pure chain keeps the sequential path and records no scheduler metrics
        self.assertNotIn("execution_metrics", plan.context)

    def test_each_action_duration_is_recorded_on_the_result(self):
        plan = Plan("g", [Action("open_application", {"app_name": "a"}), Action("open_application", {"app_name": "b"})])
        results = _SlowRuntime(delay=0.05).execute(plan)
        for result in results:
            self.assertIn("action_ms", result.data)
            self.assertGreater(result.data["action_ms"], 0)


if __name__ == "__main__":
    unittest.main()
