"""Focused tests for Part H: parallel execution of independent actions in
brain/agent_runtime.py. Uses a fake `execute_tool` at the same boundary
`Executor` already patches at in tests/test_agent_runtime.py, so no real
OS/browser automation runs.
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from brain.agent_runtime import AgentRuntime, _all_actions_independent
from brain.models import Action, PlanStatus


class AllActionsIndependentTests(unittest.TestCase):
    def test_two_or_more_safe_tools_with_no_dependencies_are_independent(self):
        actions = [
            Action(tool="open_application", args={"app_name": "discord"}),
            Action(tool="volume_down", args={"amount": 1}),
            Action(tool="open_application", args={"app_name": "spotify"}),
        ]
        self.assertTrue(_all_actions_independent(actions))

    def test_single_action_is_never_treated_as_a_parallel_batch(self):
        actions = [Action(tool="open_application", args={"app_name": "discord"})]
        self.assertFalse(_all_actions_independent(actions))

    def test_dependency_wiring_disqualifies_the_whole_plan(self):
        actions = [
            Action(tool="open_application", args={"app_name": "chrome"}),
            Action(tool="type_text", args={"text": "hello"}, depends_on=[0]),
        ]
        self.assertFalse(_all_actions_independent(actions))

    def test_context_dependent_tool_disqualifies_the_whole_plan(self):
        """open_application + type_text look independent (no depends_on
        from the local planner) but type_text reads context state
        open_application writes -- must stay on the sequential path."""
        actions = [
            Action(tool="open_application", args={"app_name": "notepad"}),
            Action(tool="type_text", args={"text": "hello"}),
        ]
        self.assertFalse(_all_actions_independent(actions))

    def test_optional_actions_disqualify_the_parallel_path(self):
        actions = [
            Action(tool="open_application", args={"app_name": "chrome"}, optional=True),
            Action(tool="volume_down", args={"amount": 1}),
        ]
        self.assertFalse(_all_actions_independent(actions))


class ParallelExecutionTests(unittest.TestCase):
    def _runtime_with_fake_tools(self, delays=None):
        runtime = AgentRuntime()
        delays = delays or {}
        started = {}
        lock = threading.Lock()

        def fake_execute_tool(tool, args):
            with lock:
                started[tool] = time.perf_counter()
            time.sleep(delays.get(tool, 0.05))
            return {"success": True, "message": f"did {tool}"}

        return runtime, fake_execute_tool, started

    def test_independent_actions_actually_overlap_in_wall_clock_time(self):
        runtime, fake_execute_tool, started = self._runtime_with_fake_tools(
            delays={"open_application": 0.2, "volume_down": 0.2}
        )
        actions = [
            Action(tool="open_application", args={"app_name": "discord"}),
            Action(tool="volume_down", args={"amount": 1}),
        ]
        from brain.models import Plan
        with patch("brain.executor.execute_tool", side_effect=fake_execute_tool):
            begin = time.perf_counter()
            results = runtime.execute(Plan("open discord and lower the volume", actions))
            elapsed = time.perf_counter() - begin
        self.assertTrue(all(r.success for r in results))
        self.assertLess(elapsed, 0.35, "two 0.2s independent actions should overlap, not sum to >=0.4s")
        gap = abs(started["open_application"] - started["volume_down"])
        self.assertLess(gap, 0.15, "both actions should start at roughly the same time")

    def test_results_are_returned_in_original_action_order(self):
        runtime, fake_execute_tool, _ = self._runtime_with_fake_tools(
            delays={"open_application": 0.15, "volume_down": 0.01, "mute_volume": 0.05}
        )
        actions = [
            Action(tool="open_application", args={"app_name": "discord"}),
            Action(tool="volume_down", args={"amount": 1}),
            Action(tool="mute_volume", args={}),
        ]
        from brain.models import Plan
        with patch("brain.executor.execute_tool", side_effect=fake_execute_tool):
            results = runtime.execute(Plan("cmd", actions))
        self.assertEqual([r.tool for r in results], ["open_application", "volume_down", "mute_volume"])

    def test_dependent_plan_keeps_strict_sequential_ordering(self):
        order = []

        def fake_execute_tool(tool, args):
            order.append(tool)
            return {"success": True, "message": "ok", "pid": 123, "hwnd": 456}

        actions = [
            Action(tool="open_application", args={"app_name": "notepad"}),
            Action(tool="type_text", args={"text": "hello"}),
        ]
        from brain.models import Plan
        runtime = AgentRuntime()
        with patch("brain.executor.execute_tool", side_effect=fake_execute_tool), \
             patch("brain.agent_runtime.type_into_notepad_native", return_value={"success": True, "message": "typed"}):
            runtime.execute(Plan("open notepad and type hello", actions))
        self.assertEqual(order[0], "open_application")

    def test_one_failure_does_not_block_the_other_independent_results(self):
        def fake_execute_tool(tool, args):
            if tool == "open_application":
                return {"success": False, "message": "failed", "error": "boom"}
            return {"success": True, "message": "ok"}

        actions = [
            Action(tool="open_application", args={"app_name": "discord"}),
            Action(tool="volume_down", args={"amount": 1}),
        ]
        from brain.models import Plan
        runtime = AgentRuntime()
        with patch("brain.executor.execute_tool", side_effect=fake_execute_tool):
            plan = Plan("cmd", actions)
            results = runtime.execute(plan)
        self.assertFalse(results[0].success)
        self.assertTrue(results[1].success)
        self.assertEqual(plan.status, PlanStatus.FAILED)
        self.assertEqual(plan.completed_actions, [1])


if __name__ == "__main__":
    unittest.main()
