"""Tests for `brain/recovery.py` and its bounded use by `AgentRuntime`.

The property that matters most here is the one that is easiest to get wrong:
recovery must be strictly bounded and must never be able to feed itself.
"""
from __future__ import annotations

import unittest

from brain.agent_runtime import AgentRuntime
from brain.models import Action, ActionRisk, Plan, ToolResult
from brain.recovery import MAX_RECOVERY_ACTIONS, Recovery, plan_recovery


def _failed(tool: str, error: str, **data) -> ToolResult:
    return ToolResult(False, tool, "failed", data=data, error=error)


class StrategySelectionTests(unittest.TestCase):
    def test_an_unverified_window_proposes_waiting_again(self):
        action = Action("open_application", {"app_name": "spotify"})
        recovery = plan_recovery(action, _failed("open_application", "application_window_unverified"))
        self.assertIsNotNone(recovery)
        self.assertEqual([a.tool for a in recovery.actions], ["wait_for_window"])
        self.assertEqual(recovery.actions[0].args["app_name"], "spotify")

    def test_an_ambiguous_application_asks_instead_of_guessing(self):
        action = Action("open_application", {"app_name": "code"})
        recovery = plan_recovery(action, _failed("open_application", "ambiguous_application", candidates=["VS Code", "Code Writer"]))
        self.assertIsNotNone(recovery)
        self.assertEqual(recovery.actions, [])
        self.assertIn("VS Code", recovery.clarification)

    def test_a_success_never_triggers_recovery(self):
        self.assertIsNone(plan_recovery(Action("open_application", {"app_name": "x"}), ToolResult(True, "open_application", "ok")))

    def test_decisions_and_misattributed_failures_are_never_recovered(self):
        action = Action("open_application", {"app_name": "x"})
        for error in ("cancelled", "human_confirmation_required", "dependency_failure", "resource_timeout"):
            self.assertIsNone(plan_recovery(action, _failed("open_application", error)), error)

    def test_an_unresolved_reference_is_never_recovered(self):
        action = Action("write_text_file", {})
        self.assertIsNone(plan_recovery(action, _failed("write_text_file", "unresolved_reference: no such field")))

    def test_a_non_safe_action_is_never_recovered(self):
        action = Action("open_application", {"app_name": "x"}, risk=ActionRisk.HIGH_IMPACT)
        self.assertIsNone(plan_recovery(action, _failed("open_application", "application_window_unverified")))

    def test_an_unknown_failure_has_no_strategy(self):
        self.assertIsNone(plan_recovery(Action("open_application", {"app_name": "x"}), _failed("open_application", "something_new")))

    def test_the_recovery_action_budget_is_enforced(self):
        self.assertGreaterEqual(MAX_RECOVERY_ACTIONS, 1)
        self.assertLessEqual(MAX_RECOVERY_ACTIONS, 3, "recovery must stay strictly bounded")

    def test_an_empty_recovery_is_reported_as_no_recovery(self):
        self.assertTrue(Recovery(reason="x").is_empty())


class _RecoveringRuntime(AgentRuntime):
    """First `open_application` reports an unverified window; the follow-up
    `wait_for_window` succeeds unless configured otherwise."""

    def __init__(self, *, wait_succeeds=True):
        super().__init__(trace=False)
        self.calls: list[str] = []
        self._wait_succeeds = wait_succeeds

    def _execute_action(self, action, cancellation_token=None, plan_lock_held=False):
        self.calls.append(action.tool)
        if action.tool == "open_application":
            return _failed("open_application", "application_window_unverified")
        if action.tool == "wait_for_window":
            if self._wait_succeeds:
                return ToolResult(True, "wait_for_window", "window is up", data={"hwnd": 1})
            return _failed("wait_for_window", "window_not_found")
        return ToolResult(True, action.tool, "ok")


class RuntimeRecoveryTests(unittest.TestCase):
    def test_a_successful_recovery_makes_the_original_action_succeed(self):
        plan = Plan("open spotify", [Action("open_application", {"app_name": "spotify"})])
        runtime = _RecoveringRuntime()
        results = runtime.execute(plan)
        self.assertTrue(results[0].success)
        self.assertTrue(results[0].data["recovery_succeeded"])
        self.assertEqual(results[0].data["original_error"], "application_window_unverified")
        self.assertEqual(runtime.calls, ["open_application", "wait_for_window"])

    def test_a_failed_recovery_leaves_the_original_failure_standing(self):
        plan = Plan("open spotify", [Action("open_application", {"app_name": "spotify"})])
        runtime = _RecoveringRuntime(wait_succeeds=False)
        results = runtime.execute(plan)
        self.assertFalse(results[0].success, "a failed recovery must never invent a success")
        self.assertEqual(results[0].error, "application_window_unverified")
        self.assertFalse(results[0].data["recovery_succeeded"])

    def test_recovery_never_recurses_and_stays_bounded(self):
        # wait_for_window itself fails. If recovery were recursive, each
        # failed recovery would generate another one and this would not
        # terminate. Exactly one recovery CYCLE may happen: the original
        # action is attempted once, the recovery action is attempted only
        # within its own declared max_attempts budget, and no further
        # recovery is generated from its failure.
        plan = Plan("open spotify", [Action("open_application", {"app_name": "spotify"})])
        runtime = _RecoveringRuntime(wait_succeeds=False)
        runtime.execute(plan)
        self.assertEqual(runtime.calls.count("open_application"), 1, "the original action must not be re-driven by recovery")
        waits = runtime.calls.count("wait_for_window")
        self.assertGreaterEqual(waits, 1)
        self.assertLessEqual(waits, 3, "bounded by the recovery action's own max_attempts, never unbounded")
        self.assertEqual(set(runtime.calls), {"open_application", "wait_for_window"}, "recovery must not spawn new kinds of work")

    def test_an_ambiguous_application_becomes_a_spoken_clarification(self):
        class Ambiguous(AgentRuntime):
            def _execute_action(self, action, cancellation_token=None, plan_lock_held=False):
                return _failed("open_application", "ambiguous_application", candidates=["VS Code", "Code Writer"])

        plan = Plan("open code", [Action("open_application", {"app_name": "code"})])
        results = Ambiguous(trace=False).execute(plan)
        self.assertFalse(results[0].success)
        self.assertEqual(results[0].error, "needs_clarification")
        self.assertIn("Which one did you mean?", results[0].message)


if __name__ == "__main__":
    unittest.main()
