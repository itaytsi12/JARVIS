"""Focused tests for brain/speculative_execution.py (Parts B/C)."""
from __future__ import annotations

import unittest

from brain.models import Action
from brain.speculative_execution import (
    PartialActionLedger,
    SAFE_PARTIAL_TOOLS,
    classify_partial_route,
    reconcile_final_route,
    reconcile_local_plan_actions,
)


class ClassifyPartialRouteTests(unittest.TestCase):
    def test_safe_examples_from_the_spec_are_eligible(self):
        for text in ("open spotify", "open chrome", "open discord", "open youtube", "volume down", "mute"):
            with self.subTest(text=text):
                route = classify_partial_route(text)
                self.assertIsNotNone(route)
                self.assertIn(route["tool"], SAFE_PARTIAL_TOOLS)

    def test_unsafe_examples_from_the_spec_are_never_eligible(self):
        unsafe = (
            "send a message to john", "email alex the report", "delete the file",
            "move the file to trash", "overwrite important.docx", "shut down the computer",
            "restart the pc", "buy this now", "install this package", "uninstall chrome",
            "delete everything in the folder", "close notepad",
        )
        for text in unsafe:
            with self.subTest(text=text):
                self.assertIsNone(classify_partial_route(text))

    def test_ambiguous_or_empty_partial_is_never_eligible(self):
        for text in ("", "open", "uh", "so i want to"):
            with self.subTest(text=text):
                self.assertIsNone(classify_partial_route(text))

    def test_router_failure_degrades_to_none_never_raises(self):
        # A pathological string that some regex in the router might choke
        # on must still degrade gracefully rather than propagate.
        self.assertIsNone(classify_partial_route(")" * 5000))


class PartialActionLedgerTests(unittest.TestCase):
    def test_single_partial_is_not_enough_to_fire(self):
        ledger = PartialActionLedger(min_stable=2)
        self.assertIsNone(ledger.observe_partial("open spotify"))

    def test_two_consecutive_stable_matches_fire_exactly_once(self):
        ledger = PartialActionLedger(min_stable=2)
        self.assertIsNone(ledger.observe_partial("open spotify"))
        action = ledger.observe_partial("open spotify")
        self.assertIsNotNone(action)
        self.assertEqual(action.route["tool"], "open_application")
        self.assertIsNone(ledger.observe_partial("open spotify"), "must not fire twice for the same stable candidate")

    def test_unstable_partials_never_fire(self):
        ledger = PartialActionLedger(min_stable=2)
        self.assertIsNone(ledger.observe_partial("open spotify"))
        self.assertIsNone(ledger.observe_partial("open chrome"))  # different candidate resets stability
        self.assertIsNone(ledger.observe_partial("open discord"))

    def test_a_noisy_partial_in_between_resets_stability(self):
        ledger = PartialActionLedger(min_stable=2)
        self.assertIsNone(ledger.observe_partial("open spotify"))
        self.assertIsNone(ledger.observe_partial("uh"))  # ambiguous partial in between
        self.assertIsNone(ledger.observe_partial("open spotify"))
        self.assertIsNotNone(ledger.observe_partial("open spotify"))

    def test_destructive_text_never_fires_regardless_of_stability(self):
        ledger = PartialActionLedger(min_stable=2)
        for _ in range(5):
            self.assertIsNone(ledger.observe_partial("send a message to john saying hi"))

    def test_already_fired_lookup_by_route_and_by_tool(self):
        ledger = PartialActionLedger(min_stable=1)
        action = ledger.observe_partial("open spotify")
        self.assertIsNotNone(action)
        route = {"type": "tool", "tool": "open_application", "arguments": {"app_name": "spotify"}}
        self.assertIs(ledger.already_fired(route), action)
        self.assertIs(ledger.already_fired_tool("open_application", {"app_name": "spotify"}), action)
        self.assertIsNone(ledger.already_fired_tool("open_application", {"app_name": "chrome"}))


class ReconcileFinalRouteTests(unittest.TestCase):
    def test_final_route_matching_a_fired_action_is_suppressed(self):
        ledger = PartialActionLedger(min_stable=1)
        ledger.observe_partial("open spotify")
        final_route = {"type": "tool", "tool": "open_application", "arguments": {"app_name": "spotify"}}
        route, matched = reconcile_final_route(ledger, final_route)
        self.assertIsNone(route)
        self.assertIsNotNone(matched)

    def test_changed_final_intent_still_executes_and_old_action_is_left_alone(self):
        ledger = PartialActionLedger(min_stable=1)
        ledger.observe_partial("open spotify")
        final_route = {"type": "tool", "tool": "open_application", "arguments": {"app_name": "chrome"}}
        route, matched = reconcile_final_route(ledger, final_route)
        self.assertEqual(route, final_route)
        self.assertIsNone(matched)

    def test_no_ledger_leaves_route_untouched(self):
        final_route = {"type": "tool", "tool": "open_application", "arguments": {"app_name": "chrome"}}
        route, matched = reconcile_final_route(None, final_route)
        self.assertEqual(route, final_route)
        self.assertIsNone(matched)

    def test_non_tool_route_types_are_never_reconciled_here(self):
        ledger = PartialActionLedger(min_stable=1)
        ledger.observe_partial("open spotify")
        final_route = {"type": "plan", "message": "open spotify and play my liked songs"}
        route, matched = reconcile_final_route(ledger, final_route)
        self.assertEqual(route, final_route)
        self.assertIsNone(matched)


class ReconcileLocalPlanActionsTests(unittest.TestCase):
    def test_leading_already_fired_action_is_dropped(self):
        ledger = PartialActionLedger(min_stable=1)
        ledger.observe_partial("open spotify")
        actions = [
            Action(tool="open_application", args={"app_name": "spotify"}),
            Action(tool="volume_up", args={"amount": 1}),
        ]
        remaining = reconcile_local_plan_actions(ledger, actions)
        self.assertEqual([a.tool for a in remaining], ["volume_up"])

    def test_full_match_can_empty_the_plan(self):
        ledger = PartialActionLedger(min_stable=1)
        ledger.observe_partial("open spotify")
        actions = [Action(tool="open_application", args={"app_name": "spotify"})]
        remaining = reconcile_local_plan_actions(ledger, actions)
        self.assertEqual(remaining, [])

    def test_non_matching_first_action_is_never_dropped(self):
        ledger = PartialActionLedger(min_stable=1)
        ledger.observe_partial("open spotify")
        actions = [
            Action(tool="volume_up", args={"amount": 1}),
            Action(tool="open_application", args={"app_name": "spotify"}),
        ]
        remaining = reconcile_local_plan_actions(ledger, actions)
        self.assertEqual([a.tool for a in remaining], ["volume_up", "open_application"])

    def test_dependent_action_is_never_trimmed(self):
        ledger = PartialActionLedger(min_stable=1)
        ledger.observe_partial("open spotify")
        actions = [Action(tool="open_application", args={"app_name": "spotify"}, depends_on=[0])]
        remaining = reconcile_local_plan_actions(ledger, actions)
        self.assertEqual(len(remaining), 1, "an action with dependency wiring must never be silently dropped")

    def test_no_ledger_or_empty_actions_is_a_no_op(self):
        actions = [Action(tool="open_application", args={"app_name": "spotify"})]
        self.assertEqual(reconcile_local_plan_actions(None, actions), actions)
        self.assertEqual(reconcile_local_plan_actions(PartialActionLedger(), []), [])


if __name__ == "__main__":
    unittest.main()
