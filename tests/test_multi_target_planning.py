"""Tests for coordinated targets and connector-aware dependencies in
`brain/task_planner.py`.

Two related defects motivated these:

- "open Spotify and VS Code" planned only Spotify. The second target was
  silently dropped, because the clause splitter only splits before a command
  VERB and "vs code" is not one, and the single-clause path then bailed out.
- Every clause was wired to depend on the previous one regardless of whether
  anything actually required that order, so nothing a user asked for could
  ever be scheduled concurrently.

The fixes are general mechanisms -- a coordinated object list, and a
dependency derived from the connector plus what the tool needs -- so these
tests assert the mechanism, not the example sentences.
"""
from __future__ import annotations

import unittest

from brain.execution_graph import build_waves, describe_schedule
from brain.task_planner import (
    create_task_plan,
    segment_sequential_commands,
    segment_with_connectors,
    split_coordinated_targets,
    states_an_order,
)


class SegmentationCompatibilityTests(unittest.TestCase):
    """The pre-existing splitting behaviour must be unchanged."""

    def test_a_connector_inside_a_quoted_payload_is_not_a_split_point(self):
        self.assertEqual(segment_sequential_commands('type "hello and then save it"'), ['type "hello and then save it"'])

    def test_a_connector_not_followed_by_a_command_verb_is_not_a_split_point(self):
        self.assertEqual(segment_sequential_commands("type rock and roll"), ["type rock and roll"])

    def test_ordinary_sequences_still_split(self):
        self.assertEqual(segment_sequential_commands("open notepad and type hello"), ["open notepad", "type hello"])


class ConnectorTests(unittest.TestCase):
    def test_segments_report_how_they_were_joined(self):
        self.assertEqual(segment_with_connectors("open notepad and type hello"), [("", "open notepad"), (" and ", "type hello")])

    def test_explicit_sequencing_words_state_an_order(self):
        for connector in (", then ", " then ", " and then ", " after that ", " next "):
            self.assertTrue(states_an_order(connector), connector)

    def test_a_bare_and_or_comma_does_not_state_an_order(self):
        self.assertFalse(states_an_order(" and "))
        self.assertFalse(states_an_order(""))


class CoordinatedTargetTests(unittest.TestCase):
    def test_a_list_of_known_targets_splits(self):
        self.assertEqual(split_coordinated_targets("spotify and vs code"), ["spotify", "vs code"])
        self.assertEqual(split_coordinated_targets("chrome, spotify and notepad"), ["chrome", "spotify", "notepad"])

    def test_a_phrase_that_merely_contains_and_is_left_alone(self):
        # A document called "my report and notes" must never be torn in half:
        # splitting only happens when EVERY piece is independently a known app
        # or website.
        self.assertEqual(split_coordinated_targets("my report and notes"), ["my report and notes"])

    def test_a_single_target_is_returned_unchanged(self):
        self.assertEqual(split_coordinated_targets("spotify"), ["spotify"])


class MultiTargetPlanTests(unittest.TestCase):
    def _tools(self, plan):
        return [(action.tool, action.args.get("app_name") or action.args.get("url")) for action in plan.actions]

    def test_both_coordinated_targets_are_planned(self):
        plan = create_task_plan("open spotify and vs code")
        apps = [args for tool, args in self._tools(plan) if tool == "open_application"]
        self.assertEqual(len(apps), 2, f"both targets must be planned, got {self._tools(plan)}")
        self.assertNotEqual(apps[0], apps[1])

    def test_coordinated_launches_carry_no_dependency_on_each_other(self):
        plan = create_task_plan("open chrome and spotify")
        launches = [action for action in plan.actions if action.tool == "open_application"]
        self.assertTrue(all(not action.depends_on for action in launches))

    def test_coordinated_launches_are_scheduled_in_one_parallel_wave(self):
        plan = create_task_plan("open chrome and spotify")
        first_wave = describe_schedule(plan.actions)[0]
        self.assertEqual(len(first_wave["parallel"]), 2)
        self.assertEqual(first_wave["sequential"], [])

    def test_a_window_dependent_clause_still_waits_for_its_window(self):
        # "type" needs the right window in front, so it must NOT float into an
        # earlier wave no matter how the clause was joined.
        plan = create_task_plan("open notepad and type hello")
        tools = [action.tool for action in plan.actions]
        self.assertEqual(tools, ["open_application", "wait_for_window", "type_text"])
        self.assertEqual(build_waves(plan.actions), [[0], [1], [2]])

    def test_an_explicit_then_preserves_order_for_a_self_contained_clause(self):
        plan = create_task_plan("open notepad, then type hello")
        self.assertEqual(build_waves(plan.actions), [[0], [1], [2]])

    def test_each_launched_app_still_waits_for_its_own_window(self):
        plan = create_task_plan("open chrome and spotify")
        for index, action in enumerate(plan.actions):
            if action.tool == "wait_for_window":
                dependency = plan.actions[action.depends_on[0]]
                self.assertEqual(dependency.tool, "open_application")
                self.assertEqual(dependency.args["app_name"], action.args["app_name"])


if __name__ == "__main__":
    unittest.main()
