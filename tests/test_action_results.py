"""Tests for `brain/action_results.py` -- passing one action's result into a
later action's arguments, and the `ToolResult` field view that makes it work.
"""
from __future__ import annotations

import unittest

from brain.action_results import (
    ReferenceError_,
    is_reference,
    plan_has_references,
    reference_targets,
    resolve_arg_references,
    result_artifacts,
    result_field,
    result_metadata,
    result_status,
    result_summary,
    result_text,
    with_reference_dependencies,
)
from brain.agent_runtime import AgentRuntime
from brain.models import Action, Plan, ToolResult


def _ref(action: int, field: str = "summary") -> dict:
    return {"__from_result__": {"action": action, "field": field}}


class ResultFieldViewTests(unittest.TestCase):
    def setUp(self):
        self.result = ToolResult(
            True, "run_command", "2 tests failed",
            data={
                "failures": ["a::one", "b::two"],
                "stdout": "collected 5 items\nFAILED a::one",
                "exit_code": 1,
                "path": "report.txt",
            },
        )

    def test_status_and_success(self):
        self.assertEqual(result_status(self.result), "ok")
        self.assertEqual(result_field(self.result, "success"), True)
        self.assertEqual(result_status(ToolResult(False, "t", "no", error="boom")), "failed")

    def test_summary_prefers_a_tool_supplied_summary(self):
        self.assertEqual(result_summary(self.result), "2 tests failed")
        withsummary = ToolResult(True, "t", "msg", data={"summary": "explicit"})
        self.assertEqual(result_summary(withsummary), "explicit")

    def test_summary_falls_back_to_the_error_rather_than_inventing_one(self):
        self.assertEqual(result_summary(ToolResult(False, "t", "", error="disk_full")), "disk_full")

    def test_text_prefers_structured_payload_over_the_short_message(self):
        self.assertIn("collected 5 items", result_text(self.result))
        self.assertEqual(result_text(ToolResult(True, "t", "just a message")), "just a message")

    def test_dotted_data_paths_and_list_indices(self):
        self.assertEqual(result_field(self.result, "data.exit_code"), 1)
        self.assertEqual(result_field(self.result, "data.failures"), ["a::one", "b::two"])
        self.assertEqual(result_field(self.result, "data.failures.0"), "a::one")

    def test_artifacts_and_metadata_views(self):
        self.assertEqual(result_artifacts(self.result), ["report.txt"])
        self.assertNotIn("stdout", result_metadata(self.result))
        self.assertIn("exit_code", result_metadata(self.result))

    def test_an_unknown_field_raises_rather_than_returning_none(self):
        with self.assertRaises(ReferenceError_):
            result_field(self.result, "not_a_field")
        with self.assertRaises(ReferenceError_):
            result_field(self.result, "data.nope")


class ReferenceResolutionTests(unittest.TestCase):
    def setUp(self):
        self.results = {
            0: ToolResult(True, "run_command", "2 tests failed", data={"failures": ["a::one", "b::two"], "exit_code": 1}),
        }

    def test_is_reference_only_matches_the_exact_marker_shape(self):
        self.assertTrue(is_reference(_ref(0)))
        self.assertFalse(is_reference({"__from_result__": {"action": 0}, "extra": 1}))
        self.assertFalse(is_reference({"action": 0}))
        self.assertFalse(is_reference("plain string"))

    def test_a_reference_is_replaced_by_the_named_field(self):
        args = resolve_arg_references({"note": _ref(0, "summary")}, self.results)
        self.assertEqual(args["note"], "2 tests failed")

    def test_a_list_field_renders_as_lines_for_a_text_argument(self):
        args = resolve_arg_references({"contents": _ref(0, "data.failures")}, self.results)
        self.assertEqual(args["contents"], "a::one\nb::two")

    def test_references_nested_in_lists_and_dicts_are_resolved(self):
        args = resolve_arg_references({"outer": {"inner": [_ref(0, "data.exit_code")]}}, self.results)
        self.assertEqual(args["outer"]["inner"][0], 1)

    def test_non_reference_arguments_pass_through_untouched(self):
        args = resolve_arg_references({"path": "x.txt", "n": 3, "flag": True}, self.results)
        self.assertEqual(args, {"path": "x.txt", "n": 3, "flag": True})

    def test_referencing_a_result_that_does_not_exist_raises(self):
        with self.assertRaises(ReferenceError_):
            resolve_arg_references({"x": _ref(7)}, self.results)

    def test_a_malformed_reference_raises_rather_than_being_ignored(self):
        with self.assertRaises(ReferenceError_):
            resolve_arg_references({"x": {"__from_result__": {"field": "summary"}}}, self.results)
        with self.assertRaises(ReferenceError_):
            resolve_arg_references({"x": {"__from_result__": {"action": -1}}}, self.results)


class ImpliedDependencyTests(unittest.TestCase):
    def test_reference_targets_are_discovered_anywhere_in_the_arguments(self):
        action = Action("write_text_file", {"a": _ref(0), "b": {"c": [_ref(2)]}})
        self.assertEqual(reference_targets(action), {0, 2})

    def test_a_reference_implies_a_dependency_even_when_not_declared(self):
        actions = [Action("run_command", {}), Action("write_text_file", {"contents": _ref(0)})]
        self.assertEqual(actions[1].depends_on, [])
        updated = with_reference_dependencies(actions)
        self.assertEqual(updated[1].depends_on, [0])

    def test_an_already_declared_dependency_is_not_duplicated(self):
        actions = [Action("run_command", {}), Action("w", {"c": _ref(0)}, depends_on=[0])]
        self.assertEqual(with_reference_dependencies(actions)[1].depends_on, [0])

    def test_plan_has_references_detects_the_mechanism_is_in_use(self):
        self.assertFalse(plan_has_references([Action("open_application", {"app_name": "x"})]))
        self.assertTrue(plan_has_references([Action("w", {"c": _ref(0)})]))


class _RefRuntime(AgentRuntime):
    def __init__(self):
        super().__init__(trace=False)
        self.seen: dict[str, dict] = {}

    def _execute_action(self, action, cancellation_token=None, plan_lock_held=False):
        self.seen[action.tool] = dict(action.args)
        if action.tool == "run_command":
            return ToolResult(True, "run_command", "2 tests failed", data={"failures": ["a::one", "b::two"], "summary": "2 tests failed"})
        return ToolResult(True, action.tool, "ok")


class ResultPassingThroughTheRuntimeTests(unittest.TestCase):
    def test_a_later_action_receives_the_earlier_structured_result(self):
        plan = Plan("run the tests and write the failures down", [
            Action("run_command", {"command": "pytest"}),
            Action("write_text_file", {"path": "f.txt", "contents": _ref(0, "data.failures")}, depends_on=[0]),
        ])
        runtime = _RefRuntime()
        results = runtime.execute(plan)
        self.assertTrue(all(r.success for r in results))
        self.assertEqual(runtime.seen["write_text_file"]["contents"], "a::one\nb::two")

    def test_a_bad_reference_fails_only_the_action_that_made_it(self):
        plan = Plan("g", [
            Action("run_command", {"command": "pytest"}),
            Action("write_text_file", {"path": "f.txt", "contents": _ref(0, "data.missing")}, depends_on=[0]),
        ])
        results = _RefRuntime().execute(plan)
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)
        self.assertIn("unresolved_reference", results[1].error)

    def test_an_undeclared_reference_still_executes_in_the_right_order(self):
        plan = Plan("g", [
            Action("run_command", {"command": "pytest"}),
            Action("write_text_file", {"path": "f.txt", "contents": _ref(0, "summary")}),
        ])
        runtime = _RefRuntime()
        results = runtime.execute(plan)
        self.assertTrue(all(r.success for r in results))
        self.assertEqual(runtime.seen["write_text_file"]["contents"], "2 tests failed")


if __name__ == "__main__":
    unittest.main()
