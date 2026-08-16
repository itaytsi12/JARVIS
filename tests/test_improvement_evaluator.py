import unittest

from brain.improvement_attempt_models import ReproductionType, EvaluationResult
from brain.improvement_diff_analysis import DiffAnalysis
from brain.improvement_evaluator import compare_regression, evaluate, parse_failed_tests
from brain.improvement_repro import ReproductionResult


def _diff(**overrides) -> DiffAnalysis:
    defaults = dict(files_changed=["brain/agent.py"], change_scope="source_only")
    defaults.update(overrides)
    return DiffAnalysis(**defaults)


def _repro(reproduced, method=ReproductionType.INTEGRATION_REPRO.value, attempted=True) -> ReproductionResult:
    return ReproductionResult(method=method, attempted=attempted, reproduced=reproduced)


_PASS = {"exit_code": 0, "output": "5 passed in 1.2s"}


class ParseFailedTestsTests(unittest.TestCase):
    def test_extracts_failed_node_ids_from_short_summary(self):
        output = (
            "..F..F.. [100%]\n"
            "=========================== FAILURES ===========================\n"
            "...\n"
            "======================= short test summary info =======================\n"
            "FAILED tests/test_a.py::test_one - AssertionError: boom\n"
            "FAILED tests/test_b.py::test_two - ValueError\n"
            "2 failed, 6 passed in 1.0s\n"
        )
        self.assertEqual(parse_failed_tests(output), {"tests/test_a.py::test_one", "tests/test_b.py::test_two"})

    def test_clean_pass_has_no_failed_tests(self):
        self.assertEqual(parse_failed_tests("8 passed in 0.5s"), set())

    def test_empty_output_is_safe(self):
        self.assertEqual(parse_failed_tests(""), set())
        self.assertEqual(parse_failed_tests(None), set())


class CompareRegressionTests(unittest.TestCase):
    def test_no_change_between_before_and_after_is_no_regression(self):
        before = {"output": "3 passed"}
        after = {"output": "3 passed"}
        comparison = compare_regression(before, after)
        self.assertFalse(comparison.has_new_failures)

    def test_preexisting_failure_present_in_both_is_not_a_new_regression(self):
        before = {"output": "FAILED tests/test_a.py::test_flaky - Error\n1 failed"}
        after = {"output": "FAILED tests/test_a.py::test_flaky - Error\n1 failed"}
        comparison = compare_regression(before, after)
        self.assertFalse(comparison.has_new_failures)
        self.assertIn("tests/test_a.py::test_flaky", comparison.preexisting_failures)

    def test_new_failure_after_the_change_is_a_regression(self):
        before = {"output": "3 passed"}
        after = {"output": "FAILED tests/test_a.py::test_new_break - Error\n1 failed, 2 passed"}
        comparison = compare_regression(before, after)
        self.assertTrue(comparison.has_new_failures)
        self.assertIn("tests/test_a.py::test_new_break", comparison.new_failures)

    def test_resolved_failure_is_tracked_separately_from_new_failures(self):
        before = {"output": "FAILED tests/test_a.py::test_old_break - Error\n1 failed"}
        after = {"output": "3 passed"}
        comparison = compare_regression(before, after)
        self.assertFalse(comparison.has_new_failures)
        self.assertIn("tests/test_a.py::test_old_break", comparison.resolved_failures)

    def test_missing_before_or_after_returns_none(self):
        self.assertIsNone(compare_regression(None, {"output": ""}))
        self.assertIsNone(compare_regression({"output": ""}, None))


class EvaluateGateTests(unittest.TestCase):
    def test_full_evidence_reaches_improved_and_ready_for_review(self):
        outcome = evaluate(
            diff=_diff(),
            before_repro=_repro(True),
            after_repro=_repro(False),
            focused_tests_result=_PASS,
            regression_before=_PASS,
            regression_after=_PASS,
        )
        self.assertEqual(outcome.result, EvaluationResult.IMPROVED.value)
        self.assertTrue(outcome.ready_for_review)
        self.assertTrue(all(outcome.acceptance_gates.values()))

    def test_generated_test_differential_can_substitute_for_live_reproduction(self):
        outcome = evaluate(
            diff=_diff(change_scope="mixed", generated_tests=["tests/test_a.py"]),
            before_repro=_repro(None, method=ReproductionType.LOG_EVIDENCE_ONLY.value),
            after_repro=None,
            focused_tests_result=_PASS,
            regression_before=_PASS,
            regression_after=_PASS,
            generated_test_differential=True,
        )
        self.assertEqual(outcome.result, EvaluationResult.IMPROVED.value)
        self.assertTrue(outcome.ready_for_review)

    def test_new_regression_failure_overrides_everything_else_as_regressed(self):
        outcome = evaluate(
            diff=_diff(),
            before_repro=_repro(True),
            after_repro=_repro(False),
            focused_tests_result=_PASS,
            regression_before={"output": "10 passed"},
            regression_after={"output": "FAILED tests/test_x.py::test_break - Error\n1 failed, 9 passed"},
        )
        self.assertEqual(outcome.result, EvaluationResult.REGRESSED.value)
        self.assertTrue(outcome.regression_detected)
        self.assertFalse(outcome.ready_for_review)

    def test_no_change_at_all_is_not_improved(self):
        outcome = evaluate(
            diff=_diff(change_scope="none", files_changed=[]),
            before_repro=_repro(True),
            after_repro=_repro(True),
            focused_tests_result=None,
            regression_before=_PASS,
            regression_after=_PASS,
        )
        self.assertEqual(outcome.result, EvaluationResult.NOT_IMPROVED.value)
        self.assertFalse(outcome.ready_for_review)

    def test_suspicious_diff_is_inconclusive_never_improved_even_with_clean_tests(self):
        suspicious = _diff(diff_suspicious=True, diff_suspicious_reasons=["touches sensitive path(s): brain/task_supervisor.py"])
        outcome = evaluate(
            diff=suspicious,
            before_repro=_repro(True),
            after_repro=_repro(False),
            focused_tests_result=_PASS,
            regression_before=_PASS,
            regression_after=_PASS,
        )
        self.assertEqual(outcome.result, EvaluationResult.INCONCLUSIVE.value)
        self.assertFalse(outcome.ready_for_review)
        self.assertNotEqual(outcome.result, EvaluationResult.IMPROVED.value)

    def test_unauthorized_commit_is_inconclusive_never_improved(self):
        tampered = _diff(unauthorized_commit=True, diff_suspicious=True, diff_suspicious_reasons=["worktree HEAD has moved"])
        outcome = evaluate(
            diff=tampered,
            before_repro=_repro(True),
            after_repro=_repro(False),
            focused_tests_result=_PASS,
            regression_before=_PASS,
            regression_after=_PASS,
        )
        self.assertEqual(outcome.result, EvaluationResult.INCONCLUSIVE.value)
        self.assertFalse(outcome.ready_for_review)

    def test_passing_tests_alone_without_behavioral_confirmation_is_inconclusive(self):
        """The critical invariant: a clean diff and passing tests are NEVER
        by themselves sufficient for IMPROVED/ready_for_review."""
        outcome = evaluate(
            diff=_diff(),
            before_repro=_repro(None, method=ReproductionType.LOG_EVIDENCE_ONLY.value),
            after_repro=None,
            focused_tests_result=_PASS,
            regression_before=_PASS,
            regression_after=_PASS,
            generated_test_differential=None,
        )
        self.assertEqual(outcome.result, EvaluationResult.INCONCLUSIVE.value)
        self.assertFalse(outcome.ready_for_review)
        self.assertFalse(outcome.acceptance_gates["behavioral_improvement_confirmed"])

    def test_failed_focused_tests_blocks_ready_for_review_even_with_behavioral_confirmation(self):
        outcome = evaluate(
            diff=_diff(),
            before_repro=_repro(True),
            after_repro=_repro(False),
            focused_tests_result={"exit_code": 1, "output": "FAILED tests/test_a.py::test_x - Error"},
            regression_before=_PASS,
            regression_after=_PASS,
        )
        self.assertFalse(outcome.ready_for_review)
        self.assertFalse(outcome.acceptance_gates["focused_tests_passed"])

    def test_missing_regression_evidence_blocks_ready_for_review(self):
        outcome = evaluate(
            diff=_diff(),
            before_repro=_repro(True),
            after_repro=_repro(False),
            focused_tests_result=_PASS,
            regression_before=None,
            regression_after=None,
        )
        self.assertFalse(outcome.ready_for_review)
        self.assertFalse(outcome.acceptance_gates["regression_no_new_failures"])

    def test_confidence_is_the_transparent_fraction_of_satisfied_gates(self):
        outcome = evaluate(
            diff=_diff(),
            before_repro=_repro(True),
            after_repro=_repro(False),
            focused_tests_result=_PASS,
            regression_before=_PASS,
            regression_after=_PASS,
        )
        satisfied = sum(outcome.acceptance_gates.values())
        self.assertEqual(outcome.confidence, round(satisfied / len(outcome.acceptance_gates), 2))
        self.assertEqual(outcome.confidence, 1.0)

    def test_agent_claim_alone_cannot_force_improved(self):
        """Simulates a coding agent that reports success verbosely but left
        no real evidence behind -- evaluate() never looks at agent prose at
        all, so this must stay INCONCLUSIVE regardless of what a stdout
        summary might have claimed."""
        outcome = evaluate(
            diff=_diff(change_scope="none", files_changed=[]),
            before_repro=_repro(None, method=ReproductionType.LOG_EVIDENCE_ONLY.value),
            after_repro=None,
            focused_tests_result=None,
            regression_before=None,
            regression_after=None,
        )
        self.assertNotEqual(outcome.result, EvaluationResult.IMPROVED.value)
        self.assertFalse(outcome.ready_for_review)


if __name__ == "__main__":
    unittest.main()
