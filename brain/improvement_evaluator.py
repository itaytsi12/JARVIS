"""Evidence-based, gate-driven evaluation of a single ImprovementAttempt.

This is the module the project's hard invariants are aimed at most directly:
no attempt may reach READY_FOR_REVIEW because the coding agent *said* it
fixed something, and no attempt may reach READY_FOR_REVIEW merely because a
test suite exited 0. `evaluate()` only ever certifies an attempt once every
gate in `acceptance_gates` is independently True, and each gate is computed
from a structural fact (a diff, a reproduction result, a before/after test
comparison) -- never from agent-authored prose.

The one gate that matters most, `behavioral_improvement_confirmed`, requires
one of two concrete forms of proof:
  1. The original problem was independently reproduced as a failure
     (`before_repro.reproduced is True`) and, after the coding agent's
     change, the same reproduction no longer fails
     (`after_repro.reproduced is False`); or
  2. The coding agent wrote a new regression test that demonstrably FAILS
     against the unpatched base commit and PASSES against its own fix
     (`generated_test_differential is True`) -- the standard "did you prove
     it" bar for an autonomous patch when live reproduction isn't possible
     (see brain/improvement_repro.py's docstring for why that's common).
Anything short of one of those two is INCONCLUSIVE, never IMPROVED, no
matter how clean the diff or how many unrelated tests pass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from brain.improvement_attempt_models import EvaluationResult
from brain.improvement_diff_analysis import DiffAnalysis
from brain.improvement_repro import ReproductionResult

_FAILED_TEST_LINE = re.compile(r"^FAILED\s+(\S+)", re.MULTILINE)


def parse_failed_tests(pytest_output: str) -> set[str]:
    """Extract failing test node ids from pytest's `-q` short summary
    section (the "FAILED tests/x.py::test_y - ..." lines). This section is
    always emitted at the end of the run, which matters because
    `SafeCommandRunner` truncates captured output to its tail -- the summary
    survives truncation even when earlier output doesn't."""
    return set(_FAILED_TEST_LINE.findall(pytest_output or ""))


@dataclass
class RegressionComparison:
    new_failures: set[str] = field(default_factory=set)
    resolved_failures: set[str] = field(default_factory=set)
    preexisting_failures: set[str] = field(default_factory=set)

    @property
    def has_new_failures(self) -> bool:
        return bool(self.new_failures)


def compare_regression(before_result: dict[str, Any] | None, after_result: dict[str, Any] | None) -> RegressionComparison | None:
    """Diff two full-suite runs by failing test NAME, not by exit code, so a
    pre-existing/environmental failure (e.g. a test needing a dependency
    that isn't installed) is never blamed on this attempt -- only a failure
    that is new after the attempt's change counts as a regression."""
    if before_result is None or after_result is None:
        return None
    before_failed = parse_failed_tests(before_result.get("output", ""))
    after_failed = parse_failed_tests(after_result.get("output", ""))
    return RegressionComparison(
        new_failures=after_failed - before_failed,
        resolved_failures=before_failed - after_failed,
        preexisting_failures=before_failed & after_failed,
    )


@dataclass
class EvaluationOutcome:
    result: str
    confidence: float
    reason: str
    acceptance_gates: dict[str, bool]
    regression_detected: bool
    ready_for_review: bool


def evaluate(
    *,
    diff: DiffAnalysis,
    before_repro: ReproductionResult,
    after_repro: ReproductionResult | None,
    focused_tests_result: dict[str, Any] | None,
    regression_before: dict[str, Any] | None,
    regression_after: dict[str, Any] | None,
    generated_test_differential: bool | None = None,
) -> EvaluationOutcome:
    regression = compare_regression(regression_before, regression_after)
    regression_detected = bool(regression and regression.has_new_failures)

    behavioral_confirmed = bool(
        (before_repro.reproduced is True and after_repro is not None and after_repro.reproduced is False)
        or generated_test_differential is True
    )

    gates = {
        "no_unauthorized_commit": not diff.unauthorized_commit,
        "diff_not_suspicious": not diff.diff_suspicious,
        "change_made": diff.change_scope != "none",
        "focused_tests_passed": bool(focused_tests_result) and focused_tests_result.get("exit_code") == 0,
        "regression_no_new_failures": regression is not None and not regression.has_new_failures,
        "behavioral_improvement_confirmed": behavioral_confirmed,
    }

    satisfied = sum(gates.values())
    confidence = round(satisfied / len(gates), 2)
    ready_for_review = all(gates.values())

    if regression_detected:
        result = EvaluationResult.REGRESSED.value
        reason = f"new test failures introduced: {sorted(regression.new_failures)}"
    elif not gates["change_made"]:
        result = EvaluationResult.NOT_IMPROVED.value
        reason = "coding agent produced no change to the worktree"
    elif not gates["no_unauthorized_commit"] or not gates["diff_not_suspicious"]:
        result = EvaluationResult.INCONCLUSIVE.value
        reason = f"diff flagged as untrustworthy: {'; '.join(diff.diff_suspicious_reasons) or 'unauthorized commit detected'}"
    elif ready_for_review:
        result = EvaluationResult.IMPROVED.value
        reason = "behavioral evidence confirms the original problem is fixed, focused and regression tests pass, and the diff is clean"
    elif not gates["behavioral_improvement_confirmed"]:
        result = EvaluationResult.INCONCLUSIVE.value
        reason = (
            "no independent evidence that the original problem is actually fixed "
            "(neither a confirmed before/after reproduction nor a differential generated test) -- "
            "tests passing alone is not sufficient evidence of improvement"
        )
    else:
        missing = [name for name, ok in gates.items() if not ok]
        result = EvaluationResult.INCONCLUSIVE.value
        reason = f"not all acceptance gates satisfied: {', '.join(missing)}"

    return EvaluationOutcome(
        result=result,
        confidence=confidence,
        reason=reason,
        acceptance_gates=gates,
        regression_detected=regression_detected,
        ready_for_review=ready_for_review and result == EvaluationResult.IMPROVED.value,
    )
