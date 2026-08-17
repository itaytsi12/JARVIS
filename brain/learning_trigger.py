"""Decide whether a completed ImprovementAttempt should trigger a voice
learning-approval offer (Phase 2), and compute the dedup fingerprint used to
suppress repeated offers for the same underlying capability (Phase 5).

Deliberately reuses `brain/improvement_evaluator.py`'s existing gate-driven
verdict instead of re-deriving "was this really a genuine, safe, reusable
fix" from scratch: `ImprovementAttempt.status == READY_FOR_REVIEW` can only
ever be set by `brain/improvement_orchestrator.py::run_attempt` after EVERY
acceptance gate in `attempt.acceptance_gates` was independently True --
including `behavioral_improvement_confirmed`, which itself requires either a
confirmed before/after reproduction or a differential test that fails on the
unpatched base and passes on the fix. That already rules out every case
Phase 2 says must never trigger an offer (environmental failure, ambiguity,
cancellation, safety refusal, partial/unverified fixes, Claude failure) --
none of those can reach READY_FOR_REVIEW in the first place. This module
only adds two things the orchestrator doesn't already guarantee: that the
teacher was actually the real Claude Code adapter (not a fake/local one),
and that no equivalent job is already in flight.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from brain.improvement_attempt_models import AttemptStatus, EvaluationResult, ImprovementAttempt
from brain.improvement_models import stable_hash
from brain.improvement_triage import _HIGH_RISK_SUBSYSTEMS
from brain.learning_models import LearningJob
from brain.learning_store import LearningJobStore, get_learning_job_store

REAL_TEACHER_PROVIDER = "claude_code"


@dataclass
class LearningOfferDecision:
    should_offer: bool
    reason: str
    fingerprint: str | None = None
    existing_job: LearningJob | None = None
    task_family: str = ""


def task_family_fingerprint(attempt: ImprovementAttempt) -> str:
    """A stable identity for the underlying VERIFIED CAPABILITY, deliberately
    excluding raw request wording -- built from `attempt.candidate_fingerprint`
    (already an evidence-derived, wording-free identity computed by
    `brain/improvement_observer.py::_fingerprint`: gap_type + subsystem +
    tool/exception/block-reason shape) plus the shape of the verified
    strategy itself (change_scope, whether a differential regression test
    was produced). Two different-wording requests that were fixed the same
    underlying way hash identically; two requests that merely sound similar
    but were fixed differently do not.
    """
    payload = {
        "candidate_fingerprint": attempt.candidate_fingerprint,
        "gap_type": attempt.gap_type,
        "subsystem": attempt.subsystem,
        "change_scope": attempt.change_scope,
        "has_generated_test": bool(attempt.generated_tests),
    }
    return stable_hash(payload)[:24]


def evaluate_learning_offer(
    attempt: ImprovementAttempt, *, store: LearningJobStore | None = None,
) -> LearningOfferDecision:
    """Pure decision function (aside from the dedup store read): does this
    attempt warrant asking "Do you want me to learn how to do that, sir?"
    Never speaks, never writes -- the caller decides what to do with the
    decision."""
    if attempt.status != AttemptStatus.READY_FOR_REVIEW.value:
        return LearningOfferDecision(False, f"attempt status is {attempt.status!r}, not READY_FOR_REVIEW")

    if attempt.agent_provider != REAL_TEACHER_PROVIDER:
        return LearningOfferDecision(
            False, f"teacher provider was {attempt.agent_provider!r}; a learning offer requires the real Claude teacher",
        )

    if attempt.evaluation != EvaluationResult.IMPROVED.value:
        return LearningOfferDecision(False, f"evaluation result was {attempt.evaluation!r}, not IMPROVED")

    gates = attempt.acceptance_gates or {}
    if not gates or not all(gates.values()):
        missing = [name for name, ok in gates.items() if not ok]
        return LearningOfferDecision(False, f"not all acceptance gates satisfied: {missing}")

    # Extra, deliberately conservative defense-in-depth: the same
    # high-risk-subsystem set brain/improvement_triage.py already treats
    # cautiously. A READY_FOR_REVIEW attempt for one of these subsystems is
    # already rare (repro refuses live replay for them), but a learning
    # offer is a *further* commitment (voice-approved, feeds training data),
    # so it gets its own explicit check rather than trusting inherited gates
    # transitively.
    if attempt.subsystem in _HIGH_RISK_SUBSYSTEMS:
        return LearningOfferDecision(False, f"subsystem {attempt.subsystem!r} is high-risk; learning offers are not made for it")

    fingerprint = task_family_fingerprint(attempt)
    store = store or get_learning_job_store()
    existing = store.find_active_by_fingerprint(fingerprint)
    if existing is not None:
        return LearningOfferDecision(
            False, f"an equivalent learning job already exists (status={existing.learning_status})",
            fingerprint=fingerprint, existing_job=existing, task_family=fingerprint,
        )

    return LearningOfferDecision(True, "verified teacher fix is eligible for a learning offer", fingerprint=fingerprint, task_family=fingerprint)
