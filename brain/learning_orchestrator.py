"""The "approve" and "start learning" halves of the voice-approved continual
learning pipeline: `handle_verified_teacher_success` (Phases 2, 5, 6, 7) and
`start_learning` (Phases 10-13, 16, 19, 21, 22, 25).

Deliberately never imports anything from `voice/` -- this codebase's
existing layering is `voice/*` depends on `brain/*`, never the reverse (see
`voice/background_assistant.py`'s imports of `brain.agent`/`brain.router`).
`handle_verified_teacher_success` accepts `request_approval` as an injected
callable instead of importing `voice.learning_approval` directly, so it
stays voice-agnostic and trivially testable with a fake.

Reuses `brain.task_supervisor.CancellationToken` -- the SAME primitive
`brain/improvement_orchestrator.py::run_attempt` already uses -- for
cancellation (Phase 20). Long-running-task bookkeeping (Phase 19,
"integrate with TaskSupervisor") is deliberately left to the caller via
`brain.task_supervisor.register_interactive_task`/`unregister_interactive_task`,
the same lightweight interactive-task registry
`voice/background_assistant.py` already uses for other long voice-triggered
actions -- not the heavier `TaskSupervisor`/`TaskStore` class, which is
built around a different problem (an autonomous edit/test coding loop with
a `ReasoningBackend`) that a batch training pipeline doesn't fit.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from brain.experience_store import ExperienceRecord, ExperienceStore, get_experience_store
from brain.improvement_attempt_models import ImprovementAttempt
from brain.improvement_coding_agent import CodingAgent
from brain.learning_dataset import build_dataset_version
from brain.learning_evaluation import Benchmark, PromotionGateConfig, evaluate_candidate
from brain.learning_models import ApprovalStatus, LearningJob, LearningJobStatus
from brain.learning_package import extract_learning_package, load_learning_package, save_learning_package
from brain.learning_store import LearningJobStore, get_learning_job_store
from brain.learning_trigger import evaluate_learning_offer
from brain.learning_training import (
    ModelRegistry, ModelVersion, TrainingBackend, TrainingConfig, TrainingPolicy,
    get_model_registry, policy_allows_training, run_pre_training_checks,
)
from brain.learning_validator import validate_variants
from brain.learning_variation import VariationConfig, generate_variants
from brain.task_supervisor import CancellationToken
from training_data.sanitizer import sanitize_text

log = logging.getLogger("jarvis.learning")

# LearningJobStatus values that mean "a previous start_learning run was
# interrupted mid-flight" -- recovered (truthfully, per Phase 25) back to
# READY_FOR_TRAINING at the top of the next run, rather than staying stuck
# forever or being silently marked complete.
_RECOVERABLE_IN_PROGRESS_STATUSES = {
    LearningJobStatus.PREPARING_DATA.value, LearningJobStatus.GENERATING_VARIANTS.value,
    LearningJobStatus.VALIDATING_DATA.value, LearningJobStatus.TRAINING.value,
    LearningJobStatus.EVALUATING.value, LearningJobStatus.PROMOTING.value,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApprovalResultLike(Protocol):
    outcome: Any  # str or str-enum; compared by value below
    transcript: str | None


ApprovalRequester = Callable[..., ApprovalResultLike]


# ----------------------------------------------------------------------
# Phase 2/5/6/7: verified teacher success -> voice offer -> LearningJob
# ----------------------------------------------------------------------

@dataclass
class LearningOfferOutcome:
    offered: bool
    job: LearningJob | None = None
    approval_outcome: str | None = None
    reason: str = ""


def handle_verified_teacher_success(
    attempt: ImprovementAttempt,
    *,
    request_approval: ApprovalRequester,
    job_store: LearningJobStore | None = None,
    experience_store: ExperienceStore | None = None,
    timeout_seconds: float | None = None,
    cancellation_token: CancellationToken | None = None,
) -> LearningOfferOutcome:
    """Call this once for every completed `ImprovementAttempt` (not only
    successful ones) -- Phase 2's eligibility gate and Phase 5's dedup both
    run through `evaluate_learning_offer` here, so most attempts return
    immediately with `offered=False` and never touch voice/approval at all.
    Never raises for an ordinary ineligible/declined/timed-out outcome.
    """
    job_store = job_store or get_learning_job_store()
    experience_store = experience_store or get_experience_store()

    decision = evaluate_learning_offer(attempt, store=job_store)
    if not decision.should_offer:
        return LearningOfferOutcome(offered=False, reason=decision.reason)

    kwargs: dict[str, Any] = {}
    if timeout_seconds is not None:
        kwargs["timeout_seconds"] = timeout_seconds
    if cancellation_token is not None:
        kwargs["cancellation_token"] = cancellation_token
    result = request_approval(**kwargs)
    outcome_value = getattr(result.outcome, "value", result.outcome)

    job = LearningJob(
        learning_job_id=uuid.uuid4().hex, created_at=_now(), updated_at=_now(),
        candidate_id=attempt.candidate_id, improvement_attempt_id=attempt.attempt_id,
        original_request=sanitize_text(attempt.original_request or ""), task_family=decision.task_family,
        subsystem=attempt.subsystem, gap_type=attempt.gap_type, claude_teacher_used=True,
        verification_evidence={"acceptance_gates": attempt.acceptance_gates, "evaluator_reason": sanitize_text(attempt.evaluator_reason or "")},
        approval_requested_at=_now(), approval_source="voice", fingerprint=decision.fingerprint,
    )

    if outcome_value == "APPROVED":
        job.approval_status = ApprovalStatus.APPROVED.value
        job.approval_response = sanitize_text(result.transcript or "")
        job.approved_at = _now()
        job.learning_status = LearningJobStatus.APPROVED.value
        job_store.create(job)
        # Phase 6 + 7: deterministic, local, instant -- never gated behind
        # "start learning" (that's reserved for Claude-costing variation
        # generation + training).
        package = extract_learning_package(attempt, learning_job_id=job.learning_job_id)
        save_learning_package(package)
        experience_store.store(ExperienceRecord.from_package(package, experience_id=uuid.uuid4().hex))
        job.learning_status = LearningJobStatus.READY_FOR_TRAINING.value
        job_store.update(job)
        return LearningOfferOutcome(True, job, outcome_value, "approved and queued for training")

    if outcome_value == "DECLINED":
        job.approval_status = ApprovalStatus.DECLINED.value
        job.approval_response = sanitize_text(result.transcript or "")
        job.learning_status = LearningJobStatus.DECLINED.value
        job_store.create(job)
        return LearningOfferOutcome(True, job, outcome_value, "declined by voice")

    # TIMED_OUT or CANCELLED both mean "no" (Phase 3: "timeout = NO").
    job.approval_status = ApprovalStatus.TIMED_OUT.value
    job.learning_status = LearningJobStatus.APPROVAL_TIMED_OUT.value
    job_store.create(job)
    return LearningOfferOutcome(True, job, outcome_value, "no valid answer within the approval window")


# ----------------------------------------------------------------------
# Phases 10-13, 16, 19, 21, 22, 25: "start learning"
# ----------------------------------------------------------------------

@dataclass
class LearningRunSummary:
    run_id: str
    started_at: str
    job_count: int = 0
    status: str = "PREPARING_DATA"
    completed_at: str | None = None
    dataset_version: str | None = None
    training_run_id: str | None = None
    candidate_model_version: str | None = None
    promoted: bool | None = None
    reasons: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


ProgressCallback = Callable[[str, str], None]


def _report(callback: ProgressCallback | None, status: str, detail: str = "") -> None:
    if callback is None:
        return
    try:
        callback(status, detail)
    except Exception:
        log.exception("learning progress callback failed")


def start_learning(
    *,
    coding_agent: CodingAgent,
    repository_root: str,
    backend: TrainingBackend,
    benchmark: Benchmark,
    job_store: LearningJobStore | None = None,
    model_registry: ModelRegistry | None = None,
    policy: TrainingPolicy | None = None,
    training_config: TrainingConfig | None = None,
    variation_config: VariationConfig | None = None,
    gate_config: PromotionGateConfig | None = None,
    dataset_root: str | None = None,
    package_root: str | None = None,
    explicit_command: bool = True,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
) -> LearningRunSummary:
    """The deterministic "Hey Jarvis, start learning" pipeline. Never raises
    -- every failure mode (empty queue, policy block, pre-training checks
    failing, backend failure, cancellation) ends in a returned
    `LearningRunSummary` with an honest `status`/`reasons`/`error`, never a
    propagated exception and never a corrupted registry/job state."""
    job_store = job_store or get_learning_job_store()
    model_registry = model_registry or get_model_registry()
    policy = policy or TrainingPolicy()
    training_config = training_config or TrainingConfig()

    summary = LearningRunSummary(run_id=uuid.uuid4().hex, started_at=_now())

    def cancelled() -> bool:
        return bool(cancellation_token is not None and cancellation_token.cancelled)

    def cancel_summary(jobs_to_release: list[LearningJob]) -> LearningRunSummary:
        for job in jobs_to_release:
            job.learning_status = LearningJobStatus.READY_FOR_TRAINING.value
            job_store.update(job)
        summary.status = "CANCELLED"
        summary.completed_at = _now()
        return summary

    try:
        # Phase 25: recover anything an interrupted previous run left
        # mid-flight, truthfully, before gathering this run's batch.
        for job in job_store.query(limit=2000):
            if job.learning_status in _RECOVERABLE_IN_PROGRESS_STATUSES:
                job.learning_status = LearningJobStatus.READY_FOR_TRAINING.value
                job_store.update(job)

        jobs = job_store.query_trainable()
        summary.job_count = len(jobs)
        if not jobs:
            summary.status = "COMPLETED"
            summary.completed_at = _now()
            summary.reasons = ["no approved learning jobs to train on"]
            return summary

        allowed, policy_reason = policy_allows_training(
            policy, job_count=len(jobs), example_count=len(jobs), explicit_command=explicit_command,
        )
        if not allowed:
            summary.status = "FAILED"
            summary.completed_at = _now()
            summary.reasons = [policy_reason]
            return summary

        if cancelled():
            return cancel_summary(jobs)

        _report(progress_callback, "PREPARING_DATA", f"{len(jobs)} approved job(s)")
        for job in jobs:
            job.learning_status = LearningJobStatus.PREPARING_DATA.value
            job_store.update(job)

        batch: list[tuple[LearningJob, Any, list[tuple[Any, Any]]]] = []
        for job in jobs:
            if cancelled():
                return cancel_summary(jobs)
            package = load_learning_package(job.learning_job_id, root=package_root)
            if package is None:
                log.warning("no saved LearningPackage for job %s -- excluded from this dataset build", job.learning_job_id)
                batch.append((job, None, []))
                continue

            _report(progress_callback, "GENERATING_VARIANTS", job.learning_job_id)
            job.learning_status = LearningJobStatus.GENERATING_VARIANTS.value
            job_store.update(job)
            variants = generate_variants(
                package, coding_agent=coding_agent, repository_root=repository_root,
                config=variation_config or VariationConfig(),
            )
            job.variants_generated = len(variants)
            job_store.update(job)

            if cancelled():
                return cancel_summary(jobs)

            _report(progress_callback, "VALIDATING_DATA", job.learning_job_id)
            job.learning_status = LearningJobStatus.VALIDATING_DATA.value
            job_store.update(job)
            verified, _unverified = validate_variants(variants)
            job.variants_verified = len(verified)
            job_store.update(job)

            verified_ids = {r.variant_id for r in verified}
            pairs = [(v, r) for v in variants for r in verified if v.variant_id == r.variant_id and v.variant_id in verified_ids]
            batch.append((job, package, pairs))

        if cancelled():
            return cancel_summary(jobs)

        real_batch = [(job, package, pairs) for job, package, pairs in batch if package is not None]
        manifest = build_dataset_version(real_batch, dataset_root=dataset_root)
        summary.dataset_version = manifest.dataset_version
        for job in jobs:
            job.dataset_version_added_to = manifest.dataset_version
            job_store.update(job)

        check = run_pre_training_checks(job_count=len(jobs), example_count=manifest.example_count, backend=backend, config=training_config)
        if not check.ready:
            summary.status = "FAILED"
            summary.completed_at = _now()
            summary.reasons = list(check.reasons)
            summary.error = "pre-training checks failed"
            if check.plan is not None:
                summary.reasons.append(f"training plan for the configured backend: {check.plan}")
            # Data already built and versioned is kept (Phase 16's "never
            # lose approved teacher data" applies just as much to a training
            # run that never got to start as to one that finished badly).
            for job in jobs:
                job.learning_status = LearningJobStatus.READY_FOR_TRAINING.value
                job_store.update(job)
            return summary

        if cancelled():
            return cancel_summary(jobs)

        _report(progress_callback, "TRAINING", manifest.dataset_version)
        for job in jobs:
            job.learning_status = LearningJobStatus.TRAINING.value
            job_store.update(job)
        result = backend.run(manifest.jsonl_path, training_config, cancellation_token=cancellation_token)
        summary.training_run_id = result.training_run_id

        if result.exit_status == "cancelled" or cancelled():
            return cancel_summary(jobs)
        if result.exit_status != "completed":
            summary.status = "FAILED"
            summary.completed_at = _now()
            summary.error = result.error
            for job in jobs:
                job.learning_status = LearningJobStatus.READY_FOR_TRAINING.value
                job.training_run_id = result.training_run_id
                job_store.update(job)
            return summary

        _report(progress_callback, "EVALUATING", result.model_version or "")
        for job in jobs:
            job.learning_status = LearningJobStatus.EVALUATING.value
            job_store.update(job)
        candidate = ModelVersion(
            model_version=result.model_version, dataset_version=manifest.dataset_version,
            training_run_id=result.training_run_id, created_at=_now(), metrics=result.metrics,
            base_model=result.base_model, adapter_path=result.checkpoint_path, config_hash=result.config_hash,
        )
        model_registry.record(candidate)
        active = model_registry.get_active()
        outcome = evaluate_candidate(
            old_model_version=active.model_version if active else None,
            new_model_version=candidate.model_version, benchmark=benchmark, gate_config=gate_config,
        )
        model_registry.mark_evaluated(candidate.model_version, outcome.new_metrics.to_dict())
        summary.candidate_model_version = candidate.model_version
        summary.promoted = outcome.promote
        summary.reasons = [outcome.reason]

        if outcome.promote:
            _report(progress_callback, "PROMOTING", candidate.model_version)
            for job in jobs:
                job.learning_status = LearningJobStatus.PROMOTING.value
                job_store.update(job)
            model_registry.promote(candidate.model_version)
            for job in jobs:
                job.learning_status = LearningJobStatus.TRAINED.value
                job.training_run_id = result.training_run_id
                job.model_version_result = candidate.model_version
                job_store.update(job)
        else:
            # Phase 16: never replace the active model; never lose the
            # approved teacher data -- jobs go back to READY_FOR_TRAINING so
            # a future run can try again with more/different data.
            model_registry.reject(candidate.model_version, outcome.reason)
            for job in jobs:
                job.learning_status = LearningJobStatus.READY_FOR_TRAINING.value
                job.training_run_id = result.training_run_id
                job_store.update(job)

        summary.status = "COMPLETED"
        summary.completed_at = _now()
        _report(progress_callback, "COMPLETED", summary.status)
        return summary

    except Exception as exc:  # crash recovery: never propagate, never leave jobs stuck
        log.exception("start_learning crashed")
        summary.status = "FAILED"
        summary.completed_at = _now()
        summary.error = f"{type(exc).__name__}: {exc}"
        try:
            for job in job_store.query(limit=2000):
                if job.learning_status in _RECOVERABLE_IN_PROGRESS_STATUSES:
                    job.learning_status = LearningJobStatus.READY_FOR_TRAINING.value
                    job_store.update(job)
        except Exception:
            log.exception("failed to recover job statuses after start_learning crash")
        return summary
