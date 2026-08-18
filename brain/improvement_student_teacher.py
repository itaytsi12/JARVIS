"""Student-first, teacher-fallback coding-task orchestration (Part A).

The missing "front door" between a live coding-task request and the
existing self-improvement pipeline: resolves the ACTIVE local student model
(`brain.learning_training.ModelRegistry`), retrieves relevant verified
experiences (`brain.experience_store`), attempts the task with the student
FIRST via the EXACT SAME isolated-worktree/diff-analysis/evaluation
machinery already used for the Claude teacher
(`brain.improvement_orchestrator.run_attempt` -- called once per agent,
never a second worktree/verification implementation), and only calls the
Claude teacher when the student is unavailable, fails to load, or fails the
task. Teacher success still flows through the EXISTING voice-approved
learning trigger (`brain.learning_orchestrator.handle_verified_teacher_success`)
-- never a second approval implementation.

Sits alongside `brain/improvement_orchestrator.py` in the same
`brain/improvement_*.py` family (see CLAUDE.md) -- this module is one layer
above it, not a competitor.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from brain.experience_store import ExperienceStore, ScoredExperience, get_experience_store, retrieve_relevant_experiences
from brain.improvement_attempt_models import AttemptStatus, ImprovementAttempt
from brain.improvement_attempt_store import ImprovementAttemptStore
from brain.improvement_coding_agent import ClaudeCodeAdapter, CodingAgent
from brain.improvement_models import ImprovementCandidate
from brain.improvement_orchestrator import OrchestratorConfig, run_attempt
from brain.learning_orchestrator import ApprovalRequester, LearningOfferOutcome, handle_verified_teacher_success
from brain.learning_store import LearningJobStore
from brain.learning_training import ModelRegistry, ModelVersion, get_model_registry
from brain.student_trajectory_store import StudentTrajectory, StudentTrajectoryStore, get_student_trajectory_store
from brain.task_supervisor import CancellationToken

log = logging.getLogger("jarvis.student_teacher")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StudentTeacherConfig:
    """Bounded resource limits for the student's attempt (Phase A11: never
    let a weak local model burn unlimited time before falling back). The
    teacher attempt reuses `brain.improvement_orchestrator.OrchestratorConfig`'s
    own defaults unless overridden."""
    student_max_revision_rounds: int = 1
    student_coding_agent_timeout_seconds: float = 300.0
    student_focused_test_timeout_seconds: float = 120.0
    student_regression_test_timeout_seconds: float = 300.0
    student_max_total_seconds: float = 900.0
    student_max_new_tokens: int = 512
    teacher_config: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    experience_top_k: int = 5

    def student_orchestrator_config(self) -> OrchestratorConfig:
        return OrchestratorConfig(
            max_revision_rounds=self.student_max_revision_rounds,
            coding_agent_timeout_seconds=self.student_coding_agent_timeout_seconds,
            focused_test_timeout_seconds=self.student_focused_test_timeout_seconds,
            regression_test_timeout_seconds=self.student_regression_test_timeout_seconds,
            max_total_seconds=self.student_max_total_seconds,
        )


@dataclass
class CodingTaskResult:
    """Answers every Phase A10 observability question structurally --
    queryable without reading raw terminal logs (also persisted: see
    `brain/student_trajectory_store.py` for the student side and the
    existing `LearningJobStore`/`ImprovementAttemptStore` for the teacher
    side)."""
    task: str
    candidate_id: str
    started_at: str
    student_available: bool = False
    student_model_version: str | None = None
    student_used: bool = False
    student_succeeded: bool = False
    student_skip_reason: str | None = None
    student_attempt_id: str | None = None
    teacher_used: bool = False
    teacher_succeeded: bool = False
    teacher_attempt_id: str | None = None
    solved_by: str = "none"  # "student" | "teacher" | "none"
    learning_offer: LearningOfferOutcome | None = None
    experiences_used: list[str] = field(default_factory=list)
    high_value_example: bool = False
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_active_student(model_registry: ModelRegistry | None = None) -> ModelVersion | None:
    """Only a model whose registry status is literally `ACTIVE` is ever
    returned (Phase A2) -- `CANDIDATE`/`EVALUATED`/`REJECTED`/`REPLACED`/
    `ARCHIVED` are never eligible. `ModelRegistry.get_active()` already
    enforces "at most one ACTIVE row" as an invariant; this function adds
    no new logic beyond calling it, deliberately -- there is exactly one
    place `ACTIVE` gets assigned (`ModelRegistry.promote`), and this must
    never second-guess it."""
    registry = model_registry or get_model_registry()
    return registry.get_active()


def build_student_agent(active: ModelVersion, *, max_iterations: int = 2, max_new_tokens: int = 512) -> CodingAgent | None:
    """Constructs `training.code_model.student_adapter.LocalCodingModelAdapter`
    from the ACTIVE registry entry's OWN recorded `base_model`/`adapter_path`
    -- never a hardcoded smoke-test model (Phase A3). Returns `None` (never
    raises) if the artifact can't be loaded, so the caller can fall back to
    Claude truthfully instead of crashing the whole task."""
    if not active.base_model:
        log.warning("[student] ACTIVE model %s has no recorded base_model; cannot construct a student agent", active.model_version)
        return None
    try:
        from training.code_model.student_adapter import LocalCodingModelAdapter
        return LocalCodingModelAdapter.from_checkpoint(
            active.base_model, active.adapter_path, max_iterations=max_iterations, max_new_tokens=max_new_tokens,
        )
    except Exception as exc:
        log.warning("[student] failed to load ACTIVE model %s: %s: %s", active.model_version, type(exc).__name__, exc)
        return None


def _experience_context_block(experiences: list[ScoredExperience]) -> str:
    """Phase A4: a SMALL, bounded slice of prior verified strategies -- text
    only, never the whole experience database."""
    if not experiences:
        return ""
    lines = ["Relevant prior verified fixes (context only -- do not assume they apply verbatim to this task):"]
    for scored in experiences:
        record = scored.experience
        lines.append(f"- {record.reusable_strategy} (problem family: {record.problem_family})")
    return "\n".join(lines)


def _with_extra_evidence(candidate: ImprovementCandidate, extra: str) -> ImprovementCandidate:
    payload = candidate.to_dict()
    payload["classification_reason"] = (payload.get("classification_reason") or "") + ("\n" + extra if extra else "")
    return ImprovementCandidate.from_dict(payload)


def _student_failure_evidence(attempt: ImprovementAttempt) -> str:
    """Phase A7: observable evidence only (files touched, diff summary,
    focused/regression test results, evaluator reason) -- never hidden
    reasoning, because this codebase never captures that for any coding
    agent in the first place."""
    lines = [
        "A local student model already attempted this task and did not succeed. Its observable evidence:",
        f"- final status: {attempt.status}",
        f"- files touched: {attempt.files_changed or '(none)'}",
        f"- diff summary: {attempt.diff_summary or '(no change)'}",
    ]
    if attempt.focused_tests_result:
        lines.append(f"- focused test result: exit_code={attempt.focused_tests_result.get('exit_code')}")
    if attempt.regression_result:
        lines.append(f"- regression test result: exit_code={attempt.regression_result.get('exit_code')}")
    if attempt.evaluator_reason:
        lines.append(f"- evaluator reason: {attempt.evaluator_reason}")
    if attempt.error:
        lines.append(f"- error: {attempt.error}")
    lines.append("You do not need to repeat the student's exact failed approach -- consider a different strategy.")
    return "\n".join(lines)


def _record_trajectory(
    store: StudentTrajectoryStore, *, candidate_id: str, task: str, quality_label: str,
    student_model_version: str | None, student_attempt_id: str | None, teacher_attempt_id: str | None,
    solved_by: str, high_value: bool, evidence: dict[str, Any],
) -> StudentTrajectory:
    trajectory = StudentTrajectory(
        trajectory_id=uuid.uuid4().hex, created_at=_now(), candidate_id=candidate_id, task=task,
        quality_label=quality_label, student_model_version=student_model_version,
        student_attempt_id=student_attempt_id, teacher_attempt_id=teacher_attempt_id,
        solved_by=solved_by, high_value=high_value, evidence=evidence,
    )
    store.record(trajectory)
    return trajectory


def run_coding_task(
    task: str,
    *,
    repository_root: str,
    model_registry: ModelRegistry | None = None,
    experience_store: ExperienceStore | None = None,
    attempt_store: ImprovementAttemptStore | None = None,
    job_store: LearningJobStore | None = None,
    trajectory_store: StudentTrajectoryStore | None = None,
    request_approval: ApprovalRequester | None = None,
    config: StudentTeacherConfig | None = None,
    cancellation_token: CancellationToken | None = None,
    gap_type: str = "CODE_CAPABILITY_GAP",
    subsystem: str | None = None,
    teacher_agent: CodingAgent | None = None,
) -> CodingTaskResult:
    """The Part A production flow. Never raises -- every failure mode
    (no student, student load failure, student task failure, teacher
    failure) ends in an honest, fully-populated `CodingTaskResult`, mirroring
    `brain.improvement_orchestrator.run_attempt`'s own "never propagate"
    guarantee.

    `teacher_agent` defaults to the real `ClaudeCodeAdapter()` -- the only
    reason it's an injectable parameter at all is so tests can supply a
    `FakeCodingAgent` instead, exactly like every other test in this
    codebase's improvement-pipeline suite; production callers should never
    pass it."""
    config = config or StudentTeacherConfig()
    model_registry = model_registry or get_model_registry()
    experience_store = experience_store or get_experience_store()
    trajectory_store = trajectory_store or get_student_trajectory_store()
    started = _now()
    candidate_id = uuid.uuid4().hex

    candidate = ImprovementCandidate(
        candidate_id=candidate_id, created_at=started, first_seen=started, last_seen=started,
        gap_type=gap_type, confidence=0.9, subsystem=subsystem,
        raw_request=task, normalized_goal=task,
        classification_reason="direct user coding-task request",
    )

    result = CodingTaskResult(task=task, candidate_id=candidate_id, started_at=started)

    try:
        experiences = retrieve_relevant_experiences(
            task, subsystem=subsystem, gap_type=gap_type, top_k=config.experience_top_k, store=experience_store,
        )
    except Exception:
        log.exception("[student-teacher] experience retrieval failed; proceeding without context")
        experiences = []
    result.experiences_used = [scored.experience.experience_id for scored in experiences]
    experience_block = _experience_context_block(experiences)

    student_attempt: ImprovementAttempt | None = None
    active = resolve_active_student(model_registry)
    if active is None:
        result.student_skip_reason = "no ACTIVE student model"
    else:
        result.student_available = True
        result.student_model_version = active.model_version
        student_agent = build_student_agent(active, max_new_tokens=config.student_max_new_tokens)
        if student_agent is None:
            result.student_skip_reason = f"ACTIVE model {active.model_version!r} could not be loaded"
        else:
            result.student_used = True
            student_candidate = _with_extra_evidence(candidate, experience_block) if experience_block else candidate
            student_attempt = run_attempt(
                student_candidate, repository_root=repository_root, coding_agent=student_agent,
                attempt_store=attempt_store, config=config.student_orchestrator_config(), cancellation_token=cancellation_token,
            )
            result.student_attempt_id = student_attempt.attempt_id
            result.student_succeeded = student_attempt.status == AttemptStatus.READY_FOR_REVIEW.value

    if result.student_used and student_attempt is not None:
        _record_trajectory(
            trajectory_store, candidate_id=candidate_id, task=task,
            quality_label="REAL_VERIFIED_STUDENT" if result.student_succeeded else "REAL_FAILED_STUDENT",
            student_model_version=result.student_model_version, student_attempt_id=student_attempt.attempt_id,
            teacher_attempt_id=None, solved_by="student" if result.student_succeeded else "none", high_value=False,
            evidence={
                "status": student_attempt.status, "revision_rounds": student_attempt.revision_rounds,
                "agent_model_calls": student_attempt.agent_model_calls,
                "duration_ms": student_attempt.duration_ms, "evaluation": student_attempt.evaluation,
            },
        )

    if result.student_succeeded:
        result.solved_by = "student"
        result.completed_at = _now()
        return result

    # Teacher fallback (Phase A7): student unavailable, failed to load, or
    # failed the task. Claude receives the original task plus observable
    # student-failure evidence, never hidden reasoning.
    result.teacher_used = True
    teacher_candidate = candidate
    if student_attempt is not None:
        teacher_candidate = _with_extra_evidence(teacher_candidate, _student_failure_evidence(student_attempt))
    if experience_block:
        teacher_candidate = _with_extra_evidence(teacher_candidate, experience_block)

    teacher_attempt = run_attempt(
        teacher_candidate, repository_root=repository_root, coding_agent=teacher_agent or ClaudeCodeAdapter(),
        attempt_store=attempt_store, config=config.teacher_config, cancellation_token=cancellation_token,
    )
    result.teacher_attempt_id = teacher_attempt.attempt_id
    result.teacher_succeeded = teacher_attempt.status == AttemptStatus.READY_FOR_REVIEW.value

    if result.teacher_succeeded:
        result.solved_by = "teacher"
        # Phase A9: the highest-value kind of teacher example -- a genuine,
        # verified student failure on the exact same task, followed by a
        # verified teacher success.
        result.high_value_example = result.student_used
        if request_approval is not None:
            high_value_reason = (
                f"local student model {result.student_model_version} attempted this task and failed "
                f"(status={student_attempt.status if student_attempt else 'unknown'}); Claude teacher then succeeded"
                if result.high_value_example else None
            )
            offer = handle_verified_teacher_success(
                teacher_attempt, request_approval=request_approval, job_store=job_store,
                experience_store=experience_store, cancellation_token=cancellation_token,
                high_value=result.high_value_example, high_value_reason=high_value_reason,
            )
            result.learning_offer = offer
    else:
        _record_trajectory(
            trajectory_store, candidate_id=candidate_id, task=task, quality_label="REAL_FAILED_TEACHER",
            student_model_version=result.student_model_version, student_attempt_id=result.student_attempt_id,
            teacher_attempt_id=teacher_attempt.attempt_id, solved_by="none", high_value=False,
            evidence={"status": teacher_attempt.status, "error": teacher_attempt.error},
        )

    result.completed_at = _now()
    return result
