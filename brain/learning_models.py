"""Structured representation of one voice-approved continual-learning job.

A `LearningJob` represents ONE task the user has explicitly approved (by
voice) for learning, after JARVIS's existing self-improvement pipeline
(`brain/improvement_*.py`) already independently verified a teacher (Claude)
solution reached `READY_FOR_REVIEW`. This module defines the schema only;
see `brain/learning_store.py` for persistence, `brain/learning_trigger.py`
for when a job should be offered, and `brain/learning_orchestrator.py` for
how approved jobs turn into a training run.

This is a new, distinct concern from `ImprovementAttempt` (a single
teacher-fix attempt) on purpose: an attempt is "did Claude fix this one
bug," which already exists and is not duplicated here. A `LearningJob` is
"did the user say yes to learning from that fix," which is a downstream,
separately-approved, separately-lifecycled record -- one attempt produces
at most one learning job, but a learning job's own lifecycle (approval,
dataset inclusion, training, promotion) is independent of the attempt's.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

LEARNING_JOB_SCHEMA_VERSION = 1


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DECLINED = "DECLINED"
    TIMED_OUT = "TIMED_OUT"


class LearningJobStatus(str, Enum):
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    DECLINED = "DECLINED"
    APPROVAL_TIMED_OUT = "APPROVAL_TIMED_OUT"
    PREPARING_DATA = "PREPARING_DATA"
    GENERATING_VARIANTS = "GENERATING_VARIANTS"
    VALIDATING_DATA = "VALIDATING_DATA"
    READY_FOR_TRAINING = "READY_FOR_TRAINING"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    PROMOTING = "PROMOTING"
    TRAINED = "TRAINED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    ARCHIVED = "ARCHIVED"


# Statuses that will never spontaneously change again without a new,
# explicit action (a fresh approval, a fresh "start learning" run). Mirrors
# brain/improvement_attempt_models.py's TERMINAL_ATTEMPT_STATUSES pattern.
TERMINAL_LEARNING_JOB_STATUSES = {
    LearningJobStatus.DECLINED.value,
    LearningJobStatus.APPROVAL_TIMED_OUT.value,
    LearningJobStatus.TRAINED.value,
    LearningJobStatus.FAILED.value,
    LearningJobStatus.CANCELLED.value,
    LearningJobStatus.ARCHIVED.value,
}

# Statuses a "start learning" run should gather up as its input batch.
TRAINABLE_LEARNING_JOB_STATUSES = {
    LearningJobStatus.APPROVED.value,
    LearningJobStatus.READY_FOR_TRAINING.value,
}


class DataQualityLabel(str, Enum):
    """Strict provenance/quality labels for every example that can end up in
    a training dataset (Phase 23). Never mixed blindly -- see
    brain/learning_dataset.py for how each is weighted/filtered."""
    REAL_VERIFIED_TEACHER = "REAL_VERIFIED_TEACHER"
    REAL_VERIFIED_STUDENT = "REAL_VERIFIED_STUDENT"
    REAL_FAILED_STUDENT = "REAL_FAILED_STUDENT"
    REAL_FAILED_TEACHER = "REAL_FAILED_TEACHER"
    SYNTHETIC_DERIVED = "SYNTHETIC_DERIVED"
    SYNTHETIC_VERIFIED = "SYNTHETIC_VERIFIED"


@dataclass
class LearningJob:
    # ------------------------------------------------------------------
    # IDENTITY
    # ------------------------------------------------------------------
    learning_job_id: str
    created_at: str
    updated_at: str
    candidate_id: str
    improvement_attempt_id: str
    trajectory_id: str | None = None

    # ------------------------------------------------------------------
    # SOURCE (frozen snapshot of the verified attempt this job came from)
    # ------------------------------------------------------------------
    original_request: str = ""
    task_family: str = ""
    subsystem: str | None = None
    gap_type: str = ""
    claude_teacher_used: bool = False
    verification_evidence: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # APPROVAL
    # ------------------------------------------------------------------
    approval_status: str = ApprovalStatus.PENDING.value
    approval_requested_at: str | None = None
    approval_deadline: str | None = None
    approval_response: str | None = None
    approval_source: str = "voice"
    approved_at: str | None = None

    # ------------------------------------------------------------------
    # LEARNING
    # ------------------------------------------------------------------
    learning_status: str = LearningJobStatus.PENDING_APPROVAL.value
    variation_generation_status: str = "NOT_STARTED"
    variants_generated: int = 0
    variants_verified: int = 0
    dataset_version_added_to: str | None = None
    training_run_id: str | None = None
    model_version_result: str | None = None

    # ------------------------------------------------------------------
    # DEDUPLICATION (Phase 5) -- a stable identity for the underlying
    # verified capability, deliberately excluding raw wording, so different
    # phrasings of the same fix never create two approval prompts.
    # ------------------------------------------------------------------
    fingerprint: str = ""

    # ------------------------------------------------------------------
    # ACTIVE-LEARNING PRIORITY (Part A, Phase A9) -- set when this job's
    # teacher fix followed a genuine, verified STUDENT failure on the same
    # task: the highest-value kind of teacher example, since it's proof the
    # current student model specifically could not do this. Read (never
    # written) by `brain/learning_dataset.py` so a future training pass can
    # prioritize/oversample these rows -- purely additive metadata, the
    # dataset row itself is unchanged either way.
    # ------------------------------------------------------------------
    high_value: bool = False
    high_value_reason: str | None = None

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------
    error: str | None = None
    schema_version: int = LEARNING_JOB_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LearningJob":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})
