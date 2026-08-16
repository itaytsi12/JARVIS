"""Structured representation of one autonomous improvement attempt.

An `ImprovementAttempt` records everything about a single try at repairing
an `ImprovementCandidate`: the isolated workspace it ran in, whether the
original problem was reproduced, what the coding agent did, what validation
found, and the final evidence-based evaluation. See
`brain/improvement_orchestrator.py` for how attempts are driven end to end.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

ATTEMPT_SCHEMA_VERSION = 1


class AttemptStatus(str, Enum):
    QUEUED = "QUEUED"
    REPRODUCING = "REPRODUCING"
    IMPROVING = "IMPROVING"
    VALIDATING = "VALIDATING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    FIX_FAILED = "FIX_FAILED"
    FIX_PARTIAL = "FIX_PARTIAL"
    NOT_REPRODUCIBLE = "NOT_REPRODUCIBLE"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


TERMINAL_ATTEMPT_STATUSES = {
    AttemptStatus.READY_FOR_REVIEW.value, AttemptStatus.FIX_FAILED.value,
    AttemptStatus.FIX_PARTIAL.value, AttemptStatus.NOT_REPRODUCIBLE.value,
    AttemptStatus.VERIFICATION_FAILED.value, AttemptStatus.REJECTED.value,
    AttemptStatus.BLOCKED.value, AttemptStatus.CANCELLED.value,
}


class ReproductionType(str, Enum):
    UNIT_REPRO = "UNIT_REPRO"
    INTEGRATION_REPRO = "INTEGRATION_REPRO"
    BEHAVIORAL_REPLAY = "BEHAVIORAL_REPLAY"
    LOG_EVIDENCE_ONLY = "LOG_EVIDENCE_ONLY"
    MANUAL_ENVIRONMENT_REQUIRED = "MANUAL_ENVIRONMENT_REQUIRED"
    UNSAFE_TO_REPLAY = "UNSAFE_TO_REPLAY"


class EvaluationResult(str, Enum):
    IMPROVED = "IMPROVED"
    NOT_IMPROVED = "NOT_IMPROVED"
    REGRESSED = "REGRESSED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class ImprovementAttempt:
    # ------------------------------------------------------------------
    # IDENTITY
    # ------------------------------------------------------------------
    attempt_id: str
    candidate_id: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    status: str = AttemptStatus.QUEUED.value

    # ------------------------------------------------------------------
    # CANDIDATE SNAPSHOT (frozen at attempt creation -- never re-derived
    # from a live candidate row so an attempt's evidence stays stable even
    # if the candidate accrues more occurrences later)
    # ------------------------------------------------------------------
    candidate_fingerprint: str = ""
    gap_type: str = ""
    confidence: float = 0.0
    subsystem: str | None = None
    original_request: str = ""
    evidence_reference: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # WORKSPACE
    # ------------------------------------------------------------------
    repository_root: str = ""
    base_commit: str = ""
    base_branch: str = ""
    worktree_path: str | None = None
    worktree_branch: str | None = None
    workspace_state: str = "unallocated"

    # ------------------------------------------------------------------
    # REPRODUCTION
    # ------------------------------------------------------------------
    reproduction_attempted: bool = False
    reproduction_method: str | None = None
    reproduced: bool | None = None
    reproduction_evidence: dict[str, Any] = field(default_factory=dict)
    reproduction_skip_reason: str | None = None

    # ------------------------------------------------------------------
    # AGENT
    # ------------------------------------------------------------------
    agent_provider: str | None = None
    agent_model: str | None = None
    agent_invocation_mode: str | None = None
    agent_model_calls: int = 0
    agent_task_prompt_reference: str | None = None
    agent_started_at: str | None = None
    agent_ended_at: str | None = None
    agent_exit_status: str | None = None
    revision_rounds: int = 0

    # ------------------------------------------------------------------
    # PATCH
    # ------------------------------------------------------------------
    files_changed: list[str] = field(default_factory=list)
    files_added: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    diff_summary: str = ""
    lines_added: int = 0
    lines_removed: int = 0
    generated_tests: list[str] = field(default_factory=list)
    change_scope: str = "unknown"
    diff_suspicious: bool = False
    diff_suspicious_reasons: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------
    focused_tests_requested: list[str] = field(default_factory=list)
    focused_tests_result: dict[str, Any] = field(default_factory=dict)
    regression_result: dict[str, Any] = field(default_factory=dict)
    behavioral_replay_result: dict[str, Any] = field(default_factory=dict)
    before_state: dict[str, Any] = field(default_factory=dict)
    after_state: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------
    evaluation: str = EvaluationResult.INCONCLUSIVE.value
    regression_detected: bool = False
    evaluation_confidence: float = 0.0
    evaluator_reason: str = ""
    acceptance_gates: dict[str, bool] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------
    duration_ms: float | None = None
    error: str | None = None
    trace_references: list[str] = field(default_factory=list)
    trajectory_interaction_id: str | None = None
    schema_version: int = ATTEMPT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImprovementAttempt":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})
