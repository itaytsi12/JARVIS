"""Structured evidence for JARVIS's self-improvement observation foundation.

An `ImprovementCandidate` records what a REAL execution actually did --
never what a route or plan merely intended -- so a future skill learner or
coding agent can understand what happened without replaying the whole
session. See `brain/improvement_observer.py` for how candidates are built
and `brain/improvement_store.py` for how they're persisted.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


SCHEMA_VERSION = 1


class GapType(str, Enum):
    SKILL_GAP = "SKILL_GAP"
    CODE_CAPABILITY_GAP = "CODE_CAPABILITY_GAP"
    EXECUTION_BUG = "EXECUTION_BUG"
    ENVIRONMENTAL_FAILURE = "ENVIRONMENTAL_FAILURE"
    USER_INPUT_OR_AMBIGUITY = "USER_INPUT_OR_AMBIGUITY"
    UNKNOWN = "UNKNOWN"


class CandidateStatus(str, Enum):
    NEW = "NEW"
    TRIAGED = "TRIAGED"
    IGNORED = "IGNORED"
    READY_FOR_IMPROVEMENT = "READY_FOR_IMPROVEMENT"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass
class ImprovementCandidate:
    # ------------------------------------------------------------------
    # IDENTITY
    # ------------------------------------------------------------------
    candidate_id: str
    created_at: str
    first_seen: str
    last_seen: str
    occurrence_count: int = 1
    status: str = CandidateStatus.NEW.value

    # ------------------------------------------------------------------
    # REQUEST (sanitized before this object is ever constructed)
    # ------------------------------------------------------------------
    raw_request: str = ""
    normalized_goal: str = ""
    requested_clauses: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # EXECUTION -- real evidence only, never intended/planned actions
    # ------------------------------------------------------------------
    route_type: str | None = None
    route_source: str | None = None
    planner_source: str | None = None
    executed_actions: list[dict[str, Any]] = field(default_factory=list)
    executed_tool_names: list[str] = field(default_factory=list)
    success: bool = False
    verified: bool = False
    partial: bool = False
    duration_ms: float | None = None
    model_calls: int = 0
    resource_wait_ms: float | None = None

    # ------------------------------------------------------------------
    # FAILURE / EVIDENCE
    # ------------------------------------------------------------------
    exception_type: str | None = None
    exception_message: str | None = None
    verification_failure_reason: str | None = None
    unmet_goal: str | None = None
    plan_rejection_reason: str | None = None
    plan_rejection_errors: list[str] = field(default_factory=list)
    unsupported_capability: str | None = None
    observations: list[str] = field(default_factory=list)
    recovery_evidence: str | None = None

    # ------------------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------------------
    gap_type: str = GapType.UNKNOWN.value
    confidence: float = 0.0
    classification_reason: str = ""
    classifier_source: str = "deterministic_v1"

    # ------------------------------------------------------------------
    # ENVIRONMENT
    # ------------------------------------------------------------------
    platform: str = ""
    subsystem: str | None = None
    tool_involved: str | None = None
    environment_notes: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------
    fingerprint: str = ""
    task_id: str | None = None
    interaction_id: str | None = None
    source: str = "real_execution"
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImprovementCandidate":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})
