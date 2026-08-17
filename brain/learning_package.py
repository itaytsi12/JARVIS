"""LearningPackage extraction (Phase 6): converts a VERIFIED
`ImprovementAttempt` into a structured, generalizable learning package.

Everything here is built deterministically from structured evidence already
recorded on the attempt (diff analysis, reproduction results, evaluator
gates) -- exactly the same "conservative, evidence-driven, no LLM call"
design already used by `brain/improvement_classifier.py` and
`brain/improvement_evaluator.py`. This module never asks Claude to
summarize itself and never stores anything resembling hidden chain-of-
thought, because this codebase never captures that in the first place --
`brain/improvement_coding_agent.py::ClaudeCodeAdapter` only ever records
`stdout_summary` (a sanitized, truncated JSON result string) and the
structural `git diff`. "Observable actions" here means exactly that
existing, already-sanitized evidence, not anything reconstructed after the
fact.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from brain.improvement_attempt_models import ImprovementAttempt
from brain.learning_trigger import task_family_fingerprint
from training_data.sanitizer import sanitize_text

LEARNING_PACKAGE_SCHEMA_VERSION = 1
DEFAULT_PACKAGE_ROOT = Path("data") / "learning_packages"


@dataclass
class LearningPackage:
    # ------------------------------------------------------------------
    # IDENTITY / LINKAGE
    # ------------------------------------------------------------------
    learning_job_id: str
    improvement_attempt_id: str
    problem_family: str

    # ------------------------------------------------------------------
    # PROBLEM
    # ------------------------------------------------------------------
    original_task: str
    subsystem: str | None
    gap_type: str
    root_cause_category: str

    # ------------------------------------------------------------------
    # OBSERVABLE CLAUDE ACTIONS / EVIDENCE (structural only, never hidden
    # reasoning -- this codebase never records that)
    # ------------------------------------------------------------------
    files_changed: list[str] = field(default_factory=list)
    files_added: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    diff_summary: str = ""
    generated_tests: list[str] = field(default_factory=list)
    revision_rounds: int = 0
    agent_model_calls: int = 0

    # ------------------------------------------------------------------
    # BEFORE / AFTER BEHAVIOR
    # ------------------------------------------------------------------
    before_behavior: dict[str, Any] = field(default_factory=dict)
    after_behavior: dict[str, Any] = field(default_factory=dict)
    reproduction_method: str | None = None

    # ------------------------------------------------------------------
    # VERIFICATION EVIDENCE
    # ------------------------------------------------------------------
    acceptance_gates: dict[str, bool] = field(default_factory=dict)
    evaluator_reason: str = ""

    # ------------------------------------------------------------------
    # GENERALIZATION GUIDANCE -- deterministic summaries built from the
    # evidence above, never a free-form Claude narrative.
    # ------------------------------------------------------------------
    reusable_strategy: str = ""
    applicability_conditions: list[str] = field(default_factory=list)
    failure_patterns: list[str] = field(default_factory=list)
    do_not_generalize: list[str] = field(default_factory=list)

    schema_version: int = LEARNING_PACKAGE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LearningPackage":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


def _root_cause_category(attempt: ImprovementAttempt) -> str:
    if attempt.change_scope == "test_only":
        return "test_only_fix"
    if attempt.reproduction_method == "INTEGRATION_REPRO" and attempt.before_state.get("reproduced") is True:
        return f"{attempt.gap_type}_with_live_reproduction"
    if attempt.generated_tests:
        return f"{attempt.gap_type}_with_differential_test"
    return attempt.gap_type or "unclassified"


def _reusable_strategy_summary(attempt: ImprovementAttempt) -> str:
    scope = attempt.change_scope or "unknown"
    files = ", ".join(attempt.files_changed[:5]) or "no files recorded"
    proof = (
        "a live before/after reproduction (the original failure reproduced, then stopped reproducing after the fix)"
        if attempt.before_state.get("reproduced") is True and attempt.after_state.get("reproduced") is False
        else "a generated regression test proven to fail on the unpatched base and pass after the fix"
    )
    return (
        f"A {scope} change touching {files} resolved a {attempt.gap_type or 'unclassified'} issue"
        f"{f' in the {attempt.subsystem} subsystem' if attempt.subsystem else ''}, confirmed by {proof} "
        f"in {attempt.revision_rounds} revision round(s)."
    )


def _applicability_conditions(attempt: ImprovementAttempt) -> list[str]:
    conditions = [f"gap_type={attempt.gap_type}"]
    if attempt.subsystem:
        conditions.append(f"subsystem={attempt.subsystem}")
    conditions.append(f"change_scope={attempt.change_scope}")
    if attempt.generated_tests:
        conditions.append("a differential regression test exists and is the primary proof of correctness")
    return conditions


def _failure_patterns(attempt: ImprovementAttempt) -> list[str]:
    patterns = []
    if attempt.revision_rounds > 1:
        patterns.append(
            f"the first {attempt.revision_rounds - 1} revision round(s) did not pass acceptance gates; "
            "only the final revision generalizes, not the intermediate attempts"
        )
    return patterns


def _do_not_generalize(attempt: ImprovementAttempt) -> list[str]:
    """Explicit boundaries so a future training pass doesn't over-fit to
    incidental details of this one occurrence."""
    notes = [
        "exact file paths and identifier names are specific to this occurrence -- "
        "generalize the root-cause shape (Phase 8's variation generation), not the literal names",
    ]
    if attempt.change_scope == "test_only":
        notes.append("this attempt only changed tests; it does not demonstrate a source-code repair pattern")
    return notes


def extract_learning_package(attempt: ImprovementAttempt, *, learning_job_id: str) -> LearningPackage:
    """Pure function: no I/O, no Claude call, no randomness. Every field is
    either copied from `attempt` (already sanitized upstream throughout the
    orchestrator/observer pipeline) or re-sanitized here as defense in depth
    (Phase 24) before this package can be persisted or shown to a variation-
    generation prompt."""
    return LearningPackage(
        learning_job_id=learning_job_id,
        improvement_attempt_id=attempt.attempt_id,
        problem_family=task_family_fingerprint(attempt),
        original_task=sanitize_text(attempt.original_request or ""),
        subsystem=attempt.subsystem,
        gap_type=attempt.gap_type,
        root_cause_category=_root_cause_category(attempt),
        files_changed=list(attempt.files_changed),
        files_added=list(attempt.files_added),
        files_deleted=list(attempt.files_deleted),
        diff_summary=sanitize_text(attempt.diff_summary or ""),
        generated_tests=list(attempt.generated_tests),
        revision_rounds=attempt.revision_rounds,
        agent_model_calls=attempt.agent_model_calls,
        before_behavior=dict(attempt.before_state or {}),
        after_behavior=dict(attempt.after_state or {}),
        reproduction_method=attempt.reproduction_method,
        acceptance_gates=dict(attempt.acceptance_gates or {}),
        evaluator_reason=sanitize_text(attempt.evaluator_reason or ""),
        reusable_strategy=_reusable_strategy_summary(attempt),
        applicability_conditions=_applicability_conditions(attempt),
        failure_patterns=_failure_patterns(attempt),
        do_not_generalize=_do_not_generalize(attempt),
    )


_TEST_PACKAGE_ROOT: Path | None = None


def _package_root(root: str | Path | None = None) -> Path:
    if root:
        return Path(root)
    env = os.getenv("LEARNING_PACKAGE_DIR")
    if env:
        return Path(env)
    global _TEST_PACKAGE_ROOT
    # Same "never touch the real data/ directory from an automated test that
    # forgot to override the path" guarantee already used by every SQLite
    # store in this codebase (see e.g. brain/learning_store.py's
    # get_learning_job_store) -- one shared temp directory per pytest
    # process, not a real, persistent location.
    if "pytest" in sys.modules:
        if _TEST_PACKAGE_ROOT is None:
            _TEST_PACKAGE_ROOT = Path(tempfile.mkdtemp(prefix="jarvis-learning-packages-pytest-"))
        return _TEST_PACKAGE_ROOT
    return DEFAULT_PACKAGE_ROOT


def save_learning_package(package: LearningPackage, *, root: str | Path | None = None) -> Path:
    """Package extraction is cheap/deterministic/local (Phase 6), so it is
    always persisted as a plain JSON file the moment a job is approved --
    no separate database table needed for something this simple, and it
    keeps the package trivially inspectable for debugging (Phase 33)."""
    directory = _package_root(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{package.learning_job_id}.json"
    path.write_text(json.dumps(package.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_learning_package(learning_job_id: str, *, root: str | Path | None = None) -> LearningPackage | None:
    path = _package_root(root) / f"{learning_job_id}.json"
    if not path.exists():
        return None
    return LearningPackage.from_dict(json.loads(path.read_text(encoding="utf-8")))
