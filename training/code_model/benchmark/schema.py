"""Versioned benchmark task schema (Phase 13).

Each task's fixture lives under
`training/code_model/benchmark/fixtures/<fixture_dir>/`::

    repo/                    the GIVEN starting repository state (buggy or
                              incomplete code the agent is handed) -- copied
                              into a fresh temp directory and `git init`'d
                              at evaluation time, never a nested git repo
                              committed into JARVIS's own history.
    repo/tests/...            agent-VISIBLE tests, if any (part of the given
                              repo state -- the agent can see and run them,
                              same as a real repository)
    harness/<hidden_test>      an INDEPENDENT acceptance test, NOT part of
                              `repo/` -- copied in by the benchmark runner
                              only AFTER the agent's patch, so the agent
                              cannot special-case or read the exact
                              assertion it will be judged against (Phase 13:
                              "the solution must NOT be exposed to the model
                              during evaluation").

`task.json` (this schema, serialized) lives alongside `repo/`/`harness/` in
each fixture directory and is loaded by `load_tasks_from_directory`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BENCHMARK_SCHEMA_VERSION = 1

# Phase 12's full category list. Only a subset has real fixture tasks today
# (see training/code_model/benchmark/fixtures/) -- CATEGORIES lists every
# category this schema supports so more fixtures can be added later without
# a schema change, not a claim that all are already covered.
CATEGORIES = {
    "syntax_runtime_bug", "logical_bug", "cross_file_bug", "multi_class_bug",
    "regression_bug", "feature_implementation", "state_management_bug",
    "api_misuse", "refactoring", "test_writing", "bug_localization",
}


@dataclass
class BenchmarkTask:
    task_id: str
    category: str
    description: str
    fixture_dir: str
    hidden_test_path: str
    visible_test_paths: list[str] = field(default_factory=list)
    timeout_seconds: float = 120.0
    constraints: list[str] = field(default_factory=list)
    # `test_writing` category: the hidden acceptance test alone is not
    # enough -- a patch that fixes behavior but adds no test must not score
    # as solved. Checked against `DiffAnalysis.generated_tests` (already
    # computed by `brain.improvement_diff_analysis.analyze_diff`).
    require_new_test: bool = False
    # `refactoring` category: passing tests alone is not enough -- a
    # no-op "fix" must not score as solved. `min_line_reduction` is a
    # mechanical, implementation-agnostic proxy for "a real structural
    # change happened" (checked against `structural_check_path`'s line
    # count before vs. after), deliberately not tied to one exact refactor
    # shape.
    structural_check_path: str | None = None
    min_line_reduction: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = BENCHMARK_SCHEMA_VERSION

    def fixture_fingerprint(self) -> str:
        """A stable identity for this task's *content* (not just its id) --
        used by `training/code_model/leakage.py` (Phase 18) to detect a
        training example that matches a held-out benchmark task, even if
        someone later renames the task."""
        from brain.improvement_models import stable_hash
        return stable_hash({"task_id": self.task_id, "category": self.category, "fixture_dir": self.fixture_dir})[:24]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkTask":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


def load_task(fixture_dir: Path) -> BenchmarkTask:
    payload = json.loads((fixture_dir / "task.json").read_text(encoding="utf-8"))
    return BenchmarkTask.from_dict(payload)


def load_tasks_from_directory(fixtures_root: str | Path) -> list[BenchmarkTask]:
    """Scans every immediate subdirectory of `fixtures_root` for a
    `task.json` and loads it. Never raises for a subdirectory that isn't a
    task (no `task.json`) -- it's just skipped."""
    root = Path(fixtures_root)
    if not root.is_dir():
        return []
    tasks = []
    for child in sorted(root.iterdir()):
        task_file = child / "task.json"
        if task_file.exists():
            tasks.append(load_task(child))
    return tasks
