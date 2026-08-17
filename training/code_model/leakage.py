"""Benchmark leakage prevention (Phase 18).

A dataset build must never silently include an example whose content
matches a held-out benchmark task -- checked by task_id, fixture-content
hash, and root-cause fingerprint, since a synthetic variant
(`brain/learning_variation.py`) could in principle regenerate something
close to a benchmark fixture without ever literally reusing its task_id.

Integrates with `brain.learning_dataset.build_dataset_version` through its
`example_filter` hook (added specifically for this, defaulting to `None`/
no-op for every existing caller) rather than duplicating that function's
file-writing logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from brain.improvement_models import stable_hash
from brain.learning_dataset import DatasetExample
from training.code_model.benchmark.schema import BenchmarkTask, load_tasks_from_directory


def _fixture_content_hashes(task: BenchmarkTask, fixtures_root: Path) -> set[str]:
    """Content hash of every file in the task's given repo state plus its
    hidden test -- a dataset example containing byte-identical content to
    one of these is almost certainly the benchmark task itself (or a
    trivial copy of it), not independent training data."""
    hashes: set[str] = set()
    for subdir in ("repo", "harness"):
        base = fixtures_root / task.fixture_dir / subdir
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                try:
                    hashes.add(stable_hash(path.read_text(encoding="utf-8", errors="replace")))
                except Exception:
                    continue
    return hashes


def build_leakage_index(fixtures_root: str | Path | None = None) -> dict[str, str]:
    """Maps a lookup key (`"task_id:..."`, `"fingerprint:..."`, or
    `"content:<hash>"`) to the benchmark `task_id` it identifies, across
    every held-out fixture. Built fresh each call (fixtures are few and
    small; no caching complexity needed)."""
    from training.code_model.benchmark.runner import DEFAULT_FIXTURES_ROOT

    root = Path(fixtures_root) if fixtures_root else DEFAULT_FIXTURES_ROOT
    index: dict[str, str] = {}
    for task in load_tasks_from_directory(root):
        index[f"task_id:{task.task_id}"] = task.task_id
        index[f"fingerprint:{task.fixture_fingerprint()}"] = task.task_id
        for content_hash in _fixture_content_hashes(task, root):
            index[f"content:{content_hash}"] = task.task_id
    return index


def _example_content_strings(example: DatasetExample) -> list[str]:
    payload = example.payload or {}
    strings: list[str] = []
    for key in ("original_task", "diff_summary", "description"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            strings.append(value)
    for file_map_key in ("before_files", "after_files"):
        file_map = payload.get(file_map_key)
        if isinstance(file_map, dict):
            strings.extend(str(v) for v in file_map.values() if v)
    return strings


def check_example_for_leakage(example: DatasetExample, leakage_index: dict[str, str]) -> str | None:
    """Returns the matching benchmark `task_id` if `example` appears to
    leak it, else `None`. Never raises."""
    for text in _example_content_strings(example):
        content_hash = stable_hash(text)
        matched = leakage_index.get(f"content:{content_hash}")
        if matched:
            return matched
    return None


@dataclass
class LeakageFilterResult:
    clean: list[DatasetExample]
    quarantined: list[tuple[DatasetExample, str]]

    @property
    def quarantined_count(self) -> int:
        return len(self.quarantined)


def filter_leaking_examples(
    examples: list[DatasetExample], *, fixtures_root: str | Path | None = None, leakage_index: dict[str, str] | None = None,
) -> LeakageFilterResult:
    """Splits `examples` into (clean, quarantined) -- quarantined entries
    always carry the matched benchmark `task_id` as their reason, never
    silently dropped without explanation."""
    index = leakage_index if leakage_index is not None else build_leakage_index(fixtures_root)
    clean: list[DatasetExample] = []
    quarantined: list[tuple[DatasetExample, str]] = []
    for example in examples:
        match = check_example_for_leakage(example, index)
        if match:
            quarantined.append((example, match))
        else:
            clean.append(example)
    return LeakageFilterResult(clean=clean, quarantined=quarantined)


def make_dataset_example_filter(*, fixtures_root: str | Path | None = None):
    """Returns a callable suitable for
    `brain.learning_dataset.build_dataset_version(..., example_filter=...)`.
    Quarantined examples are logged, never silently discarded without a
    trace -- inspect `filter_leaking_examples` directly for the full
    (example, task_id) pairs if a caller needs them."""
    index = build_leakage_index(fixtures_root)

    def _filter(examples: list[DatasetExample]) -> list[DatasetExample]:
        result = filter_leaking_examples(examples, leakage_index=index)
        if result.quarantined:
            import logging
            log = logging.getLogger("jarvis.learning")
            for example, task_id in result.quarantined:
                log.warning("[leakage] quarantined dataset example %s: matches benchmark task %r", example.example_id, task_id)
        return result.clean

    return _filter
