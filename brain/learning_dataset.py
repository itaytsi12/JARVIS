"""Dataset building and versioning (Phase 10).

Combines everything an approved learning batch has produced -- verified
teacher fixes (`LearningPackage`) and their SYNTHETIC_VERIFIED variants --
into one deduplicated, immutable, versioned dataset manifest + JSONL file on
disk under `data/learning_datasets/`. "Immutable" here means literally
that: once a version is written, `build_dataset_version` never overwrites an
existing version directory; a new build always gets a new version id.

SYNTHETIC_DERIVED (unverified) examples are deliberately never included in
the default build -- Phase 9's whole point is that only mechanically proven
variants may enter the trainable pool. A caller that explicitly wants to
inspect rejected variants uses `brain/learning_validator.py`'s own
(verified, unverified) return value directly; this module only ever writes
the trustworthy side to disk.

No general-purpose "high-quality coding dataset" file exists anywhere in
this repository today (the only training/data/*.jsonl this project has is
scoped to intent classification, a different problem entirely -- see
CLAUDE.md). `base_dataset_paths` is therefore an explicit, optional,
empty-by-default parameter: if/when such a base coding dataset is ever
added, it plugs in here without changing this module's contract.
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain.improvement_models import canonical_json, stable_hash
from brain.learning_models import DataQualityLabel, LearningJob
from brain.learning_package import LearningPackage
from brain.learning_variation import GeneratedVariant
from brain.learning_validator import VariantValidationResult

DATASET_SCHEMA_VERSION = 1
DEFAULT_DATASET_ROOT = Path("data") / "learning_datasets"

_TEST_DATASET_ROOT: Path | None = None


def _default_dataset_root() -> Path:
    # Same "never touch the real data/ directory from an automated test
    # that forgot to override the path" guarantee used throughout this
    # codebase's SQLite-backed stores (see brain/learning_store.py).
    global _TEST_DATASET_ROOT
    if "pytest" in sys.modules:
        if _TEST_DATASET_ROOT is None:
            _TEST_DATASET_ROOT = Path(tempfile.mkdtemp(prefix="jarvis-learning-datasets-pytest-"))
        return _TEST_DATASET_ROOT
    return DEFAULT_DATASET_ROOT


@dataclass
class DatasetExample:
    example_id: str
    quality_label: str
    learning_job_id: str | None
    variant_id: str | None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetManifest:
    dataset_version: str
    created_at: str
    example_count: int
    quality_label_counts: dict[str, int]
    source_learning_job_ids: list[str]
    content_hash: str
    jsonl_path: str
    schema_version: int = DATASET_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _real_teacher_example(job: LearningJob, package: LearningPackage) -> DatasetExample:
    payload = {
        "problem_family": package.problem_family,
        "subsystem": package.subsystem,
        "gap_type": package.gap_type,
        "root_cause_category": package.root_cause_category,
        "original_task": package.original_task,
        "reusable_strategy": package.reusable_strategy,
        "applicability_conditions": package.applicability_conditions,
        "do_not_generalize": package.do_not_generalize,
        "files_changed": package.files_changed,
        "diff_summary": package.diff_summary,
        "before_behavior": package.before_behavior,
        "after_behavior": package.after_behavior,
    }
    example_id = stable_hash({"kind": "real_teacher", "job": job.learning_job_id, "payload": payload})[:32]
    return DatasetExample(example_id, DataQualityLabel.REAL_VERIFIED_TEACHER.value, job.learning_job_id, None, payload)


def _synthetic_example(job: LearningJob, variant: GeneratedVariant, validation: VariantValidationResult) -> DatasetExample:
    payload = {
        "description": variant.description,
        "before_files": variant.before_files,
        "after_files": variant.after_files,
        "test_file": variant.manifest.get("test_file"),
        "validation_reason": validation.reason,
    }
    example_id = stable_hash({"kind": "synthetic_verified", "job": job.learning_job_id, "variant": variant.variant_id})[:32]
    return DatasetExample(example_id, DataQualityLabel.SYNTHETIC_VERIFIED.value, job.learning_job_id, variant.variant_id, payload)


def _dedupe_examples(examples: list[DatasetExample]) -> list[DatasetExample]:
    seen: set[str] = set()
    out = []
    for example in examples:
        content_hash = stable_hash(example.payload)
        if content_hash in seen:
            continue
        seen.add(content_hash)
        out.append(example)
    return out


def collect_examples(
    batch: list[tuple[LearningJob, LearningPackage, list[tuple[GeneratedVariant, VariantValidationResult]]]],
) -> list[DatasetExample]:
    """One example per approved job's verified teacher fix, plus one example
    per SYNTHETIC_VERIFIED variant. Everything not `.verified` is excluded
    here (Phase 9's separation, enforced structurally rather than by
    caller discipline)."""
    examples: list[DatasetExample] = []
    for job, package, variants in batch:
        examples.append(_real_teacher_example(job, package))
        for variant, validation in variants:
            if validation.verified and validation.quality_label == DataQualityLabel.SYNTHETIC_VERIFIED.value:
                examples.append(_synthetic_example(job, variant, validation))
    return _dedupe_examples(examples)


def next_dataset_version(dataset_root: Path | None = None) -> str:
    root = dataset_root or _default_dataset_root()
    root.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in root.glob("*.manifest.json"):
        name = path.name[: -len(".manifest.json")]
        if name.startswith("v") and name[1:].isdigit():
            n = max(n, int(name[1:]))
    return f"v{n + 1}"


def build_dataset_version(
    batch: list[tuple[LearningJob, LearningPackage, list[tuple[GeneratedVariant, VariantValidationResult]]]],
    *,
    dataset_root: Path | str | None = None,
    dataset_version: str | None = None,
    example_filter: Any = None,
) -> DatasetManifest:
    """Write one new, immutable dataset version to disk and return its
    manifest. Never mutates or overwrites a prior version -- raises
    FileExistsError if `dataset_version` is explicitly given and already
    exists, exactly to protect that immutability guarantee.

    `example_filter`, if given, is a `Callable[[list[DatasetExample]],
    list[DatasetExample]]` applied immediately after `collect_examples`,
    before anything is written -- the hook
    `training/code_model/leakage.py` (Phase 18) uses to quarantine any
    example that matches a held-out benchmark fixture. Defaults to `None`
    (no filtering), so every existing caller's behavior is unchanged.
    """
    root = Path(dataset_root) if dataset_root else _default_dataset_root()
    root.mkdir(parents=True, exist_ok=True)
    version = dataset_version or next_dataset_version(root)
    jsonl_path = root / f"{version}.jsonl"
    manifest_path = root / f"{version}.manifest.json"
    if jsonl_path.exists() or manifest_path.exists():
        raise FileExistsError(f"dataset version {version!r} already exists at {root}")

    examples = collect_examples(batch)
    if example_filter is not None:
        examples = example_filter(examples)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(canonical_json(example.to_dict()) + "\n")

    label_counts: dict[str, int] = {}
    for example in examples:
        label_counts[example.quality_label] = label_counts.get(example.quality_label, 0) + 1

    manifest = DatasetManifest(
        dataset_version=version,
        created_at=_now(),
        example_count=len(examples),
        quality_label_counts=label_counts,
        source_learning_job_ids=sorted({job.learning_job_id for job, _, _ in batch}),
        content_hash=stable_hash([e.to_dict() for e in examples]),
        jsonl_path=str(jsonl_path),
    )
    manifest_path.write_text(canonical_json(manifest.to_dict()), encoding="utf-8")
    return manifest


def load_dataset_manifest(dataset_version: str, *, dataset_root: Path | str | None = None) -> DatasetManifest | None:
    root = Path(dataset_root) if dataset_root else _default_dataset_root()
    manifest_path = root / f"{dataset_version}.manifest.json"
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    known = set(DatasetManifest.__dataclass_fields__)
    return DatasetManifest(**{k: v for k, v in payload.items() if k in known})
