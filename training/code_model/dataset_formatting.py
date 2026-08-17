"""SFT-ready example formatting (Phase 3).

Extends -- never modifies or replaces -- the existing
`brain/learning_dataset.py` JSONL format: reads the same immutable,
versioned `DatasetManifest`/JSONL rows that module already produces (one
row per `REAL_VERIFIED_TEACHER` teacher fix or `SYNTHETIC_VERIFIED`
variant -- see its docstring for why `SYNTHETIC_DERIVED`/unverified rows
never appear there in the first place) and produces a SEPARATE, downstream
prompt/response JSONL suitable for causal-LM fine-tuning. The original
dataset manifest/JSONL remains the audit trail and source of truth; this is
a pure, repeatable, re-derivable transformation of it -- re-running it
never touches the original file.

No hidden chain-of-thought is required or invented here (this codebase
never captures Claude's private reasoning -- see
`brain/learning_package.py`'s docstring). The "rationale" shown to the
model is always `reusable_strategy`, a deterministic, evidence-derived
summary string already computed by `brain/learning_package.py` -- never a
free-form narrative generated for this purpose.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from brain.learning_models import DataQualityLabel

SFT_SCHEMA_VERSION = 1

_SYSTEM_PREAMBLE = (
    "You are a careful software engineer fixing a real bug in a codebase. "
    "You are given the task and relevant repository evidence. Respond with "
    "the smallest patch that fixes the problem, grounded only in the "
    "evidence given -- never invent files, behavior, or reasoning not "
    "shown to you."
)


@dataclass
class SFTExample:
    example_id: str
    quality_label: str
    prompt: str
    response: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SFTExample":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


def _format_real_teacher_example(row: dict[str, Any]) -> SFTExample:
    payload = row.get("payload") or {}
    files_block = "\n".join(f"- {f}" for f in payload.get("files_changed") or []) or "(no files recorded)"
    prompt = (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"TASK:\n{payload.get('original_task', '')}\n\n"
        f"SUBSYSTEM: {payload.get('subsystem')}\n"
        f"ROOT CAUSE CATEGORY: {payload.get('root_cause_category')}\n"
        f"RELEVANT FILES:\n{files_block}\n\n"
        f"BEFORE BEHAVIOR (observed evidence): {payload.get('before_behavior')}\n"
    )
    response = (
        f"STRATEGY: {payload.get('reusable_strategy', '')}\n\n"
        f"APPLICABILITY: {'; '.join(payload.get('applicability_conditions') or [])}\n\n"
        f"PATCH SUMMARY:\n{payload.get('diff_summary', '')}\n\n"
        f"AFTER BEHAVIOR (observed evidence): {payload.get('after_behavior')}\n"
    )
    return SFTExample(
        row["example_id"], row["quality_label"], prompt, response,
        {"learning_job_id": row.get("learning_job_id"), "kind": "real_teacher"},
    )


def _format_synthetic_example(row: dict[str, Any]) -> SFTExample:
    payload = row.get("payload") or {}
    before_files = payload.get("before_files") or {}
    after_files = payload.get("after_files") or {}
    before_block = "\n\n".join(f"--- {path} ---\n{content}" for path, content in before_files.items())
    after_block = "\n\n".join(f"--- {path} ---\n{content}" for path, content in after_files.items())
    prompt = (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"TASK: {payload.get('description', '')}\n\n"
        f"REPOSITORY STATE (buggy):\n{before_block}\n"
    )
    response = f"FIXED REPOSITORY STATE:\n{after_block}\n"
    return SFTExample(
        row["example_id"], row["quality_label"], prompt, response,
        {"variant_id": row.get("variant_id"), "kind": "synthetic_verified"},
    )


_FORMATTERS: dict[str, Callable[[dict[str, Any]], SFTExample]] = {
    DataQualityLabel.REAL_VERIFIED_TEACHER.value: _format_real_teacher_example,
    DataQualityLabel.SYNTHETIC_VERIFIED.value: _format_synthetic_example,
}


def format_examples_for_sft(dataset_jsonl_path: str | Path) -> list[SFTExample]:
    """Reads `brain/learning_dataset.py`'s existing JSONL format and returns
    one `SFTExample` per row it knows how to format. A row with an
    unrecognized `quality_label` is skipped, never guessed at -- exactly
    the same "don't fabricate structure" discipline as the rest of this
    codebase's evidence-driven modules."""
    examples: list[SFTExample] = []
    path = Path(dataset_jsonl_path)
    if not path.exists():
        return examples
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        formatter = _FORMATTERS.get(row.get("quality_label"))
        if formatter is None:
            continue
        examples.append(formatter(row))
    return examples


def write_sft_jsonl(examples: list[SFTExample], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    return out_path


def build_sft_dataset(dataset_jsonl_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Convenience one-shot: format `dataset_jsonl_path` (a
    `brain.learning_dataset.DatasetManifest.jsonl_path`) and write the
    result next to it as `<same-stem>.sft.jsonl` unless `output_path` is
    given."""
    examples = format_examples_for_sft(dataset_jsonl_path)
    if output_path is None:
        source = Path(dataset_jsonl_path)
        output_path = source.with_suffix("").with_suffix(".sft.jsonl")
    return write_sft_jsonl(examples, output_path)
