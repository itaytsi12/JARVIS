"""Synthetic variant validation (Phase 9).

Every variant `brain/learning_variation.py` produces starts labeled
SYNTHETIC_DERIVED (untrusted). Only a variant that is MECHANICALLY proven
here -- syntax-checked, then its own test shown to fail cleanly against
`before/` and pass against `after/` -- is promoted to SYNTHETIC_VERIFIED.
Everything else stays SYNTHETIC_DERIVED and is kept out of the
highest-quality training pool (see `brain/learning_dataset.py`).

Reuses `brain.task_supervisor.SafeCommandRunner`, the same sandboxed
subprocess runner the rest of the improvement pipeline already uses for
running pytest -- no second subprocess-execution implementation.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from brain.learning_models import DataQualityLabel
from brain.learning_variation import GeneratedVariant
from brain.task_supervisor import SafeCommandRunner


@dataclass
class VariantValidationResult:
    variant_id: str
    verified: bool
    quality_label: str
    reason: str
    before_exit_code: int | None = None
    after_exit_code: int | None = None


def _compile_check(files: dict[str, str]) -> str | None:
    for relpath, content in files.items():
        if relpath.endswith(".py"):
            try:
                compile(content, relpath, "exec")
            except SyntaxError as exc:
                return f"{relpath}: {exc}"
    return None


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for relpath, content in files.items():
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def validate_variant(
    variant: GeneratedVariant, *, runner: SafeCommandRunner | None = None, timeout_seconds: float = 60.0,
) -> VariantValidationResult:
    """Never raises for an ordinary "couldn't validate" outcome -- that's
    an honest, unverified VariantValidationResult, not an error."""
    runner = runner or SafeCommandRunner()
    test_file = variant.manifest.get("test_file")
    if not test_file:
        return VariantValidationResult(variant.variant_id, False, DataQualityLabel.SYNTHETIC_DERIVED.value, "manifest missing test_file")

    for label, files in (("before", variant.before_files), ("after", variant.after_files)):
        if not files:
            return VariantValidationResult(variant.variant_id, False, DataQualityLabel.SYNTHETIC_DERIVED.value, f"{label} file set is empty")
        error = _compile_check(files)
        if error:
            return VariantValidationResult(variant.variant_id, False, DataQualityLabel.SYNTHETIC_DERIVED.value, f"{label}: syntax error in {error}")

    with tempfile.TemporaryDirectory(prefix="jarvis-variant-before-") as before_dir, \
         tempfile.TemporaryDirectory(prefix="jarvis-variant-after-") as after_dir:
        _write_tree(Path(before_dir), variant.before_files)
        _write_tree(Path(after_dir), variant.after_files)

        try:
            before_result = runner.run(["python", "-m", "pytest", test_file, "-q"], before_dir, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return VariantValidationResult(variant.variant_id, False, DataQualityLabel.SYNTHETIC_DERIVED.value, "before-test timed out")
        if before_result["exit_code"] != 1:
            return VariantValidationResult(
                variant.variant_id, False, DataQualityLabel.SYNTHETIC_DERIVED.value,
                f"before-test did not cleanly fail (exit_code={before_result['exit_code']})",
                before_exit_code=before_result["exit_code"],
            )

        try:
            after_result = runner.run(["python", "-m", "pytest", test_file, "-q"], after_dir, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return VariantValidationResult(
                variant.variant_id, False, DataQualityLabel.SYNTHETIC_DERIVED.value, "after-test timed out",
                before_exit_code=before_result["exit_code"],
            )
        if after_result["exit_code"] != 0:
            return VariantValidationResult(
                variant.variant_id, False, DataQualityLabel.SYNTHETIC_DERIVED.value,
                f"after-test did not pass (exit_code={after_result['exit_code']})",
                before_exit_code=before_result["exit_code"], after_exit_code=after_result["exit_code"],
            )

    return VariantValidationResult(
        variant.variant_id, True, DataQualityLabel.SYNTHETIC_VERIFIED.value,
        "before-test failed cleanly and after-test passed", before_result["exit_code"], after_result["exit_code"],
    )


def validate_variants(
    variants: list[GeneratedVariant], *, runner: SafeCommandRunner | None = None, timeout_seconds: float = 60.0,
) -> tuple[list[VariantValidationResult], list[VariantValidationResult]]:
    """Returns (verified, unverified) -- deliberately two separate lists, so
    a caller can never accidentally mix them into one pool without an
    explicit choice to do so."""
    runner = runner or SafeCommandRunner()
    results = [validate_variant(v, runner=runner, timeout_seconds=timeout_seconds) for v in variants]
    verified = [r for r in results if r.verified]
    unverified = [r for r in results if not r.verified]
    return verified, unverified
