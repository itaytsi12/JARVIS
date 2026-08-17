import tempfile
import unittest
from pathlib import Path

from brain.learning_dataset import build_dataset_version
from brain.learning_models import DataQualityLabel, LearningJob
from brain.learning_package import LearningPackage
from brain.learning_validator import VariantValidationResult
from brain.learning_variation import GeneratedVariant
from training.code_model.dataset_formatting import build_sft_dataset, format_examples_for_sft, write_sft_jsonl


def _job(job_id="job-1") -> LearningJob:
    return LearningJob(
        learning_job_id=job_id, created_at="t", updated_at="t",
        candidate_id="cand-1", improvement_attempt_id="att-1", fingerprint="fp-1",
    )


def _package(job_id="job-1") -> LearningPackage:
    return LearningPackage(
        learning_job_id=job_id, improvement_attempt_id="att-1", problem_family="fam-1",
        original_task="fix the off-by-one bug in the paginator", subsystem="filesystem", gap_type="EXECUTION_BUG",
        root_cause_category="EXECUTION_BUG_with_differential_test", reusable_strategy="Changed paginate.py to use <= instead of <.",
        applicability_conditions=["gap_type=EXECUTION_BUG"], files_changed=["paginate.py"],
        diff_summary="1 file changed, +1/-1", before_behavior={"reproduced": True}, after_behavior={"reproduced": False},
    )


def _verified_variant() -> tuple[GeneratedVariant, VariantValidationResult]:
    variant = GeneratedVariant(
        variant_id="v1", description="off-by-one in a different loop", manifest={"test_file": "tests/test_x.py"},
        before_files={"src.py": "def f():\n    return 1\n"}, after_files={"src.py": "def f():\n    return 2\n"},
    )
    return variant, VariantValidationResult("v1", True, DataQualityLabel.SYNTHETIC_VERIFIED.value, "ok", 1, 0)


class FormatExamplesForSftTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "datasets"

    def tearDown(self):
        self.temp.cleanup()

    def test_formats_real_teacher_and_synthetic_examples(self):
        batch = [(_job(), _package(), [_verified_variant()])]
        manifest = build_dataset_version(batch, dataset_root=self.root)
        examples = format_examples_for_sft(manifest.jsonl_path)
        self.assertEqual(len(examples), 2)
        kinds = {e.metadata["kind"] for e in examples}
        self.assertEqual(kinds, {"real_teacher", "synthetic_verified"})

    def test_real_teacher_prompt_contains_task_and_response_contains_strategy(self):
        batch = [(_job(), _package(), [])]
        manifest = build_dataset_version(batch, dataset_root=self.root)
        examples = format_examples_for_sft(manifest.jsonl_path)
        example = examples[0]
        self.assertIn("off-by-one bug in the paginator", example.prompt)
        self.assertIn("Changed paginate.py", example.response)
        self.assertIn("paginate.py", example.prompt)

    def test_synthetic_prompt_contains_buggy_code_response_contains_fixed_code(self):
        batch = [(_job(), _package(), [_verified_variant()])]
        manifest = build_dataset_version(batch, dataset_root=self.root)
        examples = format_examples_for_sft(manifest.jsonl_path)
        synthetic = next(e for e in examples if e.metadata["kind"] == "synthetic_verified")
        self.assertIn("return 1", synthetic.prompt)
        self.assertIn("return 2", synthetic.response)

    def test_missing_dataset_file_returns_empty_list(self):
        self.assertEqual(format_examples_for_sft(self.root / "does-not-exist.jsonl"), [])

    def test_write_sft_jsonl_round_trips(self):
        batch = [(_job(), _package(), [])]
        manifest = build_dataset_version(batch, dataset_root=self.root)
        examples = format_examples_for_sft(manifest.jsonl_path)
        out = write_sft_jsonl(examples, self.root / "out.sft.jsonl")
        self.assertTrue(out.exists())
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)

    def test_build_sft_dataset_convenience_wrapper(self):
        batch = [(_job(), _package(), [_verified_variant()])]
        manifest = build_dataset_version(batch, dataset_root=self.root)
        out = build_sft_dataset(manifest.jsonl_path)
        self.assertTrue(out.exists())
        self.assertTrue(out.name.endswith(".sft.jsonl"))

    def test_original_dataset_jsonl_is_never_modified(self):
        batch = [(_job(), _package(), [_verified_variant()])]
        manifest = build_dataset_version(batch, dataset_root=self.root)
        original_content = Path(manifest.jsonl_path).read_text(encoding="utf-8")
        build_sft_dataset(manifest.jsonl_path)
        self.assertEqual(Path(manifest.jsonl_path).read_text(encoding="utf-8"), original_content)


if __name__ == "__main__":
    unittest.main()
