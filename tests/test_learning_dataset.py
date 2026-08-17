import tempfile
import unittest
from pathlib import Path

from brain.learning_dataset import build_dataset_version, collect_examples, load_dataset_manifest, next_dataset_version
from brain.learning_models import DataQualityLabel, LearningJob
from brain.learning_package import LearningPackage
from brain.learning_validator import VariantValidationResult
from brain.learning_variation import GeneratedVariant


def _job(job_id="job-1") -> LearningJob:
    return LearningJob(
        learning_job_id=job_id, created_at="t", updated_at="t",
        candidate_id="cand-1", improvement_attempt_id="att-1", fingerprint="fp-1",
    )


def _package(job_id="job-1") -> LearningPackage:
    return LearningPackage(
        learning_job_id=job_id, improvement_attempt_id="att-1", problem_family="fam-1",
        original_task="fix the thing", subsystem="filesystem", gap_type="EXECUTION_BUG",
        root_cause_category="EXECUTION_BUG_with_differential_test", reusable_strategy="strategy",
    )


def _verified_variant(variant_id="v1") -> tuple[GeneratedVariant, VariantValidationResult]:
    variant = GeneratedVariant(
        variant_id=variant_id, description="desc", manifest={"test_file": "tests/test_x.py"},
        before_files={"src.py": "x=1"}, after_files={"src.py": "x=2"},
    )
    validation = VariantValidationResult(variant_id, True, DataQualityLabel.SYNTHETIC_VERIFIED.value, "ok", 1, 0)
    return variant, validation


def _unverified_variant(variant_id="v-bad") -> tuple[GeneratedVariant, VariantValidationResult]:
    variant = GeneratedVariant(
        variant_id=variant_id, description="desc", manifest={"test_file": "tests/test_x.py"},
        before_files={"src.py": "bad"}, after_files={"src.py": "bad"},
    )
    validation = VariantValidationResult(variant_id, False, DataQualityLabel.SYNTHETIC_DERIVED.value, "syntax error")
    return variant, validation


class CollectExamplesTests(unittest.TestCase):
    def test_includes_real_teacher_and_verified_synthetic_only(self):
        batch = [(_job(), _package(), [_verified_variant(), _unverified_variant()])]
        examples = collect_examples(batch)
        labels = sorted(e.quality_label for e in examples)
        self.assertEqual(labels, sorted([DataQualityLabel.REAL_VERIFIED_TEACHER.value, DataQualityLabel.SYNTHETIC_VERIFIED.value]))

    def test_unverified_variants_never_appear(self):
        batch = [(_job(), _package(), [_unverified_variant()])]
        examples = collect_examples(batch)
        self.assertEqual(len(examples), 1)  # only the real teacher example
        self.assertEqual(examples[0].quality_label, DataQualityLabel.REAL_VERIFIED_TEACHER.value)

    def test_duplicate_content_across_jobs_is_deduplicated(self):
        batch = [
            (_job("job-1"), _package("job-1"), []),
            (_job("job-2"), _package("job-1"), []),  # identical package payload
        ]
        examples = collect_examples(batch)
        self.assertEqual(len(examples), 1)


class BuildDatasetVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "datasets"

    def tearDown(self):
        self.temp.cleanup()

    def test_creates_manifest_and_jsonl(self):
        batch = [(_job(), _package(), [_verified_variant()])]
        manifest = build_dataset_version(batch, dataset_root=self.root)
        self.assertEqual(manifest.example_count, 2)
        self.assertTrue(Path(manifest.jsonl_path).exists())
        loaded = load_dataset_manifest(manifest.dataset_version, dataset_root=self.root)
        self.assertEqual(loaded, manifest)

    def test_versions_increment_and_are_immutable(self):
        batch = [(_job(), _package(), [])]
        m1 = build_dataset_version(batch, dataset_root=self.root)
        m2 = build_dataset_version(batch, dataset_root=self.root)
        self.assertNotEqual(m1.dataset_version, m2.dataset_version)
        with self.assertRaises(FileExistsError):
            build_dataset_version(batch, dataset_root=self.root, dataset_version=m1.dataset_version)

    def test_next_dataset_version_starts_at_v1(self):
        self.assertEqual(next_dataset_version(self.root), "v1")

    def test_source_job_ids_recorded(self):
        batch = [(_job("job-a"), _package("job-a"), []), (_job("job-b"), _package("job-b"), [])]
        manifest = build_dataset_version(batch, dataset_root=self.root)
        self.assertEqual(manifest.source_learning_job_ids, ["job-a", "job-b"])

    def test_missing_manifest_returns_none(self):
        self.assertIsNone(load_dataset_manifest("v-does-not-exist", dataset_root=self.root))


if __name__ == "__main__":
    unittest.main()
