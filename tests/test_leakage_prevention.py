import tempfile
import unittest
from pathlib import Path

from brain.learning_dataset import DatasetExample, build_dataset_version
from brain.learning_models import DataQualityLabel, LearningJob
from brain.learning_package import LearningPackage
from training.code_model.benchmark.runner import DEFAULT_FIXTURES_ROOT
from training.code_model.benchmark.schema import load_tasks_from_directory
from training.code_model.leakage import (
    build_leakage_index, check_example_for_leakage, filter_leaking_examples, make_dataset_example_filter,
)


def _job(job_id="job-1") -> LearningJob:
    return LearningJob(learning_job_id=job_id, created_at="t", updated_at="t", candidate_id="c1", improvement_attempt_id="a1", fingerprint="fp1")


class LeakageIndexTests(unittest.TestCase):
    def test_index_covers_every_shipped_fixture(self):
        index = build_leakage_index()
        tasks = load_tasks_from_directory(DEFAULT_FIXTURES_ROOT)
        for task in tasks:
            self.assertIn(f"task_id:{task.task_id}", index)


class CheckExampleForLeakageTests(unittest.TestCase):
    def setUp(self):
        self.index = build_leakage_index()

    def test_example_containing_verbatim_fixture_content_is_caught(self):
        buggy_calc_source = (DEFAULT_FIXTURES_ROOT / "syntax_runtime_bug_off_by_sign" / "repo" / "calc.py").read_text(encoding="utf-8")
        example = DatasetExample(
            example_id="ex-1", quality_label=DataQualityLabel.SYNTHETIC_VERIFIED.value,
            learning_job_id="job-1", variant_id="v1",
            payload={"before_files": {"calc.py": buggy_calc_source}, "after_files": {"calc.py": "def add(a, b):\n    return a + b\n"}},
        )
        match = check_example_for_leakage(example, self.index)
        self.assertEqual(match, "syntax_runtime_bug_off_by_sign")

    def test_unrelated_example_is_not_flagged(self):
        example = DatasetExample(
            example_id="ex-2", quality_label=DataQualityLabel.REAL_VERIFIED_TEACHER.value,
            learning_job_id="job-1", variant_id=None,
            payload={"original_task": "completely unrelated whatsapp send failure", "diff_summary": "1 file changed"},
        )
        self.assertIsNone(check_example_for_leakage(example, self.index))


class FilterLeakingExamplesTests(unittest.TestCase):
    def test_clean_and_quarantined_are_separated(self):
        leaked_source = (DEFAULT_FIXTURES_ROOT / "logical_bug_parity" / "repo" / "parity.py").read_text(encoding="utf-8")
        leaking = DatasetExample("ex-leak", DataQualityLabel.SYNTHETIC_VERIFIED.value, "job-1", "v1", {"before_files": {"parity.py": leaked_source}, "after_files": {}})
        clean_one = DatasetExample("ex-clean", DataQualityLabel.REAL_VERIFIED_TEACHER.value, "job-1", None, {"original_task": "fix the browser tab handle"})
        result = filter_leaking_examples([leaking, clean_one])
        self.assertEqual([e.example_id for e in result.clean], ["ex-clean"])
        self.assertEqual([e.example_id for e, _ in result.quarantined], ["ex-leak"])
        self.assertEqual(result.quarantined[0][1], "logical_bug_parity")


class DatasetBuilderLeakageIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "datasets"

    def tearDown(self):
        self.temp.cleanup()

    def test_build_dataset_version_quarantines_leaking_variant_via_filter_hook(self):
        from brain.learning_validator import VariantValidationResult
        from brain.learning_variation import GeneratedVariant

        leaked_source = (DEFAULT_FIXTURES_ROOT / "feature_implementation_reverse_words" / "harness" / "hidden_test.py").read_text(encoding="utf-8")
        package = LearningPackage(
            learning_job_id="job-1", improvement_attempt_id="a1", problem_family="fam-1",
            original_task="fix something else entirely", subsystem="s", gap_type="EXECUTION_BUG",
            root_cause_category="x", reusable_strategy="y",
        )
        leaking_variant = GeneratedVariant(
            variant_id="v1", description="looks like a normal variant", manifest={"test_file": "tests/test_x.py"},
            before_files={"strings_util.py": "def f():\n    pass\n"},
            after_files={"strings_util.py": leaked_source},  # accidentally contains the held-out hidden test verbatim
        )
        validation = VariantValidationResult("v1", True, DataQualityLabel.SYNTHETIC_VERIFIED.value, "ok", 1, 0)
        batch = [(_job(), package, [(leaking_variant, validation)])]

        manifest_without_filter = build_dataset_version(batch, dataset_root=self.root, dataset_version="v-nofilter")
        self.assertEqual(manifest_without_filter.example_count, 2)  # real teacher example + the leaking synthetic one

        manifest_with_filter = build_dataset_version(
            batch, dataset_root=self.root, dataset_version="v-filtered", example_filter=make_dataset_example_filter(),
        )
        self.assertEqual(manifest_with_filter.example_count, 1)  # the leaking synthetic example was quarantined

    def test_no_filter_given_behaves_exactly_as_before_phase_18(self):
        package = LearningPackage(
            learning_job_id="job-1", improvement_attempt_id="a1", problem_family="fam-1",
            original_task="fix something", subsystem="s", gap_type="EXECUTION_BUG",
            root_cause_category="x", reusable_strategy="y",
        )
        batch = [(_job(), package, [])]
        manifest = build_dataset_version(batch, dataset_root=self.root)
        self.assertEqual(manifest.example_count, 1)


if __name__ == "__main__":
    unittest.main()
