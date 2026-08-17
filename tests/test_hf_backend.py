"""Focused tests for training/code_model/hf_backend.py (Phase 29).

Genuine training-mechanics tests (model load, LoRA injection, tokenizer,
forward/backward, checkpoint save, resume, cancellation) require the real
ML stack (torch/transformers/peft/accelerate) and are skipped -- not
failed -- when that stack isn't importable, exactly like
`brain/learning_training.py::run_pre_training_checks` never crashes on a
missing dependency. Run these for real via the dedicated venv:

    .venv-code-model\\Scripts\\python -m pytest tests/test_hf_backend.py -q

Config-only tests (no torch needed) always run, in any environment.
"""
import json
import tempfile
import unittest
from pathlib import Path

from brain.learning_dataset import build_dataset_version
from brain.learning_models import LearningJob
from brain.learning_package import LearningPackage
from brain.learning_training import TrainingConfig
from training.code_model.config import load_config

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import peft  # noqa: F401
    _ML_STACK_AVAILABLE = True
except Exception:
    _ML_STACK_AVAILABLE = False

from training.code_model.hf_backend import HuggingFaceLoRATrainingBackend, TrainingRunRecord, load_run_record


def _job(job_id="job-1") -> LearningJob:
    return LearningJob(learning_job_id=job_id, created_at="t", updated_at="t", candidate_id="c1", improvement_attempt_id="a1", fingerprint="fp1")


def _package(job_id="job-1") -> LearningPackage:
    return LearningPackage(
        learning_job_id=job_id, improvement_attempt_id="a1", problem_family="fam-1",
        original_task="fix the off-by-one bug", subsystem="filesystem", gap_type="EXECUTION_BUG",
        root_cause_category="EXECUTION_BUG_with_differential_test", reusable_strategy="strategy text",
    )


class TrainingRunRecordTests(unittest.TestCase):
    """No torch needed -- pure dataclass/JSON serialization."""

    def test_round_trips_through_dict(self):
        record = TrainingRunRecord(
            training_run_id="r1", dataset_version="v1", base_model="m", config_hash="h",
            checkpoint_path="/tmp/x", current_step=3, status="RUNNING",
        )
        restored = TrainingRunRecord.from_dict(json.loads(json.dumps(record.to_dict())))
        self.assertEqual(restored, record)

    def test_load_run_record_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_run_record(tmp))


class InvalidModelTests(unittest.TestCase):
    def test_is_available_reports_missing_ml_stack_or_bad_model_without_raising(self):
        config = load_config("small_smoke_test")
        config.base_model.model_id = "definitely/not-a-real-model-id-xyz-123"
        backend = HuggingFaceLoRATrainingBackend(config)
        available, reason = backend.is_available()
        self.assertFalse(available)
        self.assertIsInstance(reason, str)
        self.assertTrue(reason)


@unittest.skipUnless(_ML_STACK_AVAILABLE, "requires torch/transformers/peft -- run via .venv-code-model")
class RealTrainingMechanicsTests(unittest.TestCase):
    """Every test in this class performs REAL model loading and REAL
    forward/backward training steps against sshleifer/tiny-gpt2 (a public,
    few-hundred-KB test fixture model) -- never a mock, never a stub."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset_manifest = build_dataset_version([(_job(), _package(), [])], dataset_root=self.root / "datasets")

    def tearDown(self):
        self.temp.cleanup()

    def _config(self, **overrides):
        config = load_config("small_smoke_test")
        for key, value in overrides.items():
            setattr(config.training, key, value)
        return config

    def test_is_available_true_for_the_smoke_config(self):
        backend = HuggingFaceLoRATrainingBackend(self._config())
        available, reason = backend.is_available()
        self.assertTrue(available, reason)

    def test_real_training_run_completes_and_saves_a_lora_adapter(self):
        backend = HuggingFaceLoRATrainingBackend(self._config(max_steps=2, save_steps=2))
        result = backend.run(self.dataset_manifest.jsonl_path, TrainingConfig(checkpoint_dir=str(self.root / "checkpoints")))
        self.assertEqual(result.exit_status, "completed", result.error)
        self.assertIsNotNone(result.model_version)
        adapter_dir = Path(result.checkpoint_path)
        self.assertTrue((adapter_dir / "adapter_config.json").exists())
        self.assertTrue((adapter_dir / "adapter_model.safetensors").exists())
        config_payload = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config_payload["peft_type"], "LORA")
        self.assertIn("train_loss", result.metrics)

    def test_training_result_is_persisted_truthfully(self):
        backend = HuggingFaceLoRATrainingBackend(self._config(max_steps=2, save_steps=2))
        result = backend.run(self.dataset_manifest.jsonl_path, TrainingConfig(checkpoint_dir=str(self.root / "checkpoints")))
        record = load_run_record(self.root / "checkpoints" / result.training_run_id)
        self.assertEqual(record.status, "COMPLETED")
        self.assertEqual(record.current_step, 2)

    def test_missing_dataset_examples_fails_without_crashing(self):
        empty_dataset = self.root / "datasets" / "empty.jsonl"
        empty_dataset.parent.mkdir(parents=True, exist_ok=True)
        empty_dataset.write_text("", encoding="utf-8")
        backend = HuggingFaceLoRATrainingBackend(self._config(max_steps=2))
        result = backend.run(str(empty_dataset), TrainingConfig(checkpoint_dir=str(self.root / "checkpoints")))
        self.assertEqual(result.exit_status, "failed")
        self.assertIn("no formatted SFT examples", result.error)

    def test_cancellation_stops_training_and_never_marks_completed(self):
        backend = HuggingFaceLoRATrainingBackend(self._config(max_steps=6, save_steps=2))

        class CancelAfterN:
            def __init__(self, n):
                self.n, self.calls = n, 0

            @property
            def cancelled(self):
                self.calls += 1
                return self.calls >= self.n

        token = CancelAfterN(2)
        result = backend.run(self.dataset_manifest.jsonl_path, TrainingConfig(checkpoint_dir=str(self.root / "checkpoints")), cancellation_token=token)
        self.assertEqual(result.exit_status, "cancelled")
        self.assertNotEqual(result.exit_status, "completed")
        record = load_run_record(self.root / "checkpoints" / result.training_run_id)
        self.assertEqual(record.status, "CANCELLED")
        self.assertTrue(Path(result.checkpoint_path).exists())

    def test_resume_continues_from_a_prior_interrupted_run(self):
        backend = HuggingFaceLoRATrainingBackend(self._config(max_steps=4, save_steps=1))

        class CancelAfterN:
            def __init__(self, n):
                self.n, self.calls = n, 0

            @property
            def cancelled(self):
                self.calls += 1
                return self.calls >= self.n

        first = backend.run(self.dataset_manifest.jsonl_path, TrainingConfig(checkpoint_dir=str(self.root / "checkpoints")), cancellation_token=CancelAfterN(2))
        self.assertEqual(first.exit_status, "cancelled")
        prior_run_dir = self.root / "checkpoints" / first.training_run_id

        class NeverCancel:
            cancelled = False

        second = backend.run(
            self.dataset_manifest.jsonl_path,
            TrainingConfig(checkpoint_dir=str(self.root / "checkpoints"), resume_from=str(prior_run_dir)),
            cancellation_token=NeverCancel(),
        )
        self.assertEqual(second.exit_status, "completed", second.error)
        self.assertEqual(second.resumed_from, str(prior_run_dir))

    def test_checkpoint_path_is_never_silently_overwritten_across_runs(self):
        backend = HuggingFaceLoRATrainingBackend(self._config(max_steps=1, save_steps=1))
        result_a = backend.run(self.dataset_manifest.jsonl_path, TrainingConfig(checkpoint_dir=str(self.root / "checkpoints")))
        result_b = backend.run(self.dataset_manifest.jsonl_path, TrainingConfig(checkpoint_dir=str(self.root / "checkpoints")))
        self.assertNotEqual(result_a.training_run_id, result_b.training_run_id)
        self.assertNotEqual(result_a.checkpoint_path, result_b.checkpoint_path)
        self.assertTrue(Path(result_a.checkpoint_path).exists())
        self.assertTrue(Path(result_b.checkpoint_path).exists())


if __name__ == "__main__":
    unittest.main()
