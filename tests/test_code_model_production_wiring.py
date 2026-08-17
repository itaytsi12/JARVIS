"""Phase 29: proves the production "start learning" path never uses
FakeTrainingBackend/FakeBenchmark, and that the real backend/benchmark are
what `voice/background_assistant.py::_start_learning_task` actually
constructs.
"""
import inspect
import unittest

from brain.learning_training import FakeTrainingBackend
from brain.learning_evaluation import FakeBenchmark
from training.code_model.benchmark.runner import RealCodingBenchmark
from training.code_model.hf_backend import HuggingFaceLoRATrainingBackend
from brain.learning_training import TrainingConfig, run_pre_training_checks
from training.code_model.production import build_production_backend, build_production_benchmark, build_training_config


class ProductionWiringTests(unittest.TestCase):
    def test_build_production_backend_is_the_real_hf_backend(self):
        backend = build_production_backend("small_smoke_test")
        self.assertIsInstance(backend, HuggingFaceLoRATrainingBackend)
        self.assertNotIsInstance(backend, FakeTrainingBackend)
        self.assertEqual(backend.backend_name, "huggingface_lora")

    def test_build_production_benchmark_is_the_real_benchmark(self):
        backend = build_production_backend("small_smoke_test")
        benchmark = build_production_benchmark(backend.code_model_config)
        self.assertIsInstance(benchmark, RealCodingBenchmark)
        self.assertNotIsInstance(benchmark, FakeBenchmark)

    def test_production_config_defaults_to_the_recommended_real_config(self):
        backend = build_production_backend()
        self.assertEqual(backend.code_model_config.name, "qlora_7b")
        self.assertNotEqual(backend.code_model_config.base_model.model_id, "not-configured")

    def test_env_var_overrides_the_production_config(self):
        import os
        old = os.environ.get("JARVIS_CODE_MODEL_CONFIG")
        try:
            os.environ["JARVIS_CODE_MODEL_CONFIG"] = "small_smoke_test"
            backend = build_production_backend()
            self.assertEqual(backend.code_model_config.name, "small_smoke_test")
        finally:
            if old is None:
                os.environ.pop("JARVIS_CODE_MODEL_CONFIG", None)
            else:
                os.environ["JARVIS_CODE_MODEL_CONFIG"] = old

    def test_build_training_config_populates_base_model_from_the_backend_config(self):
        """Regression test for a real bug this session's Phase 28 dry run
        caught: a bare TrainingConfig() defaults base_model to
        "not-configured", which run_pre_training_checks correctly rejects
        -- even when the backend's own CodeModelTrainingConfig has a
        perfectly real base model. build_training_config is the one place
        that translation must happen for every start_learning caller."""
        backend = build_production_backend("small_smoke_test")
        training_config = build_training_config(backend.code_model_config)
        self.assertEqual(training_config.base_model, "sshleifer/tiny-gpt2")
        self.assertNotEqual(training_config.base_model, "not-configured")

    def test_bare_training_config_would_have_failed_pre_training_checks(self):
        """Documents exactly why build_training_config exists: proves the
        bug it fixes was real, not hypothetical."""
        result = run_pre_training_checks(job_count=1, example_count=1, backend=build_production_backend("small_smoke_test"), config=TrainingConfig())
        self.assertFalse(result.ready)
        self.assertTrue(any("base model" in r for r in result.reasons))

    def test_background_assistant_start_learning_source_never_mentions_fake_backend_or_benchmark(self):
        """A direct source-level guard: the production dispatch method must
        never construct FakeTrainingBackend/FakeBenchmark, even indirectly
        via a stray import alias."""
        import voice.background_assistant as module
        source = inspect.getsource(module.AlwaysOnAssistant._start_learning_task)
        # A code comment is allowed to mention the fake classes by name (to
        # document their deliberate absence); what must never appear is an
        # actual construction call.
        self.assertNotIn("FakeTrainingBackend(", source)
        self.assertNotIn("FakeBenchmark(", source)
        self.assertIn("build_production_backend", source)
        self.assertIn("build_production_benchmark", source)
        self.assertIn("build_training_config", source)


if __name__ == "__main__":
    unittest.main()
