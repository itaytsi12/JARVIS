import tempfile
import unittest
from pathlib import Path

from brain.learning_training import (
    ConfiguredCloudTrainingBackend, FakeTrainingBackend, ModelRegistry, ModelVersion,
    PreTrainingCheckResult, TrainingConfig, TrainingPolicy,
    policy_allows_training, run_pre_training_checks,
)
from brain.task_supervisor import CancellationToken


class FakeTrainingBackendTests(unittest.TestCase):
    def test_completes_and_produces_a_model_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "v1.jsonl"
            dataset.write_text("{}\n")
            backend = FakeTrainingBackend()
            result = backend.run(str(dataset), TrainingConfig(base_model="fake-base"))
        self.assertEqual(result.exit_status, "completed")
        self.assertIsNotNone(result.model_version)

    def test_missing_dataset_file_fails_cleanly(self):
        backend = FakeTrainingBackend()
        result = backend.run("/does/not/exist.jsonl", TrainingConfig(base_model="fake-base"))
        self.assertEqual(result.exit_status, "failed")

    def test_cancellation_before_start_is_honored(self):
        token = CancellationToken()
        token.cancel()
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "v1.jsonl"
            dataset.write_text("{}\n")
            backend = FakeTrainingBackend()
            result = backend.run(str(dataset), TrainingConfig(base_model="fake-base"), cancellation_token=token)
        self.assertEqual(result.exit_status, "cancelled")


class ConfiguredCloudTrainingBackendTests(unittest.TestCase):
    def test_unconfigured_backend_is_unavailable(self):
        backend = ConfiguredCloudTrainingBackend()
        available, reason = backend.is_available()
        self.assertFalse(available)

    def test_configured_but_unauthorized_never_dispatches(self):
        backend = ConfiguredCloudTrainingBackend(provider_configured=True, authorized=False, provider_name="acme-gpu")
        result = backend.run("dataset.jsonl", TrainingConfig(base_model="base"))
        self.assertEqual(result.exit_status, "blocked")
        self.assertIn("authorization", result.error)
        self.assertIn("plan", result.metrics)

    def test_build_plan_never_executes_anything(self):
        backend = ConfiguredCloudTrainingBackend(provider_configured=True, provider_name="acme-gpu")
        plan = backend.build_plan(TrainingConfig(base_model="base-model"))
        self.assertEqual(plan["backend"], "acme-gpu")
        self.assertEqual(plan["base_model"], "base-model")


class PreTrainingCheckTests(unittest.TestCase):
    def test_all_conditions_satisfied_is_ready(self):
        backend = FakeTrainingBackend()
        result = run_pre_training_checks(job_count=3, example_count=10, backend=backend, config=TrainingConfig(base_model="base"), min_free_disk_bytes=1)
        self.assertTrue(result.ready)
        self.assertEqual(result.reasons, [])

    def test_no_jobs_is_not_ready(self):
        backend = FakeTrainingBackend()
        result = run_pre_training_checks(job_count=0, example_count=10, backend=backend, config=TrainingConfig(base_model="base"), min_free_disk_bytes=1)
        self.assertFalse(result.ready)
        self.assertTrue(any("no approved" in r for r in result.reasons))

    def test_never_crashes_when_backend_is_unavailable(self):
        backend = ConfiguredCloudTrainingBackend()  # unconfigured
        result = run_pre_training_checks(job_count=1, example_count=1, backend=backend, config=TrainingConfig(base_model="base"), min_free_disk_bytes=1)
        self.assertFalse(result.ready)
        self.assertIsNotNone(result.plan)

    def test_no_base_model_configured_is_reported(self):
        backend = FakeTrainingBackend()
        result = run_pre_training_checks(job_count=1, example_count=1, backend=backend, config=TrainingConfig(), min_free_disk_bytes=1)
        self.assertFalse(result.ready)
        self.assertTrue(any("base model" in r for r in result.reasons))


class TrainingPolicyTests(unittest.TestCase):
    def test_explicit_command_always_overrides_manual_only(self):
        policy = TrainingPolicy(mode="manual_only")
        allowed, _ = policy_allows_training(policy, job_count=0, example_count=0, explicit_command=True)
        self.assertTrue(allowed)

    def test_manual_only_blocks_without_explicit_command(self):
        policy = TrainingPolicy(mode="manual_only")
        allowed, _ = policy_allows_training(policy, job_count=5, example_count=50, explicit_command=False)
        self.assertFalse(allowed)

    def test_minimum_examples_threshold(self):
        policy = TrainingPolicy(mode="minimum_examples", minimum_examples=10)
        blocked, _ = policy_allows_training(policy, job_count=1, example_count=5, explicit_command=False)
        allowed, _ = policy_allows_training(policy, job_count=1, example_count=10, explicit_command=False)
        self.assertFalse(blocked)
        self.assertTrue(allowed)

    def test_minimum_learning_jobs_threshold(self):
        policy = TrainingPolicy(mode="minimum_learning_jobs", minimum_learning_jobs=3)
        blocked, _ = policy_allows_training(policy, job_count=2, example_count=100, explicit_command=False)
        allowed, _ = policy_allows_training(policy, job_count=3, example_count=100, explicit_command=False)
        self.assertFalse(blocked)
        self.assertTrue(allowed)

    def test_never_trains_after_every_single_approval_under_manual_only(self):
        policy = TrainingPolicy()  # default is manual_only
        allowed, reason = policy_allows_training(policy, job_count=1, example_count=1, explicit_command=False)
        self.assertFalse(allowed)


class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.registry = ModelRegistry(Path(self.temp.name) / "registry.sqlite3")

    def tearDown(self):
        self.registry.close()
        self.temp.cleanup()

    def _candidate(self, version="m1"):
        return ModelVersion(model_version=version, dataset_version="v1", training_run_id="run-1", created_at="t", metrics={"solve_rate": 0.5})

    def test_record_then_get(self):
        self.registry.record(self._candidate())
        self.assertEqual(self.registry.get("m1").model_version, "m1")

    def test_no_active_model_initially(self):
        self.assertIsNone(self.registry.get_active())

    def test_promote_sets_active(self):
        self.registry.record(self._candidate())
        self.registry.promote("m1")
        active = self.registry.get_active()
        self.assertEqual(active.model_version, "m1")
        self.assertEqual(active.status, "ACTIVE")

    def test_promoting_new_version_demotes_the_old_one(self):
        self.registry.record(self._candidate("m1"))
        self.registry.promote("m1")
        self.registry.record(self._candidate("m2"))
        self.registry.promote("m2")
        self.assertEqual(self.registry.get_active().model_version, "m2")
        self.assertEqual(self.registry.get("m1").status, "REPLACED")

    def test_reject_never_becomes_active(self):
        self.registry.record(self._candidate())
        self.registry.reject("m1", "worse than baseline")
        self.assertIsNone(self.registry.get_active())
        self.assertEqual(self.registry.get("m1").status, "REJECTED")
        self.assertEqual(self.registry.get("m1").rejection_reason, "worse than baseline")

    def test_promote_unknown_version_raises(self):
        with self.assertRaises(KeyError):
            self.registry.promote("does-not-exist")

    def test_survives_new_connection_same_file(self):
        path = Path(self.temp.name) / "persist.sqlite3"
        r1 = ModelRegistry(path)
        r1.record(self._candidate())
        r1.promote("m1")
        r1.close()
        r2 = ModelRegistry(path)
        try:
            self.assertEqual(r2.get_active().model_version, "m1")
        finally:
            r2.close()


if __name__ == "__main__":
    unittest.main()
