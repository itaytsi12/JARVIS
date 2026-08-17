import tempfile
import unittest
from pathlib import Path

from brain.improvement_attempt_models import AttemptStatus
from brain.improvement_attempt_store import ImprovementAttemptStore
from brain.improvement_coding_agent import FakeCodingAgent
from brain.improvement_models import GapType
from brain.improvement_orchestrator import run_attempt
from brain.experience_store import ExperienceStore, retrieve_relevant_experiences
from brain.learning_dataset import DEFAULT_DATASET_ROOT
from brain.learning_evaluation import BenchmarkMetrics, FakeBenchmark
from brain.learning_models import ApprovalStatus, LearningJobStatus
from brain.learning_orchestrator import handle_verified_teacher_success, start_learning
from brain.learning_package import DEFAULT_PACKAGE_ROOT, load_learning_package
from brain.learning_store import LearningJobStore
from brain.learning_training import FakeTrainingBackend, ModelRegistry, TrainingConfig, TrainingPolicy
from brain.learning_variation import VariationConfig
from voice.learning_approval import LearningApprovalOutcome, LearningApprovalResult

from tests.test_learning_trigger import RealTeacherAgent, _apply_real_fix, _candidate, _fast_config, _init_repo
from tests.test_learning_variation import _write_variant


class FakeApprovalRequester:
    def __init__(self, outcome: LearningApprovalOutcome, transcript: str | None = None):
        self.outcome = outcome
        self.transcript = transcript
        self.calls = 0

    def __call__(self, **kwargs) -> LearningApprovalResult:
        self.calls += 1
        return LearningApprovalResult(self.outcome, self.transcript, 1.0)


class HandleVerifiedTeacherSuccessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = _init_repo(Path(self.temp.name))
        self.attempt_store = ImprovementAttemptStore(Path(self.temp.name) / "attempts.sqlite3")
        self.job_store = LearningJobStore(Path(self.temp.name) / "jobs.sqlite3")
        self.experience_store = ExperienceStore(Path(self.temp.name) / "exp.sqlite3")
        self.package_root = Path(self.temp.name) / "packages"

    def tearDown(self):
        self.attempt_store.close()
        self.job_store.close()
        self.experience_store.close()
        self.temp.cleanup()

    def _ready_attempt(self, **overrides):
        candidate = _candidate(**overrides)
        agent = RealTeacherAgent(apply=_apply_real_fix)
        attempt = run_attempt(candidate, repository_root=str(self.repo), coding_agent=agent, attempt_store=self.attempt_store, config=_fast_config())
        self.assertEqual(attempt.status, AttemptStatus.READY_FOR_REVIEW.value)
        return attempt

    def test_ineligible_attempt_never_calls_approval(self):
        candidate_agent = FakeCodingAgent()  # fake, non-teacher provider
        from tests.test_learning_trigger import _candidate as make_candidate
        attempt = run_attempt(make_candidate(), repository_root=str(self.repo), coding_agent=candidate_agent, attempt_store=self.attempt_store, config=_fast_config())
        requester = FakeApprovalRequester(LearningApprovalOutcome.APPROVED)
        outcome = handle_verified_teacher_success(attempt, request_approval=requester, job_store=self.job_store, experience_store=self.experience_store)
        self.assertFalse(outcome.offered)
        self.assertEqual(requester.calls, 0)

    def test_approved_creates_ready_for_training_job_and_experience(self):
        # No package_root override needed: brain/learning_package.py's
        # _package_root() automatically uses an isolated per-process temp
        # directory whenever "pytest" is in sys.modules, exactly like every
        # SQLite-backed store in this codebase already does -- so this
        # never writes into the real repo's data/ directory even though
        # handle_verified_teacher_success() is called with no root override.
        attempt = self._ready_attempt()
        requester = FakeApprovalRequester(LearningApprovalOutcome.APPROVED, "yes jarvis")
        outcome = handle_verified_teacher_success(attempt, request_approval=requester, job_store=self.job_store, experience_store=self.experience_store)
        self.assertTrue(outcome.offered)
        self.assertEqual(outcome.job.learning_status, LearningJobStatus.READY_FOR_TRAINING.value)
        self.assertEqual(outcome.job.approval_status, ApprovalStatus.APPROVED.value)
        persisted = self.job_store.get(outcome.job.learning_job_id)
        self.assertEqual(persisted.learning_status, LearningJobStatus.READY_FOR_TRAINING.value)
        package = load_learning_package(outcome.job.learning_job_id)
        self.assertIsNotNone(package)
        results = retrieve_relevant_experiences(attempt.original_request or "fix the thing", store=self.experience_store, top_k=3)
        self.assertTrue(results)

    def test_declined_never_creates_experience(self):
        attempt = self._ready_attempt()
        requester = FakeApprovalRequester(LearningApprovalOutcome.DECLINED, "no jarvis")
        outcome = handle_verified_teacher_success(attempt, request_approval=requester, job_store=self.job_store, experience_store=self.experience_store)
        self.assertEqual(outcome.job.learning_status, LearningJobStatus.DECLINED.value)
        self.assertEqual(self.experience_store.count(), 0)

    def test_timed_out_is_treated_as_no(self):
        attempt = self._ready_attempt()
        requester = FakeApprovalRequester(LearningApprovalOutcome.TIMED_OUT, None)
        outcome = handle_verified_teacher_success(attempt, request_approval=requester, job_store=self.job_store, experience_store=self.experience_store)
        self.assertEqual(outcome.job.learning_status, LearningJobStatus.APPROVAL_TIMED_OUT.value)
        self.assertEqual(outcome.job.approval_status, ApprovalStatus.TIMED_OUT.value)
        self.assertEqual(self.experience_store.count(), 0)

    def test_duplicate_solved_problem_does_not_prompt_again(self):
        attempt = self._ready_attempt()
        requester1 = FakeApprovalRequester(LearningApprovalOutcome.APPROVED, "yes jarvis")
        handle_verified_teacher_success(attempt, request_approval=requester1, job_store=self.job_store, experience_store=self.experience_store)

        attempt2 = self._ready_attempt(candidate_id="cand-2")
        requester2 = FakeApprovalRequester(LearningApprovalOutcome.APPROVED, "yes jarvis")
        outcome2 = handle_verified_teacher_success(attempt2, request_approval=requester2, job_store=self.job_store, experience_store=self.experience_store)
        self.assertFalse(outcome2.offered)
        self.assertEqual(requester2.calls, 0)


class StartLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.job_store = LearningJobStore(Path(self.temp.name) / "jobs.sqlite3")
        self.model_registry = ModelRegistry(Path(self.temp.name) / "registry.sqlite3")
        self.repo = _init_repo(Path(self.temp.name))
        self.dataset_root = Path(self.temp.name) / "datasets"
        self.package_root = Path(self.temp.name) / "packages"

    def tearDown(self):
        self.job_store.close()
        self.model_registry.close()
        self.temp.cleanup()

    def _approved_job(self, job_id="job-1", seed=1):
        from brain.learning_models import LearningJob
        import brain.learning_package as learning_package_module
        from brain.learning_package import LearningPackage, save_learning_package

        job = LearningJob(
            learning_job_id=job_id, created_at="t", updated_at="t",
            candidate_id=f"cand-{job_id}", improvement_attempt_id=f"att-{job_id}",
            fingerprint=f"fp-{job_id}", learning_status=LearningJobStatus.READY_FOR_TRAINING.value,
            approval_status=ApprovalStatus.APPROVED.value,
        )
        self.job_store.create(job)
        package = LearningPackage(
            learning_job_id=job_id, improvement_attempt_id=f"att-{job_id}", problem_family=f"fam-{job_id}",
            original_task="fix the thing", subsystem="filesystem", gap_type="EXECUTION_BUG",
            root_cause_category="EXECUTION_BUG_with_differential_test", reusable_strategy="strategy",
        )
        save_learning_package(package, root=self.package_root)
        return job

    def _variant_agent(self, seed=1):
        def apply(workspace: Path):
            _write_variant(workspace, "v1", "variant", seed=seed)
        return FakeCodingAgent(apply=apply)

    def test_empty_queue_completes_without_training(self):
        summary = start_learning(
            coding_agent=self._variant_agent(), repository_root=str(self.repo),
            backend=FakeTrainingBackend(), benchmark=FakeBenchmark(),
            job_store=self.job_store, model_registry=self.model_registry,
            dataset_root=self.dataset_root, package_root=self.package_root,
        )
        self.assertEqual(summary.job_count, 0)
        self.assertEqual(summary.status, "COMPLETED")

    def test_full_run_promotes_better_candidate(self):
        self._approved_job("job-1", seed=1)
        benchmark = FakeBenchmark(default=BenchmarkMetrics(solve_rate=0.8, regression_rate=0.0, behavioral_acceptance_rate=0.8))
        backend = FakeTrainingBackend(improve=True)
        summary = start_learning(
            coding_agent=self._variant_agent(seed=1), repository_root=str(self.repo),
            backend=backend, benchmark=benchmark, job_store=self.job_store, model_registry=self.model_registry,
            dataset_root=self.dataset_root, package_root=self.package_root, training_config=TrainingConfig(base_model="fake-base"),
        )
        self.assertEqual(summary.status, "COMPLETED")
        self.assertTrue(summary.promoted)
        self.assertIsNotNone(summary.dataset_version)
        job = self.job_store.get("job-1")
        self.assertEqual(job.learning_status, LearningJobStatus.TRAINED.value)
        self.assertEqual(job.model_version_result, summary.candidate_model_version)
        active = self.model_registry.get_active()
        self.assertEqual(active.model_version, summary.candidate_model_version)

    def test_worse_candidate_is_rejected_and_job_stays_available(self):
        self._approved_job("job-1", seed=1)
        # seed the registry with an already-active, strong baseline
        from brain.learning_training import ModelVersion
        self.model_registry.record(ModelVersion(model_version="baseline", dataset_version="v0", training_run_id="r0", created_at="t"))
        self.model_registry.promote("baseline")
        benchmark = FakeBenchmark(scores={"baseline": BenchmarkMetrics(solve_rate=0.9, behavioral_acceptance_rate=0.9)}, default=BenchmarkMetrics(solve_rate=0.1, behavioral_acceptance_rate=0.1))
        backend = FakeTrainingBackend(improve=False)
        summary = start_learning(
            coding_agent=self._variant_agent(seed=1), repository_root=str(self.repo),
            backend=backend, benchmark=benchmark, job_store=self.job_store, model_registry=self.model_registry,
            dataset_root=self.dataset_root, package_root=self.package_root, training_config=TrainingConfig(base_model="fake-base"),
        )
        self.assertEqual(summary.status, "COMPLETED")
        self.assertFalse(summary.promoted)
        self.assertEqual(self.model_registry.get_active().model_version, "baseline")
        job = self.job_store.get("job-1")
        self.assertEqual(job.learning_status, LearningJobStatus.READY_FOR_TRAINING.value)  # kept for future experiments

    def test_manual_only_policy_blocks_implicit_run(self):
        self._approved_job("job-1")
        summary = start_learning(
            coding_agent=self._variant_agent(), repository_root=str(self.repo),
            backend=FakeTrainingBackend(), benchmark=FakeBenchmark(),
            job_store=self.job_store, model_registry=self.model_registry, dataset_root=self.dataset_root, package_root=self.package_root,
            policy=TrainingPolicy(mode="manual_only"), explicit_command=False,
        )
        self.assertEqual(summary.status, "FAILED")
        job = self.job_store.get("job-1")
        self.assertEqual(job.learning_status, LearningJobStatus.READY_FOR_TRAINING.value)  # untouched

    def test_explicit_command_overrides_manual_only(self):
        self._approved_job("job-1")
        summary = start_learning(
            coding_agent=self._variant_agent(), repository_root=str(self.repo),
            backend=FakeTrainingBackend(), benchmark=FakeBenchmark(),
            job_store=self.job_store, model_registry=self.model_registry, dataset_root=self.dataset_root, package_root=self.package_root,
            policy=TrainingPolicy(mode="manual_only"), explicit_command=True, training_config=TrainingConfig(base_model="fake-base"),
        )
        self.assertEqual(summary.status, "COMPLETED")

    def test_pre_training_check_failure_never_crashes_and_keeps_jobs_available(self):
        self._approved_job("job-1")
        unconfigured_backend = FakeTrainingBackend()
        summary = start_learning(
            coding_agent=self._variant_agent(), repository_root=str(self.repo),
            backend=unconfigured_backend, benchmark=FakeBenchmark(),
            job_store=self.job_store, model_registry=self.model_registry, dataset_root=self.dataset_root, package_root=self.package_root,
            training_config=TrainingConfig(base_model=""),  # no base model configured
        )
        self.assertEqual(summary.status, "FAILED")
        self.assertTrue(summary.reasons)
        job = self.job_store.get("job-1")
        self.assertEqual(job.learning_status, LearningJobStatus.READY_FOR_TRAINING.value)

    def test_progress_callback_reports_phase_sequence(self):
        self._approved_job("job-1", seed=1)
        phases = []
        start_learning(
            coding_agent=self._variant_agent(seed=1), repository_root=str(self.repo),
            backend=FakeTrainingBackend(improve=True), benchmark=FakeBenchmark(default=BenchmarkMetrics(solve_rate=0.9, behavioral_acceptance_rate=0.9)),
            job_store=self.job_store, model_registry=self.model_registry, dataset_root=self.dataset_root, package_root=self.package_root,
            progress_callback=lambda status, detail: phases.append(status), training_config=TrainingConfig(base_model="fake-base"),
        )
        self.assertIn("PREPARING_DATA", phases)
        self.assertIn("TRAINING", phases)
        self.assertIn("COMPLETED", phases)

    def test_cancellation_before_training_releases_jobs(self):
        from brain.task_supervisor import CancellationToken
        self._approved_job("job-1")
        token = CancellationToken()
        token.cancel()
        summary = start_learning(
            coding_agent=self._variant_agent(), repository_root=str(self.repo),
            backend=FakeTrainingBackend(), benchmark=FakeBenchmark(),
            job_store=self.job_store, model_registry=self.model_registry, dataset_root=self.dataset_root, package_root=self.package_root,
            cancellation_token=token,
        )
        self.assertEqual(summary.status, "CANCELLED")
        self.assertEqual(self.job_store.get("job-1").learning_status, LearningJobStatus.READY_FOR_TRAINING.value)

    def test_interrupted_previous_run_is_recovered_not_lost(self):
        job = self._approved_job("job-1")
        job.learning_status = LearningJobStatus.TRAINING.value  # simulate a crash mid-run
        self.job_store.update(job)
        summary = start_learning(
            coding_agent=self._variant_agent(seed=1), repository_root=str(self.repo),
            backend=FakeTrainingBackend(improve=True), benchmark=FakeBenchmark(default=BenchmarkMetrics(solve_rate=0.9, behavioral_acceptance_rate=0.9)),
            job_store=self.job_store, model_registry=self.model_registry, dataset_root=self.dataset_root, package_root=self.package_root, training_config=TrainingConfig(base_model="fake-base"),
        )
        self.assertEqual(summary.job_count, 1)
        self.assertEqual(summary.status, "COMPLETED")


if __name__ == "__main__":
    unittest.main()
