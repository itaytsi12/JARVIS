"""Phase 31: full end-to-end simulation of the voice-approved continual
learning pipeline, entirely with fakes (no real Claude quota, no real
training, no physical microphone):

student task -> fails -> teacher solves -> verification succeeds ->
JARVIS asks learning approval -> fake voice "yes jarvis" -> LearningJob
APPROVED -> experience available immediately -> "start learning" ->
orchestrator gathers job -> variants generated -> validated dataset built
-> fake training produces candidate -> benchmark shows improvement ->
candidate promoted -> LearningJob TRAINED.

Also proves "No Jarvis" never reaches the training queue.
"""
import tempfile
import unittest
from pathlib import Path

from brain.experience_store import ExperienceStore, retrieve_relevant_experiences
from brain.improvement_attempt_models import AttemptStatus
from brain.improvement_attempt_store import ImprovementAttemptStore
from brain.learning_evaluation import BenchmarkMetrics, FakeBenchmark
from brain.learning_models import LearningJobStatus
from brain.learning_orchestrator import handle_verified_teacher_success, start_learning
from brain.learning_store import LearningJobStore
from brain.learning_training import FakeTrainingBackend, ModelRegistry, TrainingConfig
from voice.learning_approval import LearningApprovalOutcome, LearningApprovalResult

from tests.test_learning_trigger import RealTeacherAgent, _apply_real_fix, _candidate, _fast_config, _init_repo
from tests.test_learning_variation import _write_variant


class ApprovalScript:
    """Simulates a fake voice response, matching the ApprovalRequester
    protocol `brain.learning_orchestrator.handle_verified_teacher_success`
    expects (an injected callable, never a real microphone)."""

    def __init__(self, outcome: LearningApprovalOutcome, transcript: str | None = None):
        self.outcome = outcome
        self.transcript = transcript
        self.calls = 0

    def __call__(self, **kwargs) -> LearningApprovalResult:
        self.calls += 1
        return LearningApprovalResult(self.outcome, self.transcript, 1.0)


class FullLoopEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = _init_repo(Path(self.temp.name))
        self.attempt_store = ImprovementAttemptStore(Path(self.temp.name) / "attempts.sqlite3")
        self.job_store = LearningJobStore(Path(self.temp.name) / "jobs.sqlite3")
        self.experience_store = ExperienceStore(Path(self.temp.name) / "experiences.sqlite3")
        self.model_registry = ModelRegistry(Path(self.temp.name) / "registry.sqlite3")
        self.package_root = Path(self.temp.name) / "packages"
        self.dataset_root = Path(self.temp.name) / "datasets"

    def tearDown(self):
        self.attempt_store.close()
        self.job_store.close()
        self.experience_store.close()
        self.model_registry.close()
        self.temp.cleanup()

    def _teacher_solves_and_verifies(self, **candidate_overrides):
        """Student/local path is represented by the candidate itself having
        been created from a real execution failure (brain/improvement_observer.py's
        real job in production); here we start directly from an eligible
        candidate -- exactly what a genuinely failed local attempt produces
        -- and drive it through the REAL orchestrator with a teacher agent
        that reports the real provider name, so every gate
        brain/learning_trigger.py checks is genuinely satisfied."""
        candidate = _candidate(**candidate_overrides)
        teacher = RealTeacherAgent(apply=_apply_real_fix)
        attempt = run_attempt = None
        from brain.improvement_orchestrator import run_attempt as _run_attempt
        attempt = _run_attempt(candidate, repository_root=str(self.repo), coding_agent=teacher, attempt_store=self.attempt_store, config=_fast_config())
        self.assertEqual(attempt.status, AttemptStatus.READY_FOR_REVIEW.value, "teacher fix must be independently verified before this test proceeds")
        return attempt

    def test_full_approved_loop_reaches_trained_and_promoted(self):
        # No package_root override needed anywhere below: brain/learning_package.py's
        # _package_root() automatically isolates to a per-process temp
        # directory whenever "pytest" is in sys.modules (same convention as
        # every SQLite-backed store here), so the approval step's save and
        # start_learning's later load consistently agree without either
        # call needing an explicit root -- and neither ever touches the
        # real repo's data/ directory.
        # 1. student fails -> teacher solves -> independently verified
        attempt = self._teacher_solves_and_verifies()

        # 2. JARVIS asks; fake voice answers "yes jarvis"
        approval = ApprovalScript(LearningApprovalOutcome.APPROVED, "yes jarvis")
        offer = handle_verified_teacher_success(
            attempt, request_approval=approval, job_store=self.job_store, experience_store=self.experience_store,
        )
        self.assertTrue(offer.offered)
        self.assertEqual(offer.job.learning_status, LearningJobStatus.READY_FOR_TRAINING.value)

        # 3. experience is retrievable immediately, before any retraining
        experiences = retrieve_relevant_experiences(attempt.original_request or "fix the thing", store=self.experience_store, top_k=3)
        self.assertTrue(experiences, "approved fix must be immediately retrievable as experience")

        # 4. later: "Hey Jarvis, start learning"
        def variant_apply(workspace: Path):
            _write_variant(workspace, "v1", "generalized variant", seed=1)
        from brain.improvement_coding_agent import FakeCodingAgent
        variant_agent = FakeCodingAgent(apply=variant_apply)

        benchmark = FakeBenchmark(default=BenchmarkMetrics(solve_rate=0.85, regression_rate=0.0, behavioral_acceptance_rate=0.85))
        backend = FakeTrainingBackend(improve=True)

        summary = start_learning(
            coding_agent=variant_agent, repository_root=str(self.repo), backend=backend, benchmark=benchmark,
            job_store=self.job_store, model_registry=self.model_registry,
            dataset_root=self.dataset_root,
            training_config=TrainingConfig(base_model="fake-base"),
        )

        self.assertEqual(summary.status, "COMPLETED")
        self.assertEqual(summary.job_count, 1)
        self.assertTrue(summary.promoted)
        self.assertIsNotNone(summary.dataset_version)

        final_job = self.job_store.get(offer.job.learning_job_id)
        self.assertEqual(final_job.learning_status, LearningJobStatus.TRAINED.value)
        self.assertEqual(final_job.model_version_result, summary.candidate_model_version)

        active = self.model_registry.get_active()
        self.assertEqual(active.model_version, summary.candidate_model_version)
        self.assertEqual(active.status, "ACTIVE")

    def test_no_jarvis_never_reaches_the_training_queue(self):
        attempt = self._teacher_solves_and_verifies()
        approval = ApprovalScript(LearningApprovalOutcome.DECLINED, "no jarvis")
        offer = handle_verified_teacher_success(
            attempt, request_approval=approval, job_store=self.job_store, experience_store=self.experience_store,
        )
        self.assertEqual(offer.job.learning_status, LearningJobStatus.DECLINED.value)
        self.assertEqual(self.experience_store.count(), 0)

        # "start learning" afterward must find nothing to train on
        summary = start_learning(
            coding_agent=None, repository_root=str(self.repo),
            backend=FakeTrainingBackend(), benchmark=FakeBenchmark(),
            job_store=self.job_store, model_registry=self.model_registry, dataset_root=self.dataset_root,
        )
        self.assertEqual(summary.job_count, 0)
        self.assertIsNone(self.model_registry.get_active())


if __name__ == "__main__":
    unittest.main()
