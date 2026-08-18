"""Part A tests (Phase A13): student-first, teacher-fallback coding-task
orchestration. Uses FakeCodingAgent throughout for both student and teacher
roles (no real model loading, no real Claude quota) -- `build_student_agent`
is patched to inject a fake student `CodingAgent`, and `teacher_agent` is
passed explicitly as a fake, matching this codebase's existing
improvement-pipeline test conventions.
"""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brain.improvement_attempt_models import AttemptStatus
from brain.improvement_attempt_store import ImprovementAttemptStore
from brain.improvement_coding_agent import CodingAgentConstraints, CodingAgentResult, FakeCodingAgent
from brain.improvement_student_teacher import (
    CodingTaskResult, StudentTeacherConfig, resolve_active_student, run_coding_task,
)
from brain.learning_models import DataQualityLabel
from brain.learning_store import LearningJobStore
from brain.learning_training import ModelRegistry, ModelVersion
from brain.student_trajectory_store import StudentTrajectoryStore
from brain.experience_store import ExperienceStore
from voice.learning_approval import LearningApprovalOutcome, LearningApprovalResult


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True)


def _init_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "value.py").write_text("VALUE = 1\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_sample.py").write_text("def test_sample():\n    assert True\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "initial")
    return repo


def _fast_config(**overrides) -> StudentTeacherConfig:
    from brain.improvement_orchestrator import OrchestratorConfig
    defaults = dict(
        student_coding_agent_timeout_seconds=30.0, student_focused_test_timeout_seconds=30.0,
        student_regression_test_timeout_seconds=60.0,
        teacher_config=OrchestratorConfig(coding_agent_timeout_seconds=30.0, focused_test_timeout_seconds=30.0, regression_test_timeout_seconds=60.0),
    )
    defaults.update(overrides)
    return StudentTeacherConfig(**defaults)


def _apply_real_fix(workspace: Path) -> None:
    (workspace / "value.py").write_text("VALUE = 2\n")
    (workspace / "tests" / "test_generated_fix.py").write_text("from value import VALUE\n\n\ndef test_value_is_fixed():\n    assert VALUE == 2\n")


class RealTeacherAgent(FakeCodingAgent):
    provider_name = "claude_code"


class TimeoutAgent:
    provider_name = "local_student_model"

    def run(self, task, constraints):
        return CodingAgentResult(exit_status="timeout", provider=self.provider_name, error="student attempt exceeded its bounded timeout")


class FakeApproval:
    def __init__(self, outcome=LearningApprovalOutcome.APPROVED, transcript="yes jarvis"):
        self.outcome = outcome
        self.transcript = transcript
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return LearningApprovalResult(self.outcome, self.transcript, 1.0)


class StudentTeacherTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = _init_repo(Path(self.temp.name))
        self.attempt_store = ImprovementAttemptStore(Path(self.temp.name) / "attempts.sqlite3")
        self.job_store = LearningJobStore(Path(self.temp.name) / "jobs.sqlite3")
        self.experience_store = ExperienceStore(Path(self.temp.name) / "experiences.sqlite3")
        self.trajectory_store = StudentTrajectoryStore(Path(self.temp.name) / "trajectories.sqlite3")
        self.model_registry = ModelRegistry(Path(self.temp.name) / "registry.sqlite3")

    def tearDown(self):
        self.attempt_store.close()
        self.job_store.close()
        self.experience_store.close()
        self.trajectory_store.close()
        self.model_registry.close()
        self.temp.cleanup()

    def _active_model(self, model_version="student-v1"):
        version = ModelVersion(
            model_version=model_version, dataset_version="v1", training_run_id="run-1", created_at="t",
            base_model="fake-base-model", adapter_path="/fake/adapter/path",
        )
        self.model_registry.record(version)
        self.model_registry.promote(model_version)
        return version

    def _run(self, *, student_agent=None, teacher_agent=None, request_approval=None, config=None):
        patcher = patch("brain.improvement_student_teacher.build_student_agent", return_value=student_agent)
        with patcher:
            return run_coding_task(
                "fix the bug in value.py", repository_root=str(self.repo),
                model_registry=self.model_registry, experience_store=self.experience_store,
                attempt_store=self.attempt_store, job_store=self.job_store, trajectory_store=self.trajectory_store,
                request_approval=request_approval, config=config or _fast_config(),
                teacher_agent=teacher_agent or RealTeacherAgent(),
            )


class NoActiveStudentTests(StudentTeacherTestCase):
    def test_no_active_student_uses_claude(self):
        result = self._run(teacher_agent=RealTeacherAgent(apply=_apply_real_fix))
        self.assertFalse(result.student_available)
        self.assertFalse(result.student_used)
        self.assertTrue(result.teacher_used)
        self.assertEqual(result.solved_by, "teacher")
        self.assertEqual(result.student_skip_reason, "no ACTIVE student model")


class StudentSuccessTests(StudentTeacherTestCase):
    def test_active_student_succeeds_claude_not_used(self):
        self._active_model()
        teacher = RealTeacherAgent()
        with patch.object(RealTeacherAgent, "run", wraps=teacher.run) as teacher_run:
            result = self._run(student_agent=FakeCodingAgent(apply=_apply_real_fix), teacher_agent=teacher)
        teacher_run.assert_not_called()
        self.assertTrue(result.student_available)
        self.assertTrue(result.student_used)
        self.assertTrue(result.student_succeeded)
        self.assertFalse(result.teacher_used)
        self.assertEqual(result.solved_by, "student")

    def test_student_success_trajectory_is_labeled_real_verified_student(self):
        self._active_model("student-v2")
        result = self._run(student_agent=FakeCodingAgent(apply=_apply_real_fix))
        trajectories = self.trajectory_store.query(quality_label=DataQualityLabel.REAL_VERIFIED_STUDENT.value)
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(trajectories[0].student_model_version, "student-v2")
        self.assertEqual(trajectories[0].solved_by, "student")

    def test_student_success_never_offers_learning_approval(self):
        self._active_model()
        approval = FakeApproval()
        result = self._run(student_agent=FakeCodingAgent(apply=_apply_real_fix), request_approval=approval)
        self.assertEqual(approval.calls, 0)
        self.assertIsNone(result.learning_offer)


class StudentFailureFallbackTests(StudentTeacherTestCase):
    def test_student_fails_verification_falls_back_to_claude(self):
        self._active_model()
        result = self._run(student_agent=FakeCodingAgent(), teacher_agent=RealTeacherAgent(apply=_apply_real_fix))  # no apply -> no change -> fails
        self.assertTrue(result.student_used)
        self.assertFalse(result.student_succeeded)
        self.assertTrue(result.teacher_used)
        self.assertTrue(result.teacher_succeeded)
        self.assertEqual(result.solved_by, "teacher")

    def test_student_crashes_falls_back_to_claude(self):
        self._active_model()
        result = self._run(student_agent=FakeCodingAgent(crash=True), teacher_agent=RealTeacherAgent(apply=_apply_real_fix))
        self.assertTrue(result.student_used)
        self.assertFalse(result.student_succeeded)
        self.assertTrue(result.teacher_succeeded)

    def test_student_times_out_falls_back_to_claude(self):
        self._active_model()
        result = self._run(student_agent=TimeoutAgent(), teacher_agent=RealTeacherAgent(apply=_apply_real_fix))
        self.assertTrue(result.student_used)
        self.assertFalse(result.student_succeeded)
        self.assertTrue(result.teacher_succeeded)

    def test_student_regression_falls_back_to_claude(self):
        self._active_model()

        def break_something(workspace: Path):
            (workspace / "value.py").write_text("VALUE = 2\n")
            (workspace / "tests" / "test_generated_fix.py").write_text("from value import VALUE\n\n\ndef test_value_is_fixed():\n    assert VALUE == 2\n")
            (workspace / "tests" / "test_sample.py").write_text("def test_sample():\n    assert False, 'regressed'\n")

        result = self._run(student_agent=FakeCodingAgent(apply=break_something), teacher_agent=RealTeacherAgent(apply=_apply_real_fix))
        self.assertTrue(result.student_used)
        self.assertFalse(result.student_succeeded)
        self.assertTrue(result.teacher_succeeded)

    def test_active_model_artifact_missing_gracefully_falls_back(self):
        self._active_model()
        result = self._run(student_agent=None, teacher_agent=RealTeacherAgent(apply=_apply_real_fix))
        self.assertTrue(result.student_available)
        self.assertFalse(result.student_used)  # never even attempted -- couldn't load
        self.assertIn("could not be loaded", result.student_skip_reason)
        self.assertTrue(result.teacher_succeeded)
        self.assertEqual(result.solved_by, "teacher")


class TeacherApprovalTests(StudentTeacherTestCase):
    def test_claude_succeeds_after_student_failure_learning_approval_triggered(self):
        self._active_model("student-v3")
        approval = FakeApproval(LearningApprovalOutcome.APPROVED, "yes jarvis")
        result = self._run(student_agent=FakeCodingAgent(), teacher_agent=RealTeacherAgent(apply=_apply_real_fix), request_approval=approval)
        self.assertEqual(approval.calls, 1)
        self.assertIsNotNone(result.learning_offer)
        self.assertTrue(result.learning_offer.offered)
        self.assertTrue(result.high_value_example)
        self.assertIsNotNone(result.learning_offer.job)
        self.assertTrue(result.learning_offer.job.high_value)

    def test_claude_also_fails_no_learning_approval(self):
        self._active_model()
        approval = FakeApproval()
        result = self._run(student_agent=FakeCodingAgent(), teacher_agent=RealTeacherAgent(), request_approval=approval)  # neither applies a fix
        self.assertFalse(result.teacher_succeeded)
        self.assertEqual(approval.calls, 0)
        self.assertIsNone(result.learning_offer)

    def test_high_value_example_marked_when_student_failed_and_teacher_succeeded(self):
        self._active_model()
        result = self._run(student_agent=FakeCodingAgent(), teacher_agent=RealTeacherAgent(apply=_apply_real_fix), request_approval=FakeApproval())
        self.assertTrue(result.high_value_example)

    def test_not_high_value_when_no_student_was_available(self):
        approval = FakeApproval()
        result = self._run(teacher_agent=RealTeacherAgent(apply=_apply_real_fix), request_approval=approval)
        self.assertFalse(result.high_value_example)
        self.assertFalse(result.learning_offer.job.high_value)


class ModelEligibilityTests(StudentTeacherTestCase):
    def test_rejected_model_is_never_active(self):
        self.model_registry.record(ModelVersion(model_version="m1", dataset_version="v1", training_run_id="r1", created_at="t"))
        self.model_registry.reject("m1", "worse")
        self.assertIsNone(resolve_active_student(self.model_registry))

    def test_candidate_model_is_never_active(self):
        self.model_registry.record(ModelVersion(model_version="m1", dataset_version="v1", training_run_id="r1", created_at="t"))
        self.assertIsNone(resolve_active_student(self.model_registry))

    def test_only_active_status_resolves(self):
        self._active_model("the-active-one")
        active = resolve_active_student(self.model_registry)
        self.assertEqual(active.model_version, "the-active-one")


class ExperienceRetrievalTests(StudentTeacherTestCase):
    def test_relevant_experiences_are_retrieved_and_bounded(self):
        from brain.experience_store import ExperienceRecord
        for i in range(10):
            self.experience_store.store(ExperienceRecord(
                experience_id=f"exp-{i}", learning_job_id="job-1", problem_family="fam-1",
                subsystem=None, gap_type="CODE_CAPABILITY_GAP", original_task="fix the bug in value.py",
                reusable_strategy="use the correct constant value",
            ))
        result = self._run(teacher_agent=RealTeacherAgent(apply=_apply_real_fix), config=_fast_config(experience_top_k=3))
        self.assertLessEqual(len(result.experiences_used), 3)
        self.assertGreater(len(result.experiences_used), 0)


if __name__ == "__main__":
    unittest.main()
