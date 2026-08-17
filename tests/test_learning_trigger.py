import subprocess
import tempfile
import unittest
from pathlib import Path

from brain.improvement_attempt_store import ImprovementAttemptStore
from brain.improvement_coding_agent import FakeCodingAgent
from brain.improvement_models import GapType, ImprovementCandidate
from brain.improvement_orchestrator import OrchestratorConfig, run_attempt
from brain.improvement_attempt_models import AttemptStatus
from brain.learning_models import LearningJob, LearningJobStatus
from brain.learning_store import LearningJobStore
from brain.learning_trigger import evaluate_learning_offer, task_family_fingerprint


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


def _candidate(**overrides) -> ImprovementCandidate:
    defaults = dict(
        candidate_id="cand-1", created_at="t", first_seen="t", last_seen="t",
        gap_type=GapType.EXECUTION_BUG.value, confidence=0.9, subsystem="filesystem",
        raw_request="fix the thing", normalized_goal="fix the thing",
        exception_type="RuntimeError", exception_message="boom",
    )
    defaults.update(overrides)
    return ImprovementCandidate(**defaults)


def _fast_config(**overrides) -> OrchestratorConfig:
    defaults = dict(regression_test_timeout_seconds=60.0, focused_test_timeout_seconds=30.0, coding_agent_timeout_seconds=30.0)
    defaults.update(overrides)
    return OrchestratorConfig(**defaults)


def _apply_real_fix(workspace: Path) -> None:
    (workspace / "value.py").write_text("VALUE = 2\n")
    (workspace / "tests" / "test_generated_fix.py").write_text(
        "from value import VALUE\n\n\ndef test_value_is_fixed():\n    assert VALUE == 2\n"
    )


class RealTeacherAgent(FakeCodingAgent):
    """Identical behavior to FakeCodingAgent but reports the real teacher's
    provider_name, so tests can build a genuine READY_FOR_REVIEW attempt
    without spending real Claude quota while still exercising the
    "real teacher only" gate honestly."""
    provider_name = "claude_code"


class LearningTriggerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = _init_repo(Path(self.temp.name))
        self.attempt_store = ImprovementAttemptStore(Path(self.temp.name) / "attempts.sqlite3")
        self.job_store = LearningJobStore(Path(self.temp.name) / "jobs.sqlite3")

    def tearDown(self):
        self.attempt_store.close()
        self.job_store.close()
        self.temp.cleanup()

    def _ready_attempt(self, agent_cls=RealTeacherAgent, **candidate_overrides):
        candidate = _candidate(**candidate_overrides)
        agent = agent_cls(apply=_apply_real_fix)
        attempt = run_attempt(candidate, repository_root=str(self.repo), coding_agent=agent, attempt_store=self.attempt_store, config=_fast_config())
        self.assertEqual(attempt.status, AttemptStatus.READY_FOR_REVIEW.value)
        return attempt

    def test_verified_real_teacher_success_triggers_offer(self):
        attempt = self._ready_attempt()
        decision = evaluate_learning_offer(attempt, store=self.job_store)
        self.assertTrue(decision.should_offer)
        self.assertTrue(decision.fingerprint)

    def test_fake_non_teacher_provider_does_not_trigger(self):
        attempt = self._ready_attempt(agent_cls=FakeCodingAgent)
        decision = evaluate_learning_offer(attempt, store=self.job_store)
        self.assertFalse(decision.should_offer)
        self.assertIn("teacher provider", decision.reason)

    def test_failed_teacher_attempt_never_triggers(self):
        candidate = _candidate()
        agent = RealTeacherAgent()  # no apply -> no change -> FIX_FAILED
        attempt = run_attempt(candidate, repository_root=str(self.repo), coding_agent=agent, attempt_store=self.attempt_store, config=_fast_config())
        self.assertNotEqual(attempt.status, AttemptStatus.READY_FOR_REVIEW.value)
        decision = evaluate_learning_offer(attempt, store=self.job_store)
        self.assertFalse(decision.should_offer)

    def test_environmental_style_candidate_never_reaches_ready_and_never_triggers(self):
        candidate = _candidate(gap_type=GapType.ENVIRONMENTAL_FAILURE.value)
        agent = RealTeacherAgent(apply=_apply_real_fix)
        attempt = run_attempt(candidate, repository_root=str(self.repo), coding_agent=agent, attempt_store=self.attempt_store, config=_fast_config())
        self.assertEqual(attempt.status, AttemptStatus.REJECTED.value)
        decision = evaluate_learning_offer(attempt, store=self.job_store)
        self.assertFalse(decision.should_offer)

    def test_high_risk_subsystem_never_triggers_even_if_somehow_ready(self):
        attempt = self._ready_attempt(subsystem="messaging", confidence=0.95)
        # messaging is UNSAFE_TO_REPLAY in repro, so behavioral confirmation had
        # to come from the differential test path -- confirm it still got to
        # READY_FOR_REVIEW, then confirm the trigger's own extra gate blocks it.
        decision = evaluate_learning_offer(attempt, store=self.job_store)
        self.assertFalse(decision.should_offer)
        self.assertIn("high-risk", decision.reason)

    def test_duplicate_solved_problem_does_not_trigger_again(self):
        attempt = self._ready_attempt()
        fingerprint = task_family_fingerprint(attempt)
        self.job_store.create(LearningJob(
            learning_job_id="existing", created_at="t", updated_at="t",
            candidate_id="other-cand", improvement_attempt_id="other-att",
            fingerprint=fingerprint, learning_status=LearningJobStatus.APPROVED.value,
        ))
        decision = evaluate_learning_offer(attempt, store=self.job_store)
        self.assertFalse(decision.should_offer)
        self.assertIsNotNone(decision.existing_job)

    def test_declined_prior_job_does_not_suppress_a_new_offer(self):
        attempt = self._ready_attempt()
        fingerprint = task_family_fingerprint(attempt)
        self.job_store.create(LearningJob(
            learning_job_id="declined-one", created_at="t", updated_at="t",
            candidate_id="other-cand", improvement_attempt_id="other-att",
            fingerprint=fingerprint, learning_status=LearningJobStatus.DECLINED.value,
        ))
        decision = evaluate_learning_offer(attempt, store=self.job_store)
        self.assertTrue(decision.should_offer)

    def test_fingerprint_ignores_wording_but_captures_root_cause_shape(self):
        attempt_a = self._ready_attempt(raw_request="please fix the widget", normalized_goal="please fix the widget")
        attempt_b = self._ready_attempt(candidate_id="cand-2", raw_request="totally different phrasing of the same bug", normalized_goal="totally different phrasing of the same bug")
        self.assertEqual(task_family_fingerprint(attempt_a), task_family_fingerprint(attempt_b))


if __name__ == "__main__":
    unittest.main()
