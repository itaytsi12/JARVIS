import subprocess
import tempfile
import unittest
from pathlib import Path

from brain.improvement_attempt_store import ImprovementAttemptStore
from brain.improvement_models import GapType, ImprovementCandidate
from brain.improvement_orchestrator import OrchestratorConfig, run_attempt
from brain.improvement_attempt_models import AttemptStatus
from brain.learning_package import extract_learning_package
from tests.test_learning_trigger import RealTeacherAgent, _apply_real_fix, _candidate, _fast_config, _init_repo


class LearningPackageExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = _init_repo(Path(self.temp.name))
        self.attempt_store = ImprovementAttemptStore(Path(self.temp.name) / "attempts.sqlite3")

    def tearDown(self):
        self.attempt_store.close()
        self.temp.cleanup()

    def _ready_attempt(self, **overrides):
        candidate = _candidate(**overrides)
        agent = RealTeacherAgent(apply=_apply_real_fix)
        attempt = run_attempt(candidate, repository_root=str(self.repo), coding_agent=agent, attempt_store=self.attempt_store, config=_fast_config())
        self.assertEqual(attempt.status, AttemptStatus.READY_FOR_REVIEW.value)
        return attempt

    def test_extraction_captures_problem_and_evidence(self):
        attempt = self._ready_attempt()
        package = extract_learning_package(attempt, learning_job_id="job-1")
        self.assertEqual(package.learning_job_id, "job-1")
        self.assertEqual(package.improvement_attempt_id, attempt.attempt_id)
        self.assertTrue(package.problem_family)
        self.assertEqual(package.gap_type, GapType.EXECUTION_BUG.value)
        self.assertIn("value.py", package.files_changed)
        self.assertTrue(package.generated_tests)
        self.assertTrue(all(package.acceptance_gates.values()))

    def test_reusable_strategy_is_a_deterministic_summary_not_free_text_injection(self):
        attempt = self._ready_attempt()
        package_a = extract_learning_package(attempt, learning_job_id="job-1")
        package_b = extract_learning_package(attempt, learning_job_id="job-1")
        self.assertEqual(package_a.reusable_strategy, package_b.reusable_strategy)
        self.assertIn(attempt.gap_type, package_a.reusable_strategy)

    def test_no_hidden_reasoning_fields_exist(self):
        attempt = self._ready_attempt()
        package = extract_learning_package(attempt, learning_job_id="job-1")
        payload = package.to_dict()
        for forbidden in ("chain_of_thought", "reasoning", "thinking", "internal_monologue"):
            self.assertNotIn(forbidden, payload)

    def test_do_not_generalize_notes_are_present(self):
        attempt = self._ready_attempt()
        package = extract_learning_package(attempt, learning_job_id="job-1")
        self.assertTrue(package.do_not_generalize)

    def test_round_trips_through_dict(self):
        attempt = self._ready_attempt()
        package = extract_learning_package(attempt, learning_job_id="job-1")
        restored = type(package).from_dict(package.to_dict())
        self.assertEqual(restored, package)

    def test_secret_looking_original_request_is_sanitized(self):
        attempt = self._ready_attempt(raw_request="my api key is sk-abcdefghijklmnopqrstuvwx please fix this")
        package = extract_learning_package(attempt, learning_job_id="job-1")
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwx", package.original_task)


if __name__ == "__main__":
    unittest.main()
