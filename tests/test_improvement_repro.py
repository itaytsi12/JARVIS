import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brain.improvement_attempt_models import ReproductionType
from brain.improvement_models import ImprovementCandidate
from brain.improvement_repro import SUBSYSTEM_TEST_FILES, reproduce
from brain.task_supervisor import SafeCommandRunner


def _candidate(**overrides) -> ImprovementCandidate:
    defaults = dict(candidate_id="c1", created_at="t", first_seen="t", last_seen="t")
    defaults.update(overrides)
    return ImprovementCandidate(**defaults)


class UnsafeToReplayTests(unittest.TestCase):
    def test_messaging_subsystem_refuses_without_running_anything(self):
        candidate = _candidate(subsystem="messaging")
        with patch("brain.improvement_repro.SafeCommandRunner") as runner_cls:
            result = reproduce(candidate, "/does/not/matter")
        runner_cls.assert_not_called()
        self.assertEqual(result.method, ReproductionType.UNSAFE_TO_REPLAY.value)
        self.assertFalse(result.attempted)
        self.assertIsNone(result.reproduced)
        self.assertIn("real-world side effects", result.skip_reason)


class EvidenceOnlyTests(unittest.TestCase):
    def test_redacted_arguments_fall_back_to_evidence_only(self):
        candidate = _candidate(
            subsystem="browser",
            exception_type="RuntimeError",
            executed_actions=[{"tool": "browser_type", "arguments": {"text": "<REDACTED>"}}],
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = reproduce(candidate, tmp)
        self.assertEqual(result.method, ReproductionType.LOG_EVIDENCE_ONLY.value)
        self.assertTrue(result.attempted)
        self.assertIsNone(result.reproduced)
        self.assertIn("redacted", result.skip_reason)
        self.assertEqual(result.evidence["exception_type"], "RuntimeError")

    def test_unmapped_subsystem_falls_back_to_evidence_only(self):
        candidate = _candidate(subsystem="filesystem", exception_type="OSError")
        with tempfile.TemporaryDirectory() as tmp:
            result = reproduce(candidate, tmp)
        self.assertEqual(result.method, ReproductionType.LOG_EVIDENCE_ONLY.value)
        self.assertIn("no existing automated test suite", result.skip_reason)

    def test_none_subsystem_falls_back_to_evidence_only(self):
        candidate = _candidate(subsystem=None)
        with tempfile.TemporaryDirectory() as tmp:
            result = reproduce(candidate, tmp)
        self.assertEqual(result.method, ReproductionType.LOG_EVIDENCE_ONLY.value)

    def test_mapped_subsystem_but_missing_test_file_falls_back_to_evidence_only(self):
        candidate = _candidate(subsystem="browser")
        with tempfile.TemporaryDirectory() as tmp:
            result = reproduce(candidate, tmp)  # no tests/ directory created at all
        self.assertEqual(result.method, ReproductionType.LOG_EVIDENCE_ONLY.value)
        self.assertIn("was not found", result.skip_reason)


class IntegrationReproTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        (self.workspace / "tests").mkdir()
        (self.workspace / "tests" / "test_browser_agent.py").write_text("def test_placeholder():\n    assert True\n")

    def tearDown(self):
        self.temp.cleanup()

    def _candidate(self):
        return ImprovementCandidate(candidate_id="c1", created_at="t", first_seen="t", last_seen="t", subsystem="browser")

    def test_clean_pass_means_not_reproduced(self):
        candidate = self._candidate()
        fake_runner = type("R", (), {"run": lambda self, cmd, ws, timeout=300: {"exit_code": 0, "output": "1 passed"}})()
        result = reproduce(candidate, self.workspace, runner=fake_runner)
        self.assertEqual(result.method, ReproductionType.INTEGRATION_REPRO.value)
        self.assertTrue(result.attempted)
        self.assertFalse(result.reproduced)
        self.assertEqual(result.evidence["exit_code"], 0)

    def test_test_failure_means_reproduced(self):
        candidate = self._candidate()
        fake_runner = type("R", (), {"run": lambda self, cmd, ws, timeout=300: {"exit_code": 1, "output": "1 failed"}})()
        result = reproduce(candidate, self.workspace, runner=fake_runner)
        self.assertTrue(result.reproduced)
        self.assertEqual(result.evidence["exit_code"], 1)

    def test_collection_error_is_inconclusive_not_a_guess(self):
        candidate = self._candidate()
        fake_runner = type("R", (), {"run": lambda self, cmd, ws, timeout=300: {"exit_code": 2, "output": "collection error"}})()
        result = reproduce(candidate, self.workspace, runner=fake_runner)
        self.assertIsNone(result.reproduced)
        self.assertIn("not a clean pass/fail", result.skip_reason)

    def test_timeout_is_reported_without_raising(self):
        candidate = self._candidate()

        def _raise_timeout(cmd, ws, timeout=300):
            raise subprocess.TimeoutExpired(cmd="pytest", timeout=timeout)

        fake_runner = type("R", (), {"run": lambda self, cmd, ws, timeout=300: _raise_timeout(cmd, ws, timeout)})()
        result = reproduce(candidate, self.workspace, runner=fake_runner, timeout_seconds=5)
        self.assertTrue(result.attempted)
        self.assertIsNone(result.reproduced)
        self.assertIn("exceeded", result.skip_reason)

    def test_real_safe_command_runner_actually_executes_pytest(self):
        """One genuine, unmocked end-to-end check that the wiring to
        SafeCommandRunner really runs the mapped subsystem test file."""
        candidate = self._candidate()
        result = reproduce(candidate, self.workspace, runner=SafeCommandRunner(), timeout_seconds=60)
        self.assertEqual(result.method, ReproductionType.INTEGRATION_REPRO.value)
        self.assertFalse(result.reproduced)  # the placeholder test passes
        self.assertEqual(result.evidence["exit_code"], 0)


class SubsystemTestFileMapTests(unittest.TestCase):
    def test_every_mapped_test_file_actually_exists_in_this_repository(self):
        repo_root = Path(__file__).resolve().parent.parent
        for subsystem, relative_path in SUBSYSTEM_TEST_FILES.items():
            with self.subTest(subsystem=subsystem):
                self.assertTrue((repo_root / relative_path).exists(), f"{relative_path} referenced for {subsystem!r} does not exist")


if __name__ == "__main__":
    unittest.main()
