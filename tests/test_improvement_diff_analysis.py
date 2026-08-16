import subprocess
import tempfile
import unittest
from pathlib import Path

from brain.improvement_diff_analysis import analyze_diff


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True)


def _init_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "a.py").write_text("A = 1\n")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-qm", "initial")
    return repo


class DiffAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = _init_repo(Path(self.temp.name))
        self.base_commit = _git(self.repo, "rev-parse", "HEAD").stdout.strip()

    def tearDown(self):
        self.temp.cleanup()

    def test_no_changes_is_scope_none(self):
        analysis = analyze_diff(self.repo, self.base_commit)
        self.assertEqual(analysis.change_scope, "none")
        self.assertEqual(analysis.files_changed, [])
        self.assertFalse(analysis.diff_suspicious)

    def test_modified_tracked_file_is_source_only(self):
        (self.repo / "a.py").write_text("A = 2\n")
        analysis = analyze_diff(self.repo, self.base_commit)
        self.assertEqual(analysis.change_scope, "source_only")
        self.assertIn("a.py", analysis.files_changed)
        self.assertEqual(analysis.lines_added, 1)
        self.assertEqual(analysis.lines_removed, 1)

    def test_new_untracked_file_is_detected_as_addition(self):
        (self.repo / "b.py").write_text("B = 1\nB2 = 2\n")
        analysis = analyze_diff(self.repo, self.base_commit)
        self.assertIn("b.py", analysis.files_added)
        self.assertIn("b.py", analysis.files_changed)
        self.assertEqual(analysis.lines_added, 2)

    def test_deleted_tracked_file_is_detected(self):
        (self.repo / "a.py").unlink()
        analysis = analyze_diff(self.repo, self.base_commit)
        self.assertIn("a.py", analysis.files_deleted)

    def test_only_new_test_file_is_test_only_scope(self):
        (self.repo / "tests").mkdir()
        (self.repo / "tests" / "test_a.py").write_text("def test_x():\n    assert True\n")
        analysis = analyze_diff(self.repo, self.base_commit)
        self.assertEqual(analysis.change_scope, "test_only")
        self.assertEqual(analysis.generated_tests, ["tests/test_a.py"])

    def test_source_plus_test_change_is_mixed_scope(self):
        (self.repo / "a.py").write_text("A = 2\n")
        (self.repo / "tests").mkdir()
        (self.repo / "tests" / "test_a.py").write_text("def test_x():\n    assert True\n")
        analysis = analyze_diff(self.repo, self.base_commit)
        self.assertEqual(analysis.change_scope, "mixed")
        self.assertIn("tests/test_a.py", analysis.generated_tests)
        self.assertIn("a.py", analysis.files_changed)

    def test_unauthorized_commit_is_detected_and_flagged_suspicious(self):
        (self.repo / "a.py").write_text("A = 2\n")
        _git(self.repo, "add", "a.py")
        _git(self.repo, "commit", "-qm", "sneaky commit despite instructions")
        analysis = analyze_diff(self.repo, self.base_commit)
        self.assertTrue(analysis.unauthorized_commit)
        self.assertTrue(analysis.diff_suspicious)
        self.assertTrue(any("committed" in reason for reason in analysis.diff_suspicious_reasons))

    def test_touching_own_safety_module_is_flagged_suspicious(self):
        (self.repo / "brain").mkdir()
        (self.repo / "brain" / "task_supervisor.py").write_text("# tampered\n")
        analysis = analyze_diff(self.repo, self.base_commit)
        self.assertTrue(analysis.diff_suspicious)
        self.assertTrue(any("sensitive path" in reason for reason in analysis.diff_suspicious_reasons))

    def test_touching_gitignore_is_flagged_suspicious(self):
        (self.repo / ".gitignore").write_text("*.secret\n")
        analysis = analyze_diff(self.repo, self.base_commit)
        self.assertTrue(analysis.diff_suspicious)

    def test_mass_deletion_is_flagged_suspicious(self):
        for i in range(6):
            path = self.repo / f"extra_{i}.py"
            path.write_text(f"X_{i} = {i}\n")
            _git(self.repo, "add", path.name)
        _git(self.repo, "commit", "-qm", "add extras")
        base_with_extras = _git(self.repo, "rev-parse", "HEAD").stdout.strip()
        for i in range(6):
            (self.repo / f"extra_{i}.py").unlink()
        analysis = analyze_diff(self.repo, base_with_extras)
        self.assertTrue(analysis.diff_suspicious)
        self.assertTrue(any("deleted" in reason for reason in analysis.diff_suspicious_reasons))

    def test_large_diff_is_flagged_suspicious(self):
        (self.repo / "big.py").write_text("\n".join(f"x{i} = {i}" for i in range(2500)) + "\n")
        analysis = analyze_diff(self.repo, self.base_commit)
        self.assertTrue(analysis.diff_suspicious)
        self.assertTrue(any("total changed lines" in reason for reason in analysis.diff_suspicious_reasons))

    def test_small_ordinary_fix_is_not_flagged_suspicious(self):
        (self.repo / "a.py").write_text("A = 42\n")
        analysis = analyze_diff(self.repo, self.base_commit)
        self.assertFalse(analysis.diff_suspicious)
        self.assertEqual(analysis.diff_suspicious_reasons, [])

    def test_diff_summary_is_human_readable_and_matches_computed_stats(self):
        (self.repo / "a.py").write_text("A = 2\n")
        analysis = analyze_diff(self.repo, self.base_commit)
        self.assertIn(str(len(analysis.files_changed)), analysis.diff_summary)
        self.assertIn(f"+{analysis.lines_added}", analysis.diff_summary)
        self.assertIn(f"-{analysis.lines_removed}", analysis.diff_summary)

    def test_stray_pycache_artifacts_are_never_treated_as_generated_tests(self):
        """Running pytest inside a worktree (e.g. to establish a before/after
        baseline) leaves behind tests/__pycache__/*.pyc files. Even when
        .gitignore doesn't happen to be present (as in this throwaway test
        repo) to keep them out of `git status`, analyze_diff must never
        mistake a compiled bytecode cache file for a test the coding agent
        wrote -- both because it's not source the agent produced, and
        because reading it as UTF-8 text later (to run it against the base
        commit) would crash on binary content."""
        pycache = self.repo / "tests" / "__pycache__"
        pycache.mkdir(parents=True)
        (pycache / "test_sample.cpython-311.pyc").write_bytes(b"\xa7\x00\x01\x02not valid utf-8 or python")
        analysis = analyze_diff(self.repo, self.base_commit)
        self.assertEqual(analysis.generated_tests, [])
        self.assertNotIn("tests/__pycache__/test_sample.cpython-311.pyc", analysis.files_changed)

    def test_analysis_never_creates_a_real_commit_itself(self):
        (self.repo / "a.py").write_text("A = 2\n")
        before_head = _git(self.repo, "rev-parse", "HEAD").stdout.strip()
        analyze_diff(self.repo, self.base_commit)
        after_head = _git(self.repo, "rev-parse", "HEAD").stdout.strip()
        self.assertEqual(before_head, after_head)


if __name__ == "__main__":
    unittest.main()
