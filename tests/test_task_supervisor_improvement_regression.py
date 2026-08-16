"""Regression coverage for the specific `task_supervisor` behaviors the
improvement-execution infrastructure depends on: the pre-existing
`create_isolated_workspace` call signature must keep working unchanged for
any existing caller, its new optional `branch`/`base_commit` parameters must
behave correctly for the new caller (`brain.improvement_worktree`), and
`SafeCommandRunner` must keep refusing every dangerous git/shell operation
before a subprocess is ever launched.
"""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brain.task_supervisor import SafeCommandRunner, create_isolated_workspace


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


class CreateIsolatedWorkspaceBackwardCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = _init_repo(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_old_three_positional_argument_call_still_works(self):
        result = create_isolated_workspace(self.repo, self.root / "worktree-old", "abc12345")
        self.assertTrue(Path(result["workspace"]).is_dir())
        self.assertEqual(result["branch"], "jarvis/task-abc12345")

    def test_explicit_branch_overrides_suggested_branch(self):
        result = create_isolated_workspace(self.repo, self.root / "worktree-branch", "task-1", branch="jarvis/improvement/x/y")
        self.assertEqual(result["branch"], "jarvis/improvement/x/y")

    def test_explicit_base_commit_pins_worktree_to_that_commit(self):
        first_commit = _git(self.repo, "rev-parse", "HEAD").stdout.strip()
        (self.repo / "b.py").write_text("B = 2\n")
        _git(self.repo, "add", "b.py")
        _git(self.repo, "commit", "-qm", "second")
        second_commit = _git(self.repo, "rev-parse", "HEAD").stdout.strip()
        self.assertNotEqual(first_commit, second_commit)

        result = create_isolated_workspace(self.repo, self.root / "worktree-pinned", "task-2", base_commit=first_commit)
        worktree_head = _git(Path(result["workspace"]), "rev-parse", "HEAD").stdout.strip()
        self.assertEqual(worktree_head, first_commit)
        self.assertFalse((Path(result["workspace"]) / "b.py").exists())

    def test_branch_and_base_commit_can_be_combined(self):
        commit = _git(self.repo, "rev-parse", "HEAD").stdout.strip()
        result = create_isolated_workspace(self.repo, self.root / "worktree-both", "task-3", branch="jarvis/improvement/both", base_commit=commit)
        self.assertEqual(result["branch"], "jarvis/improvement/both")
        self.assertEqual(_git(Path(result["workspace"]), "rev-parse", "HEAD").stdout.strip(), commit)

    def test_existing_destination_still_raises_file_exists_error(self):
        dest = self.root / "worktree-exists"
        dest.mkdir()
        with self.assertRaises(FileExistsError):
            create_isolated_workspace(self.repo, dest, "task-4")

    def test_inspection_fields_unchanged_repository_dirty_suggested_branch_safe(self):
        result = create_isolated_workspace(self.repo, self.root / "worktree-fields", "task-5")
        self.assertEqual(set(result), {"repository", "dirty", "suggested_branch", "safe", "workspace", "branch"})
        self.assertFalse(result["dirty"])
        self.assertTrue(result["safe"])


class SafeCommandRunnerDangerousOperationTests(unittest.TestCase):
    """Every one of these must be rejected before subprocess.run is ever
    called -- allowlist-before-launch, not sandboxing-after-launch."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runner = SafeCommandRunner()

    def tearDown(self):
        self.temp.cleanup()

    def _assert_blocked(self, command):
        with patch("brain.task_supervisor.subprocess.run") as run, self.assertRaises(PermissionError):
            self.runner.run(command, self.root)
        run.assert_not_called()

    def test_git_push_is_blocked(self):
        self._assert_blocked(["git", "push", "origin", "main"])

    def test_git_merge_is_blocked(self):
        self._assert_blocked(["git", "merge", "feature-branch"])

    def test_git_reset_hard_is_blocked(self):
        self._assert_blocked(["git", "reset", "--hard", "HEAD~1"])

    def test_git_clean_force_is_blocked(self):
        self._assert_blocked(["git", "clean", "-fd"])

    def test_git_commit_is_blocked(self):
        self._assert_blocked(["git", "commit", "-am", "sneaky"])

    def test_rm_is_not_allowlisted(self):
        self._assert_blocked(["rm", "-rf", "important.py"])

    def test_del_is_not_allowlisted(self):
        self._assert_blocked(["del", "important.py"])

    def test_pip_install_is_blocked_even_though_python_is_allowlisted(self):
        self._assert_blocked(["python", "-m", "pip", "install", "something"])

    def test_read_only_git_status_still_allowed(self):
        completed = type("Result", (), {"stdout": "clean", "stderr": "", "returncode": 0})()
        with patch("brain.task_supervisor.subprocess.run", return_value=completed) as run:
            result = self.runner.run(["git", "status"], self.root)
        self.assertEqual(result["exit_code"], 0)
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
