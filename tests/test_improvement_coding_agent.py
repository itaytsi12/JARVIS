import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brain.improvement_coding_agent import (
    ClaudeCodeAdapter,
    CodingAgentConstraints,
    CodingAgentResult,
    FakeCodingAgent,
    _is_isolated_worktree,
)
from brain.improvement_worktree import create_attempt_worktree


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


class FakeCodingAgentTests(unittest.TestCase):
    """The deterministic test double must behave predictably enough that the
    rest of the pipeline's tests can trust it as ground truth."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_default_agent_completes_without_touching_the_workspace(self):
        agent = FakeCodingAgent()
        result = agent.run("do something", CodingAgentConstraints(workspace=str(self.workspace)))
        self.assertEqual(result.exit_status, "completed")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_apply_callback_receives_the_workspace_path(self):
        received = {}

        def apply(path: Path):
            received["path"] = path
            (path / "generated.py").write_text("X = 1\n")

        agent = FakeCodingAgent(apply=apply)
        agent.run("task", CodingAgentConstraints(workspace=str(self.workspace)))
        self.assertEqual(received["path"], self.workspace)
        self.assertTrue((self.workspace / "generated.py").exists())

    def test_crash_flag_produces_crashed_status_and_never_calls_apply(self):
        called = []
        agent = FakeCodingAgent(apply=lambda p: called.append(p), crash=True)
        result = agent.run("task", CodingAgentConstraints(workspace=str(self.workspace)))
        self.assertEqual(result.exit_status, "crashed")
        self.assertIsNotNone(result.error)
        self.assertEqual(called, [])

    def test_custom_model_calls_is_reflected_in_result(self):
        agent = FakeCodingAgent(model_calls=7)
        result = agent.run("task", CodingAgentConstraints(workspace=str(self.workspace)))
        self.assertEqual(result.model_calls, 7)


class CodingAgentConstraintsTests(unittest.TestCase):
    def test_default_timeout_and_forbidden_commands(self):
        constraints = CodingAgentConstraints(workspace="/tmp/x")
        self.assertEqual(constraints.timeout_seconds, 900.0)
        for forbidden in ("commit", "push", "merge", "reset --hard", "clean -f"):
            self.assertIn(forbidden, constraints.forbidden_commands)


class ProviderNeutralInterfaceTests(unittest.TestCase):
    """Any downstream caller must be able to drive an agent through the
    `CodingAgent` protocol alone -- `.provider_name` and `.run(task,
    constraints)` -- without knowing whether it's talking to the fake or a
    real provider."""

    def _drive(self, agent, workspace: Path) -> CodingAgentResult:
        return agent.run("fix the bug", CodingAgentConstraints(workspace=str(workspace)))

    def test_fake_agent_satisfies_the_protocol_generically(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._drive(FakeCodingAgent(), Path(tmp))
            self.assertEqual(result.provider, "fake")
            self.assertIsInstance(result, CodingAgentResult)

    def test_claude_adapter_satisfies_the_protocol_generically_when_blocked(self):
        # No real invocation happens (workspace isn't a worktree), but the
        # interface shape must still match: same call, same result type.
        with tempfile.TemporaryDirectory() as tmp:
            result = self._drive(ClaudeCodeAdapter(), Path(tmp))
            self.assertEqual(result.provider, "claude_code")
            self.assertIsInstance(result, CodingAgentResult)


class IsolatedWorktreeInvariantTests(unittest.TestCase):
    """`_is_isolated_worktree` is the structural check ClaudeCodeAdapter
    relies on to refuse running with --dangerously-skip-permissions
    anywhere but a genuine linked git worktree."""

    def test_plain_directory_with_no_git_at_all_is_not_a_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(_is_isolated_worktree(tmp))

    def test_primary_repository_checkout_is_not_a_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            self.assertTrue((repo / ".git").is_dir())
            self.assertFalse(_is_isolated_worktree(repo))

    def test_genuine_linked_worktree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            handle = create_attempt_worktree(repo, "cand-1", "att-1")
            self.assertTrue((Path(handle.worktree_path) / ".git").is_file())
            self.assertTrue(_is_isolated_worktree(handle.worktree_path))

    def test_nonexistent_path_is_not_a_worktree(self):
        self.assertFalse(_is_isolated_worktree("Z:/definitely/does/not/exist"))


class ClaudeCodeAdapterSafetyTests(unittest.TestCase):
    """No real `claude` process may be invoked anywhere in this suite --
    every subprocess boundary is mocked."""

    def test_refuses_to_run_against_a_plain_non_worktree_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("brain.improvement_coding_agent.subprocess.run") as run:
                result = ClaudeCodeAdapter().run("task", CodingAgentConstraints(workspace=tmp))
        run.assert_not_called()
        self.assertEqual(result.exit_status, "blocked")
        self.assertIn("not an isolated git worktree", result.error)

    def test_refuses_to_run_against_the_primary_repository_checkout(self):
        """This is the specific invariant that matters most: the adapter must
        never operate against something that looks like the active JARVIS
        working tree (a normal repo checkout), even if it's a real git repo
        with a real .git directory."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            with patch("brain.improvement_coding_agent.subprocess.run") as run:
                result = ClaudeCodeAdapter().run("task", CodingAgentConstraints(workspace=str(repo)))
        run.assert_not_called()
        self.assertEqual(result.exit_status, "blocked")

    def test_proceeds_against_a_genuine_improvement_issued_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            handle = create_attempt_worktree(repo, "cand-2", "att-2")
            fake_completed = subprocess.CompletedProcess(args=[], returncode=0, stdout='{"ok": true}', stderr="")
            with patch("brain.improvement_coding_agent.subprocess.run", return_value=fake_completed) as run:
                result = ClaudeCodeAdapter().run("task", CodingAgentConstraints(workspace=handle.worktree_path))
        run.assert_called_once()
        _, kwargs = run.call_args
        self.assertEqual(kwargs["cwd"], handle.worktree_path)
        self.assertEqual(result.exit_status, "completed")

    def test_invocation_uses_dangerously_skip_permissions_only_once_worktree_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            handle = create_attempt_worktree(repo, "cand-3", "att-3")
            fake_completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="{}", stderr="")
            with patch("brain.improvement_coding_agent.subprocess.run", return_value=fake_completed) as run:
                ClaudeCodeAdapter().run("task", CodingAgentConstraints(workspace=handle.worktree_path))
        args, _ = run.call_args
        self.assertIn("--dangerously-skip-permissions", args[0])

    def test_timeout_is_reported_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            handle = create_attempt_worktree(repo, "cand-4", "att-4")
            with patch("brain.improvement_coding_agent.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=5)):
                result = ClaudeCodeAdapter().run("task", CodingAgentConstraints(workspace=handle.worktree_path, timeout_seconds=5))
        self.assertEqual(result.exit_status, "timeout")
        self.assertIn("5", result.error)

    def test_missing_executable_is_reported_as_crashed_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            handle = create_attempt_worktree(repo, "cand-5", "att-5")
            with patch("brain.improvement_coding_agent.subprocess.run", side_effect=FileNotFoundError("claude not found")):
                result = ClaudeCodeAdapter().run("task", CodingAgentConstraints(workspace=handle.worktree_path))
        self.assertEqual(result.exit_status, "crashed")
        self.assertIn("claude not found", result.error)

    def test_non_zero_exit_is_reported_as_crashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            handle = create_attempt_worktree(repo, "cand-6", "att-6")
            failed = subprocess.CompletedProcess(args=[], returncode=1, stdout="partial output", stderr="fatal: something broke")
            with patch("brain.improvement_coding_agent.subprocess.run", return_value=failed):
                result = ClaudeCodeAdapter().run("task", CodingAgentConstraints(workspace=handle.worktree_path))
        self.assertEqual(result.exit_status, "crashed")
        self.assertIn("something broke", result.error)

    def test_secret_looking_output_is_sanitized_before_reaching_the_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            handle = create_attempt_worktree(repo, "cand-7", "att-7")
            leaky = subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout="API_KEY=sk-abcdefghijklmnopqrstuvwx\nfinished ok", stderr="",
            )
            with patch("brain.improvement_coding_agent.subprocess.run", return_value=leaky):
                result = ClaudeCodeAdapter().run("task", CodingAgentConstraints(workspace=handle.worktree_path))
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwx", result.stdout_summary)

    def test_secret_looking_stderr_is_sanitized_on_crash_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            handle = create_attempt_worktree(repo, "cand-8", "att-8")
            leaky = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="",
                stderr="Authorization: Bearer sk-abcdefghijklmnopqrstuvwx failed",
            )
            with patch("brain.improvement_coding_agent.subprocess.run", return_value=leaky):
                result = ClaudeCodeAdapter().run("task", CodingAgentConstraints(workspace=handle.worktree_path))
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwx", result.error)


if __name__ == "__main__":
    unittest.main()
