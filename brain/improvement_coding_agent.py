"""Provider-neutral coding-agent abstraction.

`CodingAgent.run(task, workspace, constraints)` is the entire interface the
rest of the improvement pipeline depends on -- it never assumes a specific
provider. `ClaudeCodeAdapter` shells out to the `claude` CLI already
available in this environment, scoped to the assigned worktree only.
`FakeCodingAgent` is a deterministic, safe test double used throughout the
automated test suite so tests never spend real API budget or take
unpredictable time.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from tools.windows_process import hidden_process_kwargs
from training_data.sanitizer import sanitize_text


def _is_isolated_worktree(workspace: str | Path) -> bool:
    """The single cheap, structural check that distinguishes an isolated
    `git worktree add` checkout from a normal repository checkout (like the
    user's main JARVIS tree): a linked worktree's `.git` entry is always a
    FILE containing `gitdir: ...`, never a directory. A primary checkout's
    `.git` is always a directory. This is a Git invariant, not a convention
    this codebase invented, so it can't be accidentally satisfied by an
    ordinary repository -- including this one.
    """
    path = Path(workspace)
    git_entry = path / ".git"
    return path.is_dir() and git_entry.is_file()


@dataclass
class CodingAgentConstraints:
    """Everything a coding-agent invocation must respect. `workspace` is
    always the isolated worktree path -- never the user's main tree."""
    workspace: str
    timeout_seconds: float = 900.0
    forbidden_commands: tuple[str, ...] = ("commit", "push", "merge", "reset --hard", "clean -f")
    max_files_touched: int | None = None


@dataclass
class CodingAgentResult:
    exit_status: str  # "completed" | "crashed" | "timeout" | "no_change" | "blocked"
    provider: str
    model: str | None = None
    invocation_mode: str = "print"
    model_calls: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0
    stdout_summary: str = ""
    error: str | None = None
    raw_output_reference: str | None = None


class CodingAgent(Protocol):
    provider_name: str

    def run(self, task: str, constraints: CodingAgentConstraints) -> CodingAgentResult: ...


class FakeCodingAgent:
    """Deterministic test double. `apply` is a callable invoked with the
    workspace Path -- it performs whatever file changes the test scenario
    needs (a real patch, no patch, a crash, etc.), so tests exercise the
    real downstream diff/validation/evaluation pipeline against a real
    (if synthetic) worktree change, not a mocked one."""

    provider_name = "fake"

    def __init__(self, apply: Callable[[Path], None] | None = None, *, crash: bool = False, model_calls: int = 1):
        self._apply = apply
        self._crash = crash
        self._model_calls = model_calls

    def run(self, task: str, constraints: CodingAgentConstraints) -> CodingAgentResult:
        started = time.time()
        if self._crash:
            return CodingAgentResult(
                exit_status="crashed", provider=self.provider_name,
                started_at=started, ended_at=time.time(),
                error="FakeCodingAgent configured to simulate a crash",
            )
        if self._apply is not None:
            self._apply(Path(constraints.workspace))
        return CodingAgentResult(
            exit_status="completed", provider=self.provider_name,
            model_calls=self._model_calls,
            started_at=started, ended_at=time.time(),
            stdout_summary="fake agent applied its configured change",
        )


class ClaudeCodeAdapter:
    """Shells out to the `claude` CLI in non-interactive print mode, scoped
    to the assigned worktree via `cwd`. `--dangerously-skip-permissions` is
    used deliberately here and only here: this adapter must only ever run
    against an isolated worktree created by `brain.improvement_worktree`,
    never the user's main tree, which is exactly the sandboxed use case
    that flag is documented for.

    That guarantee is enforced here, not merely assumed: every `run` call
    first checks `_is_isolated_worktree(constraints.workspace)` and refuses
    (returning exit_status="blocked", never raising, never invoking the
    subprocess) if the workspace isn't a genuine linked worktree. This is
    the last line of defense against a future caller wiring this adapter up
    against the main repository -- accidentally or otherwise.
    """

    provider_name = "claude_code"

    def __init__(self, model: str | None = None, executable: str = "claude"):
        self.model = model
        self.executable = executable

    def run(self, task: str, constraints: CodingAgentConstraints) -> CodingAgentResult:
        started = time.time()
        if not _is_isolated_worktree(constraints.workspace):
            return CodingAgentResult(
                exit_status="blocked", provider=self.provider_name, model=self.model,
                started_at=started, ended_at=time.time(),
                error=(
                    f"refusing to run: {constraints.workspace!r} is not an isolated git worktree "
                    "(--dangerously-skip-permissions is only ever safe against a worktree created by "
                    "brain.improvement_worktree, never a primary repository checkout)"
                ),
            )
        args = [self.executable, "-p", task, "--output-format", "json", "--dangerously-skip-permissions"]
        if self.model:
            args += ["--model", self.model]
        try:
            result = subprocess.run(
                args, cwd=constraints.workspace, text=True, capture_output=True,
                timeout=constraints.timeout_seconds, **hidden_process_kwargs(),
            )
        except subprocess.TimeoutExpired:
            return CodingAgentResult(
                exit_status="timeout", provider=self.provider_name, model=self.model,
                started_at=started, ended_at=time.time(),
                error=f"coding agent exceeded {constraints.timeout_seconds}s",
            )
        except Exception as exc:
            return CodingAgentResult(
                exit_status="crashed", provider=self.provider_name, model=self.model,
                started_at=started, ended_at=time.time(),
                error=sanitize_text(f"{type(exc).__name__}: {exc}"),
            )
        ended = time.time()
        if result.returncode != 0:
            return CodingAgentResult(
                exit_status="crashed", provider=self.provider_name, model=self.model,
                started_at=started, ended_at=ended,
                error=sanitize_text((result.stderr or "")[-2000:]),
                stdout_summary=sanitize_text((result.stdout or "")[-2000:]),
            )
        return CodingAgentResult(
            exit_status="completed", provider=self.provider_name, model=self.model,
            started_at=started, ended_at=ended,
            stdout_summary=sanitize_text((result.stdout or "")[-4000:]),
        )
