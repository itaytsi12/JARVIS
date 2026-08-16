"""Isolated Git worktree management for autonomous improvement attempts.

Every code-improvement attempt must run outside the user's active working
tree. This module is a thin, improvement-specific layer over the existing
`brain.task_supervisor.prepare_coding_workspace`/`create_isolated_workspace`
primitives (which already do the real `git worktree add` work, correctly
scoped and using the project's standard hidden-subprocess/redaction
conventions) -- it adds unique branch/path naming, ownership tracking, and
the "never overwrite / never touch the user's active tree" guarantees this
feature specifically needs. It does not reimplement worktree creation.
"""
from __future__ import annotations

import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from brain.task_supervisor import create_isolated_workspace, prepare_coding_workspace
from tools.windows_process import hidden_process_kwargs


class WorktreeBlocked(RuntimeError):
    """A safe worktree could not be created; the caller must report BLOCKED
    truthfully rather than proceeding."""
    pass


@dataclass
class WorktreeHandle:
    attempt_id: str
    repository_root: str
    base_commit: str
    base_branch: str
    worktree_path: str
    worktree_branch: str
    dirty_base_tree: bool


_OWNERSHIP: dict[str, str] = {}  # worktree_path (resolved, casefolded) -> attempt_id
_OWNERSHIP_LOCK = threading.Lock()


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, **hidden_process_kwargs())


def discover_repository_root(start: str | Path) -> str:
    """Resolve the real repository root from any path inside it, without
    modifying anything. Raises WorktreeBlocked if `start` isn't inside a
    Git repository at all."""
    result = _run_git(Path(start).resolve(), "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        raise WorktreeBlocked(f"not a git repository: {start!r}")
    return result.stdout.strip()


def current_base_commit(repository_root: str | Path) -> str:
    result = _run_git(Path(repository_root), "rev-parse", "HEAD")
    if result.returncode != 0:
        raise WorktreeBlocked("could not resolve the current HEAD commit")
    return result.stdout.strip()


def current_branch(repository_root: str | Path) -> str:
    result = _run_git(Path(repository_root), "branch", "--show-current")
    return result.stdout.strip() if result.returncode == 0 else ""


def is_dirty(repository_root: str | Path) -> bool:
    """Read-only detection of uncommitted changes in the user's active tree.
    Never stashes, resets, or otherwise touches them."""
    result = _run_git(Path(repository_root), "status", "--porcelain")
    if result.returncode != 0:
        raise WorktreeBlocked("could not inspect working tree status")
    return bool(result.stdout.strip())


def _short(value: str, length: int = 8) -> str:
    return (value or uuid.uuid4().hex)[:length]


def create_attempt_worktree(
    repository_root: str | Path,
    candidate_id: str,
    attempt_id: str,
    *,
    worktrees_root: str | Path | None = None,
) -> WorktreeHandle:
    """Create a fresh, uniquely-named worktree pinned to the repository's
    CURRENT HEAD commit -- never the user's uncommitted changes, and never
    a moving target. Raises WorktreeBlocked (never a bare exception) if a
    safe worktree cannot be created, so the caller can record BLOCKED
    truthfully instead of guessing.
    """
    repo = Path(repository_root).resolve()
    if not repo.is_dir():
        raise WorktreeBlocked(f"repository root does not exist: {repo}")
    try:
        real_root = Path(discover_repository_root(repo))
    except WorktreeBlocked:
        raise
    if real_root.resolve() != repo:
        # `repository_root` must be the true top-level -- refuse to guess.
        raise WorktreeBlocked(f"{repo} is not the repository root (found {real_root})")

    dirty = is_dirty(repo)
    base_commit = current_base_commit(repo)
    base_branch = current_branch(repo)

    branch = f"jarvis/improvement/{_short(candidate_id)}/{_short(attempt_id)}"
    dest = Path(worktrees_root or (repo / ".jarvis-improvement-worktrees")) / f"{_short(candidate_id)}-{_short(attempt_id)}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        raise WorktreeBlocked(f"worktree path already exists: {dest}")
    if _branch_exists(repo, branch):
        raise WorktreeBlocked(f"branch already exists: {branch}")

    try:
        result = create_isolated_workspace(str(repo), str(dest), attempt_id, branch=branch, base_commit=base_commit)
    except FileExistsError as exc:
        raise WorktreeBlocked(f"worktree path collision: {exc}") from exc
    except RuntimeError as exc:
        raise WorktreeBlocked(f"git worktree add failed: {exc}") from exc

    handle = WorktreeHandle(
        attempt_id=attempt_id,
        repository_root=str(repo),
        base_commit=base_commit,
        base_branch=base_branch,
        worktree_path=result["workspace"],
        worktree_branch=result["branch"],
        dirty_base_tree=dirty,
    )
    with _OWNERSHIP_LOCK:
        _OWNERSHIP[str(Path(handle.worktree_path).resolve()).casefold()] = attempt_id
    return handle


def _branch_exists(repo: Path, branch: str) -> bool:
    result = _run_git(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}")
    return result.returncode == 0


def owns_worktree(attempt_id: str, worktree_path: str | Path) -> bool:
    with _OWNERSHIP_LOCK:
        return _OWNERSHIP.get(str(Path(worktree_path).resolve()).casefold()) == attempt_id


def validate_worktree(handle: WorktreeHandle) -> bool:
    """Confirm a worktree path genuinely belongs to the expected repository
    (defense against a stale/tampered handle being reused).

    A worktree's `--git-common-dir` points back at the *main* repository's
    `.git` directory (absolute, when queried from inside the worktree); the
    main repository's own `--git-common-dir` is just `.git` (relative to
    itself). Resolving both to absolute paths and comparing is the real
    linkage check -- `Path.__truediv__` already discards the left operand
    when the right one is absolute, so this works uniformly for both forms.
    """
    path = Path(handle.worktree_path)
    if not path.is_dir():
        return False
    worktree_common = _run_git(path, "rev-parse", "--git-common-dir")
    if worktree_common.returncode != 0:
        return False
    main_common = _run_git(Path(handle.repository_root), "rev-parse", "--git-common-dir")
    if main_common.returncode != 0:
        return False
    resolved_from_worktree = (path / worktree_common.stdout.strip()).resolve()
    resolved_from_main = (Path(handle.repository_root) / main_common.stdout.strip()).resolve()
    return resolved_from_worktree == resolved_from_main


def cleanup_worktree(handle: WorktreeHandle, *, attempt_id: str, force: bool = False) -> bool:
    """Remove a worktree -- only for attempts that own it, and only meant
    to be called for FAILED/ABANDONED attempts. A READY_FOR_REVIEW worktree
    must be left in place by the caller (this function doesn't decide
    that policy; the orchestrator does)."""
    if not owns_worktree(attempt_id, handle.worktree_path):
        raise WorktreeBlocked("attempt does not own this worktree; refusing to remove it")
    repo = Path(handle.repository_root)
    args = ["worktree", "remove", handle.worktree_path] + (["--force"] if force else [])
    result = _run_git(repo, *args)
    if result.returncode != 0:
        return False
    _run_git(repo, "branch", "-D", handle.worktree_branch)
    with _OWNERSHIP_LOCK:
        _OWNERSHIP.pop(str(Path(handle.worktree_path).resolve()).casefold(), None)
    return True


def list_worktrees(repository_root: str | Path) -> list[dict]:
    result = _run_git(Path(repository_root), "worktree", "list", "--porcelain")
    if result.returncode != 0:
        return []
    entries, current = [], {}
    for line in result.stdout.splitlines():
        if not line.strip():
            if current:
                entries.append(current)
            current = {}
            continue
        if " " in line:
            key, value = line.split(" ", 1)
            current[key] = value
        else:
            current[line] = True
    if current:
        entries.append(current)
    return entries
