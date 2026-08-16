"""Structural analysis of what a coding-agent invocation actually changed in
an attempt worktree, computed straight from `git diff` -- never from the
coding agent's own summary of its work. `brain/improvement_coding_agent.py`
tells the agent not to commit/push/merge/reset; this module is the check
that verifies that instruction was actually honored, and flags anything
that looks like the agent touched its own safety rails.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from tools.windows_process import hidden_process_kwargs

# Files whose modification is inherently suspicious for an autonomous
# improvement attempt: a coding agent editing its own sandbox/safety
# machinery, CI config, or secrets is exactly the failure mode isolation is
# meant to prevent, even though the worktree itself can't literally escape
# its sandbox by editing these paths (they only take effect in the main
# tree once a human reviews and merges the branch).
_SENSITIVE_PATH_MARKERS = (
    "brain/improvement_worktree.py",
    "brain/improvement_coding_agent.py",
    "brain/improvement_orchestrator.py",
    "brain/improvement_evaluator.py",
    "brain/task_supervisor.py",
    ".github/workflows/",
    ".gitignore",
    ".env",
)

_LARGE_DIFF_LINE_THRESHOLD = 2000
_MANY_FILES_THRESHOLD = 25
_MANY_DELETIONS_THRESHOLD = 5


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, **hidden_process_kwargs())


@dataclass
class DiffAnalysis:
    files_changed: list[str] = field(default_factory=list)
    files_added: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0
    diff_summary: str = ""
    change_scope: str = "none"  # "none" | "test_only" | "source_only" | "mixed"
    generated_tests: list[str] = field(default_factory=list)
    diff_suspicious: bool = False
    diff_suspicious_reasons: list[str] = field(default_factory=list)
    unauthorized_commit: bool = False


def _parse_numstat(output: str) -> dict[str, tuple[int, int]]:
    stats: dict[str, tuple[int, int]] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        added_raw, removed_raw, path = parts
        added = int(added_raw) if added_raw.isdigit() else 0
        removed = int(removed_raw) if removed_raw.isdigit() else 0
        stats[path] = (added, removed)
    return stats


def _parse_name_status(output: str) -> tuple[list[str], list[str], list[str]]:
    added, deleted, modified = [], [], []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status, paths = parts[0], parts[1:]
        if status.startswith("A"):
            added.append(paths[0])
        elif status.startswith("D"):
            deleted.append(paths[0])
        elif status.startswith("R"):
            # rename: old path is effectively deleted, new path added.
            deleted.append(paths[0])
            added.append(paths[-1])
        else:
            modified.append(paths[-1])
    return added, deleted, modified


def _is_bytecode_cache_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return "__pycache__/" in normalized or normalized.endswith((".pyc", ".pyo"))


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.startswith("tests/") and normalized.endswith(".py") and Path(normalized).name.startswith("test_")


def analyze_diff(worktree_path: str | Path, base_commit: str) -> DiffAnalysis:
    """Compute a DiffAnalysis for everything that changed in `worktree_path`
    relative to `base_commit` -- staged, unstaged, AND untracked files.
    Never raises for an empty/clean diff; that's just `change_scope="none"`.
    """
    repo = Path(worktree_path)
    analysis = DiffAnalysis()

    head = _run_git(repo, "rev-parse", "HEAD").stdout.strip()
    if head and head != base_commit:
        analysis.unauthorized_commit = True
        analysis.diff_suspicious = True
        analysis.diff_suspicious_reasons.append(
            f"worktree HEAD ({head[:12]}) has moved past base_commit ({base_commit[:12]}) -- "
            "the coding agent appears to have committed despite being instructed not to"
        )

    # `-N` marks untracked files as intent-to-add (empty blob -> real content)
    # WITHOUT staging their actual content for a commit, so `git diff` can see
    # them as pure additions alongside tracked changes. This never commits or
    # pushes anything; it only affects this worktree's index.
    _run_git(repo, "add", "-A", "-N")

    numstat = _parse_numstat(_run_git(repo, "diff", "--numstat", base_commit).stdout)
    added, deleted, modified = _parse_name_status(_run_git(repo, "diff", "--name-status", base_commit).stdout)

    # __pycache__/*.pyc noise from tests that ran inside this worktree is not
    # a real change a coding agent made -- normally .gitignore keeps it out
    # of `git status` entirely, but this filter doesn't depend on that; a
    # bytecode cache artifact must never be mistaken for a "generated test".
    added = [p for p in added if not _is_bytecode_cache_path(p)]
    deleted = [p for p in deleted if not _is_bytecode_cache_path(p)]
    modified = [p for p in modified if not _is_bytecode_cache_path(p)]

    analysis.files_added = sorted(added)
    analysis.files_deleted = sorted(deleted)
    analysis.files_changed = sorted(set(added) | set(deleted) | set(modified))

    for path, (added_lines, removed_lines) in numstat.items():
        if _is_bytecode_cache_path(path):
            continue
        analysis.lines_added += added_lines
        analysis.lines_removed += removed_lines

    analysis.generated_tests = [p for p in analysis.files_changed if _is_test_path(p)]

    if not analysis.files_changed:
        analysis.change_scope = "none"
    elif all(_is_test_path(p) for p in analysis.files_changed):
        analysis.change_scope = "test_only"
    elif not analysis.generated_tests:
        analysis.change_scope = "source_only"
    else:
        analysis.change_scope = "mixed"

    analysis.diff_summary = (
        f"{len(analysis.files_changed)} file(s) changed, +{analysis.lines_added}/-{analysis.lines_removed} "
        f"({analysis.change_scope})"
    )

    if len(analysis.files_changed) > _MANY_FILES_THRESHOLD:
        analysis.diff_suspicious = True
        analysis.diff_suspicious_reasons.append(f"{len(analysis.files_changed)} files touched, more than expected for a single targeted fix")

    if analysis.lines_added + analysis.lines_removed > _LARGE_DIFF_LINE_THRESHOLD:
        analysis.diff_suspicious = True
        analysis.diff_suspicious_reasons.append(
            f"{analysis.lines_added + analysis.lines_removed} total changed lines, unusually large for an autonomous fix"
        )

    if len(analysis.files_deleted) > _MANY_DELETIONS_THRESHOLD:
        analysis.diff_suspicious = True
        analysis.diff_suspicious_reasons.append(f"{len(analysis.files_deleted)} files deleted")

    touched_sensitive = [p for p in analysis.files_changed if any(marker in p.replace("\\", "/") for marker in _SENSITIVE_PATH_MARKERS)]
    if touched_sensitive:
        analysis.diff_suspicious = True
        analysis.diff_suspicious_reasons.append(f"touches sensitive path(s): {', '.join(touched_sensitive)}")

    return analysis
