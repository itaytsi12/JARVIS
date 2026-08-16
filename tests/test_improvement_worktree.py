import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

from brain.improvement_worktree import (
    WorktreeBlocked,
    cleanup_worktree,
    create_attempt_worktree,
    current_base_commit,
    current_branch,
    discover_repository_root,
    is_dirty,
    list_worktrees,
    owns_worktree,
    validate_worktree,
)


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


class ImprovementWorktreeInspectionTests(unittest.TestCase):
    """Read-only helpers must never modify the repository they inspect."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = _init_repo(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_discover_repository_root_from_nested_path(self):
        nested = self.repo / "nested" / "dir"
        nested.mkdir(parents=True)
        self.assertEqual(Path(discover_repository_root(nested)).resolve(), self.repo.resolve())

    def test_discover_repository_root_rejects_non_repo(self):
        outside = Path(self.temp.name) / "not_a_repo"
        outside.mkdir()
        with self.assertRaises(WorktreeBlocked):
            discover_repository_root(outside)

    def test_current_base_commit_matches_head(self):
        expected = _git(self.repo, "rev-parse", "HEAD").stdout.strip()
        self.assertEqual(current_base_commit(self.repo), expected)

    def test_current_branch_matches_git(self):
        expected = _git(self.repo, "branch", "--show-current").stdout.strip()
        self.assertEqual(current_branch(self.repo), expected)

    def test_is_dirty_false_on_clean_repo(self):
        self.assertFalse(is_dirty(self.repo))

    def test_is_dirty_true_with_untracked_file_and_leaves_it_untouched(self):
        (self.repo / "scratch.txt").write_text("untracked")
        self.assertTrue(is_dirty(self.repo))
        # Read-only detection must never stash/clean/reset the file away.
        self.assertTrue((self.repo / "scratch.txt").exists())
        self.assertEqual((self.repo / "scratch.txt").read_text(), "untracked")


class CreateAttemptWorktreeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = _init_repo(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_creates_worktree_at_unique_path_with_expected_branch_naming(self):
        handle = create_attempt_worktree(self.repo, "candidate-aaaa1111", "attempt-bbbb2222")
        self.assertTrue(Path(handle.worktree_path).is_dir())
        self.assertTrue(handle.worktree_branch.startswith("jarvis/improvement/"))
        self.assertEqual(handle.worktree_branch.count("/"), 3)  # jarvis/improvement/<candidate>/<attempt>

    def test_base_commit_matches_repository_head_at_creation_time(self):
        head = current_base_commit(self.repo)
        handle = create_attempt_worktree(self.repo, "cand-1", "att-1")
        self.assertEqual(handle.base_commit, head)
        worktree_head = _git(Path(handle.worktree_path), "rev-parse", "HEAD").stdout.strip()
        self.assertEqual(worktree_head, head)

    def test_two_attempts_for_same_candidate_get_unique_paths_and_branches(self):
        first = create_attempt_worktree(self.repo, "cand-shared", "att-one")
        second = create_attempt_worktree(self.repo, "cand-shared", "att-two")
        self.assertNotEqual(first.worktree_path, second.worktree_path)
        self.assertNotEqual(first.worktree_branch, second.worktree_branch)
        self.assertTrue(Path(first.worktree_path).is_dir())
        self.assertTrue(Path(second.worktree_path).is_dir())

    def test_dirty_base_tree_flag_reflects_uncommitted_changes_and_leaves_them_untouched(self):
        (self.repo / "dirty.txt").write_text("uncommitted")
        handle = create_attempt_worktree(self.repo, "cand-dirty", "att-dirty")
        self.assertTrue(handle.dirty_base_tree)
        # The user's active tree must be completely unaffected by worktree creation.
        self.assertTrue((self.repo / "dirty.txt").exists())
        self.assertEqual((self.repo / "dirty.txt").read_text(), "uncommitted")
        status = _git(self.repo, "status", "--porcelain").stdout
        self.assertIn("dirty.txt", status)

    def test_clean_base_tree_flag_false(self):
        handle = create_attempt_worktree(self.repo, "cand-clean", "att-clean")
        self.assertFalse(handle.dirty_base_tree)

    def test_repository_root_must_be_the_true_top_level(self):
        subdir = self.repo / "nested"
        subdir.mkdir()
        with self.assertRaises(WorktreeBlocked):
            create_attempt_worktree(subdir, "cand-x", "att-x")

    def test_invalid_repository_raises_worktree_blocked(self):
        not_a_repo = self.root / "not_a_repo"
        not_a_repo.mkdir()
        with self.assertRaises(WorktreeBlocked):
            create_attempt_worktree(not_a_repo, "cand-x", "att-x")

    def test_nonexistent_repository_root_raises_worktree_blocked(self):
        with self.assertRaises(WorktreeBlocked):
            create_attempt_worktree(self.root / "does_not_exist", "cand-x", "att-x")

    def test_destination_collision_raises_worktree_blocked(self):
        handle = create_attempt_worktree(self.repo, "cand-collide", "att-collide")
        with self.assertRaises(WorktreeBlocked):
            create_attempt_worktree(self.repo, "cand-collide", "att-collide")
        # The first, successful worktree must remain intact after the collision.
        self.assertTrue(Path(handle.worktree_path).is_dir())

    def test_branch_collision_raises_worktree_blocked_even_after_path_is_freed(self):
        handle = create_attempt_worktree(self.repo, "cand-branch", "att-branch")
        # Free up the destination path without removing the branch, to isolate
        # the branch-collision check from the path-collision check.
        _git(self.repo, "worktree", "remove", "--force", handle.worktree_path)
        with self.assertRaises(WorktreeBlocked):
            create_attempt_worktree(self.repo, "cand-branch", "att-branch")

    def test_worktree_creation_does_not_change_active_branch_of_main_repo(self):
        before = current_branch(self.repo)
        create_attempt_worktree(self.repo, "cand-branchcheck", "att-branchcheck")
        self.assertEqual(current_branch(self.repo), before)

    def test_worktree_creation_never_commits_pushes_merges_or_resets_main_tree(self):
        # The default worktrees_root lives under the repo as an untracked
        # directory (`.jarvis-improvement-worktrees/`), so it is expected to
        # show up as untracked in `git status`. What must NEVER happen is any
        # change to a *tracked* file, or any new commit -- that's the actual
        # "don't touch the user's active tree" guarantee.
        before_log = _git(self.repo, "log", "--oneline").stdout
        before_diff = _git(self.repo, "diff", "HEAD").stdout
        create_attempt_worktree(self.repo, "cand-safety", "att-safety")
        self.assertEqual(_git(self.repo, "log", "--oneline").stdout, before_log)
        self.assertEqual(_git(self.repo, "diff", "HEAD").stdout, before_diff)
        untracked = _git(self.repo, "status", "--porcelain").stdout
        self.assertTrue(all(".jarvis-improvement-worktrees" in line for line in untracked.splitlines() if line.strip()))

    def test_successful_worktree_remains_on_disk_without_explicit_cleanup(self):
        handle = create_attempt_worktree(self.repo, "cand-keep", "att-keep")
        self.assertTrue(Path(handle.worktree_path).is_dir())
        entries = list_worktrees(self.repo)
        self.assertTrue(any(Path(e.get("worktree", "")).resolve() == Path(handle.worktree_path).resolve() for e in entries))

    def test_concurrent_worktree_creation_for_distinct_attempts_all_succeed_uniquely(self):
        results: list = []
        errors: list = []
        lock = threading.Lock()

        def worker(index: int):
            try:
                handle = create_attempt_worktree(self.repo, "cand-concurrent", f"att-{index}")
                with lock:
                    results.append(handle)
            except Exception as exc:  # pragma: no cover - failure path surfaced via assertion
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 5)
        paths = {Path(h.worktree_path).resolve() for h in results}
        branches = {h.worktree_branch for h in results}
        self.assertEqual(len(paths), 5)
        self.assertEqual(len(branches), 5)


class WorktreeOwnershipAndValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = _init_repo(self.root)
        self.handle = create_attempt_worktree(self.repo, "cand-own", "att-own")

    def tearDown(self):
        self.temp.cleanup()

    def test_owns_worktree_true_for_creating_attempt(self):
        self.assertTrue(owns_worktree("att-own", self.handle.worktree_path))

    def test_owns_worktree_false_for_unrelated_attempt(self):
        self.assertFalse(owns_worktree("some-other-attempt", self.handle.worktree_path))

    def test_validate_worktree_true_for_genuine_worktree(self):
        self.assertTrue(validate_worktree(self.handle))

    def test_validate_worktree_false_for_nonexistent_path(self):
        from dataclasses import replace
        tampered = replace(self.handle, worktree_path=str(self.root / "does_not_exist"))
        self.assertFalse(validate_worktree(tampered))

    def test_validate_worktree_false_for_unrelated_directory(self):
        from dataclasses import replace
        unrelated = self.root / "unrelated"
        unrelated.mkdir()
        tampered = replace(self.handle, worktree_path=str(unrelated))
        self.assertFalse(validate_worktree(tampered))


class CleanupWorktreeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = _init_repo(self.root)
        self.handle = create_attempt_worktree(self.repo, "cand-cleanup", "att-cleanup")

    def tearDown(self):
        self.temp.cleanup()

    def test_cleanup_refuses_for_non_owning_attempt(self):
        with self.assertRaises(WorktreeBlocked):
            cleanup_worktree(self.handle, attempt_id="not-the-owner")
        # Refusal must leave the worktree completely intact.
        self.assertTrue(Path(self.handle.worktree_path).is_dir())
        self.assertTrue(owns_worktree("att-cleanup", self.handle.worktree_path))

    def test_cleanup_removes_worktree_directory_and_branch(self):
        result = cleanup_worktree(self.handle, attempt_id="att-cleanup")
        self.assertTrue(result)
        self.assertFalse(Path(self.handle.worktree_path).exists())
        branches = _git(self.repo, "branch", "--list", self.handle.worktree_branch).stdout
        self.assertEqual(branches.strip(), "")

    def test_cleanup_clears_ownership_so_stale_handle_cannot_be_reused(self):
        cleanup_worktree(self.handle, attempt_id="att-cleanup")
        self.assertFalse(owns_worktree("att-cleanup", self.handle.worktree_path))

    def test_cleanup_never_touches_main_repository_working_tree(self):
        (self.repo / "a.py").write_text("A = 1  # untouched\n")
        before = (self.repo / "a.py").read_text()
        cleanup_worktree(self.handle, attempt_id="att-cleanup")
        self.assertEqual((self.repo / "a.py").read_text(), before)


if __name__ == "__main__":
    unittest.main()
