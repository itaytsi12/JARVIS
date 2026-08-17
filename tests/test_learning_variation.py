import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from brain.improvement_coding_agent import FakeCodingAgent
from brain.learning_package import LearningPackage
from brain.learning_variation import VariationConfig, generate_variants


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True)


def _init_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "value.py").write_text("VALUE = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "initial")
    return repo


def _package(**overrides) -> LearningPackage:
    defaults = dict(
        learning_job_id="job-1", improvement_attempt_id="att-1", problem_family="fam-1",
        original_task="fix the off-by-one bug", subsystem="filesystem", gap_type="EXECUTION_BUG",
        root_cause_category="EXECUTION_BUG_with_differential_test",
        reusable_strategy="strategy summary",
    )
    defaults.update(overrides)
    return LearningPackage(**defaults)


def _write_variant(workspace: Path, variant_id: str, description: str, seed: int = 0) -> None:
    base = workspace / ".jarvis-learning-variants" / variant_id
    (base / "before").mkdir(parents=True, exist_ok=True)
    (base / "after").mkdir(parents=True, exist_ok=True)
    (base / "manifest.json").write_text(json.dumps({"description": description, "test_file": "tests/test_x.py"}))
    (base / "before" / "tests").mkdir(exist_ok=True)
    (base / "before" / "src.py").write_text(f"def f():\n    return {seed}\n")
    (base / "before" / "tests" / "test_x.py").write_text(f"from src import f\ndef test_x():\n    assert f() == {seed + 1}\n")
    (base / "after" / "tests").mkdir(exist_ok=True)
    (base / "after" / "src.py").write_text(f"def f():\n    return {seed + 1}\n")
    (base / "after" / "tests" / "test_x.py").write_text(f"from src import f\ndef test_x():\n    assert f() == {seed + 1}\n")


class GenerateVariantsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = _init_repo(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_collects_variants_written_by_the_agent(self):
        def apply(workspace: Path):
            _write_variant(workspace, "v1", "first variant", seed=1)
            _write_variant(workspace, "v2", "second variant", seed=2)

        agent = FakeCodingAgent(apply=apply)
        variants = generate_variants(_package(), coding_agent=agent, repository_root=str(self.repo), config=VariationConfig(max_claude_calls=1, variants_per_call=5, max_total_variants=5))
        self.assertEqual(len(variants), 2)
        ids = {v.variant_id for v in variants}
        self.assertEqual(ids, {"v1", "v2"})
        self.assertTrue(all(v.quality_label == "SYNTHETIC_DERIVED" for v in variants))

    def test_variants_capture_different_code_structures_not_wording(self):
        def apply(workspace: Path):
            _write_variant(workspace, "v1", "off by one in loop bound")
        agent = FakeCodingAgent(apply=apply)
        variants = generate_variants(_package(), coding_agent=agent, repository_root=str(self.repo), config=VariationConfig(max_claude_calls=1, max_total_variants=1))
        self.assertEqual(len(variants), 1)
        self.assertIn("src.py", variants[0].before_files)
        self.assertIn("tests/test_x.py", variants[0].before_files)
        self.assertNotEqual(variants[0].before_files["src.py"], variants[0].after_files["src.py"])

    def test_duplicate_content_across_calls_is_deduplicated(self):
        call_count = {"n": 0}

        def apply(workspace: Path):
            call_count["n"] += 1
            _write_variant(workspace, f"v{call_count['n']}", "same content every time")

        agent = FakeCodingAgent(apply=apply)
        variants = generate_variants(_package(), coding_agent=agent, repository_root=str(self.repo), config=VariationConfig(max_claude_calls=3, variants_per_call=1, max_total_variants=3))
        # every call writes byte-identical before/after content under a new
        # id -- content-hash dedup must collapse them to one.
        self.assertEqual(len(variants), 1)

    def test_respects_max_claude_calls_cost_control(self):
        calls = []

        def apply(workspace: Path):
            calls.append(1)
            # never write anything -> forces the loop to keep calling until
            # max_claude_calls, since "no new variants" also breaks early;
            # write a uniquely-named empty dir with no manifest to avoid
            # tripping the early-stop-on-no-new-variants path differently
            # than the call-count path we're testing.
            pass

        agent = FakeCodingAgent(apply=apply)
        generate_variants(_package(), coding_agent=agent, repository_root=str(self.repo), config=VariationConfig(max_claude_calls=2, max_total_variants=5))
        self.assertEqual(len(calls), 1)  # first call yields zero variants -> stops immediately, never wastes a 2nd call

    def test_stops_once_max_total_variants_reached(self):
        def apply(workspace: Path):
            for i in range(10):
                _write_variant(workspace, f"v{i}", f"variant {i}", seed=i)

        agent = FakeCodingAgent(apply=apply)
        variants = generate_variants(_package(), coding_agent=agent, repository_root=str(self.repo), config=VariationConfig(max_claude_calls=5, variants_per_call=3, max_total_variants=4))
        self.assertLessEqual(len(variants), 4)

    def test_no_variants_written_returns_empty_list_without_raising(self):
        agent = FakeCodingAgent()  # no apply -> nothing written
        variants = generate_variants(_package(), coding_agent=agent, repository_root=str(self.repo), config=VariationConfig(max_claude_calls=1))
        self.assertEqual(variants, [])

    def test_worktree_is_cleaned_up_after_generation(self):
        def apply(workspace: Path):
            _write_variant(workspace, "v1", "variant")
        agent = FakeCodingAgent(apply=apply)
        generate_variants(_package(), coding_agent=agent, repository_root=str(self.repo), config=VariationConfig(max_claude_calls=1))
        worktrees_dir = self.repo / ".jarvis-improvement-worktrees"
        remaining = list(worktrees_dir.iterdir()) if worktrees_dir.exists() else []
        self.assertEqual(remaining, [])


if __name__ == "__main__":
    unittest.main()
