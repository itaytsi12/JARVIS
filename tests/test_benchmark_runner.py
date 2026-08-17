"""Real coding-benchmark harness tests (Phase 27, 29). Uses
`FakeCodingAgent` throughout, per Phase 27's explicit allowance ("run the
real benchmark harness against a deterministic fake coding agent... prove
acceptance-test execution and scoring are real. Do not fake benchmark
scores"). Every test here genuinely creates a fresh git repo per task,
genuinely runs pytest as a subprocess, and genuinely computes solve/fail
outcomes from that real execution -- nothing here injects a score.
"""
import unittest
from pathlib import Path

from brain.improvement_coding_agent import FakeCodingAgent
from training.code_model.benchmark.runner import DEFAULT_FIXTURES_ROOT, RealCodingBenchmark, aggregate_metrics
from training.code_model.benchmark.schema import BenchmarkTask, load_tasks_from_directory


def _correct_fix_agent() -> FakeCodingAgent:
    def apply(workspace: Path) -> None:
        if (workspace / "calc.py").exists():
            (workspace / "calc.py").write_text("def add(a, b):\n    return a + b\n")
        elif (workspace / "parity.py").exists():
            (workspace / "parity.py").write_text("def is_even(n):\n    return n % 2 == 0\n")
        elif (workspace / "orders.py").exists():
            (workspace / "orders.py").write_text(
                "from formatting import format_price\n\n\ndef order_summary(item_name, price_cents):\n"
                "    return f\"{item_name}: {format_price(price_cents)}\"\n"
            )
        elif (workspace / "inventory.py").exists():
            (workspace / "inventory.py").write_text(
                "def total_value(items):\n    return sum(item[\"price\"] * item[\"qty\"] for item in items)\n\n\n"
                "def apply_discount(items, percent):\n    for item in items:\n        item[\"price\"] = item[\"price\"] * (1 - percent / 100)\n    return items\n"
            )
        elif (workspace / "strings_util.py").exists():
            (workspace / "strings_util.py").write_text('def reverse_words(sentence):\n    return " ".join(reversed(sentence.split()))\n')
    return FakeCodingAgent(apply=apply)


class FixtureLoadingTests(unittest.TestCase):
    def test_all_shipped_fixtures_load(self):
        tasks = load_tasks_from_directory(DEFAULT_FIXTURES_ROOT)
        self.assertEqual(len(tasks), 5)
        ids = {t.task_id for t in tasks}
        self.assertEqual(ids, {
            "syntax_runtime_bug_off_by_sign", "logical_bug_parity", "cross_file_bug_price_formatting",
            "regression_bug_discount", "feature_implementation_reverse_words",
        })

    def test_categories_span_multiple_distinct_categories(self):
        tasks = load_tasks_from_directory(DEFAULT_FIXTURES_ROOT)
        categories = {t.category for t in tasks}
        self.assertGreaterEqual(len(categories), 5)

    def test_every_task_has_a_hidden_test_file_not_in_the_visible_repo(self):
        tasks = load_tasks_from_directory(DEFAULT_FIXTURES_ROOT)
        for task in tasks:
            with self.subTest(task_id=task.task_id):
                hidden_path = DEFAULT_FIXTURES_ROOT / task.fixture_dir / "harness" / task.hidden_test_path
                visible_path = DEFAULT_FIXTURES_ROOT / task.fixture_dir / "repo" / task.hidden_test_path
                self.assertTrue(hidden_path.exists())
                self.assertFalse(visible_path.exists(), "hidden test must not also be part of the given repo state")

    def test_missing_fixtures_directory_returns_empty_list(self):
        self.assertEqual(load_tasks_from_directory("/does/not/exist"), [])


class RealBenchmarkExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tasks = load_tasks_from_directory(DEFAULT_FIXTURES_ROOT)

    def test_correct_fix_is_genuinely_solved(self):
        benchmark = RealCodingBenchmark(self.tasks, agent_factory=lambda v: _correct_fix_agent())
        run_result = benchmark.run("correct")
        for result in run_result.task_results:
            with self.subTest(task_id=result.task_id):
                self.assertTrue(result.solved, result.error)
                self.assertTrue(result.acceptance_passed)
        self.assertEqual(run_result.metrics.solve_rate, 1.0)

    def test_no_change_agent_is_genuinely_unsolved(self):
        benchmark = RealCodingBenchmark(self.tasks, agent_factory=lambda v: FakeCodingAgent())
        run_result = benchmark.run("no-op")
        for result in run_result.task_results:
            with self.subTest(task_id=result.task_id):
                self.assertFalse(result.solved)
                self.assertFalse(result.patch_applied)
        self.assertEqual(run_result.metrics.solve_rate, 0.0)

    def test_partial_patch_that_fails_hidden_test_is_not_solved(self):
        def wrong_fix(workspace: Path):
            if (workspace / "calc.py").exists():
                (workspace / "calc.py").write_text("def add(a, b):\n    return a * b\n")  # still wrong

        single_task = [t for t in self.tasks if t.task_id == "syntax_runtime_bug_off_by_sign"]
        benchmark = RealCodingBenchmark(single_task, agent_factory=lambda v: FakeCodingAgent(apply=wrong_fix))
        run_result = benchmark.run("wrong-fix")
        self.assertFalse(run_result.task_results[0].solved)
        self.assertTrue(run_result.task_results[0].patch_applied)
        self.assertFalse(run_result.task_results[0].acceptance_passed)

    def test_regression_breaking_fix_is_detected(self):
        def break_regression(workspace: Path):
            if (workspace / "inventory.py").exists():
                (workspace / "inventory.py").write_text(
                    "def total_value(items):\n    raise RuntimeError('broken')\n\n\n"
                    "def apply_discount(items, percent):\n    for item in items:\n        item[\"price\"] *= (1 - percent / 100)\n    return items\n"
                )
        single_task = [t for t in self.tasks if t.task_id == "regression_bug_discount"]
        benchmark = RealCodingBenchmark(single_task, agent_factory=lambda v: FakeCodingAgent(apply=break_regression))
        run_result = benchmark.run("regressing-fix")
        result = run_result.task_results[0]
        self.assertFalse(result.regression_passed)
        self.assertFalse(result.solved)  # a regressing fix is never "solved" even if the new feature works

    def test_crashing_agent_is_recorded_not_raised(self):
        crashing_agent = FakeCodingAgent(crash=True)
        single_task = self.tasks[:1]
        benchmark = RealCodingBenchmark(single_task, agent_factory=lambda v: crashing_agent)
        run_result = benchmark.run("crashing")
        self.assertFalse(run_result.task_results[0].solved)
        self.assertEqual(run_result.task_results[0].agent_exit_status, "crashed")

    def test_each_task_gets_a_fresh_isolated_workspace(self):
        seen_workspaces = []

        def record_workspace(workspace: Path):
            seen_workspaces.append(str(workspace))

        benchmark = RealCodingBenchmark(self.tasks, agent_factory=lambda v: FakeCodingAgent(apply=record_workspace))
        benchmark.run("isolation-check")
        self.assertEqual(len(seen_workspaces), len(set(seen_workspaces)))  # every task got a distinct workspace
        for workspace in seen_workspaces:
            self.assertFalse(Path(workspace).exists())  # disposable workspace destroyed after the run

    def test_solution_files_are_never_present_in_the_agent_visible_repo(self):
        """The hidden test file must never leak into what the agent can see
        BEFORE it submits its patch (Phase 13's "solution must not be
        exposed during evaluation")."""
        captured = {}

        def snapshot(workspace: Path):
            captured["files"] = {p.name for p in workspace.rglob("*") if p.is_file()}

        single_task = [t for t in self.tasks if t.task_id == "syntax_runtime_bug_off_by_sign"]
        benchmark = RealCodingBenchmark(single_task, agent_factory=lambda v: FakeCodingAgent(apply=snapshot))
        benchmark.run("visibility-check")
        self.assertNotIn("hidden_test.py", captured["files"])
        self.assertNotIn("_benchmark_hidden_test.py", captured["files"])


class AggregateMetricsTests(unittest.TestCase):
    def test_empty_results_do_not_divide_by_zero(self):
        metrics = aggregate_metrics([])
        self.assertEqual(metrics.solve_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
