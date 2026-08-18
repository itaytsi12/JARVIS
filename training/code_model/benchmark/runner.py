"""Real coding-benchmark execution (Phase 14-16).

Implements `brain.learning_evaluation.Benchmark` for real -- this is the
module that replaces `FakeBenchmark` in the production "start learning"
path. For every task: create a fresh isolated workspace from the fixture,
hand the task to a real `CodingAgent`, run the independent hidden
acceptance test AND the visible regression tests, inspect the actual diff,
record a real result, then destroy the disposable workspace. Reuses
`brain.improvement_diff_analysis.analyze_diff` and
`brain.task_supervisor.SafeCommandRunner` -- the SAME diff-analysis and
sandboxed-subprocess machinery the self-improvement pipeline already uses
-- no second implementation of either.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from brain.improvement_coding_agent import CodingAgent, CodingAgentConstraints
from brain.improvement_diff_analysis import analyze_diff
from brain.learning_evaluation import BenchmarkMetrics
from brain.task_supervisor import SafeCommandRunner
from training.code_model.benchmark.schema import BenchmarkTask, load_tasks_from_directory

DEFAULT_FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True)


def _force_writable_and_retry(func, path, exc_info) -> None:
    """`shutil.rmtree` error handler: git marks every object file read-only
    on Windows (mode 0o444), which the default handler can't delete. Clear
    the read-only bit and retry the failing operation once."""
    import os
    import stat

    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def _cleanup_workspace(workspace: Path) -> None:
    """Best-effort, but with real handling for the one specific, recurring
    Windows failure mode this actually hits: git object files created
    read-only, which plain `shutil.rmtree` cannot remove. A couple of short
    retries additionally covers a just-exited pytest subprocess transiently
    holding a handle open. If it still fails after that, this stays
    best-effort exactly like the rest of this codebase's disposable-
    worktree cleanup (`brain/improvement_orchestrator.py::_maybe_cleanup`)
    -- never worth failing the benchmark run over."""
    for attempt in range(3):
        try:
            shutil.rmtree(workspace, onexc=_force_writable_and_retry)
        except TypeError:
            # Python < 3.12 uses the older `onerror` callback signature.
            shutil.rmtree(workspace, onerror=_force_writable_and_retry)
        except Exception:
            pass
        if not workspace.exists():
            return
        time.sleep(0.2 * (attempt + 1))


@dataclass
class TaskResult:
    task_id: str
    category: str
    solved: bool
    patch_applied: bool
    acceptance_passed: bool | None
    regression_passed: bool | None
    iterations: int
    runtime_seconds: float
    tool_calls: int
    agent_exit_status: str
    error: str | None = None
    test_added: bool | None = None
    structural_change_detected: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkRunResult:
    model_version: str
    task_results: list[TaskResult] = field(default_factory=list)
    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)

    def to_dict(self) -> dict:
        return {"model_version": self.model_version, "task_results": [r.to_dict() for r in self.task_results], "metrics": self.metrics.to_dict()}


def _run_task(task: BenchmarkTask, agent: CodingAgent, fixtures_root: Path, *, safe_runner: SafeCommandRunner | None = None) -> TaskResult:
    """Never raises for an ordinary task failure -- a crash, a timeout, an
    agent that produces no patch are all honest `TaskResult`s, not
    exceptions."""
    safe_runner = safe_runner or SafeCommandRunner()
    started = time.perf_counter()
    workspace = Path(tempfile.mkdtemp(prefix=f"jarvis-benchmark-{task.task_id}-"))
    try:
        repo_source = fixtures_root / task.fixture_dir / "repo"
        if not repo_source.is_dir():
            return TaskResult(task.task_id, task.category, False, False, None, None, 0, time.perf_counter() - started, 0, "crashed", f"fixture repo not found: {repo_source}")
        shutil.copytree(repo_source, workspace, dirs_exist_ok=True)

        init = _git(workspace, "init", "-q")
        if init.returncode != 0:
            return TaskResult(task.task_id, task.category, False, False, None, None, 0, time.perf_counter() - started, 0, "crashed", f"git init failed: {init.stderr}")
        _git(workspace, "config", "user.email", "benchmark@example.invalid")
        _git(workspace, "config", "user.name", "Benchmark")
        _git(workspace, "add", "-A")
        _git(workspace, "commit", "-qm", "initial fixture state")
        base_commit = _git(workspace, "rev-parse", "HEAD").stdout.strip()

        original_line_count = None
        if task.structural_check_path:
            check_file = workspace / task.structural_check_path
            if check_file.exists():
                original_line_count = len([ln for ln in check_file.read_text(encoding="utf-8").splitlines() if ln.strip()])

        agent_result = agent.run(task.description, CodingAgentConstraints(workspace=str(workspace), timeout_seconds=task.timeout_seconds))
        tool_calls = getattr(agent_result, "model_calls", 0) or 0

        diff = analyze_diff(str(workspace), base_commit)
        patch_applied = diff.change_scope != "none"

        test_added: bool | None = None
        if task.require_new_test:
            test_added = bool(diff.generated_tests)

        structural_change_detected: bool | None = None
        if task.structural_check_path and original_line_count is not None:
            check_file = workspace / task.structural_check_path
            new_line_count = len([ln for ln in check_file.read_text(encoding="utf-8").splitlines() if ln.strip()]) if check_file.exists() else 0
            structural_change_detected = (original_line_count - new_line_count) >= task.min_line_reduction

        acceptance_passed: bool | None = None
        hidden_source = fixtures_root / task.fixture_dir / "harness" / task.hidden_test_path
        if hidden_source.exists():
            hidden_dest = workspace / "_benchmark_hidden_test.py"
            hidden_dest.write_text(hidden_source.read_text(encoding="utf-8"), encoding="utf-8")
            try:
                acceptance_result = safe_runner.run(["python", "-m", "pytest", "_benchmark_hidden_test.py", "-q"], str(workspace), timeout=task.timeout_seconds)
                acceptance_passed = acceptance_result["exit_code"] == 0
            except subprocess.TimeoutExpired:
                acceptance_passed = False
            except Exception:
                acceptance_passed = False

        regression_passed: bool | None = None
        if task.visible_test_paths:
            try:
                regression_result = safe_runner.run(["python", "-m", "pytest", *task.visible_test_paths, "-q"], str(workspace), timeout=task.timeout_seconds)
                regression_passed = regression_result["exit_code"] == 0
            except Exception:
                regression_passed = False

        solved = bool(
            patch_applied and acceptance_passed is True and regression_passed is not False
            and test_added is not False and structural_change_detected is not False
        )
        return TaskResult(
            task.task_id, task.category, solved, patch_applied, acceptance_passed, regression_passed,
            iterations=1, runtime_seconds=time.perf_counter() - started, tool_calls=tool_calls,
            agent_exit_status=agent_result.exit_status,
            error=agent_result.error if agent_result.exit_status != "completed" else None,
            test_added=test_added, structural_change_detected=structural_change_detected,
        )
    except Exception as exc:
        return TaskResult(task.task_id, task.category, False, False, None, None, 0, time.perf_counter() - started, 0, "crashed", f"{type(exc).__name__}: {exc}")
    finally:
        _cleanup_workspace(workspace)


def _category_solve_rate(results: list[TaskResult], categories: set[str]) -> float:
    subset = [r for r in results if r.category in categories]
    if not subset:
        return 0.0
    return sum(1 for r in subset if r.solved) / len(subset)


def aggregate_metrics(results: list[TaskResult]) -> BenchmarkMetrics:
    n = len(results) or 1
    solved = sum(1 for r in results if r.solved)
    patch_applied = sum(1 for r in results if r.patch_applied)
    acceptance_known = [r for r in results if r.acceptance_passed is not None]
    regression_known = [r for r in results if r.regression_passed is not None]
    timeouts_and_crashes = sum(1 for r in results if r.agent_exit_status in ("timeout", "crashed"))
    return BenchmarkMetrics(
        solve_rate=solved / n,
        bug_localization_rate=patch_applied / n,  # proxy: did the agent touch the repo at all -- a real,
        # cheap, structural signal (same "change_scope != none" gate used throughout brain/improvement_evaluator.py),
        # not a claim of exact line-level fault localization, which this harness doesn't attempt to measure.
        logical_bug_repair_rate=_category_solve_rate(results, {"logical_bug"}),
        multi_file_repair_rate=_category_solve_rate(results, {"cross_file_bug", "multi_class_bug"}),
        regression_rate=(1 - sum(1 for r in regression_known if r.regression_passed) / len(regression_known)) if regression_known else 0.0,
        focused_test_success_rate=(sum(1 for r in acceptance_known if r.acceptance_passed) / len(acceptance_known)) if acceptance_known else 0.0,
        full_suite_success_rate=(sum(1 for r in regression_known if r.regression_passed) / len(regression_known)) if regression_known else 0.0,
        behavioral_acceptance_rate=solved / n,
        average_iterations=sum(r.iterations for r in results) / n,
        tool_usage_rate=sum(r.tool_calls for r in results) / n,
    )


class RealCodingBenchmark:
    """Implements `brain.learning_evaluation.Benchmark`'s exact protocol
    (`evaluate(model_version) -> BenchmarkMetrics`) for real. `agent_factory`
    maps a `model_version` string to a real `CodingAgent` instance -- e.g.
    `training.code_model.student_adapter.LocalCodingModelAdapter.from_checkpoint`
    for the trained candidate, the same factory with `adapter_path=None`
    for the untrained base model (Phase 16's baseline comparison arm), or
    (for tests/smoke-runs only, per Phase 27) `FakeCodingAgent`."""

    def __init__(self, tasks: list[BenchmarkTask] | None = None, *, agent_factory: Callable[[str], CodingAgent], fixtures_root: str | Path | None = None):
        self.fixtures_root = Path(fixtures_root) if fixtures_root else DEFAULT_FIXTURES_ROOT
        self.tasks = tasks if tasks is not None else load_tasks_from_directory(self.fixtures_root)
        self.agent_factory = agent_factory

    def run(self, model_version: str) -> BenchmarkRunResult:
        agent = self.agent_factory(model_version)
        results = [_run_task(task, agent, self.fixtures_root) for task in self.tasks]
        return BenchmarkRunResult(model_version=model_version, task_results=results, metrics=aggregate_metrics(results))

    def evaluate(self, model_version: str) -> BenchmarkMetrics:
        return self.run(model_version).metrics
