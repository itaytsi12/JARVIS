"""Held-out benchmark evaluation and promotion gates (Phases 15, 16).

Measures the OLD ACTIVE model against the NEW CANDIDATE model on a fixed
benchmark protocol; the candidate is promoted ONLY if every gate in
`PromotionGateConfig` is satisfied. Training loss/metrics returned by
`brain/learning_training.py`'s `TrainingRunResult` are never consulted here
-- `evaluate_candidate` only trusts a fresh `Benchmark.evaluate()` call for
both models, exactly so "training loss alone means nothing" (Phase 15) is
structurally true, not just documented.

`Benchmark` is a small protocol -- a real held-out coding benchmark plugs in
later without touching `brain/learning_orchestrator.py`. `FakeBenchmark` is
the deterministic double used by tests and the bounded dry run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class BenchmarkMetrics:
    solve_rate: float = 0.0
    bug_localization_rate: float = 0.0
    logical_bug_repair_rate: float = 0.0
    multi_file_repair_rate: float = 0.0
    regression_rate: float = 0.0
    focused_test_success_rate: float = 0.0
    full_suite_success_rate: float = 0.0
    behavioral_acceptance_rate: float = 0.0
    average_iterations: float = 0.0
    tool_usage_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Benchmark(Protocol):
    def evaluate(self, model_version: str) -> BenchmarkMetrics: ...


class FakeBenchmark:
    """Deterministic double: returns a canned `BenchmarkMetrics` per model
    version (or `default` for anything unlisted). No real model inference,
    no GPU, no external benchmark suite."""

    def __init__(self, scores: dict[str, BenchmarkMetrics] | None = None, default: BenchmarkMetrics | None = None):
        self.scores = scores or {}
        self.default = default if default is not None else BenchmarkMetrics()
        self.calls: list[str] = []

    def evaluate(self, model_version: str) -> BenchmarkMetrics:
        self.calls.append(model_version)
        return self.scores.get(model_version, self.default)


@dataclass
class PromotionGateConfig:
    min_solve_rate_improvement: float = 0.02
    max_regression_rate_increase: float = 0.0
    min_behavioral_acceptance_rate: float = 0.5


@dataclass
class EvaluationOutcome:
    old_model_version: str | None
    new_model_version: str
    old_metrics: BenchmarkMetrics
    new_metrics: BenchmarkMetrics
    gates: dict[str, bool] = field(default_factory=dict)
    promote: bool = False
    reason: str = ""


def evaluate_candidate(
    *,
    old_model_version: str | None,
    new_model_version: str,
    benchmark: Benchmark,
    gate_config: PromotionGateConfig | None = None,
) -> EvaluationOutcome:
    """Pure aside from the two `benchmark.evaluate()` calls. When there is
    no prior active model (the very first model version ever produced), the
    baseline is a zero `BenchmarkMetrics()` -- the candidate still has to
    clear every gate on its own merits, not merely "exist"."""
    gate_config = gate_config or PromotionGateConfig()
    new_metrics = benchmark.evaluate(new_model_version)
    old_metrics = benchmark.evaluate(old_model_version) if old_model_version else BenchmarkMetrics()

    gates = {
        "solve_rate_improved": (new_metrics.solve_rate - old_metrics.solve_rate) >= gate_config.min_solve_rate_improvement,
        "no_new_regressions": (new_metrics.regression_rate - old_metrics.regression_rate) <= gate_config.max_regression_rate_increase,
        "behavioral_acceptance_sufficient": new_metrics.behavioral_acceptance_rate >= gate_config.min_behavioral_acceptance_rate,
    }
    promote = all(gates.values())
    if promote:
        reason = "candidate model satisfies every promotion gate against the current active model"
    else:
        missing = [name for name, ok in gates.items() if not ok]
        reason = f"candidate did not satisfy: {', '.join(missing)}"

    return EvaluationOutcome(
        old_model_version=old_model_version, new_model_version=new_model_version,
        old_metrics=old_metrics, new_metrics=new_metrics, gates=gates, promote=promote, reason=reason,
    )
