import unittest

from brain.learning_evaluation import BenchmarkMetrics, FakeBenchmark, PromotionGateConfig, evaluate_candidate


class EvaluateCandidateTests(unittest.TestCase):
    def test_better_candidate_is_promoted(self):
        benchmark = FakeBenchmark(scores={
            "old": BenchmarkMetrics(solve_rate=0.5, regression_rate=0.0, behavioral_acceptance_rate=0.6),
            "new": BenchmarkMetrics(solve_rate=0.7, regression_rate=0.0, behavioral_acceptance_rate=0.7),
        })
        outcome = evaluate_candidate(old_model_version="old", new_model_version="new", benchmark=benchmark)
        self.assertTrue(outcome.promote)
        self.assertTrue(all(outcome.gates.values()))

    def test_worse_candidate_is_not_promoted(self):
        benchmark = FakeBenchmark(scores={
            "old": BenchmarkMetrics(solve_rate=0.6, regression_rate=0.0, behavioral_acceptance_rate=0.6),
            "new": BenchmarkMetrics(solve_rate=0.4, regression_rate=0.0, behavioral_acceptance_rate=0.6),
        })
        outcome = evaluate_candidate(old_model_version="old", new_model_version="new", benchmark=benchmark)
        self.assertFalse(outcome.promote)
        self.assertFalse(outcome.gates["solve_rate_improved"])

    def test_candidate_with_new_regressions_is_rejected_even_if_solve_rate_improved(self):
        benchmark = FakeBenchmark(scores={
            "old": BenchmarkMetrics(solve_rate=0.5, regression_rate=0.0, behavioral_acceptance_rate=0.6),
            "new": BenchmarkMetrics(solve_rate=0.9, regression_rate=0.1, behavioral_acceptance_rate=0.9),
        })
        outcome = evaluate_candidate(old_model_version="old", new_model_version="new", benchmark=benchmark)
        self.assertFalse(outcome.promote)
        self.assertFalse(outcome.gates["no_new_regressions"])

    def test_marginal_improvement_below_threshold_is_not_promoted(self):
        benchmark = FakeBenchmark(scores={
            "old": BenchmarkMetrics(solve_rate=0.50, regression_rate=0.0, behavioral_acceptance_rate=0.6),
            "new": BenchmarkMetrics(solve_rate=0.505, regression_rate=0.0, behavioral_acceptance_rate=0.6),
        })
        outcome = evaluate_candidate(old_model_version="old", new_model_version="new", benchmark=benchmark, gate_config=PromotionGateConfig(min_solve_rate_improvement=0.02))
        self.assertFalse(outcome.promote)

    def test_no_prior_active_model_uses_zero_baseline(self):
        benchmark = FakeBenchmark(scores={"new": BenchmarkMetrics(solve_rate=0.3, regression_rate=0.0, behavioral_acceptance_rate=0.5)})
        outcome = evaluate_candidate(old_model_version=None, new_model_version="new", benchmark=benchmark)
        self.assertEqual(outcome.old_metrics, BenchmarkMetrics())
        self.assertTrue(outcome.promote)

    def test_training_metrics_are_never_consulted_only_fresh_benchmark_calls(self):
        benchmark = FakeBenchmark(scores={
            "old": BenchmarkMetrics(solve_rate=0.5, behavioral_acceptance_rate=0.6),
            "new": BenchmarkMetrics(solve_rate=0.9, behavioral_acceptance_rate=0.9),
        })
        evaluate_candidate(old_model_version="old", new_model_version="new", benchmark=benchmark)
        self.assertEqual(sorted(benchmark.calls), ["new", "old"])

    def test_low_behavioral_acceptance_blocks_promotion(self):
        benchmark = FakeBenchmark(scores={
            "old": BenchmarkMetrics(solve_rate=0.3, behavioral_acceptance_rate=0.3),
            "new": BenchmarkMetrics(solve_rate=0.9, behavioral_acceptance_rate=0.2),
        })
        outcome = evaluate_candidate(old_model_version="old", new_model_version="new", benchmark=benchmark)
        self.assertFalse(outcome.promote)
        self.assertFalse(outcome.gates["behavioral_acceptance_sufficient"])


if __name__ == "__main__":
    unittest.main()
