"""CLI: `python -m training.code_model.evaluate --model <model_version> [--config <name>]`

Runs the real benchmark against ONE model version and prints its metrics,
without making any promotion decision -- that's `start_learning`'s job
(Phase 25).
"""
from __future__ import annotations

import argparse
import json

from training.code_model.config import load_config
from training.code_model.production import build_production_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained model version against the real held-out benchmark.")
    parser.add_argument("--model", required=True, help="model_version to evaluate (looked up in the model registry for its adapter path)")
    parser.add_argument("--config", default="qlora_7b")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code_model_config = load_config(args.config)
    benchmark = build_production_benchmark(code_model_config)
    metrics = benchmark.evaluate(args.model)
    print(json.dumps(metrics.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
