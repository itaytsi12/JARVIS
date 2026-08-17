"""CLI: `python -m training.code_model.benchmark --model <model_version> [--config <name>]`

Runs the FULL benchmark (every fixture task) against one model version and
prints per-task results plus aggregate metrics (Phase 25).
"""
from __future__ import annotations

import argparse
import json

from training.code_model.config import load_config
from training.code_model.production import build_production_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the real coding benchmark against a model version.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="qlora_7b")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code_model_config = load_config(args.config)
    benchmark = build_production_benchmark(code_model_config)
    run_result = benchmark.run(args.model)
    print(json.dumps(run_result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
