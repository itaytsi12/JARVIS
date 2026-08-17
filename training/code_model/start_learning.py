"""CLI: `python -m training.code_model.start_learning --repository-root <path>`

Drives the exact same pipeline as the voice command "Hey Jarvis, start
learning" (`voice/background_assistant.py::_start_learning_task`, via the
same `training/code_model/production.py` wiring) -- for debugging without a
microphone (Phase 25).
"""
from __future__ import annotations

import argparse
import json

from brain.improvement_coding_agent import ClaudeCodeAdapter
from brain.learning_orchestrator import start_learning
from brain.learning_training import TrainingPolicy
from training.code_model.production import build_production_backend, build_production_benchmark, build_training_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug/CLI equivalent of 'Hey Jarvis, start learning' -- no microphone needed.")
    parser.add_argument("--repository-root", required=True, help="repository root the teacher / variation-generation coding agent operates against")
    parser.add_argument("--config", default=None, help="override JARVIS_CODE_MODEL_CONFIG for this run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backend = build_production_backend(args.config)
    benchmark = build_production_benchmark(backend.code_model_config)

    def report(status: str, detail: str) -> None:
        print(f"[{status}] {detail}")

    summary = start_learning(
        coding_agent=ClaudeCodeAdapter(),
        repository_root=args.repository_root,
        backend=backend,
        benchmark=benchmark,
        training_config=build_training_config(backend.code_model_config),
        policy=TrainingPolicy(mode="manual_only"),
        explicit_command=True,
        progress_callback=report,
    )
    print(json.dumps(summary.to_dict(), indent=2, default=str))
    return 0 if summary.status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
