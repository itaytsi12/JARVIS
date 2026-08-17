"""CLI: `python -m training.code_model.train --config <name> --dataset <jsonl_path>`

Runs `training/code_model/hf_backend.py`'s real backend directly -- no
voice involvement -- for manual/CLI-driven training runs and debugging
without a microphone (Phase 25).
"""
from __future__ import annotations

import argparse
import json
import sys

from brain.learning_training import TrainingConfig
from training.code_model.config import list_configs, load_config
from training.code_model.hf_backend import HuggingFaceLoRATrainingBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train JARVIS's local coding student model with real LoRA/QLoRA.")
    parser.add_argument("--config", required=True, help=f"config name ({', '.join(list_configs())}) or a path to a YAML file")
    parser.add_argument("--dataset", required=True, help="path to a dataset JSONL file (brain.learning_dataset.DatasetManifest.jsonl_path)")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume-from", default=None, help="a previous run's checkpoint directory to resume from")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--base-model", default=None, help="override the config's base_model.model_id for this run")
    parser.add_argument("--force", action="store_true", help="run even if the hardware feasibility check reports LOCAL_TRAINING_NOT_FEASIBLE")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    code_model_config = load_config(args.config)
    backend = HuggingFaceLoRATrainingBackend(code_model_config)
    available, reason = backend.is_available()
    if not available and not args.force:
        print(f"NOT FEASIBLE: {reason}", file=sys.stderr)
        print("Pass --force to attempt it anyway, or use a smaller config / configure a cloud backend.", file=sys.stderr)
        return 1

    training_config = TrainingConfig(
        base_model=args.base_model or "not-configured",
        checkpoint_dir=args.checkpoint_dir or code_model_config.training.output_dir,
        resume_from=args.resume_from,
        max_steps=args.max_steps,
    )
    result = backend.run(args.dataset, training_config)
    print(json.dumps({
        "exit_status": result.exit_status, "training_run_id": result.training_run_id,
        "model_version": result.model_version, "checkpoint_path": result.checkpoint_path,
        "metrics": result.metrics, "error": result.error, "resumed_from": result.resumed_from,
    }, indent=2))
    return 0 if result.exit_status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
