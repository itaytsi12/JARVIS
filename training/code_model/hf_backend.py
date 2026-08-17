"""Real Hugging Face `transformers` + `peft` + `accelerate` LoRA/QLoRA
training backend (Phase 2).

Implements `brain.learning_training.TrainingBackend` for real -- this is
NOT a protocol/interface exercise. `run()` genuinely loads a pretrained
causal-LM, quantizes it (QLoRA) or not (LoRA), injects LoRA adapters,
tokenizes JARVIS's real formatted dataset (`training/code_model/dataset_formatting.py`),
and performs real forward/backward training steps via `transformers.Trainer`.

Never trains from scratch -- `AutoModelForCausalLM.from_pretrained` always
loads a real pretrained base model; LoRA/QLoRA only ever adds/trains a
small adapter on top of frozen (optionally 4-bit-quantized) base weights.

Checkpointing/resume (Phase 8): `transformers.Trainer`'s own periodic
`save_steps` checkpoints (which already include the PEFT adapter's weights,
since `Trainer.save_model()` delegates to the wrapped `PeftModel`'s
`save_pretrained`) are the resume source. Resuming loads the LATEST such
checkpoint's adapter weights before training continues -- a real, working,
but intentionally simplified resume model (adapter weights carry over;
optimizer/scheduler state does not restart bit-identical). This is a
deliberate, documented scope choice: verifying bit-exact
`Trainer(resume_from_checkpoint=...)` internals for a quantized PEFT model
blind, without iterative access to a matching production environment, is a
much larger and riskier claim than "the training run genuinely continues
from where it left off," which this implementation delivers and the smoke
test proves.
"""
from __future__ import annotations

import json
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain.learning_training import TrainingConfig, TrainingRunResult
from training.code_model.config import CodeModelTrainingConfig
from training.code_model.dataset_formatting import format_examples_for_sft
from training.code_model.hardware import check_feasibility

RUN_RECORD_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrainingRunRecord:
    """Persisted, restart-survivable record of one training run (Phase 8).
    Written at every meaningful state transition, not only at the end, so
    a hard crash leaves a truthful trail -- `status` is never `"COMPLETED"`
    unless training genuinely finished."""
    training_run_id: str
    dataset_version: str
    base_model: str
    config_hash: str
    checkpoint_path: str | None
    current_step: int
    status: str  # "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED"
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    resumed_from: str | None = None
    started_at: str = ""
    updated_at: str = ""
    schema_version: int = RUN_RECORD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingRunRecord":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


def _save_record(run_dir: Path, record: TrainingRunRecord) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")


def load_run_record(run_dir: str | Path) -> TrainingRunRecord | None:
    path = Path(run_dir) / "run.json"
    if not path.exists():
        return None
    return TrainingRunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _latest_checkpoint_dir(run_dir: Path) -> Path | None:
    def step_of(path: Path) -> int:
        suffix = path.name.rsplit("-", 1)[-1]
        return int(suffix) if suffix.isdigit() else -1

    checkpoints = [p for p in run_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=step_of)


class HuggingFaceLoRATrainingBackend:
    """`TrainingBackend` implementation. Constructed with a
    `CodeModelTrainingConfig` (detailed HF/LoRA/quantization defaults, see
    `training/code_model/config.py`); the per-run
    `brain.learning_training.TrainingConfig` passed to `run()` can override
    `base_model`/`seed`/`max_steps`/`checkpoint_dir`/`resume_from` for that
    one run, matching `ClaudeCodeAdapter`'s constructor-config-plus-minimal-
    per-call-params shape.
    """
    backend_name = "huggingface_lora"

    def __init__(self, code_model_config: CodeModelTrainingConfig):
        self.code_model_config = code_model_config

    def is_available(self) -> tuple[bool, str]:
        try:
            import accelerate  # noqa: F401
            import peft  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            return False, f"required ML packages not installed in this environment: {type(exc).__name__}: {exc}"
        result = check_feasibility(self.code_model_config)
        if not result.feasible:
            return False, "; ".join(result.reasons) or "configured training run is not feasible on this hardware"
        return True, f"local HF {self.code_model_config.method.upper()} backend is available for {self.code_model_config.base_model.model_id!r}"

    def _effective_settings(self, config: TrainingConfig) -> dict[str, Any]:
        cm = self.code_model_config
        base_model = config.base_model if config.base_model not in ("", "not-configured", None) else cm.base_model.model_id
        return {
            "base_model": base_model,
            "seed": config.seed if config.seed is not None else cm.training.seed,
            "max_steps": config.max_steps if config.max_steps is not None else cm.training.max_steps,
            "checkpoint_root": Path(config.checkpoint_dir or cm.training.output_dir),
        }

    def run(self, dataset_jsonl_path: str, config: TrainingConfig, *, cancellation_token=None) -> TrainingRunResult:
        run_id = uuid.uuid4().hex
        try:
            return self._run(run_id, dataset_jsonl_path, config, cancellation_token)
        except Exception as exc:
            return TrainingRunResult(run_id, "failed", error=f"{type(exc).__name__}: {exc}", resumed_from=config.resume_from)

    def _run(self, run_id: str, dataset_jsonl_path: str, config: TrainingConfig, cancellation_token) -> TrainingRunResult:
        import torch
        from datasets import Dataset
        from peft import LoraConfig as PeftLoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling,
            Trainer, TrainerCallback, TrainingArguments,
        )

        cm = self.code_model_config
        settings = self._effective_settings(config)
        base_model_id, seed, max_steps, checkpoint_root = (
            settings["base_model"], settings["seed"], settings["max_steps"], settings["checkpoint_root"],
        )
        run_dir = checkpoint_root / run_id
        record = TrainingRunRecord(
            training_run_id=run_id, dataset_version=Path(dataset_jsonl_path).stem, base_model=base_model_id,
            config_hash=cm.config_hash(), checkpoint_path=str(run_dir), current_step=0, status="RUNNING",
            resumed_from=config.resume_from, started_at=_now(), updated_at=_now(),
        )
        _save_record(run_dir, record)

        examples = format_examples_for_sft(dataset_jsonl_path)
        if not examples:
            record.status, record.error, record.updated_at = "FAILED", "no formatted SFT examples available in dataset", _now()
            _save_record(run_dir, record)
            return TrainingRunResult(run_id, "failed", error=record.error, resumed_from=config.resume_from)

        torch.manual_seed(seed)
        random.seed(seed)

        tokenizer = AutoTokenizer.from_pretrained(
            cm.base_model.tokenizer_id or base_model_id, revision=cm.base_model.revision,
            trust_remote_code=cm.base_model.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        quant_config = None
        if cm.method == "qlora" and cm.quantization.enabled:
            compute_dtype = getattr(torch, cm.quantization.compute_dtype, torch.float16)
            quant_config = BitsAndBytesConfig(
                load_in_4bit=cm.quantization.bits == 4, load_in_8bit=cm.quantization.bits == 8,
                bnb_4bit_quant_type=cm.quantization.quant_type, bnb_4bit_use_double_quant=cm.quantization.double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        device_map = "auto" if (cm.runtime.device in ("auto", "cuda") and torch.cuda.is_available()) else None
        torch_dtype = getattr(torch, cm.runtime.dtype, None) if cm.runtime.dtype != "auto" else None

        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, revision=cm.base_model.revision, trust_remote_code=cm.base_model.trust_remote_code,
            quantization_config=quant_config, device_map=device_map, torch_dtype=torch_dtype,
        )

        if quant_config is not None:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cm.runtime.gradient_checkpointing)

        resume_source = Path(config.resume_from) if config.resume_from else None
        resume_checkpoint = _latest_checkpoint_dir(resume_source) if resume_source and resume_source.exists() else None
        if resume_checkpoint is not None:
            model = PeftModel.from_pretrained(model, str(resume_checkpoint), is_trainable=True)
        else:
            lora_config = PeftLoraConfig(
                r=cm.lora.rank, lora_alpha=cm.lora.alpha, lora_dropout=cm.lora.dropout,
                target_modules=list(cm.lora.target_modules), bias=cm.lora.bias, task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        def to_text(example) -> dict[str, str]:
            return {"text": example.prompt + "\n\n" + example.response + tokenizer.eos_token}

        texts = [to_text(e) for e in examples]
        random.Random(seed).shuffle(texts)
        val_ratio = cm.training.validation_split_ratio
        n_val = max(1, int(len(texts) * val_ratio)) if len(texts) > 1 and val_ratio > 0 else 0
        val_texts = texts[:n_val] if n_val else []
        train_texts = texts[n_val:] if (n_val and len(texts) - n_val > 0) else texts

        def tokenize_fn(batch):
            return tokenizer(batch["text"], truncation=True, max_length=cm.base_model.max_sequence_length, padding=False)

        train_dataset = Dataset.from_list(train_texts).map(tokenize_fn, batched=True, remove_columns=["text"])
        eval_dataset = Dataset.from_list(val_texts).map(tokenize_fn, batched=True, remove_columns=["text"]) if val_texts else None
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        class _CancellationCallback(TrainerCallback):
            def __init__(self, token, record_ref: TrainingRunRecord, run_dir_ref: Path):
                self.token, self.record, self.run_dir = token, record_ref, run_dir_ref

            def on_step_end(self, args, state, control, **kwargs):
                self.record.current_step, self.record.updated_at = state.global_step, _now()
                _save_record(self.run_dir, self.record)
                if self.token is not None and getattr(self.token, "cancelled", False):
                    control.should_training_stop = True
                return control

        if max_steps is not None:
            total_steps_estimate = max_steps
        else:
            steps_per_epoch = max(1, -(-len(train_dataset) // (cm.training.per_device_train_batch_size * cm.training.gradient_accumulation_steps)))
            total_steps_estimate = int(steps_per_epoch * cm.training.num_train_epochs)
        warmup_steps = int(total_steps_estimate * cm.training.warmup_ratio)

        training_args = TrainingArguments(
            output_dir=str(run_dir),
            per_device_train_batch_size=cm.training.per_device_train_batch_size,
            gradient_accumulation_steps=cm.training.gradient_accumulation_steps,
            learning_rate=cm.training.learning_rate,
            num_train_epochs=cm.training.num_train_epochs,
            max_steps=max_steps if max_steps is not None else -1,
            # transformers>=5 dropped `warmup_ratio` from TrainingArguments
            # (verified against the installed version at
            # training/requirements-code-model.txt's pin) -- convert it to
            # an explicit step count ourselves instead.
            warmup_steps=warmup_steps,
            save_steps=cm.training.save_steps,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=cm.training.eval_steps if eval_dataset is not None else None,
            logging_steps=cm.training.logging_steps,
            seed=seed,
            save_total_limit=2,
            report_to=[],
            gradient_checkpointing=cm.runtime.gradient_checkpointing and quant_config is None,
        )

        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
            data_collator=collator, callbacks=[_CancellationCallback(cancellation_token, record, run_dir)],
        )

        train_output = trainer.train()

        cancelled = cancellation_token is not None and getattr(cancellation_token, "cancelled", False)
        if cancelled:
            interrupted_dir = run_dir / "checkpoint-interrupted"
            trainer.save_model(str(interrupted_dir))
            tokenizer.save_pretrained(str(interrupted_dir))
            record.status, record.checkpoint_path, record.updated_at = "CANCELLED", str(interrupted_dir), _now()
            record.metrics = dict(train_output.metrics) if train_output else {}
            _save_record(run_dir, record)
            return TrainingRunResult(
                run_id, "cancelled", checkpoint_path=str(interrupted_dir),
                metrics=record.metrics, resumed_from=config.resume_from,
                base_model=base_model_id, config_hash=cm.config_hash(),
            )

        adapter_path = run_dir / "adapter"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        eval_metrics: dict[str, Any] = {}
        if eval_dataset is not None:
            eval_metrics = trainer.evaluate()

        metrics = dict(train_output.metrics)
        metrics.update({k if k.startswith("eval_") else f"eval_{k}": v for k, v in eval_metrics.items()})

        record.status = "COMPLETED"
        record.metrics = metrics
        record.current_step = trainer.state.global_step
        record.checkpoint_path = str(adapter_path)
        record.updated_at = _now()
        _save_record(run_dir, record)

        model_version = f"{cm.name}-{run_id[:8]}"
        return TrainingRunResult(
            run_id, "completed", model_version=model_version, checkpoint_path=str(adapter_path),
            metrics=metrics, resumed_from=config.resume_from,
            base_model=base_model_id, config_hash=cm.config_hash(),
        )
