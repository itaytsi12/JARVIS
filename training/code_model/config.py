"""Configurable base-model / training / LoRA / quantization / runtime
settings for the real training backend (Phase 1, 5).

Deliberately NOT the same object as `brain.learning_training.TrainingConfig`
(the small, protocol-level config every `TrainingBackend` already receives
per call) -- that object stays exactly as-is so the existing
`TrainingBackend` protocol is never modified. A
`HuggingFaceLoRATrainingBackend` is instead *constructed* with a
`CodeModelTrainingConfig` (detailed HF/LoRA/quantization defaults, loaded
from one of `training/code_model/configs/*.yaml`), and the per-call
`brain.learning_training.TrainingConfig.base_model`/`.seed`/`.max_steps`/
`.checkpoint_dir`/`.resume_from` -- fields the orchestrator's own
pre-training checks already validate -- override the matching detailed
defaults for that one run when explicitly set. This mirrors
`brain/improvement_coding_agent.py::ClaudeCodeAdapter(model=None,
executable="claude")`'s constructor-config-plus-minimal-per-call-params
shape.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_SCHEMA_VERSION = 1
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


@dataclass
class BaseModelConfig:
    model_id: str = "not-configured"
    revision: str | None = None
    trust_remote_code: bool = False
    tokenizer_id: str | None = None  # defaults to model_id when None
    max_sequence_length: int = 2048


@dataclass
class TrainingHyperparams:
    output_dir: str = "data/learning_checkpoints"
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    max_steps: int | None = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    save_steps: int = 50
    eval_steps: int = 50
    logging_steps: int = 5
    seed: int = 42
    validation_split_ratio: float = 0.1


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    bias: str = "none"


@dataclass
class QuantizationConfig:
    # "qlora" method implies enabled=True; "lora" implies enabled=False.
    # Kept as its own object (rather than folded into `method`) because a
    # real backend needs bits/quant_type/compute_dtype independently
    # configurable, and because a future environment may support 8-bit but
    # not 4-bit, etc.
    enabled: bool = True
    bits: int = 4  # 4 or 8
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "float16"


@dataclass
class RuntimeConfig:
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    dtype: str = "auto"  # "auto" | "float16" | "bfloat16" | "float32"
    gradient_checkpointing: bool = True


@dataclass
class CodeModelTrainingConfig:
    name: str = "unnamed"
    method: str = "qlora"  # "qlora" | "lora" -- never "full": this backend never trains from scratch
    base_model: BaseModelConfig = field(default_factory=BaseModelConfig)
    training: TrainingHyperparams = field(default_factory=TrainingHyperparams)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    schema_version: int = CONFIG_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        """A stable identity for "this exact training configuration" --
        recorded on `ModelVersion.config_hash` (Phase 9) so two runs with
        different hyperparameters are never confused for the same recipe."""
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CodeModelTrainingConfig":
        payload = dict(payload or {})
        return cls(
            name=payload.get("name", "unnamed"),
            method=payload.get("method", "qlora"),
            base_model=BaseModelConfig(**(payload.get("base_model") or {})),
            training=TrainingHyperparams(**(payload.get("training") or {})),
            lora=LoRAConfig(**(payload.get("lora") or {})),
            quantization=QuantizationConfig(**(payload.get("quantization") or {})),
            runtime=RuntimeConfig(**(payload.get("runtime") or {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CodeModelTrainingConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(payload)


def load_config(name_or_path: str) -> CodeModelTrainingConfig:
    """`name_or_path` may be a bare config name (looked up under
    `training/code_model/configs/<name>.yaml`) or a direct path to a YAML
    file."""
    candidate = Path(name_or_path)
    if not candidate.exists():
        candidate = CONFIGS_DIR / f"{name_or_path}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(f"no such training config: {name_or_path!r} (looked at {candidate})")
    return CodeModelTrainingConfig.from_yaml(candidate)


def list_configs() -> list[str]:
    if not CONFIGS_DIR.is_dir():
        return []
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))
