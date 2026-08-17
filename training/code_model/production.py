"""Production backend/benchmark wiring for "Hey Jarvis, start learning"
(Phase 19).

`voice/background_assistant.py::_start_learning_task` calls
`build_production_backend`/`build_production_benchmark` instead of
constructing `FakeTrainingBackend`/`FakeBenchmark` directly -- this module
is the ONE place that decision is made, so it's trivial to audit that no
fake ever appears in the production path (tests remain free to construct
fakes directly; only this module feeds the voice-triggered production
call).

Which base model/config is used is controlled by the
`JARVIS_CODE_MODEL_CONFIG` environment variable (default: `"qlora_7b"`,
this project's recommended real config -- see
`training/code_model/configs/`). Swapping to a different config, or a
future stronger base model, never requires touching this module or the
orchestrator -- only the env var or a new file under `configs/`.
"""
from __future__ import annotations

import os

from brain.learning_training import TrainingConfig
from training.code_model.benchmark.runner import RealCodingBenchmark
from training.code_model.config import CodeModelTrainingConfig, load_config
from training.code_model.hf_backend import HuggingFaceLoRATrainingBackend

DEFAULT_PRODUCTION_CONFIG_NAME = "qlora_7b"


def build_production_backend(config_name: str | None = None) -> HuggingFaceLoRATrainingBackend:
    name = config_name or os.getenv("JARVIS_CODE_MODEL_CONFIG", DEFAULT_PRODUCTION_CONFIG_NAME)
    return HuggingFaceLoRATrainingBackend(load_config(name))


def build_training_config(code_model_config: CodeModelTrainingConfig) -> TrainingConfig:
    """The protocol-level `brain.learning_training.TrainingConfig` every
    `TrainingBackend.run()` call receives, populated from the backend's own
    detailed config -- in particular `base_model`, which
    `brain.learning_training.run_pre_training_checks` validates directly
    (it has no visibility into a backend's internal
    `CodeModelTrainingConfig`). Passing a bare `TrainingConfig()` here would
    leave `base_model="not-configured"` and cause pre-training checks to
    fail even when the backend is genuinely ready -- this is the one
    correct place that translation happens, reused by both the voice
    dispatch and `training/code_model/start_learning.py`'s CLI equivalent."""
    return TrainingConfig(
        base_model=code_model_config.base_model.model_id,
        seed=code_model_config.training.seed,
        max_steps=code_model_config.training.max_steps,
        checkpoint_dir=code_model_config.training.output_dir,
    )


def _agent_factory_for_config(code_model_config: CodeModelTrainingConfig):
    def factory(model_version: str):
        from brain.learning_training import get_model_registry
        from training.code_model.student_adapter import LocalCodingModelAdapter

        adapter_path = None
        base_model_id = code_model_config.base_model.model_id
        record = get_model_registry().get(model_version)
        if record is not None:
            adapter_path = record.adapter_path or adapter_path
            base_model_id = record.base_model or base_model_id
        return LocalCodingModelAdapter.from_checkpoint(
            base_model_id, adapter_path, trust_remote_code=code_model_config.base_model.trust_remote_code,
        )

    return factory


def build_production_benchmark(code_model_config: CodeModelTrainingConfig) -> RealCodingBenchmark:
    """`model_version` values the returned benchmark is asked to `evaluate`
    are looked up in the model registry to resolve the correct adapter path
    -- including the special case of an untrained baseline (no matching
    registry row / no `adapter_path`), which falls back to the bare base
    model, exactly the Phase 16 "evaluate the original/base open coding
    model" comparison arm."""
    return RealCodingBenchmark(agent_factory=_agent_factory_for_config(code_model_config))
