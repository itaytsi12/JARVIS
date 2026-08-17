"""Training backend abstraction, pre-training checks, training policy, and
the model registry (Phases 12, 13, 14).

`TrainingBackend` is a small protocol -- the same "provider-neutral
adapter" shape as `brain/improvement_coding_agent.py::CodingAgent` -- so a
real local LoRA/QLoRA backend or a real cloud-GPU backend can be plugged in
later without touching `brain/learning_orchestrator.py`. This module never
trains a model from scratch (every backend here is documented as
fine-tuning a configured, already-pretrained base model) and never spends
money on its own initiative: `ConfiguredCloudTrainingBackend` only ever
*describes* a training plan unless a caller has set explicit authorization,
enforced structurally (a boolean the caller must pass True), not merely by
convention or a docstring warning.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------
# Phase 14: training backend protocol
# ----------------------------------------------------------------------

@dataclass
class ResourceBudget:
    """Phase 21 configuration surface. This session ships no real local
    GPU/LoRA backend to enforce these against (every backend here is
    `FakeTrainingBackend` or the never-auto-dispatching
    `ConfiguredCloudTrainingBackend`), so nothing in this module currently
    throttles CPU/RAM/GPU usage -- that enforcement belongs inside a real
    `TrainingBackend.run()` implementation, which can read these fields.
    What IS real today: `brain/learning_orchestrator.py::start_learning`
    always runs on a background thread (never the voice/audio thread), and
    variant validation in `brain/learning_validator.py` already runs
    strictly sequentially, so neither can make JARVIS unresponsive or spike
    parallel resource usage as currently implemented."""
    max_cpu_percent: float | None = None
    gpu_device: str | None = None
    max_ram_mb: int | None = None
    max_parallel_validation_jobs: int = 1
    yield_to_interactive: bool = True


@dataclass
class TrainingConfig:
    base_model: str = "not-configured"
    method: str = "LoRA"
    seed: int = 42
    max_steps: int | None = None
    checkpoint_dir: str = "data/learning_checkpoints"
    resume_from: str | None = None
    resource_budget: ResourceBudget = field(default_factory=ResourceBudget)


@dataclass
class TrainingRunResult:
    training_run_id: str
    exit_status: str  # "completed" | "failed" | "cancelled" | "blocked"
    model_version: str | None = None
    checkpoint_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    resumed_from: str | None = None


class TrainingBackend(Protocol):
    backend_name: str

    def is_available(self) -> tuple[bool, str]: ...

    def run(
        self, dataset_jsonl_path: str, config: TrainingConfig, *, cancellation_token=None,
    ) -> TrainingRunResult: ...


class FakeTrainingBackend:
    """Deterministic test/dry-run double. Never touches a GPU, never reads
    the dataset file's content (only that it exists), completes instantly.
    `improve` controls whether the produced metrics should read as better
    than a baseline, so evaluation/promotion tests can exercise both the
    promote and reject paths deterministically."""
    backend_name = "fake"

    def __init__(self, *, improve: bool = True, metrics: dict[str, Any] | None = None):
        self.improve = improve
        self.metrics = metrics
        self.calls = 0

    def is_available(self) -> tuple[bool, str]:
        return True, "fake backend is always available (tests/dry-run only)"

    def run(self, dataset_jsonl_path: str, config: TrainingConfig, *, cancellation_token=None) -> TrainingRunResult:
        self.calls += 1
        run_id = uuid.uuid4().hex
        if cancellation_token is not None and getattr(cancellation_token, "cancelled", False):
            return TrainingRunResult(run_id, "cancelled", error="cancelled before training started")
        if not Path(dataset_jsonl_path).exists():
            return TrainingRunResult(run_id, "failed", error=f"dataset file not found: {dataset_jsonl_path}")
        model_version = f"student-{run_id[:8]}"
        metrics = self.metrics or {"solve_rate": 0.7 if self.improve else 0.3, "regression_rate": 0.0, "loss": 0.2}
        return TrainingRunResult(
            run_id, "completed", model_version=model_version,
            checkpoint_path=str(Path(config.checkpoint_dir) / model_version), metrics=metrics,
            resumed_from=config.resume_from,
        )


class ConfiguredCloudTrainingBackend:
    """Describes what training WOULD run on a configured cloud GPU backend,
    without ever dispatching it, unless `authorized=True` is explicitly
    passed by the caller (Phase 12: "do not automatically spend money
    without explicit configuration/authorization"). `provider_configured`
    reflects whether the backend even has connection details at all --
    independent of authorization, which is a separate, per-run decision.
    """
    backend_name = "configured_cloud"

    def __init__(self, *, provider_configured: bool = False, authorized: bool = False, provider_name: str = "unconfigured"):
        self.provider_configured = provider_configured
        self.authorized = authorized
        self.provider_name = provider_name

    def is_available(self) -> tuple[bool, str]:
        if not self.provider_configured:
            return False, "no cloud GPU training backend is configured"
        return True, f"cloud backend {self.provider_name!r} is configured"

    def build_plan(self, config: TrainingConfig) -> dict[str, Any]:
        return {
            "backend": self.provider_name,
            "base_model": config.base_model,
            "method": config.method,
            "seed": config.seed,
            "note": "this plan was NOT dispatched; set authorized=True explicitly to run it",
        }

    def run(self, dataset_jsonl_path: str, config: TrainingConfig, *, cancellation_token=None) -> TrainingRunResult:
        run_id = uuid.uuid4().hex
        if not self.provider_configured:
            return TrainingRunResult(run_id, "blocked", error="no cloud GPU training backend is configured")
        if not self.authorized:
            return TrainingRunResult(
                run_id, "blocked",
                error=(
                    "cloud training was not dispatched: explicit authorization is required. "
                    "This backend never spends money on its own initiative."
                ),
                metrics={"plan": self.build_plan(config)},
            )
        # A real cloud dispatch implementation belongs here once a specific
        # provider is chosen; this module intentionally stops short of that
        # so no session can accidentally start real billed compute.
        return TrainingRunResult(run_id, "failed", error="cloud dispatch is authorized but not yet implemented")


# ----------------------------------------------------------------------
# Phase 12: pre-training checks
# ----------------------------------------------------------------------

@dataclass
class PreTrainingCheckResult:
    ready: bool
    reasons: list[str] = field(default_factory=list)
    plan: dict[str, Any] | None = None


def run_pre_training_checks(
    *, job_count: int, example_count: int, backend: TrainingBackend, config: TrainingConfig,
    min_free_disk_bytes: int = 200 * 1024 * 1024,
) -> PreTrainingCheckResult:
    """Never raises -- every failure mode here is an honest, reported
    reason, not a crash (Phase 12: "If training cannot run on current
    hardware: DO NOT crash")."""
    reasons: list[str] = []
    if job_count <= 0:
        reasons.append("no approved learning jobs")
    if example_count <= 0:
        reasons.append("no usable training examples")
    try:
        available, why = backend.is_available()
    except Exception as exc:
        available, why = False, f"backend availability check raised: {exc}"
    if not available:
        reasons.append(f"training backend unavailable: {why}")
    if config.base_model in {"", "not-configured"}:
        reasons.append("no base model configured")
    try:
        free = shutil.disk_usage(".").free
        if free < min_free_disk_bytes:
            reasons.append(f"insufficient disk space ({free} bytes free, need {min_free_disk_bytes})")
    except Exception:
        pass  # best-effort only; never block on an environment we can't inspect

    plan = None
    if reasons and isinstance(backend, ConfiguredCloudTrainingBackend):
        plan = backend.build_plan(config)
    return PreTrainingCheckResult(ready=not reasons, reasons=reasons, plan=plan)


# ----------------------------------------------------------------------
# Phase 13: training policy
# ----------------------------------------------------------------------

@dataclass
class TrainingPolicy:
    mode: str = "manual_only"  # "manual_only" | "minimum_examples" | "minimum_learning_jobs"
    minimum_examples: int = 0
    minimum_learning_jobs: int = 0


def policy_allows_training(
    policy: TrainingPolicy, *, job_count: int, example_count: int, explicit_command: bool,
) -> tuple[bool, str]:
    """An explicit "Hey Jarvis, start learning" command always overrides a
    waiting threshold (Phase 13) -- it never overrides the pre-training
    checks in `run_pre_training_checks`, which are a separate, mandatory
    gate regardless of policy."""
    if explicit_command:
        return True, "explicit 'start learning' command overrides waiting thresholds"
    if policy.mode == "manual_only":
        return False, "policy is manual_only; training only starts from an explicit command"
    if policy.mode == "minimum_examples":
        if example_count < policy.minimum_examples:
            return False, f"only {example_count} new examples available, policy requires at least {policy.minimum_examples}"
        return True, "minimum_examples threshold satisfied"
    if policy.mode == "minimum_learning_jobs":
        if job_count < policy.minimum_learning_jobs:
            return False, f"only {job_count} approved jobs available, policy requires at least {policy.minimum_learning_jobs}"
        return True, "minimum_learning_jobs threshold satisfied"
    return False, f"unrecognized training policy mode: {policy.mode!r}"


# ----------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------

@dataclass
class ModelVersion:
    model_version: str
    dataset_version: str
    training_run_id: str
    created_at: str
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "CANDIDATE"  # "CANDIDATE" | "ACTIVE" | "REJECTED"
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelVersion":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


class ModelRegistry:
    """SQLite-backed record of every training run and which single
    `model_version` is currently ACTIVE. Promotion is the only way the
    active model ever changes -- a training run finishing is never, by
    itself, sufficient (Phase 16: training loss/completion alone means
    nothing)."""

    def __init__(self, path: str | Path | None = None):
        import os
        self.path = Path(path or os.getenv("MODEL_REGISTRY_DB_PATH") or Path.cwd() / "data" / "jarvis_model_registry.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock, self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_versions(
                    model_version TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def record(self, version: ModelVersion) -> ModelVersion:
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO model_versions VALUES(?,?,?,?)",
                (version.model_version, version.status, version.created_at, json.dumps(version.to_dict())),
            )
        return version

    def get(self, model_version: str) -> ModelVersion | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT payload_json FROM model_versions WHERE model_version=?", (model_version,)
            ).fetchone()
        return ModelVersion.from_dict(json.loads(row["payload_json"])) if row else None

    def get_active(self) -> ModelVersion | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT payload_json FROM model_versions WHERE status='ACTIVE' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return ModelVersion.from_dict(json.loads(row["payload_json"])) if row else None

    def promote(self, model_version: str) -> ModelVersion:
        """The only way `status='ACTIVE'` is ever assigned. Demotes any
        previously active version to 'REPLACED' first, inside one
        transaction, so there is never a moment with zero or two active
        models."""
        with self._lock, self.connection:
            version = self.get(model_version)
            if version is None:
                raise KeyError(f"no such model version: {model_version!r}")
            previous = self.get_active()
            if previous is not None and previous.model_version != model_version:
                previous.status = "REPLACED"
                self.connection.execute(
                    "UPDATE model_versions SET status=?, payload_json=? WHERE model_version=?",
                    (previous.status, json.dumps(previous.to_dict()), previous.model_version),
                )
            version.status = "ACTIVE"
            self.connection.execute(
                "UPDATE model_versions SET status=?, payload_json=? WHERE model_version=?",
                (version.status, json.dumps(version.to_dict()), model_version),
            )
            return version

    def reject(self, model_version: str, reason: str) -> ModelVersion:
        with self._lock, self.connection:
            version = self.get(model_version)
            if version is None:
                raise KeyError(f"no such model version: {model_version!r}")
            version.status = "REJECTED"
            version.rejection_reason = reason
            self.connection.execute(
                "UPDATE model_versions SET status=?, payload_json=? WHERE model_version=?",
                (version.status, json.dumps(version.to_dict()), model_version),
            )
            return version

    def history(self, limit: int = 200) -> list[ModelVersion]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT payload_json FROM model_versions ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [ModelVersion.from_dict(json.loads(row["payload_json"])) for row in rows]

    def close(self) -> None:
        with self._lock:
            self.connection.close()


_REGISTRY: ModelRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_model_registry() -> ModelRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            if _REGISTRY is None:
                import os
                import sys
                import tempfile
                test_path = None
                if "pytest" in sys.modules and not os.getenv("MODEL_REGISTRY_DB_PATH"):
                    test_path = Path(tempfile.mkdtemp(prefix="jarvis-model-registry-pytest-")) / "jarvis_model_registry.sqlite3"
                _REGISTRY = ModelRegistry(test_path)
    return _REGISTRY


def reset_model_registry_for_tests(path: str | Path | None = None) -> ModelRegistry:
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is not None:
            _REGISTRY.close()
        _REGISTRY = ModelRegistry(path)
    return _REGISTRY
