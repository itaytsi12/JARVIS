"""Phase 28: the strongest safe end-to-end dry run without expensive real
training.

    approved LearningJob fixture
        -> REAL dataset builder (brain.learning_dataset)
        -> REAL tiny training backend (training.code_model.hf_backend,
           small_smoke_test config: sshleifer/tiny-gpt2, real LoRA,
           real forward/backward steps)
        -> real model artifact (a real saved PEFT adapter on disk)
        -> REAL benchmark harness (training.code_model.benchmark.runner,
           the 5 real executable fixture tasks)
        -> real promotion evaluation (brain.learning_evaluation, fresh
           benchmark scores only)
        -> candidate promoted/rejected based on the ACTUAL score

No FakeTrainingBackend, no FakeBenchmark anywhere in this script. The
teacher-fix step itself uses FakeCodingAgent (Phase 27/28 explicitly allow
this -- proving training/benchmark mechanics are real doesn't require
spending Claude quota a second time; the real ClaudeCodeAdapter path was
already proven end-to-end in a prior session's smoke test).

Honest expectation: `sshleifer/tiny-gpt2` is a randomly-initialized,
non-coding test fixture model. It is expected to solve ZERO real benchmark
tasks, before or after this tiny amount of training -- and therefore to be
correctly REJECTED by the real promotion gates. That is not a failure of
this dry run; a rigged/forced promotion would be the actual failure mode
this script exists to rule out. The point is that every step -- dataset,
training, artifact, benchmark, decision -- is genuinely executed and the
decision is driven by a real (here: correctly zero) score, not fabricated.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

JARVIS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JARVIS_ROOT))

from brain.improvement_coding_agent import FakeCodingAgent  # noqa: E402
from brain.learning_dataset import build_dataset_version  # noqa: E402
from brain.learning_models import DataQualityLabel, LearningJob, LearningJobStatus  # noqa: E402
from brain.learning_package import LearningPackage  # noqa: E402
from brain.learning_store import LearningJobStore  # noqa: E402
from brain.learning_training import ModelRegistry, TrainingPolicy  # noqa: E402
from brain.learning_orchestrator import start_learning  # noqa: E402
from brain.learning_validator import VariantValidationResult  # noqa: E402
from brain.learning_variation import GeneratedVariant  # noqa: E402
from training.code_model.benchmark.runner import RealCodingBenchmark  # noqa: E402
from training.code_model.config import load_config  # noqa: E402
from training.code_model.hf_backend import HuggingFaceLoRATrainingBackend  # noqa: E402


def _write_synthetic_variant(workspace: Path) -> None:
    base = workspace / ".jarvis-learning-variants" / "v1"
    (base / "before").mkdir(parents=True, exist_ok=True)
    (base / "after").mkdir(parents=True, exist_ok=True)
    (base / "before" / "tests").mkdir(exist_ok=True)
    (base / "after" / "tests").mkdir(exist_ok=True)
    import json
    (base / "manifest.json").write_text(json.dumps({"description": "an off-by-sign bug in a helper function", "test_file": "tests/test_y.py"}))
    (base / "before" / "helper.py").write_text("def h(a, b):\n    return a - b\n")
    (base / "before" / "tests" / "test_y.py").write_text("from helper import h\ndef test_h():\n    assert h(3, 2) == 5\n")
    (base / "after" / "helper.py").write_text("def h(a, b):\n    return a + b\n")
    (base / "after" / "tests" / "test_y.py").write_text("from helper import h\ndef test_h():\n    assert h(3, 2) == 5\n")


def make_agent_factory_for_tiny_model(model_registry: ModelRegistry):
    """Real model loading for whichever `model_version` string the real
    benchmark is asked to evaluate -- resolves the trained candidate's real
    adapter path via the SAME (fixture-scoped, not the default singleton)
    model registry `start_learning` itself writes to, or falls back to the
    bare (untrained) base model for the baseline arm."""
    def factory(model_version: str):
        from training.code_model.student_adapter import LocalCodingModelAdapter

        record = model_registry.get(model_version)
        adapter_path = record.adapter_path if record else None
        return LocalCodingModelAdapter.from_checkpoint("sshleifer/tiny-gpt2", adapter_path, max_iterations=1, max_new_tokens=48)

    return factory


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="jarvis-code-model-dry-run-"))
    print(f"[dry-run] fixture root: {tmp}")
    try:
        job_store = LearningJobStore(tmp / "jobs.sqlite3")
        model_registry = ModelRegistry(tmp / "registry.sqlite3")
        package_root = tmp / "packages"
        dataset_root = tmp / "datasets"

        # --- Step 1: an approved LearningJob fixture ---
        job = LearningJob(
            learning_job_id="job-1", created_at="t", updated_at="t", candidate_id="cand-1",
            improvement_attempt_id="att-1", fingerprint="fp-1", learning_status=LearningJobStatus.READY_FOR_TRAINING.value,
        )
        job_store.create(job)
        package = LearningPackage(
            learning_job_id="job-1", improvement_attempt_id="att-1", problem_family="fam-1",
            original_task="fix the off-by-sign bug in calc.py's add function", subsystem="filesystem",
            gap_type="EXECUTION_BUG", root_cause_category="EXECUTION_BUG_with_differential_test",
            reusable_strategy="Changed the subtraction to addition in add().",
            applicability_conditions=["gap_type=EXECUTION_BUG"], files_changed=["calc.py"],
            diff_summary="1 file changed, +1/-1",
            before_behavior={"reproduced": True}, after_behavior={"reproduced": False},
        )
        from brain.learning_package import save_learning_package
        save_learning_package(package, root=package_root)
        print("[dry-run] Step 1: approved LearningJob + LearningPackage fixture created")

        # --- Steps 2-6: the real pipeline, via start_learning ---
        code_model_config = load_config("small_smoke_test")
        backend = HuggingFaceLoRATrainingBackend(code_model_config)
        benchmark = RealCodingBenchmark(agent_factory=make_agent_factory_for_tiny_model(model_registry))

        # Variation generation uses a fake coding agent that writes one real,
        # mechanically-verifiable synthetic variant -- proving the real
        # dataset builder includes both REAL_VERIFIED_TEACHER and
        # SYNTHETIC_VERIFIED rows, not a stub.
        variant_agent = FakeCodingAgent(apply=_write_synthetic_variant)
        from training.code_model.production import build_training_config

        training_config = build_training_config(code_model_config)
        training_config.checkpoint_dir = str(tmp / "checkpoints")  # keep real artifacts inside the disposable fixture, never data/

        summary = start_learning(
            coding_agent=variant_agent,
            repository_root=str(JARVIS_ROOT),  # only used to create the throwaway variation-generation worktree
            backend=backend,
            benchmark=benchmark,
            training_config=training_config,
            job_store=job_store,
            model_registry=model_registry,
            dataset_root=dataset_root,
            package_root=package_root,
            policy=TrainingPolicy(mode="manual_only"),
            explicit_command=True,
            progress_callback=lambda status, detail: print(f"[dry-run] stage: {status} {detail}"),
        )

        print(f"\n[dry-run] Step 2 (dataset): version={summary.dataset_version}")
        print(f"[dry-run] Step 3-4 (real training + artifact): status={summary.status} candidate={summary.candidate_model_version}")
        candidate = model_registry.get(summary.candidate_model_version) if summary.candidate_model_version else None
        if candidate:
            print(f"[dry-run]   real adapter_path: {candidate.adapter_path}")
            print(f"[dry-run]   adapter exists on disk: {Path(candidate.adapter_path).exists() if candidate.adapter_path else False}")
            print(f"[dry-run]   real training metrics: {candidate.metrics}")
        print(f"[dry-run] Step 5 (real benchmark + promotion decision): promoted={summary.promoted}")
        print(f"[dry-run]   reasons: {summary.reasons}")
        if candidate:
            print(f"[dry-run]   real benchmark_result recorded on candidate: {candidate.benchmark_result}")

        final_job = job_store.get("job-1")
        print(f"[dry-run] final job learning_status: {final_job.learning_status}")

        ok = (
            summary.status == "COMPLETED"
            and summary.dataset_version is not None
            and candidate is not None
            and candidate.adapter_path is not None
            and Path(candidate.adapter_path).exists()
            and "train_loss" in candidate.metrics
            and summary.promoted is not None  # a real decision was made, either way
            and candidate.benchmark_result  # real benchmark scores were recorded, not skipped
        )
        print("\n[dry-run] RESULT:", "PASS" if ok else "FAIL")
        print("[dry-run] Note: promoted=False is the EXPECTED, correct outcome for this untrained tiny fixture model --")
        print("[dry-run] the point of this run is that every stage genuinely executed and the decision used real scores.")
        return 0 if ok else 1
    finally:
        try:
            job_store.close()
            model_registry.close()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
