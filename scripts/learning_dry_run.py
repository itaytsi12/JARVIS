"""Bounded, realistic dry run of the full voice-approved continual learning
pipeline (Phase 32).

Uses the REAL, already-proven `ClaudeCodeAdapter` for exactly ONE call (the
teacher fix), against a disposable fixture git repository -- never the
active JARVIS working tree. Everything after the verified fix (voice
approval, variation generation, training, evaluation, promotion) uses
fakes, so this never spends more than one real Claude invocation and never
starts real GPU training or spends cloud money.

Run manually:  python scripts/learning_dry_run.py
Also serves as a CLI/debug entry point (Phase 33) for exercising the
pipeline without a microphone or real training backend.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

JARVIS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JARVIS_ROOT))

from brain.experience_store import ExperienceStore, retrieve_relevant_experiences  # noqa: E402
from brain.improvement_attempt_models import AttemptStatus  # noqa: E402
from brain.improvement_attempt_store import ImprovementAttemptStore  # noqa: E402
from brain.improvement_coding_agent import ClaudeCodeAdapter, FakeCodingAgent  # noqa: E402
from brain.improvement_models import GapType, ImprovementCandidate  # noqa: E402
from brain.improvement_orchestrator import OrchestratorConfig, run_attempt  # noqa: E402
from brain.learning_evaluation import BenchmarkMetrics, FakeBenchmark  # noqa: E402
from brain.learning_orchestrator import handle_verified_teacher_success, start_learning  # noqa: E402
from brain.learning_store import LearningJobStore  # noqa: E402
from brain.learning_training import FakeTrainingBackend, ModelRegistry, TrainingConfig  # noqa: E402
import brain.learning_package as learning_package_module  # noqa: E402
from voice.learning_approval import LearningApprovalOutcome, LearningApprovalResult  # noqa: E402


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result


def build_fixture_repo(root: Path) -> Path:
    repo = root / "fixture_repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "dry-run@example.invalid")
    _git(repo, "config", "user.name", "Learning Dry Run")
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b  # BUG\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add_returns_sum():\n    assert add(2, 3) == 5\n", encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "initial: add() with an obvious subtraction bug")
    return repo


class FakeApproval:
    def __call__(self, **kwargs) -> LearningApprovalResult:
        print("[dry-run] simulated voice: 'yes jarvis'")
        return LearningApprovalResult(LearningApprovalOutcome.APPROVED, "yes jarvis", 1.0)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="jarvis-learning-dry-run-"))
    original_package_root = learning_package_module.DEFAULT_PACKAGE_ROOT
    print(f"[dry-run] fixture root: {tmp}")
    try:
        repo = build_fixture_repo(tmp)
        attempt_store = ImprovementAttemptStore(tmp / "attempts.sqlite3")
        job_store = LearningJobStore(tmp / "jobs.sqlite3")
        experience_store = ExperienceStore(tmp / "experiences.sqlite3")
        model_registry = ModelRegistry(tmp / "registry.sqlite3")
        package_root = tmp / "packages"
        learning_package_module.DEFAULT_PACKAGE_ROOT = package_root

        # --- Step 1: student fails, teacher (REAL ClaudeCodeAdapter, one call) solves ---
        candidate = ImprovementCandidate(
            candidate_id="dry-run-cand", created_at="t", first_seen="t", last_seen="t",
            gap_type=GapType.EXECUTION_BUG.value, confidence=0.95, subsystem="dry_run_subsystem",
            tool_involved="calc.add", raw_request="add(2, 3) should return 5",
            normalized_goal="Fix add(a, b) in calc.py: it returns a - b instead of a + b.",
            exception_type="AssertionError", exception_message="add(2, 3) returned -1, expected 5",
            verification_failure_reason="tests/test_calc.py::test_add_returns_sum fails",
            classification_reason="deterministic arithmetic bug in add()",
        )
        from unittest.mock import patch
        with patch.dict("brain.improvement_repro.SUBSYSTEM_TEST_FILES", {"dry_run_subsystem": "tests/test_calc.py"}):
            print("[dry-run] invoking REAL ClaudeCodeAdapter (one Claude call)...")
            attempt = run_attempt(
                candidate, repository_root=str(repo), coding_agent=ClaudeCodeAdapter(),
                attempt_store=attempt_store,
                config=OrchestratorConfig(max_revision_rounds=2, coding_agent_timeout_seconds=180.0, focused_test_timeout_seconds=60.0, regression_test_timeout_seconds=60.0, max_total_seconds=600.0),
            )
        print(f"[dry-run] teacher attempt status: {attempt.status}")
        if attempt.status != AttemptStatus.READY_FOR_REVIEW.value:
            print(f"[dry-run] BLOCKER: real teacher attempt did not reach READY_FOR_REVIEW (error={attempt.error!r}) -- stopping, not faking success.")
            return 1

        # --- Step 2: voice learning offer (fake "yes jarvis") ---
        offer = handle_verified_teacher_success(attempt, request_approval=FakeApproval(), job_store=job_store, experience_store=experience_store)
        print(f"[dry-run] learning offer: offered={offer.offered} status={offer.job.learning_status if offer.job else None}")
        if not offer.offered or offer.job is None:
            print("[dry-run] BLOCKER: verified teacher fix did not trigger a learning offer.")
            return 1

        # --- Step 3: experience available immediately ---
        # A realistic future retrieval call: a new, differently-worded task
        # in the same subsystem/gap-type family should surface this fix.
        experiences = retrieve_relevant_experiences(
            "the subtract function in math.py returns the wrong total",
            subsystem="dry_run_subsystem", gap_type="EXECUTION_BUG",
            store=experience_store, top_k=3,
        )
        print(f"[dry-run] immediate experience retrieval found {len(experiences)} match(es)")

        # --- Step 4: "Hey Jarvis, start learning" -- fakes from here on ---
        def variant_apply(workspace: Path) -> None:
            base = workspace / ".jarvis-learning-variants" / "v1"
            (base / "before").mkdir(parents=True, exist_ok=True)
            (base / "after").mkdir(parents=True, exist_ok=True)
            (base / "before" / "tests").mkdir(exist_ok=True)
            (base / "after" / "tests").mkdir(exist_ok=True)
            import json
            (base / "manifest.json").write_text(json.dumps({"description": "off-by-sign arithmetic bug in a different function", "test_file": "tests/test_y.py"}))
            (base / "before" / "y.py").write_text("def g(a, b):\n    return a - b\n")
            (base / "before" / "tests" / "test_y.py").write_text("from y import g\ndef test_y():\n    assert g(4, 1) == 5\n")
            (base / "after" / "y.py").write_text("def g(a, b):\n    return a + b\n")
            (base / "after" / "tests" / "test_y.py").write_text("from y import g\ndef test_y():\n    assert g(4, 1) == 5\n")

        variant_agent = FakeCodingAgent(apply=variant_apply)
        benchmark = FakeBenchmark(default=BenchmarkMetrics(solve_rate=0.8, regression_rate=0.0, behavioral_acceptance_rate=0.8))
        backend = FakeTrainingBackend(improve=True)

        summary = start_learning(
            coding_agent=variant_agent, repository_root=str(repo), backend=backend, benchmark=benchmark,
            job_store=job_store, model_registry=model_registry, dataset_root=tmp / "datasets", package_root=package_root,
            training_config=TrainingConfig(base_model="dry-run-fake-base"),
        )
        print(f"[dry-run] start_learning summary: status={summary.status} promoted={summary.promoted} dataset_version={summary.dataset_version}")

        final_job = job_store.get(offer.job.learning_job_id)
        active_model = model_registry.get_active()
        print(f"[dry-run] final job status: {final_job.learning_status}")
        print(f"[dry-run] active model: {active_model.model_version if active_model else None}")

        ok = (
            attempt.status == AttemptStatus.READY_FOR_REVIEW.value
            and offer.offered
            and len(experiences) >= 1
            and summary.status == "COMPLETED"
            and summary.promoted is True
            and final_job.learning_status == "TRAINED"
            and active_model is not None
        )
        print("[dry-run] RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        learning_package_module.DEFAULT_PACKAGE_ROOT = original_package_root
        try:
            attempt_store.close()
            job_store.close()
            experience_store.close()
            model_registry.close()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
