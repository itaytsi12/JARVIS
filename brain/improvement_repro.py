"""Conservative, evidence-honest reproduction of an ImprovementCandidate's
original problem, run inside an isolated attempt worktree.

This module deliberately does NOT attempt generic dynamic re-invocation of
whatever tool originally failed. Two structural facts about this codebase
make that unsafe to fake:

1. `brain/improvement_observer.py` intentionally redacts content-bearing
   tool arguments (typed text, message bodies) before a candidate is ever
   persisted -- see `training_data/sanitizer.py::sanitize_action_arguments`.
   The original content that would be needed to faithfully replay those
   actions is gone by design, not by oversight. Pretending to reproduce a
   `type_text`/`send_whatsapp_message` failure from `<REDACTED>` arguments
   would just fabricate a false signal.
2. Several tool subsystems (browser, desktop UI, live application launches)
   have real, hard-to-sandbox side effects and non-deterministic
   environment dependencies (an open browser tab, window focus, OS state).
   Automatically re-driving them from a background attempt is not safe.

So this module only ever produces one of three honest outcomes, matching
`ReproductionType` in `brain/improvement_attempt_models.py`:

- UNSAFE_TO_REPLAY: the candidate's subsystem can perform real-world side
  effects (see `_UNSAFE_TO_REPLAY_SUBSYSTEMS`); nothing is executed.
- LOG_EVIDENCE_ONLY: replay isn't safe/possible (redacted arguments, no
  mapped test suite, unmapped subsystem) -- the original exception/failure
  evidence is carried forward as-is, with `reproduced=None` because nothing
  was independently confirmed.
- INTEGRATION_REPRO: for the small set of subsystems with an existing,
  already-safe-to-run automated test suite (`SUBSYSTEM_TEST_FILES`), that
  suite is actually executed inside the worktree via `SafeCommandRunner`.
  A clean pytest failure (exit code 1) is treated as `reproduced=True`; a
  clean pass (exit code 0) is `reproduced=False`; anything else (collection
  errors, timeouts) stays `reproduced=None` rather than guessing.

`reproduced=None` is a legitimate, common outcome -- it means "an
autonomous fix for this candidate cannot claim confirmed reproduction," not
an error. Downstream (`brain/improvement_evaluator.py`) must never treat it
as equivalent to `reproduced=True`.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brain.improvement_attempt_models import ReproductionType
from brain.improvement_models import ImprovementCandidate
from brain.task_supervisor import SafeCommandRunner

# Only subsystems with a test suite that is already safe to run unattended
# (no network side effects, no real messages sent, no real windows opened)
# are eligible for INTEGRATION_REPRO. Anything not listed here falls back to
# LOG_EVIDENCE_ONLY -- silence here is the safe default, not a limitation to
# work around by guessing a file name.
SUBSYSTEM_TEST_FILES: dict[str, str] = {
    "browser": "tests/test_browser_agent.py",
    "desktop_ui": "tests/test_desktop_fixes.py",
    "web_answer": "tests/test_web_questions.py",
}

# Mirrors brain/improvement_triage.py's _HIGH_RISK_SUBSYSTEMS -- kept as an
# independent constant (not imported) because triage's list is about "should
# we even consider automating this" while this one is the stricter "would
# executing anything here perform a real-world action" question repro must
# answer on its own, even if triage policy changes later.
_UNSAFE_TO_REPLAY_SUBSYSTEMS = {"messaging"}

_REDACTED_MARKER = "<REDACTED>"


@dataclass
class ReproductionResult:
    method: str
    attempted: bool
    reproduced: bool | None
    evidence: dict[str, Any] = field(default_factory=dict)
    skip_reason: str | None = None
    duration_ms: float | None = None


def _has_redacted_arguments(candidate: ImprovementCandidate) -> bool:
    for action in candidate.executed_actions or []:
        arguments = action.get("arguments") or {}
        for value in arguments.values():
            if isinstance(value, str) and _REDACTED_MARKER in value:
                return True
    return False


def _evidence_only(candidate: ImprovementCandidate, reason: str) -> ReproductionResult:
    return ReproductionResult(
        method=ReproductionType.LOG_EVIDENCE_ONLY.value,
        attempted=True,
        reproduced=None,
        evidence={
            "exception_type": candidate.exception_type,
            "exception_message": candidate.exception_message,
            "verification_failure_reason": candidate.verification_failure_reason,
            "plan_rejection_reason": candidate.plan_rejection_reason,
            "unsupported_capability": candidate.unsupported_capability,
        },
        skip_reason=reason,
    )


def reproduce(
    candidate: ImprovementCandidate,
    worktree_path: str | Path,
    *,
    runner: SafeCommandRunner | None = None,
    timeout_seconds: float = 180.0,
) -> ReproductionResult:
    """Attempt (safely, honestly) to reproduce `candidate`'s original
    problem inside `worktree_path`. Never raises for an ordinary "couldn't
    reproduce" outcome -- that's a normal `ReproductionResult`, not a
    failure of this function."""
    if candidate.subsystem in _UNSAFE_TO_REPLAY_SUBSYSTEMS:
        return ReproductionResult(
            method=ReproductionType.UNSAFE_TO_REPLAY.value, attempted=False, reproduced=None,
            skip_reason=f"subsystem {candidate.subsystem!r} can perform real-world side effects; automatic replay is refused",
        )

    if _has_redacted_arguments(candidate):
        return _evidence_only(candidate, "one or more recorded action arguments were redacted for privacy; original content is unavailable for replay")

    test_file = SUBSYSTEM_TEST_FILES.get(candidate.subsystem or "")
    if test_file is None:
        return _evidence_only(candidate, f"no existing automated test suite is mapped for subsystem {candidate.subsystem!r}")

    if not (Path(worktree_path) / test_file).exists():
        return _evidence_only(candidate, f"expected test file {test_file!r} was not found in this worktree")

    runner = runner or SafeCommandRunner()
    started = time.perf_counter()
    try:
        result = runner.run(["python", "-m", "pytest", test_file, "-q"], str(worktree_path), timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return ReproductionResult(
            method=ReproductionType.INTEGRATION_REPRO.value, attempted=True, reproduced=None,
            skip_reason=f"subsystem test suite exceeded {timeout_seconds}s",
            duration_ms=timeout_seconds * 1000,
        )
    duration_ms = (time.perf_counter() - started) * 1000

    exit_code = result["exit_code"]
    if exit_code == 0:
        reproduced: bool | None = False
        skip_reason = None
    elif exit_code == 1:
        reproduced = True
        skip_reason = None
    else:
        reproduced = None
        skip_reason = f"pytest exited {exit_code} (not a clean pass/fail -- likely a collection error, not evidence either way)"

    return ReproductionResult(
        method=ReproductionType.INTEGRATION_REPRO.value,
        attempted=True,
        reproduced=reproduced,
        evidence={"test_file": test_file, "exit_code": exit_code, "output_tail": result["output"][-2000:]},
        skip_reason=skip_reason,
        duration_ms=duration_ms,
    )
