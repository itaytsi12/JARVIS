"""Conservative triage gate and ranking for ImprovementCandidates.

Not every recorded candidate should be handed to an autonomous coding
agent. This module decides whether a candidate is even the *right kind* of
problem for source-code repair (as opposed to a skill-learning problem, an
environmental blip, or something too ambiguous to act on), and -- among
eligible candidates -- which one is worth attempting first.

No cloud/LLM call is used here; this mirrors brain/improvement_classifier.py's
deterministic, evidence-driven design.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from brain.improvement_models import CandidateStatus, GapType, ImprovementCandidate

MIN_CONFIDENCE_FOR_CODE_FIX = 0.6
MIN_OCCURRENCES_FOR_LOW_CONFIDENCE = 3  # a recurring low-confidence pattern is worth a second look


class TriageDecision(str, Enum):
    ELIGIBLE_FOR_CODE_IMPROVEMENT = "ELIGIBLE_FOR_CODE_IMPROVEMENT"
    SKILL_LEARNING_CANDIDATE = "SKILL_LEARNING_CANDIDATE"
    NEEDS_HUMAN_TRIAGE = "NEEDS_HUMAN_TRIAGE"
    NOT_ACTIONABLE = "NOT_ACTIONABLE"
    UNSAFE_TO_AUTOMATE = "UNSAFE_TO_AUTOMATE"


@dataclass
class TriageResult:
    decision: str
    reason: str


# Subsystems where an automatic replay/attempt could plausibly perform an
# irreversible real-world action (sending something, a purchase, an account
# change) if the reproduction step weren't carefully sandboxed. Triage
# doesn't refuse these outright -- brain/improvement_repro.py is the layer
# that must guarantee no unsafe replay -- but a subsystem this sensitive
# combined with low confidence is treated conservatively here.
_HIGH_RISK_SUBSYSTEMS = {"messaging"}


def triage_candidate(
    candidate: ImprovementCandidate,
    *,
    has_active_attempt: bool = False,
    has_ready_for_review_attempt: bool = False,
) -> TriageResult:
    if candidate.status == CandidateStatus.IGNORED.value:
        return TriageResult(TriageDecision.NOT_ACTIONABLE.value, "candidate was explicitly ignored")
    if has_active_attempt:
        return TriageResult(TriageDecision.NOT_ACTIONABLE.value, "an attempt is already in progress for this candidate")
    if has_ready_for_review_attempt:
        return TriageResult(TriageDecision.NOT_ACTIONABLE.value, "a READY_FOR_REVIEW attempt already exists for this candidate")

    gap_type = candidate.gap_type

    if gap_type == GapType.ENVIRONMENTAL_FAILURE.value:
        return TriageResult(TriageDecision.NOT_ACTIONABLE.value, "environmental failures are not auto-code-fixed")
    if gap_type == GapType.USER_INPUT_OR_AMBIGUITY.value:
        return TriageResult(TriageDecision.NOT_ACTIONABLE.value, "ambiguity/missing user input is not a code defect")
    if gap_type == GapType.SKILL_GAP.value:
        return TriageResult(TriageDecision.SKILL_LEARNING_CANDIDATE.value, "primitives appear sufficient; this needs a workflow/skill, not a source change")
    if gap_type == GapType.UNKNOWN.value:
        if candidate.occurrence_count >= MIN_OCCURRENCES_FOR_LOW_CONFIDENCE:
            return TriageResult(TriageDecision.NEEDS_HUMAN_TRIAGE.value, f"UNKNOWN but recurring ({candidate.occurrence_count} occurrences) -- worth a human look")
        return TriageResult(TriageDecision.NOT_ACTIONABLE.value, "insufficient evidence to classify; not auto-fixed")

    if gap_type not in {GapType.EXECUTION_BUG.value, GapType.CODE_CAPABILITY_GAP.value}:
        return TriageResult(TriageDecision.NEEDS_HUMAN_TRIAGE.value, f"unrecognized gap_type {gap_type!r}")

    if candidate.subsystem in _HIGH_RISK_SUBSYSTEMS and candidate.confidence < MIN_CONFIDENCE_FOR_CODE_FIX:
        return TriageResult(
            TriageDecision.UNSAFE_TO_AUTOMATE.value,
            f"subsystem {candidate.subsystem!r} can perform real-world side effects; confidence too low to automate safely",
        )

    if candidate.confidence < MIN_CONFIDENCE_FOR_CODE_FIX:
        return TriageResult(
            TriageDecision.NEEDS_HUMAN_TRIAGE.value,
            f"confidence {candidate.confidence:.2f} below the {MIN_CONFIDENCE_FOR_CODE_FIX} threshold for automatic {gap_type}",
        )

    return TriageResult(TriageDecision.ELIGIBLE_FOR_CODE_IMPROVEMENT.value, f"{gap_type} with confidence {candidate.confidence:.2f}")


@dataclass
class RankedCandidate:
    candidate: ImprovementCandidate
    score: float
    reasons: list[str]


def score_candidate(candidate: ImprovementCandidate) -> RankedCandidate:
    """A transparent, additive score -- not a false-precision single number
    pretending to be a calibrated probability. Each contributing signal is
    named in `reasons` so the ranking is auditable."""
    score = 0.0
    reasons: list[str] = []

    occurrence_component = min(candidate.occurrence_count, 10) * 2.0
    score += occurrence_component
    reasons.append(f"occurrence_count={candidate.occurrence_count} (+{occurrence_component:.1f})")

    confidence_component = candidate.confidence * 10.0
    score += confidence_component
    reasons.append(f"confidence={candidate.confidence:.2f} (+{confidence_component:.1f})")

    if candidate.gap_type == GapType.EXECUTION_BUG.value:
        score += 5.0
        reasons.append("gap_type=EXECUTION_BUG, core-functionality signal (+5.0)")
    elif candidate.gap_type == GapType.CODE_CAPABILITY_GAP.value:
        score += 3.0
        reasons.append("gap_type=CODE_CAPABILITY_GAP (+3.0)")

    if candidate.subsystem in {"browser", "app_lifecycle", "desktop_ui"}:
        score += 2.0
        reasons.append(f"user-facing subsystem={candidate.subsystem} (+2.0)")

    if candidate.partial:
        score += 1.0
        reasons.append("partial completion observed (+1.0)")

    return RankedCandidate(candidate=candidate, score=round(score, 2), reasons=reasons)


def rank_candidates(candidates: list[ImprovementCandidate]) -> list[RankedCandidate]:
    ranked = [score_candidate(c) for c in candidates]
    ranked.sort(key=lambda r: (r.score, r.candidate.occurrence_count, r.candidate.last_seen), reverse=True)
    return ranked


def get_best_improvement_candidate(candidates: list[ImprovementCandidate]) -> RankedCandidate | None:
    ranked = rank_candidates(candidates)
    return ranked[0] if ranked else None
