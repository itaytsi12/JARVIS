import unittest

from brain.improvement_models import CandidateStatus, GapType, ImprovementCandidate
from brain.improvement_triage import (
    MIN_CONFIDENCE_FOR_CODE_FIX,
    MIN_OCCURRENCES_FOR_LOW_CONFIDENCE,
    TriageDecision,
    get_best_improvement_candidate,
    rank_candidates,
    score_candidate,
    triage_candidate,
)


def _candidate(**overrides) -> ImprovementCandidate:
    defaults = dict(
        candidate_id="c1", created_at="t", first_seen="t", last_seen="t",
        gap_type=GapType.EXECUTION_BUG.value, confidence=0.9,
    )
    defaults.update(overrides)
    return ImprovementCandidate(**defaults)


class TriageEligibilityTests(unittest.TestCase):
    def test_strong_execution_bug_is_eligible(self):
        result = triage_candidate(_candidate(gap_type=GapType.EXECUTION_BUG.value, confidence=0.9))
        self.assertEqual(result.decision, TriageDecision.ELIGIBLE_FOR_CODE_IMPROVEMENT.value)

    def test_strong_code_capability_gap_is_eligible(self):
        result = triage_candidate(_candidate(gap_type=GapType.CODE_CAPABILITY_GAP.value, confidence=0.9))
        self.assertEqual(result.decision, TriageDecision.ELIGIBLE_FOR_CODE_IMPROVEMENT.value)

    def test_low_confidence_execution_bug_needs_human_triage_not_eligible(self):
        result = triage_candidate(_candidate(gap_type=GapType.EXECUTION_BUG.value, confidence=0.3))
        self.assertEqual(result.decision, TriageDecision.NEEDS_HUMAN_TRIAGE.value)

    def test_confidence_exactly_at_threshold_is_eligible(self):
        result = triage_candidate(_candidate(gap_type=GapType.EXECUTION_BUG.value, confidence=MIN_CONFIDENCE_FOR_CODE_FIX))
        self.assertEqual(result.decision, TriageDecision.ELIGIBLE_FOR_CODE_IMPROVEMENT.value)

    def test_skill_gap_is_never_eligible_for_code_editing_regardless_of_confidence(self):
        for confidence in (0.0, 0.5, 0.99, 1.0):
            with self.subTest(confidence=confidence):
                result = triage_candidate(_candidate(gap_type=GapType.SKILL_GAP.value, confidence=confidence))
                self.assertEqual(result.decision, TriageDecision.SKILL_LEARNING_CANDIDATE.value)
                self.assertNotEqual(result.decision, TriageDecision.ELIGIBLE_FOR_CODE_IMPROVEMENT.value)

    def test_environmental_failure_is_not_actionable(self):
        result = triage_candidate(_candidate(gap_type=GapType.ENVIRONMENTAL_FAILURE.value, confidence=0.99))
        self.assertEqual(result.decision, TriageDecision.NOT_ACTIONABLE.value)

    def test_user_input_or_ambiguity_is_not_actionable(self):
        result = triage_candidate(_candidate(gap_type=GapType.USER_INPUT_OR_AMBIGUITY.value, confidence=0.99))
        self.assertEqual(result.decision, TriageDecision.NOT_ACTIONABLE.value)

    def test_unknown_low_occurrence_is_not_actionable(self):
        result = triage_candidate(_candidate(gap_type=GapType.UNKNOWN.value, occurrence_count=1))
        self.assertEqual(result.decision, TriageDecision.NOT_ACTIONABLE.value)

    def test_unknown_recurring_goes_to_human_triage(self):
        result = triage_candidate(_candidate(gap_type=GapType.UNKNOWN.value, occurrence_count=MIN_OCCURRENCES_FOR_LOW_CONFIDENCE))
        self.assertEqual(result.decision, TriageDecision.NEEDS_HUMAN_TRIAGE.value)

    def test_unknown_never_becomes_eligible_no_matter_the_occurrence_count(self):
        result = triage_candidate(_candidate(gap_type=GapType.UNKNOWN.value, occurrence_count=1000))
        self.assertNotEqual(result.decision, TriageDecision.ELIGIBLE_FOR_CODE_IMPROVEMENT.value)


class TriageSafetyGateTests(unittest.TestCase):
    def test_already_in_progress_candidate_is_not_selected_again(self):
        result = triage_candidate(_candidate(), has_active_attempt=True)
        self.assertEqual(result.decision, TriageDecision.NOT_ACTIONABLE.value)

    def test_already_ready_for_review_candidate_is_not_selected_again(self):
        result = triage_candidate(_candidate(), has_ready_for_review_attempt=True)
        self.assertEqual(result.decision, TriageDecision.NOT_ACTIONABLE.value)

    def test_active_attempt_overrides_otherwise_eligible_signal(self):
        strong = _candidate(gap_type=GapType.EXECUTION_BUG.value, confidence=1.0)
        result = triage_candidate(strong, has_active_attempt=True)
        self.assertEqual(result.decision, TriageDecision.NOT_ACTIONABLE.value)

    def test_ignored_status_is_not_actionable(self):
        result = triage_candidate(_candidate(status=CandidateStatus.IGNORED.value, confidence=1.0))
        self.assertEqual(result.decision, TriageDecision.NOT_ACTIONABLE.value)

    def test_high_risk_subsystem_with_low_confidence_is_unsafe_to_automate(self):
        result = triage_candidate(_candidate(subsystem="messaging", confidence=0.4))
        self.assertEqual(result.decision, TriageDecision.UNSAFE_TO_AUTOMATE.value)

    def test_high_risk_subsystem_with_high_confidence_is_still_eligible(self):
        result = triage_candidate(_candidate(subsystem="messaging", confidence=0.9))
        self.assertEqual(result.decision, TriageDecision.ELIGIBLE_FOR_CODE_IMPROVEMENT.value)

    def test_unrecognized_gap_type_needs_human_triage_not_silently_eligible(self):
        result = triage_candidate(_candidate(gap_type="SOME_FUTURE_GAP_TYPE_NOT_YET_HANDLED", confidence=0.99))
        self.assertEqual(result.decision, TriageDecision.NEEDS_HUMAN_TRIAGE.value)


class TriageRankingTests(unittest.TestCase):
    def test_score_is_transparent_and_reproducible_from_named_reasons(self):
        candidate = _candidate(occurrence_count=4, confidence=0.8, gap_type=GapType.EXECUTION_BUG.value, subsystem="browser", partial=True)
        ranked = score_candidate(candidate)
        expected = min(4, 10) * 2.0 + 0.8 * 10.0 + 5.0 + 2.0 + 1.0
        self.assertAlmostEqual(ranked.score, round(expected, 2))
        self.assertEqual(len(ranked.reasons), 5)  # one reason per contributing signal, nothing opaque

    def test_occurrence_component_is_capped_at_ten(self):
        low = score_candidate(_candidate(occurrence_count=10, confidence=0.0, gap_type=GapType.UNKNOWN.value))
        high = score_candidate(_candidate(occurrence_count=1000, confidence=0.0, gap_type=GapType.UNKNOWN.value))
        self.assertEqual(low.score, high.score)

    def test_execution_bug_scores_higher_than_code_capability_gap_at_equal_confidence(self):
        bug = score_candidate(_candidate(gap_type=GapType.EXECUTION_BUG.value, confidence=0.7, occurrence_count=1))
        gap = score_candidate(_candidate(gap_type=GapType.CODE_CAPABILITY_GAP.value, confidence=0.7, occurrence_count=1))
        self.assertGreater(bug.score, gap.score)

    def test_rank_candidates_sorts_highest_score_first(self):
        weak = _candidate(candidate_id="weak", confidence=0.1, occurrence_count=1, gap_type=GapType.UNKNOWN.value)
        strong = _candidate(candidate_id="strong", confidence=1.0, occurrence_count=10, gap_type=GapType.EXECUTION_BUG.value)
        ranked = rank_candidates([weak, strong])
        self.assertEqual(ranked[0].candidate.candidate_id, "strong")

    def test_rank_candidates_tie_breaks_by_occurrence_then_recency(self):
        older = _candidate(candidate_id="older", occurrence_count=2, last_seen="2026-01-01T00:00:00+00:00")
        newer = _candidate(candidate_id="newer", occurrence_count=2, last_seen="2026-06-01T00:00:00+00:00")
        ranked = rank_candidates([older, newer])
        self.assertEqual(ranked[0].candidate.candidate_id, "newer")

    def test_get_best_improvement_candidate_returns_none_for_empty_list(self):
        self.assertIsNone(get_best_improvement_candidate([]))

    def test_get_best_improvement_candidate_returns_top_ranked(self):
        weak = _candidate(candidate_id="weak", confidence=0.1, occurrence_count=1, gap_type=GapType.UNKNOWN.value)
        strong = _candidate(candidate_id="strong", confidence=1.0, occurrence_count=10, gap_type=GapType.EXECUTION_BUG.value)
        best = get_best_improvement_candidate([weak, strong])
        self.assertEqual(best.candidate.candidate_id, "strong")


if __name__ == "__main__":
    unittest.main()
