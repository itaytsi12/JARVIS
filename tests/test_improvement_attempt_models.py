import unittest

from brain.improvement_attempt_models import (
    ATTEMPT_SCHEMA_VERSION,
    TERMINAL_ATTEMPT_STATUSES,
    AttemptStatus,
    EvaluationResult,
    ImprovementAttempt,
    ReproductionType,
)


def _attempt(**overrides) -> ImprovementAttempt:
    defaults = dict(attempt_id="attempt-1", candidate_id="candidate-1", created_at="2026-08-17T00:00:00+00:00")
    defaults.update(overrides)
    return ImprovementAttempt(**defaults)


class ImprovementAttemptDefaultsTests(unittest.TestCase):
    """Only identity fields are required; everything else has a safe, inert default."""

    def test_required_fields_only(self):
        attempt = _attempt()
        self.assertEqual(attempt.attempt_id, "attempt-1")
        self.assertEqual(attempt.candidate_id, "candidate-1")

    def test_missing_identity_field_raises(self):
        with self.assertRaises(TypeError):
            ImprovementAttempt(candidate_id="candidate-1", created_at="now")  # type: ignore[call-arg]

    def test_default_status_is_queued(self):
        self.assertEqual(_attempt().status, AttemptStatus.QUEUED.value)

    def test_default_workspace_state_is_unallocated(self):
        self.assertEqual(_attempt().workspace_state, "unallocated")

    def test_default_evaluation_is_inconclusive(self):
        self.assertEqual(_attempt().evaluation, EvaluationResult.INCONCLUSIVE.value)

    def test_default_evaluation_confidence_is_zero(self):
        self.assertEqual(_attempt().evaluation_confidence, 0.0)

    def test_default_regression_not_detected(self):
        self.assertFalse(_attempt().regression_detected)

    def test_default_reproduction_not_attempted(self):
        attempt = _attempt()
        self.assertFalse(attempt.reproduction_attempted)
        self.assertIsNone(attempt.reproduced)

    def test_default_schema_version_matches_module_constant(self):
        self.assertEqual(_attempt().schema_version, ATTEMPT_SCHEMA_VERSION)

    def test_mutable_defaults_are_not_shared_between_instances(self):
        """dataclass `field(default_factory=list/dict)` must give each attempt
        its own list/dict -- a shared mutable default would leak state
        between unrelated attempts, which for an autonomous-execution record
        would be a silent correctness bug, not just a style nit."""
        first = _attempt(attempt_id="a")
        second = _attempt(attempt_id="b")
        first.files_changed.append("brain/agent.py")
        first.acceptance_gates["focused_tests_passed"] = True
        first.evidence_reference["candidate_id"] = "a"
        self.assertEqual(second.files_changed, [])
        self.assertEqual(second.acceptance_gates, {})
        self.assertEqual(second.evidence_reference, {})


class ImprovementAttemptStatusTests(unittest.TestCase):
    """Status transitions themselves are driven by the (not-yet-written)
    orchestrator, but the schema's own status vocabulary must be internally
    consistent -- every AttemptStatus is exactly one of terminal/non-terminal,
    and no status silently falls through the cracks."""

    def test_every_status_is_classified_terminal_or_not(self):
        all_statuses = {s.value for s in AttemptStatus}
        non_terminal = all_statuses - TERMINAL_ATTEMPT_STATUSES
        self.assertEqual(non_terminal, {AttemptStatus.QUEUED.value, AttemptStatus.REPRODUCING.value,
                                         AttemptStatus.IMPROVING.value, AttemptStatus.VALIDATING.value})

    def test_ready_for_review_is_terminal(self):
        self.assertIn(AttemptStatus.READY_FOR_REVIEW.value, TERMINAL_ATTEMPT_STATUSES)

    def test_in_progress_statuses_are_not_terminal(self):
        for status in (AttemptStatus.QUEUED, AttemptStatus.REPRODUCING, AttemptStatus.IMPROVING, AttemptStatus.VALIDATING):
            self.assertNotIn(status.value, TERMINAL_ATTEMPT_STATUSES)

    def test_failure_and_rejection_statuses_are_terminal(self):
        for status in (AttemptStatus.FIX_FAILED, AttemptStatus.FIX_PARTIAL, AttemptStatus.NOT_REPRODUCIBLE,
                       AttemptStatus.VERIFICATION_FAILED, AttemptStatus.REJECTED, AttemptStatus.BLOCKED,
                       AttemptStatus.CANCELLED):
            self.assertIn(status.value, TERMINAL_ATTEMPT_STATUSES)

    def test_status_field_accepts_any_attempt_status_value(self):
        for status in AttemptStatus:
            attempt = _attempt(status=status.value)
            self.assertEqual(attempt.status, status.value)


class ImprovementAttemptSerializationTests(unittest.TestCase):
    def test_round_trip_preserves_all_fields(self):
        attempt = _attempt(
            status=AttemptStatus.VALIDATING.value,
            files_changed=["brain/agent.py", "brain/agent_runtime.py"],
            acceptance_gates={"focused_tests_passed": True, "regression_passed": False},
            evaluation=EvaluationResult.IMPROVED.value,
            reproduction_method=ReproductionType.UNIT_REPRO.value,
        )
        restored = ImprovementAttempt.from_dict(attempt.to_dict())
        self.assertEqual(restored, attempt)

    def test_to_dict_is_a_plain_json_safe_dict(self):
        payload = _attempt().to_dict()
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["status"], AttemptStatus.QUEUED.value)
        self.assertEqual(payload["files_changed"], [])

    def test_from_dict_ignores_unknown_fields(self):
        payload = _attempt().to_dict()
        payload["some_future_field_this_version_does_not_know_about"] = "value"
        restored = ImprovementAttempt.from_dict(payload)
        self.assertEqual(restored.attempt_id, "attempt-1")

    def test_from_dict_fills_defaults_for_missing_fields(self):
        minimal = {"attempt_id": "a1", "candidate_id": "c1", "created_at": "now"}
        restored = ImprovementAttempt.from_dict(minimal)
        self.assertEqual(restored.status, AttemptStatus.QUEUED.value)
        self.assertEqual(restored.files_changed, [])
        self.assertEqual(restored.acceptance_gates, {})

    def test_from_dict_round_trip_from_empty_payload_raises(self):
        # candidate_id/attempt_id/created_at are still required -- from_dict
        # must not silently invent identity for a record with none.
        with self.assertRaises(TypeError):
            ImprovementAttempt.from_dict({})


if __name__ == "__main__":
    unittest.main()
