import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from brain import agent
from brain.improvement_classifier import classify
from brain.improvement_models import CandidateStatus, GapType, ImprovementCandidate, stable_hash
from brain.improvement_observer import _fingerprint, _gather_evidence, _infer_subsystem, _skip_reason, observe
from brain.improvement_store import ImprovementStore, reset_improvement_store_for_tests
from brain.models import Action, Plan, ToolResult


def _fresh_store() -> ImprovementStore:
    tmp = Path(tempfile.mkdtemp(prefix="jarvis-improvement-test-")) / "improvements.sqlite3"
    return reset_improvement_store_for_tests(tmp)


class ImprovementClassifierTests(unittest.TestCase):
    """Phase 3/9: deterministic classification, no cloud model involved."""

    def test_real_exception_is_execution_bug(self):
        gap, confidence, reason = classify({"exception_type": "RuntimeError", "exception_message": "boom"})
        self.assertEqual(gap, GapType.EXECUTION_BUG.value)
        self.assertGreater(confidence, 0.5)

    def test_network_exception_is_environmental(self):
        gap, _, _ = classify({"exception_type": "ConnectionError", "exception_message": "network unreachable"})
        self.assertEqual(gap, GapType.ENVIRONMENTAL_FAILURE.value)

    def test_ambiguous_app_identity_is_user_input_or_ambiguity(self):
        gap, confidence, _ = classify({"block_reason": "app_identification_uncertain"})
        self.assertEqual(gap, GapType.USER_INPUT_OR_AMBIGUITY.value)
        self.assertGreater(confidence, 0.5)

    def test_clarification_required_is_user_input_or_ambiguity(self):
        gap, _, _ = classify({"block_reason": "clarification_required"})
        self.assertEqual(gap, GapType.USER_INPUT_OR_AMBIGUITY.value)

    def test_unknown_tool_is_code_capability_gap(self):
        gap, confidence, reason = classify({"unknown_tool_names": ["fold_laundry"]})
        self.assertEqual(gap, GapType.CODE_CAPABILITY_GAP.value)
        self.assertIn("fold_laundry", reason)

    def test_environmental_error_code_is_environmental_failure(self):
        gap, _, _ = classify({"failed_error_codes_environmental": ["chrome_not_found"]})
        self.assertEqual(gap, GapType.ENVIRONMENTAL_FAILURE.value)

    def test_tool_success_but_unverified_mismatch_is_execution_bug(self):
        gap, _, _ = classify({"tool_success_but_unverified_mismatch": True})
        self.assertEqual(gap, GapType.EXECUTION_BUG.value)

    def test_unrepresented_clause_is_skill_gap_not_high_confidence(self):
        gap, confidence, _ = classify({"unrepresented_clause": True})
        self.assertEqual(gap, GapType.SKILL_GAP.value)
        self.assertLess(confidence, 0.7)  # conservative, not overconfident

    def test_verification_shaped_failure_is_execution_bug(self):
        gap, _, _ = classify({"failed_error_codes": ["verification_failed"]})
        self.assertEqual(gap, GapType.EXECUTION_BUG.value)

    def test_no_structured_evidence_is_unknown_with_low_confidence(self):
        gap, confidence, _ = classify({})
        self.assertEqual(gap, GapType.UNKNOWN.value)
        self.assertLess(confidence, 0.5)

    def test_unrecognized_error_code_stays_unknown_not_guessed(self):
        gap, confidence, _ = classify({"failed_error_codes": ["some_never_seen_before_code"]})
        self.assertEqual(gap, GapType.UNKNOWN.value)
        self.assertLess(confidence, 0.5)


class ImprovementSkipPolicyTests(unittest.TestCase):
    """Phase 2/7: noise filtering."""

    def test_verified_success_is_skipped(self):
        self.assertEqual(_skip_reason({"success": True, "verified": True, "actions": []}), "verified_success")

    def test_unverified_success_is_skipped(self):
        self.assertEqual(_skip_reason({"success": True, "verified": False, "actions": []}), "success")

    def test_user_cancellation_is_skipped(self):
        outcome = {"success": False, "actions": [{"tool": "web_search", "success": False, "error": "cancelled"}]}
        self.assertEqual(_skip_reason(outcome), "user_cancelled")

    def test_approval_not_granted_is_skipped(self):
        outcome = {"success": False, "actions": [{"tool": "close_application", "success": False, "error": "human_confirmation_required"}]}
        self.assertEqual(_skip_reason(outcome), "approval_not_granted")

    def test_genuine_failure_is_not_skipped(self):
        outcome = {"success": False, "actions": [{"tool": "open_application", "success": False, "error": "application_window_unverified"}]}
        self.assertIsNone(_skip_reason(outcome))


class ImprovementSubsystemInferenceTests(unittest.TestCase):
    def test_browser_tools_map_to_browser_subsystem(self):
        self.assertEqual(_infer_subsystem(["browser_click_first_result"]), "browser")

    def test_desktop_input_tools_map_to_desktop_ui(self):
        self.assertEqual(_infer_subsystem(["type_text"]), "desktop_ui")

    def test_unknown_tool_still_gets_a_general_bucket(self):
        self.assertEqual(_infer_subsystem(["some_never_registered_tool"]), "general")

    def test_no_tools_is_none(self):
        self.assertIsNone(_infer_subsystem([]))


class ImprovementFingerprintTests(unittest.TestCase):
    """Phase 4: same root cause + different wording -> same fingerprint;
    different root cause + similar wording -> different fingerprint."""

    def test_same_root_cause_different_wording_same_fingerprint(self):
        evidence_a = {"failed_actions": [{"tool": "browser_click_first_result", "error": "verification_failed"}]}
        evidence_b = {"failed_actions": [{"tool": "browser_click_first_result", "error": "verification_failed"}]}
        fp_a = _fingerprint(evidence_a, GapType.EXECUTION_BUG.value, "browser", "browser_click_first_result")
        fp_b = _fingerprint(evidence_b, GapType.EXECUTION_BUG.value, "browser", "browser_click_first_result")
        self.assertEqual(fp_a, fp_b)

    def test_different_root_cause_different_fingerprint(self):
        evidence_a = {"failed_actions": [{"tool": "browser_click_first_result", "error": "verification_failed"}]}
        evidence_b = {"exception_type": "TargetClosedError", "failed_actions": [{"tool": "browser_click_first_result", "error": None}]}
        fp_a = _fingerprint(evidence_a, GapType.EXECUTION_BUG.value, "browser", "browser_click_first_result")
        fp_b = _fingerprint(evidence_b, GapType.EXECUTION_BUG.value, "browser", "browser_click_first_result")
        self.assertNotEqual(fp_a, fp_b)

    def test_fingerprint_excludes_raw_wording(self):
        # Only structural fields feed the fingerprint -- passing unrelated
        # "clauses"/free text must not perturb it.
        evidence = {"failed_actions": [{"tool": "x", "error": "y"}], "clauses": ["anything, wording, at, all"]}
        fp1 = _fingerprint(evidence, GapType.EXECUTION_BUG.value, "browser", "x")
        evidence["clauses"] = ["totally different wording"]
        fp2 = _fingerprint(evidence, GapType.EXECUTION_BUG.value, "browser", "x")
        self.assertEqual(fp1, fp2)


class ImprovementStorePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.store = _fresh_store()

    def _candidate(self, fingerprint="fp-1", **overrides):
        base = dict(
            candidate_id="", created_at="t", first_seen="t", last_seen="t",
            gap_type=GapType.EXECUTION_BUG.value, confidence=0.8, fingerprint=fingerprint,
        )
        base.update(overrides)
        import uuid
        base["candidate_id"] = uuid.uuid4().hex
        return ImprovementCandidate(**base)

    def test_new_candidate_is_inserted(self):
        stored, is_new = self.store.record(self._candidate())
        self.assertTrue(is_new)
        self.assertEqual(stored.occurrence_count, 1)
        self.assertEqual(self.store.count(), 1)

    def test_same_fingerprint_deduplicates_and_increments_occurrence(self):
        first, _ = self.store.record(self._candidate(fingerprint="fp-dup"))
        second, is_new = self.store.record(self._candidate(fingerprint="fp-dup"))
        self.assertFalse(is_new)
        self.assertEqual(second.candidate_id, first.candidate_id)
        self.assertEqual(second.occurrence_count, 2)
        self.assertEqual(self.store.count(), 1)

    def test_different_fingerprints_create_separate_rows(self):
        self.store.record(self._candidate(fingerprint="fp-a"))
        self.store.record(self._candidate(fingerprint="fp-b"))
        self.assertEqual(self.store.count(), 2)

    def test_query_by_id(self):
        stored, _ = self.store.record(self._candidate(fingerprint="fp-q"))
        found = self.store.get(stored.candidate_id)
        self.assertIsNotNone(found)
        self.assertEqual(found.candidate_id, stored.candidate_id)

    def test_query_by_status_and_gap_type(self):
        self.store.record(self._candidate(fingerprint="fp-1", gap_type=GapType.SKILL_GAP.value))
        self.store.record(self._candidate(fingerprint="fp-2", gap_type=GapType.EXECUTION_BUG.value))
        skill_gaps = self.store.query(gap_type=GapType.SKILL_GAP.value)
        self.assertEqual(len(skill_gaps), 1)
        self.assertEqual(skill_gaps[0].gap_type, GapType.SKILL_GAP.value)
        new_ones = self.store.query(status=CandidateStatus.NEW.value)
        self.assertEqual(len(new_ones), 2)

    def test_status_transition(self):
        stored, _ = self.store.record(self._candidate(fingerprint="fp-status"))
        self.assertTrue(self.store.set_status(stored.candidate_id, CandidateStatus.TRIAGED.value))
        updated = self.store.get(stored.candidate_id)
        self.assertEqual(updated.status, CandidateStatus.TRIAGED.value)

    def test_status_transition_survives_dedup_update(self):
        stored, _ = self.store.record(self._candidate(fingerprint="fp-persist-status"))
        self.store.set_status(stored.candidate_id, CandidateStatus.IGNORED.value)
        self.store.record(self._candidate(fingerprint="fp-persist-status"))
        updated = self.store.get(stored.candidate_id)
        self.assertEqual(updated.status, CandidateStatus.IGNORED.value)
        self.assertEqual(updated.occurrence_count, 2)

    def test_invalid_status_rejected(self):
        stored, _ = self.store.record(self._candidate(fingerprint="fp-invalid"))
        with self.assertRaises(ValueError):
            self.store.set_status(stored.candidate_id, "NOT_A_REAL_STATUS")

    def test_persists_across_new_connection_same_file(self):
        stored, _ = self.store.record(self._candidate(fingerprint="fp-restart"))
        path = self.store.path
        self.store.close()
        reopened = ImprovementStore(path)
        found = reopened.get(stored.candidate_id)
        self.assertIsNotNone(found)
        self.assertEqual(found.fingerprint, "fp-restart")
        reopened.close()

    def test_concurrent_dedup_increments_are_not_lost(self):
        errors = []
        def worker():
            try:
                self.store.record(self._candidate(fingerprint="fp-concurrent"))
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.store.count(), 1)
        stored = self.store.query()[0]
        self.assertEqual(stored.occurrence_count, 20)


class ImprovementObserverIntegrationTests(unittest.TestCase):
    """Real run_agent integration -- the actual live pipeline, mocked only
    at the tool-execution boundary, exactly like the existing regression
    tests in tests/test_action_execution_regression.py."""

    def setUp(self):
        self.store = _fresh_store()

    def test_verified_success_creates_no_candidate(self):
        route = {"type": "tool", "tool": "open_application", "arguments": {"app_name": "chrome"}}
        with patch.object(agent.executor, "execute_action", return_value=ToolResult(True, "open_application", "Opened chrome successfully.", {"pid": 1, "hwnd": 2, "verified": True})):
            agent.run_agent("open chrome", route=route)
        self.assertEqual(self.store.count(), 0)

    def test_simple_local_success_creates_no_candidate(self):
        route = {"type": "tool", "tool": "volume_up", "arguments": {"amount": 1}}
        with patch.object(agent.executor, "execute_action", return_value=ToolResult(True, "volume_up", "Volume up.", {"verified": True})):
            agent.run_agent("volume up", route=route)
        self.assertEqual(self.store.count(), 0)

    def test_real_tool_exception_creates_execution_bug_candidate(self):
        route = {"type": "tool", "tool": "open_application", "arguments": {"app_name": "notepad"}}
        with patch.object(agent.executor, "execute_action", side_effect=RuntimeError("unexpected crash")):
            with self.assertRaises(RuntimeError):
                agent.run_agent("open notepad", route=route)
        self.assertEqual(self.store.count(), 1)
        candidate = self.store.query()[0]
        self.assertEqual(candidate.gap_type, GapType.EXECUTION_BUG.value)
        self.assertEqual(candidate.exception_type, "RuntimeError")

    def test_existing_tool_crash_inside_plan_is_execution_bug(self):
        route = {"type": "local_plan", "actions": [Action("open_application", {"app_name": "notepad"})]}
        with patch.object(agent, "_execute_recorded_plan", side_effect=RuntimeError("plan execution crashed")):
            with self.assertRaises(RuntimeError):
                agent.run_agent("open notepad", route=route)
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.store.query()[0].gap_type, GapType.EXECUTION_BUG.value)

    def test_tool_success_but_verification_failure_creates_candidate(self):
        route = {"type": "local_plan", "actions": [Action("browser_click", {"target": "Save", "kind": "button"})]}
        results = [ToolResult(True, "browser_click", "Browser click completed, but no resulting page change was independently verified.", {"url": "https://example.com", "verified": False})]
        with patch.object(agent, "_execute_recorded_plan", return_value=results):
            agent.run_agent("click save", route=route)
        self.assertEqual(self.store.count(), 1)
        candidate = self.store.query()[0]
        self.assertEqual(candidate.gap_type, GapType.EXECUTION_BUG.value)
        self.assertTrue(candidate.success)  # tool-level success, but flagged anyway
        self.assertFalse(candidate.verified)

    def test_application_missing_is_environmental_failure(self):
        route = {"type": "tool", "tool": "open_application", "arguments": {"app_name": "some_never_installed_app"}}
        with patch.object(agent.executor, "execute_action", return_value=ToolResult(False, "open_application", "I don't know how to open 'some_never_installed_app' yet.", {}, "unknown_application")):
            agent.run_agent("open some never installed app", route=route)
        self.assertEqual(self.store.count(), 1)
        candidate = self.store.query()[0]
        self.assertIn(candidate.gap_type, {GapType.ENVIRONMENTAL_FAILURE.value, GapType.CODE_CAPABILITY_GAP.value})

    def test_network_exception_is_environmental_not_execution_bug(self):
        route = {"type": "tool", "tool": "open_website", "arguments": {"url": "https://example.com"}}
        with patch.object(agent.executor, "execute_action", side_effect=ConnectionError("network unreachable")):
            with self.assertRaises(ConnectionError):
                agent.run_agent("open example.com", route=route)
        candidate = self.store.query()[0]
        self.assertEqual(candidate.gap_type, GapType.ENVIRONMENTAL_FAILURE.value)

    def test_missing_clarification_is_user_input_or_ambiguity(self):
        with patch.object(agent, "should_use_task_planner", return_value=True), \
             patch.object(agent, "create_task_plan", return_value=Plan("open a music", [], context={"clarification": "Which music app do you mean?"})):
            agent.run_agent("open a music", route=None)
        self.assertEqual(self.store.count(), 1)
        candidate = self.store.query()[0]
        self.assertEqual(candidate.gap_type, GapType.USER_INPUT_OR_AMBIGUITY.value)

    def test_insufficient_evidence_stays_unknown(self):
        outcome = {"executed": True, "success": False, "verified": False, "partial": False, "actions": [{"tool": "mystery_tool", "success": False, "verified": False, "error": "totally_novel_error_code"}], "action_count": 1}
        candidate = observe(command="do something odd", route={"type": "tool"}, execution_outcome=outcome)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.gap_type, GapType.UNKNOWN.value)
        self.assertLess(candidate.confidence, 0.5)

    def test_same_underlying_bug_different_wording_deduplicates(self):
        # should_use_task_planner is forced off so the explicit route (and
        # therefore the mocked plan execution) is what actually runs for
        # both differently-worded commands -- otherwise the real task
        # planner would build two structurally different plans and this
        # would no longer be testing "same root cause, different wording".
        route = {"type": "local_plan", "actions": [Action("browser_click_first_result", {})]}
        results = [ToolResult(False, "browser_click_first_result", "Browser action could not be verified.", {"verified": False}, "verification_failed")]
        with patch.object(agent, "should_use_task_planner", return_value=False), \
             patch.object(agent, "_execute_recorded_plan", return_value=results):
            agent.run_agent("open the first Minecraft video", route=route)
            agent.run_agent("search youtube for redstone and open the top result", route=route)
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.store.query()[0].occurrence_count, 2)

    def test_different_root_causes_do_not_merge(self):
        route = {"type": "local_plan", "actions": [Action("browser_click_first_result", {})]}
        verification_failure = [ToolResult(False, "browser_click_first_result", "Browser action could not be verified.", {"verified": False}, "verification_failed")]
        with patch.object(agent, "should_use_task_planner", return_value=False), \
             patch.object(agent, "_execute_recorded_plan", return_value=verification_failure):
            agent.run_agent("open the first result", route=route)
        with patch.object(agent, "should_use_task_planner", return_value=False), \
             patch.object(agent, "_execute_recorded_plan", side_effect=RuntimeError("TargetClosedError-shaped failure")):
            with self.assertRaises(RuntimeError):
                agent.run_agent("open the first result again", route=route)
        self.assertEqual(self.store.count(), 2)

    def test_sensitive_tool_arguments_are_sanitized_before_persistence(self):
        route = {"type": "local_plan", "actions": [Action(
            "send_whatsapp_message", {"recipient": "Alice", "message": "my password is hunter2"},
            sensitive_fields={"message"},
        )]}
        results = [ToolResult(False, "send_whatsapp_message", "Failed to send.", {"verified": False}, "recipient_not_found")]
        with patch.object(agent, "_execute_recorded_plan", return_value=results):
            agent.run_agent("tell Alice my password is hunter2", route=route)
        candidate = self.store.query()[0]
        dumped = str(candidate.to_dict())
        self.assertNotIn("hunter2", dumped)

    def test_stale_route_actions_that_never_executed_are_not_recorded_as_executed(self):
        # The exact shape of the original false-success bug: route_command()
        # built a garbage multi-action local_plan, but should_use_task_planner
        # overrides it and a totally different (here: failing) plan runs
        # instead. The candidate must reflect only what really executed.
        stale_route = {"type": "local_plan", "actions": [
            Action("open_application", {"app_name": "chrome, go to youtube, search for x"}),
            Action("open_application", {"app_name": "the first result"}),
        ]}
        incomplete_plan = Plan("compound goal", [Action("browser_open_url", {"url": "https://example.com"})], context={})
        with patch.object(agent, "create_task_plan", return_value=incomplete_plan), patch.object(agent, "create_plan", return_value=[]):
            agent.run_agent("Open Chrome, go to YouTube, search for x, and open the first result.", route=stale_route)
        self.assertEqual(self.store.count(), 1)
        candidate = self.store.query()[0]
        self.assertEqual(candidate.executed_tool_names, [])
        for action in candidate.executed_actions:
            self.assertNotEqual(action.get("tool"), "open_application")

    def test_user_cancellation_creates_no_candidate(self):
        from brain.task_supervisor import CancellationToken
        token = CancellationToken(); token.cancel()
        route = {"type": "tool", "tool": "open_application", "arguments": {"app_name": "notepad"}}
        agent.run_agent("open notepad", route=route, cancellation_token=token)
        self.assertEqual(self.store.count(), 0)

    def test_safety_refusal_creates_no_candidate(self):
        route = {"type": "local_plan", "actions": [Action("close_application", {"app_name": "notepad"})]}
        results = [ToolResult(False, "close_application", "High-impact action requires confirmation.", {}, "human_confirmation_required")]
        with patch.object(agent, "_execute_recorded_plan", return_value=results):
            agent.run_agent("close notepad", route=route)
        self.assertEqual(self.store.count(), 0)

    def test_successful_lifecycle_recovery_creates_no_candidate(self):
        # The browser session recovered internally (see tools/browser_agent.py)
        # and the action still succeeded -- from execution_outcome's
        # perspective this is indistinguishable from any other success.
        route = {"type": "local_plan", "actions": [Action("browser_open_url", {"url": "https://www.youtube.com"})]}
        results = [ToolResult(True, "browser_open_url", "Browser: YouTube", {"url": "https://www.youtube.com", "verified": True})]
        with patch.object(agent, "_execute_recorded_plan", return_value=results):
            agent.run_agent("open youtube", route=route)
        self.assertEqual(self.store.count(), 0)

    def test_partial_execution_creates_candidate(self):
        route = {"type": "local_plan", "actions": [
            Action("open_application", {"app_name": "notepad"}),
            Action("open_website", {"url": "https://www.youtube.com"}),
        ]}
        results = [
            ToolResult(True, "open_application", "Opened Notepad.", {"verified": True}),
            ToolResult(False, "open_website", "Website did not launch in time.", {"verified": False}, "website_navigation_unverified"),
        ]
        with patch.object(agent, "_execute_recorded_plan", return_value=results):
            agent.run_agent("open notepad and open youtube", route=route)
        self.assertEqual(self.store.count(), 1)
        candidate = self.store.query()[0]
        self.assertTrue(candidate.partial)
        self.assertFalse(candidate.success)

    def test_existing_execution_outcome_contract_is_unchanged_for_callers(self):
        # response_formatter reads execution.get("executed")/["actions"]/
        # ["success"]/["partial"] -- the new fields must be additive only.
        route = {"type": "tool", "tool": "open_application", "arguments": {"app_name": "chrome"}}
        execution_outcome = {}
        with patch.object(agent.executor, "execute_action", return_value=ToolResult(True, "open_application", "Opened chrome successfully.", {"pid": 1, "hwnd": 2, "verified": True})):
            agent.run_agent("open chrome", route=route, execution_outcome=execution_outcome)
        self.assertTrue(execution_outcome["executed"])
        self.assertTrue(execution_outcome["success"])
        self.assertIn("actions", execution_outcome)

    def test_existing_training_data_recording_still_happens(self):
        from training_data import get_recorder
        recorder = get_recorder()
        route = {"type": "tool", "tool": "open_application", "arguments": {"app_name": "chrome"}}
        with patch.object(recorder, "record", wraps=recorder.record) as spy:
            with patch.object(agent.executor, "execute_action", return_value=ToolResult(True, "open_application", "Opened chrome successfully.", {"pid": 1, "hwnd": 2, "verified": True})):
                agent.run_agent("open chrome", route=route)
        self.assertTrue(spy.called)


class ImprovementCandidateSchemaTests(unittest.TestCase):
    def test_round_trip_to_dict_and_back(self):
        candidate = ImprovementCandidate(
            candidate_id="abc", created_at="t", first_seen="t", last_seen="t",
            fingerprint="fp", gap_type=GapType.SKILL_GAP.value,
        )
        restored = ImprovementCandidate.from_dict(candidate.to_dict())
        self.assertEqual(restored, candidate)

    def test_from_dict_ignores_unknown_fields_for_forward_compatibility(self):
        payload = ImprovementCandidate(candidate_id="x", created_at="t", first_seen="t", last_seen="t", fingerprint="fp").to_dict()
        payload["a_field_from_a_future_schema_version"] = "whatever"
        restored = ImprovementCandidate.from_dict(payload)
        self.assertEqual(restored.candidate_id, "x")


if __name__ == "__main__":
    unittest.main()
