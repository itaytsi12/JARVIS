"""Unit tests for brain/context_resolver.py -- the general, structured
conversational-reference resolver. No router/agent wiring here; see
tests/test_context_router.py and tests/test_conversational_context_integration.py
for the wired-up behavior."""
import time
import unittest

from brain.context_resolver import (
    classify_reference_shape,
    extract_result_items,
    gather_candidates,
    observe_tool_result,
    resolve_correction,
    resolve_ordinal,
    resolve_reference,
    resolve_replay,
    resolved_context_summary,
)
from brain.models import ToolResult
from brain.session_context import SessionContext


class ShapeClassificationTests(unittest.TestCase):
    def test_pronouns(self):
        for word in ("it", "It.", "this", "that?", "them", "there"):
            with self.subTest(word=word):
                self.assertEqual(classify_reference_shape(word), "pronoun")

    def test_demonstratives(self):
        for phrase in ("the app", "this window", "that project", "the file", "this browser", "the task"):
            with self.subTest(phrase=phrase):
                self.assertEqual(classify_reference_shape(phrase), "demonstrative")

    def test_ordinals(self):
        for phrase in ("the first one", "second", "the 3rd result", "the last one", "tenth file"):
            with self.subTest(phrase=phrase):
                self.assertEqual(classify_reference_shape(phrase), "ordinal")

    def test_again(self):
        for phrase in ("do it again", "try again", "again", "repeat that", "once more"):
            with self.subTest(phrase=phrase):
                self.assertEqual(classify_reference_shape(phrase), "again")

    def test_corrections(self):
        for phrase in ("no, I meant Telegram", "actually use Firefox", "actually, I meant Spotify", "not that one, I meant Chrome"):
            with self.subTest(phrase=phrase):
                self.assertEqual(classify_reference_shape(phrase), "correction")

    def test_unrelated_sentence_is_unclassified(self):
        self.assertIsNone(classify_reference_shape("open the JARVIS project"))
        self.assertIsNone(classify_reference_shape(""))
        self.assertIsNone(classify_reference_shape("   "))


class RecencyByTurnTests(unittest.TestCase):
    """The core recency mechanism: TURN order decides, not wall-clock gaps."""

    def test_most_recently_opened_app_wins_across_separate_turns(self):
        context = SessionContext()
        context.bump_turn()
        context.record_app_event("Chrome", "opened")
        context.bump_turn()
        context.record_app_event("Spotify", "opened")
        resolution = resolve_reference("it", context, {"application"})
        self.assertTrue(resolution.success)
        self.assertEqual(resolution.value, "Spotify")
        self.assertEqual(resolution.method, "resolved")

    def test_two_apps_opened_in_the_same_turn_are_ambiguous(self):
        context = SessionContext()
        context.bump_turn()
        context.record_app_event("Chrome", "opened")
        context.record_app_event("Spotify", "opened")
        resolution = resolve_reference("it", context, {"application"})
        self.assertFalse(resolution.success)
        self.assertEqual(resolution.method, "ambiguous")
        self.assertEqual({c.label for c in resolution.candidates}, {"Chrome", "Spotify"})
        self.assertIn("Chrome", resolution.clarification_question)
        self.assertIn("Spotify", resolution.clarification_question)

    def test_closing_an_app_removes_it_from_the_open_stack(self):
        context = SessionContext()
        context.bump_turn()
        context.record_app_event("Chrome", "opened")
        context.bump_turn()
        context.record_app_event("Chrome", "closed")
        context.bump_turn()
        context.record_app_event("Spotify", "opened")
        resolution = resolve_reference("it", context, {"application"})
        self.assertTrue(resolution.success)
        self.assertEqual(resolution.value, "Spotify")

    def test_closed_app_alone_leaves_nothing_to_resolve(self):
        context = SessionContext()
        context.bump_turn()
        context.record_app_event("Chrome", "opened")
        context.bump_turn()
        context.record_app_event("Chrome", "closed")
        resolution = resolve_reference("it", context, {"application"})
        self.assertFalse(resolution.success)
        self.assertEqual(resolution.method, "not_found")

    def test_no_context_at_all_is_not_found_not_a_crash(self):
        resolution = resolve_reference("it", SessionContext())
        self.assertFalse(resolution.success)
        self.assertEqual(resolution.method, "not_found")

    def test_non_reference_phrase_never_resolves(self):
        context = SessionContext()
        context.record_app_event("Chrome", "opened")
        resolution = resolve_reference("open the JARVIS project", context)
        self.assertFalse(resolution.success)


class DemonstrativeTypeInferenceTests(unittest.TestCase):
    def test_the_project_only_matches_project_candidates(self):
        context = SessionContext()
        context.bump_turn()
        context.record_app_event("Chrome", "opened")
        context.record_project("jarvis", "/repo/jarvis")
        resolution = resolve_reference("the project", context)
        self.assertTrue(resolution.success)
        self.assertEqual(resolution.entity_type, "project")
        self.assertEqual(resolution.value, "/repo/jarvis")

    def test_the_browser_resolves_to_browser_even_with_a_more_recent_app(self):
        context = SessionContext()
        context.bump_turn()
        context.browser_active = True
        context.record_app_event("browser", "opened")
        context.bump_turn()
        context.record_app_event("Spotify", "opened")
        resolution = resolve_reference("the browser", context)
        self.assertTrue(resolution.success)
        self.assertEqual(resolution.entity_type, "browser")


class TTLExpiryTests(unittest.TestCase):
    def test_stale_app_reference_is_not_used(self):
        context = SessionContext()
        context.bump_turn()
        context.record_app_event("Chrome", "opened")
        context.recent_apps[-1].at -= 100000  # far beyond APP_REFERENCE_TTL_SECONDS
        resolution = resolve_reference("it", context, {"application"})
        self.assertFalse(resolution.success)

    def test_stale_result_set_does_not_answer_an_ordinal(self):
        context = SessionContext()
        context.record_result_set([("a.txt", "a.txt"), ("b.txt", "b.txt")], source="list_files", kind="file")
        context.last_result_set.created_at -= 100000
        resolution = resolve_ordinal("the first one", context)
        self.assertFalse(resolution.success)
        self.assertEqual(resolution.error, "result_set_expired")

    def test_stale_previous_command_blocks_a_correction(self):
        context = SessionContext()
        context.record_command("open_application", {"app_name": "discord"})
        context.last_command_time -= 100000
        resolution = resolve_correction("no, I meant Telegram", context)
        self.assertFalse(resolution.success)
        self.assertEqual(resolution.error, "previous_command_expired")


class OrdinalResolutionTests(unittest.TestCase):
    def test_first_second_last_resolve_correctly(self):
        context = SessionContext()
        context.record_result_set([("a.txt", "a.txt"), ("b.txt", "b.txt"), ("c.txt", "c.txt")], source="list_files", kind="file")
        self.assertEqual(resolve_ordinal("the first one", context).value, "a.txt")
        self.assertEqual(resolve_ordinal("the second one", context).value, "b.txt")
        self.assertEqual(resolve_ordinal("the last one", context).value, "c.txt")

    def test_out_of_range_ordinal_fails_cleanly(self):
        context = SessionContext()
        context.record_result_set([("a.txt", "a.txt")], source="list_files", kind="file")
        resolution = resolve_ordinal("the third one", context)
        self.assertFalse(resolution.success)
        self.assertEqual(resolution.error, "ordinal_out_of_range")

    def test_no_result_set_fails_cleanly(self):
        resolution = resolve_ordinal("the first one", SessionContext())
        self.assertFalse(resolution.success)
        self.assertEqual(resolution.error, "no_recent_result_set")

    def test_kind_mismatch_is_rejected_rather_than_guessed(self):
        context = SessionContext()
        context.record_result_set([("FAILED tests/test_x.py::test_y", "tests/test_x.py::test_y")], source="run_command", kind="test_failure")
        resolution = resolve_ordinal("the first one", context, kind="file")
        self.assertFalse(resolution.success)
        self.assertEqual(resolution.error, "result_set_type_mismatch")

    def test_a_new_result_set_replaces_the_old_one_wholesale(self):
        context = SessionContext()
        context.record_result_set([("a.txt", "a.txt")], source="list_files", kind="file")
        context.record_result_set([("b.txt", "b.txt"), ("c.txt", "c.txt")], source="list_files", kind="file")
        self.assertEqual(resolve_ordinal("the first one", context).value, "b.txt")
        self.assertEqual(len(context.last_result_set.items), 2)


class CorrectionAndReplayTests(unittest.TestCase):
    def test_correction_carries_the_previous_tool_and_new_value(self):
        context = SessionContext()
        context.record_command("open_application", {"app_name": "discord"})
        resolution = resolve_correction("no, I meant Telegram", context)
        self.assertTrue(resolution.success)
        self.assertEqual(resolution.value["tool"], "open_application")
        self.assertEqual(resolution.value["previous_args"], {"app_name": "discord"})
        self.assertEqual(resolution.value["new_value"], "Telegram")

    def test_correction_without_a_previous_command_fails_cleanly(self):
        resolution = resolve_correction("no, I meant Telegram", SessionContext())
        self.assertFalse(resolution.success)
        self.assertEqual(resolution.error, "no_previous_command")

    def test_replay_returns_the_exact_previous_tool_call(self):
        context = SessionContext()
        context.record_command("open_website", {"url": "https://www.youtube.com"})
        resolution = resolve_replay(context)
        self.assertTrue(resolution.success)
        self.assertEqual(resolution.value, {"tool": "open_website", "args": {"url": "https://www.youtube.com"}})


class ExtractResultItemsTests(unittest.TestCase):
    def test_pytest_failed_lines_are_extracted_in_order(self):
        output = (
            "collected 3 items\n"
            "FAILED tests/test_a.py::test_one - AssertionError: boom\n"
            "FAILED tests/test_b.py::test_two - ValueError: bad\n"
            "2 failed, 1 passed\n"
        )
        items = extract_result_items(output)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0][1], "tests/test_a.py::test_one")
        self.assertEqual(items[1][1], "tests/test_b.py::test_two")

    def test_numbered_list_is_extracted(self):
        output = "Here are the files:\n1. main.py\n2. utils.py\n3. tests.py\n"
        items = extract_result_items(output)
        self.assertEqual([label for label, _ in items], ["main.py", "utils.py", "tests.py"])

    def test_unstructured_output_yields_nothing(self):
        self.assertEqual(extract_result_items("just some free text, no list here"), [])
        self.assertEqual(extract_result_items(""), [])


class ObserveToolResultTests(unittest.TestCase):
    def test_list_files_populates_a_file_result_set(self):
        context = SessionContext()
        result = ToolResult(True, "list_files", "ok", {"path": "C:/repo", "items": ["a.py", "b.py"], "verified": True})
        observe_tool_result(context, "list_files", {"path": "C:/repo"}, result)
        self.assertIsNotNone(context.last_result_set)
        self.assertEqual(context.last_result_set.kind, "file")
        self.assertEqual([item.value for item in context.last_result_set.items], ["a.py", "b.py"])

    def test_failing_run_command_populates_a_test_failure_result_set_despite_nonzero_exit(self):
        context = SessionContext()
        stdout = "FAILED tests/test_a.py::test_one - AssertionError\nFAILED tests/test_b.py::test_two - ValueError\n"
        result = ToolResult(False, "run_command", "2 failed", {"stdout": stdout, "stderr": "", "exit_code": 1, "verified": True}, error="tests_failed")
        observe_tool_result(context, "run_command", {"command": "pytest"}, result)
        self.assertIsNotNone(context.last_result_set)
        self.assertEqual(context.last_result_set.kind, "test_failure")
        self.assertEqual(len(context.last_result_set.items), 2)
        # A failed command is ALSO a real error -- both must be true.
        self.assertEqual(context.last_error, "tests_failed")

    def test_successful_command_with_no_list_shaped_output_touches_no_result_set(self):
        context = SessionContext()
        result = ToolResult(True, "run_command", "ok", {"stdout": "on branch master\nnothing to commit\n", "stderr": ""})
        observe_tool_result(context, "run_command", {"command": "git status"}, result)
        self.assertIsNone(context.last_result_set)
        self.assertEqual(context.last_command_tool, "run_command")

    def test_never_raises_on_malformed_result(self):
        context = SessionContext()
        observe_tool_result(context, "run_command", {}, object())  # not a ToolResult at all
        observe_tool_result(context, "list_files", None, None)


class ResolvedContextSummaryTests(unittest.TestCase):
    def test_summary_includes_task_project_error_and_result_set(self):
        context = SessionContext()
        context.record_project("jarvis", "/repo/jarvis")
        context.record_task("t1", "Run the tests", "failed", result_summary="2 failed, 3 passed", error="pytest exited 1")
        context.record_result_set([("FAILED a - x", "a")], source="run_command", kind="test_failure")
        summary = resolved_context_summary(context)
        self.assertEqual(summary["project_path"], "/repo/jarvis")
        self.assertEqual(summary["previous_task_goal"], "Run the tests")
        self.assertEqual(summary["previous_task_status"], "failed")
        self.assertEqual(summary["previous_task_error"], "pytest exited 1")
        self.assertEqual(summary["last_result_set"]["kind"], "test_failure")

    def test_empty_context_yields_empty_summary(self):
        self.assertEqual(resolved_context_summary(SessionContext()), {})

    def test_expired_error_is_excluded(self):
        context = SessionContext()
        context.record_error("boom", source="run_command")
        context.last_error_at -= 100000
        self.assertNotIn("last_error", resolved_context_summary(context))


if __name__ == "__main__":
    unittest.main()
