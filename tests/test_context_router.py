"""Tests for brain/context_router.py and its wiring into
brain/router.py::route_command(command, context=...).

These exercise ROUTING only (what route dict comes back), not execution --
see tests/test_conversational_context_integration.py for the full
run_agent()-level behavior (no-model-call guarantees, AgentRuntime context
handoff, dataset logging).
"""
import unittest
from unittest.mock import patch

from brain import router
from brain.router import route_command
from brain.session_context import SessionContext


def _no_cloud_fallback():
    """Nothing in these tests should ever reach the paid cloud intent
    classifier -- a "falls through" assertion only needs to know that
    context_router declined, not what the (mocked, cost-free) eventual
    fallback route looked like."""
    return patch.object(router, "classify_intent", return_value={"type": "ai", "message": "unused"})


def _opened(context, name, browser=False):
    context.bump_turn()
    if browser:
        context.browser_active = True
        context.active_app = "browser"
    context.record_app_event(name, "opened")


class BackwardCompatibilityTests(unittest.TestCase):
    """No context passed -- every existing caller must see zero change."""

    def test_close_it_without_context_falls_back_to_legacy_literal_behavior(self):
        route = route_command("close it")
        self.assertEqual(route, {"type": "tool", "tool": "close_application", "arguments": {"app_name": "it"}})

    def test_ordinary_commands_are_unaffected_by_the_context_kwarg_existing(self):
        self.assertEqual(route_command("open notepad")["tool"], "open_application")
        self.assertEqual(route_command("open notepad", context=None)["tool"], "open_application")


class ElliperticalAppActionTests(unittest.TestCase):
    def test_close_it_resolves_to_the_only_recently_opened_app(self):
        context = SessionContext()
        _opened(context, "spotify")
        route = route_command("close it", context=context)
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route["tool"], "close_application")
        self.assertEqual(route["arguments"], {"app_name": "spotify"})
        self.assertEqual(route["route_source"], "context_pronoun")

    def test_quit_that_resolves_the_same_as_close_it(self):
        context = SessionContext()
        _opened(context, "spotify")
        route = route_command("quit that", context=context)
        self.assertEqual(route["tool"], "close_application")
        self.assertEqual(route["arguments"], {"app_name": "spotify"})

    def test_close_it_prefers_the_most_recently_opened_of_two_apps(self):
        context = SessionContext()
        _opened(context, "chrome")
        _opened(context, "spotify")
        route = route_command("close it", context=context)
        self.assertEqual(route["arguments"], {"app_name": "spotify"})

    def test_close_it_asks_for_clarification_when_two_apps_share_a_turn(self):
        context = SessionContext()
        context.bump_turn()
        context.record_app_event("chrome", "opened")
        context.record_app_event("spotify", "opened")
        route = route_command("close it", context=context)
        self.assertEqual(route["type"], "clarification")
        self.assertIn("chrome", route["message"].lower())
        self.assertIn("spotify", route["message"].lower())

    def test_close_it_with_no_recent_app_falls_through_to_legacy_behavior(self):
        context = SessionContext()
        route = route_command("close it", context=context)
        # Nothing contextual resolved -> legacy literal fallback, unchanged.
        self.assertEqual(route, {"type": "tool", "tool": "close_application", "arguments": {"app_name": "it"}})

    def test_focus_it_maps_to_focus_application(self):
        context = SessionContext()
        _opened(context, "notepad")
        route = route_command("focus it", context=context)
        self.assertEqual(route["tool"], "focus_application")
        self.assertEqual(route["arguments"], {"app_name": "notepad"})


class SearchContinuationTests(unittest.TestCase):
    def test_open_chrome_then_bare_search_uses_the_browser(self):
        context = SessionContext()
        _opened(context, "chrome", browser=True)
        route = route_command("search for cats", context=context)
        self.assertEqual(route["type"], "local_plan")
        action = route["actions"][0]
        self.assertEqual(action.tool, "browser_open_url")
        self.assertIn("google.com/search", action.args["url"])
        self.assertIn("cats", action.args["url"])

    def test_search_instead_reuses_the_provider_from_the_first_search(self):
        context = SessionContext()
        _opened(context, "browser", browser=True)
        context.last_search_provider = "youtube"
        with _no_cloud_fallback():
            route = route_command("search for Batman instead", context=context)
        action = route["actions"][0]
        self.assertEqual(action.tool, "browser_open_url")
        self.assertIn("youtube.com/results", action.args["url"])
        self.assertIn("batman", action.args["url"])

    def test_provider_qualified_search_is_left_to_the_existing_deeper_pattern(self):
        context = SessionContext()
        context.browser_active = True
        route = route_command("search google for cats", context=context)
        # Falls through untouched to brain/router.py's own provider-aware
        # search_patterns -- never intercepted here.
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route["tool"], "open_website")
        self.assertIn("google.com/search?q=cats", route["arguments"]["url"])

    def test_search_without_an_active_browser_is_not_intercepted(self):
        context = SessionContext()
        with _no_cloud_fallback():
            route = route_command("search for cats", context=context)
        self.assertNotEqual(route.get("route_source"), "context_search_continuation")


class OrdinalFileReferenceTests(unittest.TestCase):
    def test_open_the_second_one_resolves_against_the_last_file_listing(self):
        context = SessionContext()
        context.record_result_set([("a.py", "C:/repo/a.py"), ("b.py", "C:/repo/b.py")], source="list_files", kind="file")
        route = route_command("open the second one", context=context)
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route["tool"], "open_path")
        self.assertEqual(route["arguments"], {"path": "C:/repo/b.py"})

    def test_ordinal_with_no_result_set_falls_through(self):
        context = SessionContext()
        route = route_command("open the second one", context=context)
        self.assertNotEqual(route.get("route_source"), "context_ordinal")

    def test_fix_the_first_one_escalates_to_agent_with_resolved_test_name(self):
        context = SessionContext()
        context.record_result_set(
            [("FAILED tests/test_a.py::test_one - AssertionError", "tests/test_a.py::test_one")],
            source="run_command", kind="test_failure",
        )
        route = route_command("fix the first one", context=context)
        self.assertEqual(route["type"], "agent_task")
        self.assertIn("tests/test_a.py::test_one", route["goal"])
        self.assertEqual(route["route_source"], "context_ordinal_followup")


class ReplayTests(unittest.TestCase):
    def test_do_it_again_replays_the_exact_previous_command(self):
        context = SessionContext()
        context.record_command("open_application", {"app_name": "spotify"})
        route = route_command("do it again", context=context)
        self.assertEqual(route, {"type": "tool", "tool": "open_application", "arguments": {"app_name": "spotify"}, "route_source": "context_replay"})

    def test_try_again_without_a_previous_command_falls_through(self):
        with _no_cloud_fallback():
            route = route_command("try again", context=SessionContext())
        self.assertNotEqual(route.get("route_source"), "context_replay")


class CorrectionTests(unittest.TestCase):
    def test_no_i_meant_corrects_the_previous_open_application(self):
        context = SessionContext()
        context.record_command("open_application", {"app_name": "discord"})
        route = route_command("no, I meant Telegram", context=context)
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route["tool"], "open_application")
        self.assertEqual(route["arguments"], {"app_name": "telegram"})
        self.assertEqual(route["corrects_previous"], {"app_name": "discord"})

    def test_actually_use_corrects_the_previous_open_application(self):
        context = SessionContext()
        context.record_command("open_application", {"app_name": "discord"})
        route = route_command("actually use Firefox", context=context)
        self.assertEqual(route["arguments"], {"app_name": "firefox"})

    def test_correction_without_a_previous_command_falls_through(self):
        with _no_cloud_fallback():
            route = route_command("no, I meant Telegram", context=SessionContext())
        self.assertNotEqual(route.get("route_source"), "context_correction")

    def test_correction_does_not_get_swallowed_by_the_comma_signal(self):
        # "no, I meant Telegram" contains a bare comma -- must not silently
        # fall through to a completely different (or missing) route.
        context = SessionContext()
        context.record_command("open_application", {"app_name": "discord"})
        route = route_command("no, I meant Telegram", context=context)
        self.assertEqual(route["type"], "tool")


class ProjectOpenTests(unittest.TestCase):
    def test_open_my_jarvis_project_resolves_to_a_real_path(self):
        route = route_command("open my JARVIS project", context=SessionContext())
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route["tool"], "open_path")
        self.assertTrue(route["arguments"]["path"])
        self.assertEqual(route["project_name"], "jarvis")

    def test_open_unknown_project_falls_through(self):
        route = route_command("open the frobnicator project", context=SessionContext())
        self.assertNotEqual(route.get("route_source"), "context_project_open")

    def test_run_it_after_opening_a_project_escalates_with_the_resolved_path(self):
        context = SessionContext()
        context.bump_turn()
        context.record_project("jarvis", "C:/repo/jarvis")
        route = route_command("run it", context=context)
        self.assertEqual(route["type"], "agent_task")
        self.assertEqual(route["route_source"], "context_followup_run")
        self.assertEqual(route["resolved_context"]["project_path"], "C:/repo/jarvis")


class ReasoningFollowupTests(unittest.TestCase):
    def test_why_did_that_fail_escalates_with_the_resolved_error(self):
        context = SessionContext()
        context.record_task("t1", "Run git status", "failed", error="fatal: not a git repository")
        route = route_command("why did that fail?", context=context)
        self.assertEqual(route["type"], "agent_task")
        self.assertEqual(route["resolved_context"]["previous_task_error"], "fatal: not a git repository")

    def test_what_does_that_mean_escalates_with_resolved_context(self):
        context = SessionContext()
        context.record_task("t1", "Run git status", "completed", result_summary="On branch master, nothing to commit")
        route = route_command("what does that mean?", context=context)
        self.assertEqual(route["type"], "agent_task")
        self.assertIn("previous_task_result_summary", route["resolved_context"])

    def test_why_did_that_fail_with_no_context_falls_through(self):
        route = route_command("why did that fail?", context=SessionContext())
        self.assertNotEqual(route.get("route_source"), "context_followup_reasoning")


if __name__ == "__main__":
    unittest.main()
