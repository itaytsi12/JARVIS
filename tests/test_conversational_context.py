"""Regression tests for brain/conversational_context.py and its wiring
into brain/router.py (bug reports 1 and 2 from the live conversational-
context test):

1. "What does that mean?" (and its variants) must resolve against the
   most relevant recent assistant output instead of going blind to
   web_answer/GPT.
2. A short browser search correction ("Batman instead.") must resolve
   deterministically to the same browser search, with zero model calls,
   whenever session context has an active, resolvable browser search.
"""
import unittest

from brain.conversational_context import (
    is_explanatory_followup,
    resolve_browser_search_correction,
    resolve_explanatory_followup,
    resolve_recent_referent,
)
from brain.router import route_command
from brain.session_context import SessionContext


class ExplanatoryFollowupDetectionTests(unittest.TestCase):
    def test_recognizes_the_reported_phrasings(self):
        for text in [
            "What does that mean?",
            "what does that mean",
            "Why?",
            "why",
            "Why did that happen?",
            "Explain that.",
            "explain this",
            "What happened?",
            "What does this mean?",
            "Tell me more about that.",
        ]:
            with self.subTest(text=text):
                self.assertTrue(is_explanatory_followup(text))

    def test_does_not_hijack_a_self_contained_question(self):
        for text in [
            "why is Chrome using so much memory",
            "explain the theory of relativity",
            "what happened in the news today",
            "tell me more about black holes",
        ]:
            with self.subTest(text=text):
                self.assertFalse(is_explanatory_followup(text))


class ResolveExplanatoryFollowupTests(unittest.TestCase):
    def test_resolves_against_last_assistant_response(self):
        context = SessionContext(last_assistant_response="git status shows 3 modified files and 2 untracked files.")
        route = resolve_explanatory_followup("What does that mean?", context)
        self.assertIsNotNone(route)
        self.assertEqual(route["type"], "contextual_question")
        self.assertEqual(route["route_source"], "conversational_context")
        self.assertIn("3 modified files", route["context_text"])

    def test_falls_back_to_last_spoken_response_when_no_assistant_response(self):
        context = SessionContext(last_spoken_response="Opened YouTube, sir.")
        route = resolve_explanatory_followup("Why?", context)
        self.assertIsNotNone(route)
        self.assertEqual(route["context_text"], "Opened YouTube, sir.")

    def test_returns_none_without_a_real_referent(self):
        self.assertIsNone(resolve_explanatory_followup("What does that mean?", SessionContext()))
        self.assertIsNone(resolve_explanatory_followup("What does that mean?", None))

    def test_returns_none_for_a_non_explanatory_command(self):
        context = SessionContext(last_assistant_response="Opened YouTube, sir.")
        self.assertIsNone(resolve_explanatory_followup("Open Notepad", context))

    def test_resolve_recent_referent_prefers_assistant_response(self):
        context = SessionContext(last_assistant_response="A", last_spoken_response="B")
        self.assertEqual(resolve_recent_referent(context), "A")


class RouterExplanatoryFollowupIntegrationTests(unittest.TestCase):
    """Sequence A from the bug report, at the router level."""

    def test_route_command_resolves_context_before_question_classification(self):
        context = SessionContext(last_assistant_response="Git status shows the working tree is clean.")
        route = route_command("What does that mean?", context)
        self.assertEqual(route["type"], "contextual_question")
        self.assertNotEqual(route["type"], "question")

    def test_route_command_without_context_falls_back_to_question(self):
        route = route_command("What does that mean?")
        self.assertEqual(route["type"], "question")

    def test_route_command_with_empty_context_falls_back_to_question(self):
        route = route_command("What does that mean?", SessionContext())
        self.assertEqual(route["type"], "question")

    def test_ordinary_action_command_is_unaffected_by_context(self):
        context = SessionContext(last_assistant_response="Something happened.")
        route = route_command("open notepad", context)
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route["tool"], "open_application")


class BrowserSearchCorrectionTests(unittest.TestCase):
    def _youtube_context(self, query="Iron Man"):
        return SessionContext(browser_active=True, last_search_provider="youtube", last_search_query=query)

    def test_bare_instead_resolves_to_new_youtube_search(self):
        route = resolve_browser_search_correction("Batman instead.", self._youtube_context())
        self.assertIsNotNone(route)
        self.assertEqual(route["type"], "local_plan")
        self.assertEqual(route["route_source"], "conversational_context_browser_correction")
        action = route["actions"][0]
        self.assertEqual(action.tool, "browser_open_url")
        self.assertIn("youtube.com/results", action.args["url"])
        self.assertIn("Batman", action.args["url"])

    def test_search_for_x_instead_resolves(self):
        route = resolve_browser_search_correction("Search for Batman instead.", self._youtube_context())
        self.assertIsNotNone(route)
        self.assertIn("Batman", route["actions"][0].args["url"])

    def test_try_x_instead_resolves(self):
        route = resolve_browser_search_correction("Try Superman instead.", self._youtube_context())
        self.assertIsNotNone(route)
        self.assertIn("Superman", route["actions"][0].args["url"])

    def test_change_that_to_x_resolves(self):
        route = resolve_browser_search_correction("Change that to Batman.", self._youtube_context())
        self.assertIsNotNone(route)
        self.assertIn("Batman", route["actions"][0].args["url"])

    def test_reuses_google_template_when_that_was_the_last_provider(self):
        context = SessionContext(browser_active=True, last_search_provider="google", last_search_query="cats")
        route = resolve_browser_search_correction("dogs instead.", context)
        self.assertIsNotNone(route)
        self.assertIn("google.com/search", route["actions"][0].args["url"])

    def test_ambiguous_something_else_is_left_for_the_agent_runtime(self):
        # No concrete replacement query -- must NOT be resolved locally, so
        # it falls through to the pre-existing (eventually agent-runtime)
        # routing instead of guessing.
        route = resolve_browser_search_correction("Search for something else.", self._youtube_context())
        self.assertIsNone(route)

    def test_no_browser_active_falls_through(self):
        context = SessionContext(browser_active=False, last_search_provider="youtube", last_search_query="Iron Man")
        self.assertIsNone(resolve_browser_search_correction("Batman instead.", context))

    def test_unknown_provider_falls_through(self):
        context = SessionContext(browser_active=True, last_search_provider="netflix", last_search_query="Iron Man")
        self.assertIsNone(resolve_browser_search_correction("Batman instead.", context))

    def test_none_context_returns_none(self):
        self.assertIsNone(resolve_browser_search_correction("Batman instead.", None))

    def test_whatsapp_recipient_correction_is_not_hijacked(self):
        # "send it to Alex instead" is brain/router.py's dedicated
        # `revise_whatsapp_recipient` pattern -- must never be captured as
        # a browser search, even with a coincidentally-active browser.
        route = resolve_browser_search_correction("send it to Alex instead", self._youtube_context())
        self.assertIsNone(route)


class RouterBrowserCorrectionIntegrationTests(unittest.TestCase):
    """Sequence B from the bug report, at the router level."""

    def test_route_command_resolves_browser_correction_deterministically(self):
        context = SessionContext(browser_active=True, last_search_provider="youtube", last_search_query="Iron Man")
        route = route_command("Batman instead.", context)
        self.assertEqual(route["type"], "local_plan")
        self.assertEqual(route["route_source"], "conversational_context_browser_correction")

    def test_whatsapp_recipient_correction_still_routes_correctly_with_browser_context(self):
        context = SessionContext(browser_active=True, last_search_provider="youtube", last_search_query="Iron Man")
        route = route_command("send it to Alex instead", context)
        self.assertEqual(route["type"], "revise_whatsapp_recipient")
        self.assertEqual(route["recipient"], "alex")

    def test_without_browser_context_falls_through_to_agent_escalation_path(self):
        # No browser_active -> not resolved locally. It should NOT crash
        # and should not be misrouted as a browser action.
        route = route_command("Batman instead.")
        self.assertNotEqual(route.get("route_source"), "conversational_context_browser_correction")


if __name__ == "__main__":
    unittest.main()
