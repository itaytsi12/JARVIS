import os
import io
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stdout
from pathlib import Path

from playwright._impl._errors import TargetClosedError as PWTargetClosedError

from tools.browser_agent import BrowserAgent, BrowserSessionError, HumanActionRequired, PageState
from brain.agent_runtime import AgentRuntime
from brain.models import Action, Plan, PlanStatus
from brain.task_planner import create_task_plan, format_plan


class BrowserLocatorSafetyTests(unittest.TestCase):
    def test_close_clears_playwright_even_if_browser_close_fails(self):
        agent=BrowserAgent();agent._browser=Mock();agent._playwright=Mock();playwright=agent._playwright;agent.page=Mock();agent._browser.close.side_effect=RuntimeError("close failed")
        with self.assertRaises(RuntimeError):agent.close()
        playwright.stop.assert_called_once_with()
        self.assertIsNone(agent.page);self.assertIsNone(agent._browser);self.assertIsNone(agent._playwright)
    def test_ambiguous_visible_semantic_target_is_rejected(self):
        visible=Mock();visible.is_visible.return_value=True
        locator=Mock();locator.count.return_value=2;locator.nth.return_value=visible
        page=Mock();page.get_by_role.return_value=locator;page.get_by_label.return_value=locator;page.get_by_placeholder.return_value=locator;page.get_by_text.return_value=locator
        agent=BrowserAgent();agent.page=page
        with self.assertRaisesRegex(LookupError,"Multiple visible"):
            agent._locator("Continue","button")

    def test_unique_visible_semantic_target_is_returned(self):
        visible=Mock();visible.is_visible.return_value=True
        locator=Mock();locator.count.return_value=1;locator.nth.return_value=visible
        page=Mock();page.get_by_role.return_value=locator;page.get_by_label.return_value=locator;page.get_by_placeholder.return_value=locator;page.get_by_text.return_value=locator
        agent=BrowserAgent();agent.page=page
        self.assertIs(agent._locator("Continue","button"),visible)

    def test_ambiguous_css_selector_is_rejected_instead_of_clicking_first(self):
        visible=Mock();visible.is_visible.return_value=True
        locator=Mock();locator.count.return_value=2;locator.nth.return_value=visible
        page=Mock();page.get_by_label.return_value=Mock(count=lambda:0);page.get_by_placeholder.return_value=Mock(count=lambda:0);page.get_by_role.return_value=Mock(count=lambda:0);page.get_by_text.return_value=Mock(count=lambda:0);page.locator.return_value=locator
        agent=BrowserAgent();agent.page=page
        with self.assertRaisesRegex(LookupError,"Multiple visible elements matched selector"):agent._locator(".continue")


class BrowserLifecycleRecoveryTests(unittest.TestCase):
    """Deterministic (mocked-Playwright) coverage of the lifecycle recovery
    and retry policy. Real-Chrome coverage of the same contract lives in
    BrowserLifecycleRealRecoveryTests below (gated behind JARVIS_BROWSER_TESTS)."""

    def _live_agent(self):
        agent = BrowserAgent()
        agent._playwright = Mock()
        agent._browser = Mock()
        agent._browser.is_connected.return_value = True
        agent.page = Mock()
        agent.page.is_closed.return_value = False
        return agent

    def test_healthy_session_is_reused_without_relaunch(self):
        agent = self._live_agent()
        original_page = agent.page
        with patch.object(agent, "_launch") as launch:
            agent.ensure_live_session()
        launch.assert_not_called()
        self.assertIs(agent.page, original_page)

    def test_safe_operation_recovers_and_retries_exactly_once_then_succeeds(self):
        agent = self._live_agent()
        calls = []
        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise PWTargetClosedError()
            return "ok"
        with patch.object(agent, "_discard_session") as discard, patch.object(agent, "ensure_live_session"):
            result = agent._run_retryable(flaky)
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 2)
        discard.assert_called_once()

    def test_safe_operation_fails_truthfully_after_second_lifecycle_error(self):
        agent = self._live_agent()
        calls = []
        def always_fails():
            calls.append(1)
            raise PWTargetClosedError()
        with patch.object(agent, "_discard_session") as discard, patch.object(agent, "ensure_live_session"):
            with self.assertRaises(BrowserSessionError):
                agent._run_retryable(always_fails)
        # Exactly one retry attempt -- never an infinite or repeated loop.
        self.assertEqual(len(calls), 2)
        discard.assert_called_once()

    def test_non_lifecycle_error_is_never_retried(self):
        agent = self._live_agent()
        calls = []
        def raises_lookup():
            calls.append(1)
            raise LookupError("no element matched")
        with patch.object(agent, "ensure_live_session"), patch.object(agent, "_discard_session") as discard:
            with self.assertRaises(LookupError):
                agent._run_retryable(raises_lookup)
        self.assertEqual(len(calls), 1)
        discard.assert_not_called()

    def test_non_retryable_action_recovers_state_but_does_not_retry_the_action(self):
        # browser_click_first_result-style actions: recovery must happen (so
        # the *next* command gets a healthy browser) but this action must
        # never be silently repeated, since its side effect on the lost page
        # can't be known.
        agent = self._live_agent()
        calls = []
        def clicks():
            calls.append(1)
            raise PWTargetClosedError()
        with patch.object(agent, "_discard_session") as discard, patch.object(agent, "ensure_live_session") as ensure:
            with self.assertRaises(BrowserSessionError):
                agent._run_once(clicks)
        self.assertEqual(len(calls), 1)
        discard.assert_called_once()
        # ensure_live_session is called once up front and once during recovery.
        self.assertEqual(ensure.call_count, 2)

    def test_locator_propagates_lifecycle_error_instead_of_treating_it_as_no_match(self):
        # A closed page mid-scan must surface as a lifecycle error so the
        # caller's retry policy can see it -- not be silently swallowed and
        # misreported as "no visible element matched".
        agent = BrowserAgent()
        page = Mock()
        broken = Mock()
        broken.count.side_effect = PWTargetClosedError()
        page.get_by_role.return_value = broken
        page.get_by_label.return_value = broken
        page.get_by_placeholder.return_value = broken
        page.get_by_text.return_value = broken
        agent.page = page
        with self.assertRaises(PWTargetClosedError):
            agent._locator("Continue", "button")

    def test_no_false_success_when_recovery_never_holds(self):
        # End-to-end through the real AgentRuntime execution path: a browser
        # action that can never recover must surface as a failed ToolResult,
        # never success=True. This is the invariant the earlier false-success
        # fix established, and browser recovery must not weaken it.
        agent = self._live_agent()
        with patch.object(agent, "open_url", side_effect=BrowserSessionError("session lost")):
            runtime = AgentRuntime(browser=agent, trace=False)
            plan = Plan("open youtube", [Action("browser_open_url", {"url": "https://www.youtube.com"})])
            results = runtime.execute(plan)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertIn("session lost", str(results[0].error))

    def test_before_snapshot_degrades_gracefully_instead_of_failing_the_action(self):
        # brain/agent_runtime.py's _browser_action reads a "before" page
        # snapshot purely for verification diffing. If the session is
        # already dead at that point, the action itself must still get a
        # chance to recover -- the diagnostic snapshot must not pre-empt it.
        agent = self._live_agent()
        agent.page.is_closed.return_value = True
        real_state = Mock(url="https://www.youtube.com", title="YouTube")
        with patch.object(agent, "open_url", return_value=real_state) as open_url:
            runtime = AgentRuntime(browser=agent, trace=False)
            result = runtime._browser_action("browser_open_url", {"url": "https://www.youtube.com"})
        open_url.assert_called_once()
        self.assertTrue(result.success)


class BrowserFirstResultSelectionTests(unittest.TestCase):
    """Deterministic (mocked-Playwright) coverage of the first-result
    selection contract. Real-DOM coverage (nav exclusion, provider-specific
    selection, real YouTube) lives in BrowserFirstResultRealTests below."""

    def test_destination_matches_rejects_regression_to_homepage(self):
        self.assertFalse(BrowserAgent._destination_matches(
            "https://www.youtube.com/watch?v=abc123", "https://www.youtube.com/"))

    def test_destination_matches_accepts_same_path(self):
        self.assertTrue(BrowserAgent._destination_matches(
            "https://www.youtube.com/watch?v=abc123", "https://www.youtube.com/watch?v=abc123"))

    def test_destination_matches_rejects_cross_domain(self):
        self.assertFalse(BrowserAgent._destination_matches(
            "https://www.youtube.com/watch?v=abc123", "https://evil.example.com/watch?v=abc123"))

    def test_click_first_result_fails_truthfully_when_no_candidate_exists(self):
        agent = BrowserAgent()
        agent._playwright = Mock();agent._browser = Mock();agent._browser.is_connected.return_value = True
        agent.page = Mock();agent.page.is_closed.return_value = False;agent.page.url = "https://example.com/results"
        with patch.object(agent, "_detect_search_provider", return_value=None), \
             patch.object(agent, "_select_generic_first_result", return_value=None):
            with self.assertRaisesRegex(LookupError, "No unambiguous first search result"):
                agent.click_first_result()

    def test_click_first_result_never_reports_success_on_wrong_destination(self):
        # Even if a candidate link is found and clicked without error, a
        # click that lands somewhere other than the selected destination
        # (the exact reported bug: clicking a result but ending up on the
        # homepage) must raise, not silently return a PageState.
        agent = BrowserAgent()
        agent._playwright = Mock();agent._browser = Mock();agent._browser.is_connected.return_value = True
        agent.page = Mock();agent.page.is_closed.return_value = False
        agent.page.url = "https://www.youtube.com/results?search_query=x"
        link = Mock()
        with patch.object(agent, "_detect_search_provider", return_value="youtube"), \
             patch.object(agent, "_select_youtube_video_result", return_value=(link, "/watch?v=abc123", "A Real Video")):
            def _after_click(*a, **k):
                agent.page.url = "https://www.youtube.com/"  # regressed to homepage
            link.click.side_effect = _after_click
            with self.assertRaisesRegex(LookupError, "did not navigate to the expected destination"):
                agent.click_first_result()
        link.click.assert_called_once()

    def test_click_first_result_succeeds_when_destination_matches(self):
        agent = BrowserAgent()
        agent._playwright = Mock();agent._browser = Mock();agent._browser.is_connected.return_value = True
        agent.page = Mock();agent.page.is_closed.return_value = False
        agent.page.url = "https://www.youtube.com/results?search_query=x"
        link = Mock()
        def _after_click(*a, **k):
            agent.page.url = "https://www.youtube.com/watch?v=abc123"
        link.click.side_effect = _after_click
        expected_state = PageState("A Real Video", "https://www.youtube.com/watch?v=abc123")
        with patch.object(agent, "_detect_search_provider", return_value="youtube"), \
             patch.object(agent, "_select_youtube_video_result", return_value=(link, "/watch?v=abc123", "A Real Video")), \
             patch.object(agent, "get_page_state", return_value=expected_state):
            state = agent.click_first_result()
        self.assertEqual(state.url, "https://www.youtube.com/watch?v=abc123")


@unittest.skipUnless(os.getenv("JARVIS_BROWSER_TESTS") == "1", "set JARVIS_BROWSER_TESTS=1 for real browser tests")
class BrowserAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = BrowserAgent(headless=True)
        cls.url = (Path(__file__).parent / "fixtures" / "agent_site.html").resolve().as_uri()

    @classmethod
    def tearDownClass(cls):
        cls.agent.close()

    def setUp(self):
        self.state = self.agent.open_url(self.url)

    def test_observation_is_concise_and_structured(self):
        self.assertEqual(self.state.title, "JARVIS Agent Test")
        self.assertLessEqual(len(self.state.interactive_elements), 30)
        self.assertTrue(any(item["name"] == "Search" for item in self.state.interactive_elements))

    def test_semantic_form_fill_and_dropdown(self):
        self.agent.type_into_field("Username", "test")
        self.agent.type_into_field("Password", "example")
        self.agent.select_option("Options", "B")
        self.assertEqual(self.agent.page.get_by_label("Username").input_value(), "test")
        self.assertEqual(self.agent.page.get_by_label("Password").input_value(), "example")
        self.assertEqual(self.agent.page.get_by_label("Options").input_value(), "B")

    def test_dynamic_element_and_handoff(self):
        self.agent.click_element("Show dynamic element", "button")
        self.agent.wait_for_element("Dynamic content is ready")
        with self.assertRaises(HumanActionRequired):
            self.agent.click_element("Continue", "button")

    def test_runtime_login_flow_redacts_password(self):
        goal = f"Open {self.url}, and log in with username test and password example."
        plan = create_task_plan(goal)
        output = io.StringIO()
        runtime = AgentRuntime(browser=self.agent, trace=True)
        with redirect_stdout(output):
            results = runtime.execute(plan)
        self.assertEqual(plan.status, PlanStatus.COMPLETED)
        self.assertTrue(all(item.success for item in results))
        self.assertNotIn("example", output.getvalue())
        self.assertNotIn("example", format_plan(plan))
        self.assertIn("<REDACTED>", output.getvalue())


@unittest.skipUnless(os.getenv("JARVIS_BROWSER_TESTS") == "1", "set JARVIS_BROWSER_TESTS=1 for real browser tests")
class BrowserLifecycleRealRecoveryTests(unittest.TestCase):
    """Real headless-Chrome coverage of the lifecycle recovery contract --
    exercises Playwright's actual is_connected()/is_closed() semantics
    rather than mocks, using the same local HTML fixture as BrowserAgentTests."""

    def setUp(self):
        self.agent = BrowserAgent(headless=True)
        self.url = (Path(__file__).parent / "fixtures" / "agent_site.html").resolve().as_uri()

    def tearDown(self):
        self.agent.close()

    def test_healthy_page_is_reused_across_calls(self):
        self.agent.open_url(self.url)
        first_page = self.agent.page
        self.agent.open_url(self.url)
        self.assertIs(self.agent.page, first_page)

    def test_page_closed_before_action_recovers_with_a_new_page(self):
        self.agent.open_url(self.url)
        dead_page = self.agent.page
        dead_page.close()
        state = self.agent.open_url(self.url)
        self.assertIsNot(self.agent.page, dead_page)
        self.assertTrue(dead_page.is_closed())
        self.assertFalse(self.agent.page.is_closed())
        self.assertEqual(state.title, "JARVIS Agent Test")

    def test_disconnected_browser_recovers_via_full_relaunch(self):
        self.agent.open_url(self.url)
        dead_browser = self.agent._browser
        dead_browser.close()
        self.assertFalse(dead_browser.is_connected())
        state = self.agent.open_url(self.url)
        self.assertIsNot(self.agent._browser, dead_browser)
        self.assertTrue(self.agent._browser.is_connected())
        self.assertFalse(self.agent.page.is_closed())
        self.assertEqual(state.title, "JARVIS Agent Test")

    def test_stale_references_are_replaced_not_reused_after_full_relaunch(self):
        self.agent.open_url(self.url)
        old_page, old_browser, old_playwright = self.agent.page, self.agent._browser, self.agent._playwright
        old_browser.close()
        self.agent.open_url(self.url)
        self.assertIsNot(self.agent.page, old_page)
        self.assertIsNot(self.agent._browser, old_browser)
        # Never a half-valid mix: whatever combination results, browser and
        # page must agree with each other.
        self.assertTrue(self.agent._browser.is_connected())
        self.assertFalse(self.agent.page.is_closed())

    def test_target_closed_error_during_open_url_reproduces_reported_bug_and_recovers(self):
        # This is the exact live failure mode from the bug report: the page
        # was closed out from under browser_open_url. ensure_live_session's
        # proactive check catches it before Playwright ever raises, and the
        # command succeeds instead of surfacing TargetClosedError forever.
        self.agent.open_url(self.url)
        self.agent.page.close()
        state = self.agent.open_url(self.url)
        self.assertEqual(state.title, "JARVIS Agent Test")

    def test_recovery_never_reports_false_success_when_relaunch_itself_fails(self):
        self.agent.open_url(self.url)
        self.agent._browser.close()
        with patch.object(self.agent, "_launch", side_effect=RuntimeError("no browser available")):
            with self.assertRaises(RuntimeError):
                self.agent.open_url(self.url)
        # A failed relaunch must not leave a half-valid session either.
        self.assertIsNone(self.agent.page)
        self.assertIsNone(self.agent._browser)

    def test_normal_browser_action_sequence_still_works_after_lifecycle_changes(self):
        self.agent.open_url(self.url)
        self.agent.click_element("Show dynamic element", "button")
        self.agent.wait_for_element("Dynamic content is ready")
        self.agent.type_into_field("Username", "test")
        self.assertEqual(self.agent.page.get_by_label("Username").input_value(), "test")


@unittest.skipUnless(os.getenv("JARVIS_BROWSER_TESTS") == "1", "set JARVIS_BROWSER_TESTS=1 for real browser tests")
class BrowserFirstResultRealTests(unittest.TestCase):
    """Real headless-Chrome coverage of first-result selection, using local
    fixtures that deliberately place navigation/chrome links ahead of the
    actual results -- the exact shape that broke the old "first visible <a>"
    logic."""

    def setUp(self):
        self.agent = BrowserAgent(headless=True)
        self.fixtures = Path(__file__).parent / "fixtures"

    def tearDown(self):
        self.agent.close()

    def test_generic_fallback_ignores_nav_links_and_selects_first_main_result(self):
        # agent_site.html has a <nav> with several non-fragment links (About)
        # positioned before <main>'s own "results" section -- the generic
        # fallback must land on the first result, never the nav chrome.
        url = (self.fixtures / "agent_site.html").resolve().as_uri()
        self.agent.open_url(url)
        state = self.agent.click_first_result()
        self.assertEqual(state.title, "First Result Article")
        self.assertIn("first-result.html", self.agent.page.url)

    def test_youtube_like_fixture_selects_first_video_ignoring_channel_and_filters(self):
        # Deterministic, non-network coverage of the YouTube-specific
        # strategy: header nav, filter chips, a channel link, and a playlist
        # link all appear before the real video results.
        url = (self.fixtures / "youtube_like_results.html").resolve().as_uri()
        self.agent.open_url(url)
        selection = self.agent._select_youtube_video_result()
        self.assertIsNotNone(selection)
        _link, href, text = selection
        self.assertIn("/watch?v=abc123def45", href)
        self.assertEqual(text, "First Real Video Result")

    def test_no_candidates_fails_truthfully(self):
        url = (self.fixtures / "no_results.html").resolve().as_uri()
        self.agent.open_url(url)
        with self.assertRaisesRegex(LookupError, "No unambiguous first search result"):
            self.agent.click_first_result()

    def test_real_youtube_search_selects_first_video_not_homepage(self):
        # End-to-end reproduction of the exact reported bug against the real
        # site: search, click the first result, and land on an actual video
        # -- never youtube.com's homepage. Network-dependent.
        self.agent.open_url("https://www.youtube.com/results?search_query=Minecraft+Redstone+Tutorial")
        state = self.agent.click_first_result()
        self.assertIn("/watch?v=", self.agent.page.url)
        self.assertNotEqual(self.agent.page.url.rstrip("/"), "https://www.youtube.com")
        self.assertTrue(state.title)


if __name__ == "__main__":
    unittest.main()
