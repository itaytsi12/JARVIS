import os
import io
import unittest
from unittest.mock import Mock
from contextlib import redirect_stdout
from pathlib import Path

from tools.browser_agent import BrowserAgent, HumanActionRequired
from brain.agent_runtime import AgentRuntime
from brain.models import PlanStatus
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


if __name__ == "__main__":
    unittest.main()
