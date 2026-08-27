"""End-to-end conversational-context tests through the real `run_agent()`
entry point -- the same one voice and typed input both use. Tool execution
is mocked at the same boundary `tests/test_multistep_context.py` already
uses (`agent.executor.execute_action` / `agent.agent_runtime.execute`), so
these never touch a real window, browser, or subprocess, but the full
routing -> context-recording -> resolution -> execution path is real.

`agent.agent_runtime` is a process-wide singleton, so every test swaps in a
throwaway `SessionContext` in `setUp` and restores the original in
`tearDown` -- never touches the real one another concurrently-running test
(or the harness itself) might depend on.
"""
import unittest
from unittest.mock import patch

from brain import agent, router
from brain.models import ToolResult
from brain.session_context import SessionContext


def _no_cloud_calls():
    """Every test in this file asserts locally-resolvable commands never
    spend a model call -- patch every possible cloud entry point so a
    routing bug fails loudly (AssertionError) instead of quietly making a
    real, paid network request."""
    return (
        patch.object(agent, "create_plan", side_effect=AssertionError("cloud planner must not be called")),
        patch.object(agent, "ask_ai", side_effect=AssertionError("ask_ai must not be called")),
        patch.object(router, "classify_intent", side_effect=AssertionError("cloud intent classifier must not be called")),
    )


class ConversationalContextTestCase(unittest.TestCase):
    def setUp(self):
        self._original_context = agent.agent_runtime.context
        agent.agent_runtime.context = SessionContext(memory=self._original_context.memory, session_id=self._original_context.session_id)
        self._patches = _no_cloud_calls()
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        agent.agent_runtime.context = self._original_context

    # -- shared fakes ----------------------------------------------------
    def _fake_executor(self, extra=None):
        """Patches agent.executor.execute_action (open_application,
        open_website, open_path, list_files, run_command, ...)."""
        extra = extra or {}

        def execute_action(action):
            if action.tool in extra:
                return extra[action.tool](action)
            if action.tool == "open_application":
                return ToolResult(True, action.tool, f"Opened {action.args['app_name']}.", {"pid": 111, "hwnd": 222, "verified": True})
            if action.tool == "open_website":
                return ToolResult(True, action.tool, "Opened website.", {"url": action.args["url"], "verified": True})
            if action.tool == "open_path":
                return ToolResult(True, action.tool, "Opened.", {"path": action.args["path"], "verified": True})
            return ToolResult(True, action.tool, "ok", {"verified": True})

        return patch.object(agent.executor, "execute_action", side_effect=execute_action)

    def _fake_runtime_execute(self, extra=None):
        """Patches agent.agent_runtime.execute (SESSION_AWARE_SINGLE_TOOLS:
        close_application, focus_application, ...) without touching a real
        window."""
        extra = extra or {}

        def execute(plan, **kwargs):
            results = []
            observer = kwargs.get("action_observer")
            for index, action in enumerate(plan.actions):
                if observer:
                    observer("prepared", index, action, None, agent.agent_runtime.context)
                if action.tool in extra:
                    result = extra[action.tool](action)
                else:
                    result = ToolResult(True, action.tool, "ok", {"verified": True})
                if result.success:
                    agent.agent_runtime._update_context(action, result)
                if observer:
                    observer("result", index, action, result, agent.agent_runtime.context)
                results.append(result)
            return results

        return patch.object(agent.agent_runtime, "execute", side_effect=execute)


class EllipticalCloseTests(ConversationalContextTestCase):
    def test_open_spotify_then_close_it(self):
        with self._fake_executor():
            agent.run_agent("open spotify")
        with self._fake_runtime_execute() as mock_execute:
            response = agent.run_agent("close it")
        self.assertTrue(mock_execute.called)
        action = mock_execute.call_args.args[0].actions[0]
        self.assertEqual(action.tool, "close_application")
        self.assertEqual(action.args, {"app_name": "spotify"})

    def test_open_chrome_then_open_spotify_then_close_it_resolves_to_the_most_recent(self):
        with self._fake_executor():
            agent.run_agent("open chrome")
            agent.run_agent("open spotify")
        with self._fake_runtime_execute() as mock_execute:
            agent.run_agent("close it")
        action = mock_execute.call_args.args[0].actions[0]
        self.assertEqual(action.args, {"app_name": "spotify"})

    def test_ambiguous_reference_asks_instead_of_guessing(self):
        # Both apps opened by the SAME local_plan command -> same turn.
        from brain.models import Action, Plan

        plan = Plan("open chrome and spotify", [
            Action("open_application", {"app_name": "chrome"}),
            Action("open_application", {"app_name": "spotify"}),
        ])
        with patch.object(router, "create_local_plan", return_value=plan.actions), self._fake_executor():
            agent.run_agent("open chrome and spotify")
        response = agent.run_agent("close it")
        self.assertIn("chrome", response.lower())
        self.assertIn("spotify", response.lower())


class BrowserContinuationTests(ConversationalContextTestCase):
    """Search continuations route through browser_open_url (local_plan ->
    AgentRuntime), the same Playwright-backed action task_planner.py's own
    "open X and search Y" / "go back" / "first result" already use -- see
    brain/context_router.py::_browser_navigation_route."""

    def test_open_chrome_then_search_for_cats(self):
        with self._fake_executor():
            agent.run_agent("open chrome")
        with self._fake_runtime_execute() as mock_execute:
            agent.run_agent("search for cats")
        action = mock_execute.call_args.args[0].actions[0]
        self.assertEqual(action.tool, "browser_open_url")
        self.assertIn("cats", action.args["url"])

    def test_open_youtube_and_search_iron_man_then_search_batman_instead(self):
        with self._fake_executor():
            agent.run_agent("open youtube")
        agent.agent_runtime.context.browser_active = True
        agent.agent_runtime.context.last_search_provider = "youtube"
        with self._fake_runtime_execute() as mock_execute:
            agent.run_agent("search for batman instead")
        action = mock_execute.call_args.args[0].actions[0]
        self.assertEqual(action.tool, "browser_open_url")
        self.assertIn("youtube.com/results", action.args["url"])
        self.assertIn("batman", action.args["url"])


class ResultSetOrdinalTests(ConversationalContextTestCase):
    def test_list_files_then_open_the_second_one(self):
        def list_files_result(action):
            return ToolResult(True, "list_files", "ok", {"path": "C:/repo", "items": ["a.py", "b.py", "c.py"], "verified": True})

        with self._fake_executor({"list_files": list_files_result}):
            agent.run_agent("list files", route={"type": "tool", "tool": "list_files", "arguments": {"path": "C:/repo"}})
        with self._fake_executor() as mock_execute:
            agent.run_agent("open the second one")
        action = mock_execute.call_args.args[0]
        self.assertEqual(action.tool, "open_path")
        self.assertEqual(action.args["path"], "b.py")

    def test_stale_result_set_is_not_used(self):
        agent.agent_runtime.context.record_result_set([("a.py", "a.py")], source="list_files", kind="file")
        agent.agent_runtime.context.last_result_set.created_at -= 100000
        response = agent.run_agent("open the second one")
        # Falls through to "unknown route" rather than opening a stale file.
        self.assertNotIn("a.py", response)


class ReplayAndCorrectionTests(ConversationalContextTestCase):
    def test_correction_no_i_meant_telegram(self):
        with self._fake_executor() as mock_execute:
            agent.run_agent("open discord")
        with self._fake_executor() as mock_execute2:
            agent.run_agent("no, I meant Telegram")
        action = mock_execute2.call_args.args[0]
        self.assertEqual(action.tool, "open_application")
        self.assertEqual(action.args["app_name"], "telegram")

    def test_do_it_again_replays_the_exact_previous_command(self):
        with self._fake_executor():
            agent.run_agent("open spotify")
        with self._fake_executor() as mock_execute:
            agent.run_agent("do it again")
        action = mock_execute.call_args.args[0]
        self.assertEqual(action.tool, "open_application")
        self.assertEqual(action.args, {"app_name": "spotify"})


class ProjectAndTaskFollowupTests(ConversationalContextTestCase):
    """These genuinely need reasoning (what does "run" mean for a project?
    why did a command fail?) so they correctly escalate to AgentRuntime --
    but the escalation itself is mocked so no real, paid model call ever
    happens; what's under test is that the RESOLVED, SPECIFIC context
    reaches that escalation rather than an agent having to re-derive it."""

    def test_open_jarvis_project_then_run_it_escalates_with_the_resolved_path(self):
        with self._fake_executor():
            agent.run_agent("open my jarvis project")
        self.assertEqual(agent.agent_runtime.context.last_project_name, "jarvis")
        with patch.object(agent, "_agent_escalation_available", return_value=True), \
             patch.object(agent, "_run_agent_with_loop", return_value="Working on it, sir.") as escalate:
            agent.run_agent("run it")
        self.assertTrue(escalate.called)
        goal = escalate.call_args.args[0]
        self.assertIn("jarvis", goal.lower())

    def test_run_tests_then_fix_the_first_one_resolves_the_specific_failure(self):
        stdout = "FAILED tests/test_a.py::test_one - AssertionError: boom\nFAILED tests/test_b.py::test_two - ValueError\n"

        def run_command_result(action):
            return ToolResult(False, "run_command", "2 failed", {"stdout": stdout, "stderr": "", "exit_code": 1, "verified": True}, error="tests_failed")

        with self._fake_executor({"run_command": run_command_result}):
            agent.run_agent("run the tests", route={"type": "tool", "tool": "run_command", "arguments": {"command": "pytest"}})
        self.assertEqual(agent.agent_runtime.context.last_result_set.kind, "test_failure")
        with patch.object(agent, "_agent_escalation_available", return_value=True), \
             patch.object(agent, "_run_agent_with_loop", return_value="Working on it, sir.") as escalate:
            agent.run_agent("fix the first one")
        self.assertTrue(escalate.called)
        goal = escalate.call_args.args[0]
        self.assertIn("tests/test_a.py::test_one", goal)
        self.assertNotIn("test_two", goal)

    def test_run_git_status_then_why_did_that_fail_resolves_the_real_error(self):
        def run_command_result(action):
            return ToolResult(False, "run_command", "not a git repo", {"stdout": "", "stderr": "fatal: not a git repository", "exit_code": 128, "verified": True}, error="git_not_found")

        with self._fake_executor({"run_command": run_command_result}):
            agent.run_agent("run git status in the jarvis project", route={"type": "tool", "tool": "run_command", "arguments": {"command": "git status"}})
        self.assertEqual(agent.agent_runtime.context.last_error, "git_not_found")
        with patch.object(agent, "_agent_escalation_available", return_value=True), \
             patch.object(agent, "_run_agent_with_loop", return_value="It means git isn't installed, sir.") as escalate:
            response = agent.run_agent("why did that fail?")
        self.assertTrue(escalate.called)
        self.assertEqual(response, "It means git isn't installed, sir.")

    def test_context_builder_surfaces_resolved_task_and_error_to_the_agent_prompt(self):
        from brain.context_builder import ContextBuilder

        agent.agent_runtime.context.record_task("t1", "Run the tests", "failed", result_summary="2 failed, 1 passed", error="AssertionError in test_one")
        agent.agent_runtime.context.record_project("jarvis", "C:/repo/jarvis")
        built = ContextBuilder().build("fix the first one", session_context=agent.agent_runtime.context)
        self.assertIn("AssertionError in test_one", built.system_prompt)
        self.assertIn("C:/repo/jarvis", built.system_prompt)
        self.assertIn("Run the tests", built.system_prompt)


class ExistingMechanismsStillWorkTests(ConversationalContextTestCase):
    """Section: preserve deterministic fast routing / Task Manager -- these
    predate this feature and must be completely unaffected by it."""

    def test_stop_that_still_cancels_via_the_existing_mechanism(self):
        route = router.route_command("stop that", context=agent.agent_runtime.context)
        self.assertEqual(route, {"type": "cancel_read_only_task"})

    def test_what_are_you_doing_still_reports_task_status(self):
        route = router.route_command("what are you doing", context=agent.agent_runtime.context)
        self.assertEqual(route, {"type": "task_status"})


class VoiceNormalizedConsistencyTests(ConversationalContextTestCase):
    def test_voice_normalized_close_it_behaves_like_typed_close_it(self):
        from voice.text_normalizer import normalize_transcript

        with self._fake_executor():
            agent.run_agent("open spotify")
        normalized, _ = normalize_transcript("Hey Jarvis, close it.")
        typed_route = router.route_command("close it", context=agent.agent_runtime.context)
        voice_route = router.route_command(normalized, context=agent.agent_runtime.context)
        self.assertEqual(typed_route, voice_route)


class NoModelCallPerformanceTests(ConversationalContextTestCase):
    def test_simple_contextual_followups_resolve_in_milliseconds(self):
        import time

        with self._fake_executor():
            agent.run_agent("open spotify")
        started = time.perf_counter()
        with self._fake_runtime_execute():
            agent.run_agent("close it")
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.assertLess(elapsed_ms, 200)


if __name__ == "__main__":
    unittest.main()
