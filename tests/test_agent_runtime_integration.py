"""End-to-end: a complex request actually reaches the AgentRuntime.

The live complaint these tests lock down was NOT "the router says the wrong
thing" -- by itself a route named `agent_task` proves nothing. It was that
real requests such as

    "Tell me what files are in the JARVIS project folder. Do not modify
     anything."
    "Read main.py and tell me what it does. Do not modify anything."
    "Run git status in the JARVIS project and tell me what changed."

went into the legacy paths (`deterministic_planner` -> `cloud_planner`) and
came back with "I couldn't create a safe local plan for that task."

So each test here drives the REAL `brain.agent.run_agent` and asserts the
whole chain: the provider is genuinely called, real tools run, their real
observations are fed back, the model continues for several steps, a final
answer is produced, and token/cost usage is recorded.

The provider is a scripted fake (`providers/mock_provider.py`), so the suite
stays offline and free -- but everything between `run_agent` and the provider
is production code, including real filesystem and terminal tools.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from providers.base import ModelResponse, Usage
from providers.mock_provider import CallableProvider, text_response, tool_response
from providers.registry import register_provider, reset_providers_for_tests


class _RecordingProvider:
    """A `ModelProvider` driven by a callable, recording every turn.

    Wraps `CallableProvider` rather than reimplementing one, and adds the
    per-call bookkeeping these tests assert on (how many turns, what the
    model saw on each turn, what it asked for).
    """

    name = "recording"

    def __init__(self, handler, model="claude-sonnet-5"):
        self._inner = CallableProvider(handler, model=model)
        self.model = model
        self.turns: list[list] = []
        self.tool_specs: list[str] = []

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self):
        return None

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model, "available": True}

    def complete(self, messages, **kwargs):
        self.turns.append(list(messages))
        if not self.tool_specs:
            self.tool_specs = [spec.name for spec in (kwargs.get("tools") or [])]
        response = self._inner.complete(messages, **kwargs)
        response.provider = self.name
        response.model = self.model
        return response

    @property
    def call_count(self) -> int:
        return len(self.turns)


def install_provider(test_case, provider) -> None:
    """Make `providers.registry` hand out `provider`, restoring the real
    registry afterwards. `tests/conftest.py` clears every credential, so
    without this the agent is (correctly) unavailable."""
    register_provider("anthropic", lambda: provider)
    test_case.addCleanup(reset_providers_for_tests)


def usage(input_tokens=1200, output_tokens=40) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens, reported=True)


class ComplexRequestReachesTheRuntimeTests(unittest.TestCase):
    """Requests that failed live must now run a real agent loop."""

    LIVE_FAILURES = (
        "Tell me what files are in the JARVIS project folder. Do not modify anything.",
        "Read main.py and tell me what it does. Do not modify anything.",
        "Inspect the JARVIS project and list the main folders and what each one is for. Do not modify anything.",
        "Run git status in the JARVIS project and tell me what changed. Do not modify anything.",
        "Inspect this JARVIS project and tell me how the main components are connected. Do not modify anything.",
    )

    def test_each_live_failure_now_invokes_the_provider_and_answers(self):
        from brain.agent import run_agent

        for command in self.LIVE_FAILURES:
            with self.subTest(command=command):
                provider = _RecordingProvider(lambda messages, **kw: text_response("Here's what I found, sir."))
                install_provider(self, provider)
                outcome = {}
                answer = run_agent(command, execution_outcome=outcome)
                self.assertGreaterEqual(provider.call_count, 1, "the provider was never called")
                self.assertEqual(answer, "Here's what I found, sir.")
                self.assertEqual(outcome["route_type"], "agent_task")
                self.assertTrue(outcome["success"])
                self.assertNotIn("couldn't create", answer.lower())

    def test_the_legacy_failure_message_is_gone(self):
        """The exact reply the user reported must not be reachable while an
        agent provider is configured."""
        from brain.agent import run_agent

        provider = _RecordingProvider(lambda messages, **kw: text_response("Done, sir."))
        install_provider(self, provider)
        for command in self.LIVE_FAILURES:
            with self.subTest(command=command):
                answer = run_agent(command)
                self.assertNotIn("safe local plan", answer)
                self.assertNotIn("couldn't create a plan", answer)

    def test_the_whole_request_is_handed_over_not_a_captured_fragment(self):
        from brain.agent import run_agent

        command = "Read main.py and tell me what it does. Do not modify anything."
        provider = _RecordingProvider(lambda messages, **kw: text_response("It's the entry point, sir."))
        install_provider(self, provider)
        run_agent(command)
        first_turn_text = " ".join(message.content for message in provider.turns[0])
        self.assertIn("main.py", first_turn_text)
        self.assertIn("Do not modify anything", first_turn_text)


class RealToolRoundTripTests(unittest.TestCase):
    """Not just "a model was called": real tools, real observations, several
    steps, and honest usage accounting."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="jarvis-agent-e2e-"))
        (self.root / "main.py").write_text("print('jarvis entry point')\n", encoding="utf-8")
        (self.root / "notes.txt").write_text("hello\n", encoding="utf-8")

    def test_a_multi_step_run_reads_a_real_file_and_answers_from_it(self):
        from brain.agent import run_agent

        target = self.root / "main.py"
        script = [
            tool_response("list_files", {"path": str(self.root)}),
            tool_response("read_text_file", {"path": str(target)}),
            text_response("main.py just prints the entry-point banner, sir."),
        ]
        seen_observations = []

        def handler(messages, **kwargs):
            if messages and messages[-1].tool_outcomes:
                seen_observations.append(messages[-1].tool_outcomes[0])
            response = script[min(len(seen_observations), len(script) - 1)]
            return ModelResponse(
                text=response.text,
                tool_calls=list(response.tool_calls),
                stop_reason=response.stop_reason,
                usage=usage(),
                estimated_cost_usd=0.004,
            )

        provider = _RecordingProvider(handler)
        install_provider(self, provider)
        outcome = {}
        answer = run_agent(f"Read {target} and tell me what it does. Do not modify anything.", execution_outcome=outcome)

        # 1. the model was called more than once -- it CONTINUED after tools
        self.assertGreaterEqual(provider.call_count, 3)
        # 2. real tools ran and real observations came back
        self.assertGreaterEqual(len(seen_observations), 2)
        self.assertIn("main.py", seen_observations[0].content)
        self.assertIn("jarvis entry point", seen_observations[1].content)
        self.assertFalse(any(item.is_error for item in seen_observations))
        # 3. a useful final answer
        self.assertEqual(answer, "main.py just prints the entry-point banner, sir.")
        # 4. usage was recorded, and WHICH tools ran is visible to callers
        self.assertGreater(outcome.get("model_calls", 0), 1)
        self.assertEqual(outcome.get("agent_tools"), ["list_files", "read_text_file"])
        self.assertEqual(outcome.get("agent_steps"), 2)
        self.assertEqual(outcome.get("model"), "claude-sonnet-5")

    def test_the_agent_is_offered_the_filesystem_terminal_and_code_tools(self):
        """A repository question is only answerable if these reach the model."""
        from brain.agent import run_agent

        provider = _RecordingProvider(lambda messages, **kw: text_response("Understood, sir."))
        install_provider(self, provider)
        run_agent("Inspect this project and explain how the components connect. Do not modify anything.")
        for tool in ("list_files", "read_text_file", "run_command", "inspect_project", "search_code"):
            with self.subTest(tool=tool):
                self.assertIn(tool, provider.tool_specs)

    def test_a_failing_tool_is_reported_honestly_and_the_model_can_adapt(self):
        from brain.agent import run_agent

        missing = self.root / "does_not_exist.py"
        calls = []

        def handler(messages, **kwargs):
            calls.append(messages)
            if len(calls) == 1:
                return tool_response("read_text_file", {"path": str(missing)})
            if len(calls) == 2:
                return tool_response("read_text_file", {"path": str(self.root / "notes.txt")})
            return text_response("The first path was wrong; notes.txt says hello, sir.")

        provider = _RecordingProvider(handler)
        install_provider(self, provider)
        answer = run_agent(f"Read {missing} and tell me what it says. Do not modify anything.")
        # The failure was observed, not swallowed, and the run continued.
        self.assertGreaterEqual(provider.call_count, 3)
        self.assertIn("notes.txt", answer)

    def test_token_and_cost_usage_is_recorded(self):
        from brain.agent import run_agent

        def handler(messages, **kwargs):
            return ModelResponse(
                text="All set, sir.",
                stop_reason="end_turn",
                usage=usage(input_tokens=999, output_tokens=11),
                estimated_cost_usd=0.0123,
            )

        provider = _RecordingProvider(handler)
        install_provider(self, provider)
        outcome = {}
        run_agent("Inspect this project and explain how it works. Do not modify anything.", execution_outcome=outcome)
        self.assertEqual(outcome.get("model_calls"), 1)


class SimpleCommandsNeverReachTheModelTests(unittest.TestCase):
    """"Do not send simple commands to Claude unnecessarily." The fast path
    must not have become slower or chattier."""

    SIMPLE = {
        "open Spotify": "open_application",
        "volume down": "volume_down",
        "volume up": "volume_up",
        "mute": "mute_volume",
        "calculate 527 * 93": "calculator",
        "inspect window": "inspect_window",
        "what time is it": "get_time",
        "press enter": "press_key",
        "close chrome": "close_application",
    }

    def test_no_model_call_and_the_same_tool_as_before(self):
        from brain.router import route_command

        provider = _RecordingProvider(lambda messages, **kw: text_response("should never happen"))
        install_provider(self, provider)
        for command, expected_tool in self.SIMPLE.items():
            with self.subTest(command=command):
                route = route_command(command)
                self.assertEqual(route["type"], "tool", f"{command!r} left the fast path: {route!r}")
                self.assertEqual(route["tool"], expected_tool)
        self.assertEqual(provider.call_count, 0)

    def test_music_commands_stay_on_the_deterministic_music_router(self):
        from brain.router import route_command

        provider = _RecordingProvider(lambda messages, **kw: text_response("should never happen"))
        install_provider(self, provider)
        for command, expected_tool in (("open music", "open_music"), ("play Israeli playlist", "music_play"), ("pause", "music_pause")):
            with self.subTest(command=command):
                route = route_command(command)
                self.assertEqual(route["type"], "tool")
                self.assertEqual(route["tool"], expected_tool)
        self.assertEqual(provider.call_count, 0)


class NoLegacyCloudCallsWhenTheAgentIsAvailableTests(unittest.TestCase):
    """"Avoid legacy cloud planning/classification calls when they are no
    longer needed." Those calls cost 15-30s live before a failure reply."""

    def test_an_unresolvable_request_skips_the_cloud_intent_router(self):
        from brain import router

        provider = _RecordingProvider(lambda messages, **kw: text_response("ok"))
        install_provider(self, provider)
        with patch.object(router, "classify_intent", side_effect=AssertionError("the cloud intent router was called")):
            route = router.route_command("frobnicate the widget in a way nothing understands")
        self.assertEqual(route["type"], "agent_task")
        self.assertEqual(route["route_source"], "no_deterministic_route")
        self.assertEqual(route.get("model_calls"), 0)

    def test_an_incomplete_local_plan_goes_to_the_agent_not_the_cloud_planner(self):
        """The rule-based planner produced nothing usable for a multi-clause
        request. That used to mean an OpenAI planning call and then, when the
        generated plan failed validation, "I couldn't build a complete
        validated plan...". No real tool runs here: the planner is stubbed so
        the test exercises the fallback decision, not Notepad."""
        from brain import agent
        from brain.models import Plan

        provider = _RecordingProvider(lambda messages, **kw: text_response("Handled it, sir."))
        install_provider(self, provider)
        command = "open notepad, type the summary, then email it to the team and confirm it sent"
        with patch.object(agent, "create_task_plan", return_value=Plan(command, [])),              patch.object(agent, "assess_plan_completeness", return_value={"complete": False, "clauses": ["a", "b", "c"]}),              patch.object(agent, "validate_goal_coverage", return_value=["incomplete"]),              patch.object(agent, "create_plan", side_effect=AssertionError("the legacy cloud planner was called")):
            outcome = {}
            answer = agent.run_agent(command, execution_outcome=outcome)
        self.assertEqual(answer, "Handled it, sir.")
        self.assertEqual(outcome["route_type"], "agent_task")
        self.assertGreaterEqual(provider.call_count, 1)

    def test_the_cloud_intent_router_still_runs_when_no_agent_is_configured(self):
        """"Claude is optional": with no provider, the legacy path is intact."""
        from brain import router

        reset_providers_for_tests()
        self.addCleanup(reset_providers_for_tests)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False), \
             patch.object(router, "classify_intent", return_value={"type": "ai", "message": "x"}) as classify:
            route = router.route_command("frobnicate the widget in a way nothing understands")
        self.assertTrue(classify.called)
        self.assertEqual(route["type"], "ai")


class TypedAndVoicePathsAgreeTests(unittest.TestCase):
    """`main.py` calls `run_agent(command)` with NO route; the voice path
    routes first. Both must reach the same place -- the typed path used to
    skip `route_command` entirely whenever `should_use_task_planner` matched."""

    def test_run_agent_without_a_route_still_reaches_the_agent_runtime(self):
        from brain.agent import run_agent
        from brain.task_planner import should_use_task_planner

        command = "Tell me what files are in the JARVIS project folder. Do not modify anything."
        # The precondition for the bug: the planner heuristic likes this
        # sentence, so it used to win before routing ever happened.
        self.assertTrue(should_use_task_planner(command))

        provider = _RecordingProvider(lambda messages, **kw: text_response("Listed them, sir."))
        install_provider(self, provider)
        outcome = {}
        answer = run_agent(command, execution_outcome=outcome)
        self.assertEqual(outcome["route_type"], "agent_task")
        self.assertEqual(answer, "Listed them, sir.")

    def test_typed_and_voice_normalized_forms_match(self):
        from brain.router import route_command
        from voice.text_normalizer import normalize_transcript

        provider = _RecordingProvider(lambda messages, **kw: text_response("ok"))
        install_provider(self, provider)
        for command in (
            "Tell me what files are in the JARVIS project folder. Do not modify anything.",
            "Read main.py and tell me what it does. Do not modify anything.",
            "Run git status in the JARVIS project and tell me what changed. Do not modify anything.",
            "open Spotify",
            "volume down",
        ):
            with self.subTest(command=command):
                spoken, _ = normalize_transcript(f"Hey Jarvis, {command}")
                typed_route = route_command(command)
                voice_route = route_command(spoken)
                self.assertEqual(typed_route["type"], voice_route["type"])
                self.assertEqual(typed_route.get("tool"), voice_route.get("tool"))


class ToolRequiringQuestionsReachTheAgentTests(unittest.TestCase):
    """"Questions requiring tools must be allowed to use AgentRuntime."
    The QUESTION route answers from the web, which cannot see this machine."""

    LOCAL_QUESTIONS = (
        "what files are in the jarvis project folder",
        "what is in the current directory",
        "how many python files are in this project",
        "what does git status say in this repository",
    )

    WEB_QUESTIONS = (
        "who is the president of france",
        "what is the weather in san francisco tomorrow",
        "tell me what the capital of japan is",
    )

    def test_local_questions_escalate_when_an_agent_is_available(self):
        from brain.router import route_command

        provider = _RecordingProvider(lambda messages, **kw: text_response("ok"))
        install_provider(self, provider)
        for question in self.LOCAL_QUESTIONS:
            with self.subTest(question=question):
                self.assertEqual(route_command(question)["type"], "agent_task")

    def test_ordinary_questions_still_use_the_web_answer_route(self):
        from brain.router import route_command

        provider = _RecordingProvider(lambda messages, **kw: text_response("ok"))
        install_provider(self, provider)
        for question in self.WEB_QUESTIONS:
            with self.subTest(question=question):
                self.assertEqual(route_command(question)["type"], "question")

    def test_without_an_agent_local_questions_keep_their_old_route(self):
        """"Claude is optional": escalating a filesystem question with no
        filesystem-capable agent gains nothing, so it must fall back to the
        route it had before -- not be taken away from the web-answer path."""
        from brain.router import route_command

        reset_providers_for_tests()
        self.addCleanup(reset_providers_for_tests)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            for question in self.LOCAL_QUESTIONS:
                with self.subTest(question=question):
                    self.assertNotEqual(route_command(question)["type"], "agent_task")


class AgentOptionalTests(unittest.TestCase):
    """With no provider configured every route must behave as it always has."""

    def test_a_complex_request_degrades_to_the_planner_without_a_provider(self):
        from brain.router import route_command

        reset_providers_for_tests()
        self.addCleanup(reset_providers_for_tests)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            route = route_command("Inspect this project and explain how it works. Do not modify anything.")
        # The router still escalates (it is provider-agnostic for this rule);
        # brain/agent.py is what degrades it to the planner.
        self.assertEqual(route["type"], "agent_task")
        self.assertEqual(route["route_source"], "complexity_guard")

    def test_simple_commands_are_identical_with_and_without_a_provider(self):
        from brain.router import route_command

        commands = ("open Spotify", "volume down", "inspect window", "what time is it", "press enter")
        provider = _RecordingProvider(lambda messages, **kw: text_response("ok"))
        install_provider(self, provider)
        with_agent = {command: route_command(command) for command in commands}
        reset_providers_for_tests()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            without_agent = {command: route_command(command) for command in commands}
        for command in commands:
            with self.subTest(command=command):
                self.assertEqual(with_agent[command]["tool"], without_agent[command]["tool"])


if __name__ == "__main__":
    unittest.main()
