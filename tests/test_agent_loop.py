"""The agent loop: action -> observation -> next action, and every safety limit.

The provider is always a fake, so these run offline and free. The tools
are real (calculator, filesystem, terminal) or injected handlers, so the
observations the loop reasons over are genuine tool results.
"""
import tempfile
import unittest
from pathlib import Path

from brain.agent_loop import (
    CANCELLED,
    COMPLETED,
    FAILURE_LIMIT,
    NO_PROVIDER,
    PROVIDER_ERROR,
    STEP_LIMIT,
    AgentLimits,
    AgentLoop,
    _observation_text,
)
from brain.context_builder import ContextBuilder
from brain.models import ToolResult
from brain.tool_catalog import ToolCatalog
from providers.base import ModelResponse, ProviderRateLimited, ToolCall, Usage
from providers.mock_provider import CallableProvider, ScriptedProvider, text_response, tool_response


def _context(request: str = "do the thing"):
    return ContextBuilder().build(request)


def _limits(**overrides) -> AgentLimits:
    defaults = dict(max_steps=10, max_action_retries=2, max_consecutive_failures=4, timeout_seconds=30.0)
    defaults.update(overrides)
    return AgentLimits(**defaults)


class LoopBehaviourTests(unittest.TestCase):
    def setUp(self):
        self.catalog = ToolCatalog()
        self.root = Path(tempfile.mkdtemp())

    def test_a_plain_answer_finishes_in_one_turn(self):
        provider = ScriptedProvider([text_response("The answer is 42, sir.")])
        run = AgentLoop(provider, self.catalog, limits=_limits()).run("what is the answer", context=_context())
        self.assertEqual(run.stop_reason, COMPLETED)
        self.assertEqual(run.answer, "The answer is 42, sir.")
        self.assertEqual(run.model_calls, 1)
        # No tool ran, so nothing was independently confirmed.
        self.assertFalse(run.verified)

    def test_the_loop_feeds_a_real_observation_back_to_the_model(self):
        seen = []

        def handler(messages, **kwargs):
            seen.append(list(messages))
            if len(seen) == 1:
                return tool_response("calculator", {"expression": "6*7"})
            return text_response("It comes to 42, sir.")

        run = AgentLoop(CallableProvider(handler), self.catalog, limits=_limits()).run(
            "what is six times seven", context=_context()
        )
        self.assertEqual(run.answer, "It comes to 42, sir.")
        second_turn = seen[1]
        observation = second_turn[-1].tool_outcomes[0].content
        self.assertIn("42", observation)
        self.assertFalse(second_turn[-1].tool_outcomes[0].is_error)

    def test_the_model_can_adapt_after_a_real_failure(self):
        target = self.root / "notes.txt"
        turns = []

        def handler(messages, **kwargs):
            # Snapshot: the loop keeps appending to the same list, so a
            # stored reference would show the final state, not this turn's.
            turns.append(list(messages))
            if len(turns) == 1:
                return tool_response("read_text_file", {"path": str(target)}, call_id="a")
            if len(turns) == 2:
                return tool_response("create_text_file", {"path": str(target), "contents": "hello"}, call_id="b")
            return text_response("The file did not exist, so I created it, sir.")

        run = AgentLoop(CallableProvider(handler), self.catalog, limits=_limits()).run(
            "make sure notes.txt exists", context=_context()
        )
        self.assertTrue(run.success)
        self.assertEqual([step.success for step in run.steps], [False, True])
        self.assertTrue(target.exists())
        # The failure was surfaced as an error observation, not hidden.
        self.assertTrue(turns[1][-1].tool_outcomes[0].is_error)
        self.assertIn("FAILED", turns[1][-1].tool_outcomes[0].content)

    def test_multi_step_work_is_not_cut_short_by_the_limits(self):
        script = [tool_response("calculator", {"expression": f"{index}+1"}, call_id=f"c{index}") for index in range(8)]
        script.append(text_response("All eight calculations are done, sir."))
        run = AgentLoop(ScriptedProvider(script), self.catalog, limits=_limits()).run("do eight things", context=_context())
        self.assertEqual(run.stop_reason, COMPLETED)
        self.assertEqual(len(run.steps), 8)

    def test_several_tool_calls_in_one_turn_all_execute(self):
        def handler(messages, **kwargs):
            if len(messages) == 1:
                return ModelResponse(
                    text="",
                    tool_calls=[
                        ToolCall("a", "calculator", {"expression": "1+1"}),
                        ToolCall("b", "calculator", {"expression": "2+2"}),
                    ],
                    stop_reason="tool_use",
                    usage=Usage(1, 1),
                )
            return text_response("Both done, sir.")

        run = AgentLoop(CallableProvider(handler), self.catalog, limits=_limits()).run("two sums", context=_context())
        self.assertEqual(len(run.steps), 2)


class SafetyLimitTests(unittest.TestCase):
    def setUp(self):
        self.catalog = ToolCatalog()

    def _always(self, name, arguments):
        return CallableProvider(
            lambda messages, **kwargs: ModelResponse(
                text="", tool_calls=[ToolCall("c1", name, arguments)], stop_reason="tool_use", usage=Usage(1, 1)
            )
        )

    def test_repeating_an_identical_failing_call_is_refused_after_the_retry_cap(self):
        provider = self._always("read_code", {"path": "/definitely/not/here.py"})
        run = AgentLoop(provider, self.catalog, limits=_limits(max_action_retries=2, max_consecutive_failures=10, max_steps=6)).run(
            "read a missing file", context=_context()
        )
        refusals = [step for step in run.steps if step.error == "refused_repeated_failure"]
        self.assertTrue(refusals)
        self.assertIn("Change the approach", refusals[0].observation)

    def test_retries_are_counted(self):
        provider = self._always("read_code", {"path": "/definitely/not/here.py"})
        run = AgentLoop(provider, self.catalog, limits=_limits(max_consecutive_failures=10, max_steps=4)).run(
            "read a missing file", context=_context()
        )
        self.assertGreater(run.retries, 0)

    def test_consecutive_failures_stop_the_loop_honestly(self):
        provider = self._always("read_code", {"path": "/definitely/not/here.py"})
        run = AgentLoop(provider, self.catalog, limits=_limits(max_consecutive_failures=3, max_steps=20)).run(
            "read a missing file", context=_context()
        )
        self.assertEqual(run.stop_reason, FAILURE_LIMIT)
        self.assertFalse(run.success)
        self.assertNotIn("done", run.answer.lower())

    def test_the_step_limit_stops_an_endless_loop(self):
        provider = CallableProvider(
            lambda messages, **kwargs: ModelResponse(
                text="", tool_calls=[ToolCall("c", "calculator", {"expression": "1+1"})], stop_reason="tool_use", usage=Usage(1, 1)
            )
        )
        run = AgentLoop(provider, self.catalog, limits=_limits(max_steps=4)).run("loop forever", context=_context())
        self.assertEqual(run.stop_reason, STEP_LIMIT)
        self.assertEqual(run.model_calls, 4)
        self.assertFalse(run.success)

    def test_the_timeout_stops_a_long_run(self):
        provider = CallableProvider(
            lambda messages, **kwargs: ModelResponse(
                text="", tool_calls=[ToolCall("c", "calculator", {"expression": "1+1"})], stop_reason="tool_use", usage=Usage(1, 1)
            )
        )
        run = AgentLoop(provider, self.catalog, limits=_limits(timeout_seconds=0.0)).run("slow work", context=_context())
        self.assertEqual(run.stop_reason, "timeout")
        self.assertFalse(run.success)

    def test_a_partial_answer_reports_what_actually_got_done(self):
        script = [tool_response("calculator", {"expression": "1+1"})] * 5
        provider = ScriptedProvider(script)
        run = AgentLoop(provider, self.catalog, limits=_limits(max_steps=3)).run("do things", context=_context())
        self.assertIn("calculator", run.answer)


class CancellationTests(unittest.TestCase):
    class Token:
        def __init__(self, cancelled=False):
            self.cancelled = cancelled

    def test_a_cancelled_token_stops_before_any_model_call(self):
        provider = ScriptedProvider([text_response("should never run")])
        run = AgentLoop(provider, ToolCatalog(), limits=_limits()).run(
            "do a thing", context=_context(), cancellation_token=self.Token(True)
        )
        self.assertEqual(run.stop_reason, CANCELLED)
        self.assertEqual(run.model_calls, 0)

    def test_cancelling_mid_run_stops_before_the_next_tool(self):
        token = self.Token(False)

        def handler(messages, **kwargs):
            token.cancelled = True
            return tool_response("calculator", {"expression": "1+1"})

        run = AgentLoop(CallableProvider(handler), ToolCatalog(), limits=_limits()).run(
            "do a thing", context=_context(), cancellation_token=token
        )
        self.assertEqual(run.stop_reason, CANCELLED)
        self.assertEqual(run.steps, [])


class ProviderFailureTests(unittest.TestCase):
    def test_a_missing_provider_is_reported_not_faked(self):
        class Unavailable:
            name = "none"
            model = ""

            def is_available(self):
                return False

        run = AgentLoop(Unavailable(), ToolCatalog(), limits=_limits()).run("do a thing", context=_context())
        self.assertEqual(run.stop_reason, NO_PROVIDER)
        self.assertFalse(run.success)
        self.assertIn("ANTHROPIC_API_KEY", run.answer)

    def test_a_rate_limit_is_surfaced_as_a_rate_limit(self):
        def handler(messages, **kwargs):
            raise ProviderRateLimited("slow down")

        run = AgentLoop(CallableProvider(handler), ToolCatalog(), limits=_limits()).run("do a thing", context=_context())
        self.assertEqual(run.stop_reason, PROVIDER_ERROR)
        self.assertIn("rate limited", run.answer.lower())


class ObservationTests(unittest.TestCase):
    def test_failure_leads_with_the_error_code(self):
        text = _observation_text(ToolResult(False, "run_command", "it broke", {"exit_code": 1}, "exit_code_1"))
        self.assertTrue(text.startswith("FAILED (exit_code_1)"))

    def test_stdout_and_exit_code_are_surfaced(self):
        text = _observation_text(ToolResult(True, "run_command", "ok", {"stdout": "1 passed", "exit_code": 0}))
        self.assertIn("1 passed", text)
        self.assertIn("exit_code: 0", text)

    def test_file_content_is_not_duplicated(self):
        body = "line one\nline two"
        text = _observation_text(ToolResult(True, "read_code", body, {"numbered_contents": body, "contents": body}))
        self.assertEqual(text.count("line one"), 1)

    def test_an_empty_result_still_says_something(self):
        self.assertIn("no detail", _observation_text(ToolResult(True, "x", "", {})))


class AccountingTests(unittest.TestCase):
    def test_tokens_and_cost_accumulate_across_turns(self):
        first = tool_response("calculator", {"expression": "1+1"}, input_tokens=100, output_tokens=10)
        first.estimated_cost_usd = 0.001
        second = text_response("Two, sir.", input_tokens=120, output_tokens=8)
        second.estimated_cost_usd = 0.002
        run = AgentLoop(ScriptedProvider([first, second]), ToolCatalog(), limits=_limits()).run("one plus one", context=_context())
        self.assertEqual(run.usage.input_tokens, 220)
        self.assertEqual(run.usage.output_tokens, 18)
        self.assertAlmostEqual(run.estimated_cost_usd, 0.003)
        self.assertTrue(run.usage.reported)

    def test_unreported_usage_stays_unreported(self):
        response = text_response("hi")
        response.usage = Usage(reported=False)
        run = AgentLoop(ScriptedProvider([response]), ToolCatalog(), limits=_limits()).run("hi", context=_context())
        self.assertFalse(run.usage.reported)


if __name__ == "__main__":
    unittest.main()
