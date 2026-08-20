"""Model providers: the vendor-neutral interface, the Anthropic adapter,
error translation, availability without a key, and usage/cost recording.

Every test here uses a fake client. No network call is ever made and no
API key is ever required.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config.settings import reload_config
from providers.anthropic_provider import AnthropicProvider, _to_api_messages, _translate_error
from providers.base import (
    Message,
    ProviderAuthError,
    ProviderRateLimited,
    ProviderRequestError,
    ProviderServerError,
    ProviderTimeout,
    ProviderUnavailable,
    ToolCall,
    ToolOutcome,
    ToolSpec,
)
from providers.mock_provider import ScriptedProvider, text_response, tool_response
from providers.registry import (
    get_agent_provider,
    provider_status,
    register_provider,
)
from providers.usage import TrackedProvider, UsageStore


class _Block:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class _Usage:
    def __init__(self, input_tokens=0, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class _Response:
    def __init__(self, content, stop_reason="end_turn", model="claude-opus-5", usage=None):
        self.content = content
        self.stop_reason = stop_reason
        self.model = model
        self.usage = usage or _Usage(100, 50)
        self.stop_details = None


class _Messages:
    def __init__(self, response, recorder):
        self._response = response
        self._recorder = recorder

    def create(self, **request):
        self._recorder.append(request)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeClient:
    def __init__(self, response):
        self.requests = []
        self.messages = _Messages(response, self.requests)

    def with_options(self, **kwargs):
        return self


class AvailabilityTests(unittest.TestCase):
    def test_provider_is_unavailable_without_a_key(self):
        provider = AnthropicProvider(api_key=None)
        self.assertFalse(provider.is_available())
        self.assertEqual(provider.unavailable_reason(), "missing_api_key")

    def test_completing_without_a_key_raises_provider_unavailable_not_a_vendor_error(self):
        provider = AnthropicProvider(api_key=None)
        with self.assertRaises(ProviderUnavailable):
            provider.complete([Message.user("hello")])

    def test_describe_never_contains_the_key(self):
        described = AnthropicProvider(api_key="sk-ant-secret").describe()
        self.assertEqual(described["credentials"], "<set>")
        self.assertNotIn("sk-ant-secret", repr(described))

    def test_registry_returns_none_when_nothing_is_configured(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            reload_config()
            register_provider("anthropic", AnthropicProvider)
            self.assertIsNone(get_agent_provider())
        reload_config()

    def test_provider_status_explains_why_a_provider_is_unusable(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            reload_config()
            register_provider("anthropic", AnthropicProvider)
            status = provider_status()
        reload_config()
        self.assertIsNone(status["active_provider"])
        reasons = [entry.get("unavailable_reason") for entry in status["providers"]]
        self.assertIn("missing_api_key", reasons)


class RequestTranslationTests(unittest.TestCase):
    def test_tool_calls_and_results_round_trip_into_api_blocks(self):
        messages = [
            Message.user("run the tests"),
            Message.assistant("Running them.", [ToolCall("call_1", "run_command", {"command": "pytest"})]),
            Message.tool_results([ToolOutcome("call_1", "exit code 0", is_error=False)]),
        ]
        payload = _to_api_messages(messages)
        self.assertEqual(payload[0], {"role": "user", "content": "run the tests"})
        self.assertEqual(payload[1]["role"], "assistant")
        self.assertEqual(payload[1]["content"][0]["type"], "text")
        self.assertEqual(payload[1]["content"][1]["type"], "tool_use")
        self.assertEqual(payload[1]["content"][1]["id"], "call_1")
        self.assertEqual(payload[2]["content"][0]["type"], "tool_result")
        self.assertEqual(payload[2]["content"][0]["tool_use_id"], "call_1")
        self.assertFalse(payload[2]["content"][0]["is_error"])

    def test_failed_tool_result_is_marked_as_an_error_for_the_model(self):
        payload = _to_api_messages([Message.tool_results([ToolOutcome("c", "FAILED", is_error=True)])])
        self.assertTrue(payload[0]["content"][0]["is_error"])

    def test_tools_and_system_prompt_are_sent(self):
        client = _FakeClient(_Response([_Block(type="text", text="hi")]))
        provider = AnthropicProvider(api_key="sk-test", client=client, model="claude-opus-5")
        provider.complete(
            [Message.user("hello")],
            system="be brief",
            tools=[ToolSpec("get_time", "the time", {"type": "object", "properties": {}, "required": []})],
        )
        request = client.requests[0]
        self.assertEqual(request["system"], "be brief")
        self.assertEqual(request["tools"][0]["name"], "get_time")
        self.assertIn("input_schema", request["tools"][0])
        self.assertEqual(request["model"], "claude-opus-5")


class ResponseParsingTests(unittest.TestCase):
    def test_text_and_tool_use_blocks_are_extracted(self):
        client = _FakeClient(
            _Response(
                [
                    _Block(type="text", text="Let me check."),
                    _Block(type="tool_use", id="c1", name="get_time", input={}),
                ],
                stop_reason="tool_use",
            )
        )
        response = AnthropicProvider(api_key="sk-test", client=client).complete([Message.user("time?")])
        self.assertEqual(response.text, "Let me check.")
        self.assertEqual(response.stop_reason, "tool_use")
        self.assertTrue(response.wants_tools)
        self.assertEqual(response.tool_calls[0].name, "get_time")

    def test_usage_and_cost_are_reported(self):
        client = _FakeClient(_Response([_Block(type="text", text="done")], usage=_Usage(1_000_000, 1_000_000)))
        response = AnthropicProvider(api_key="sk-test", client=client, model="claude-opus-5").complete([Message.user("x")])
        self.assertEqual(response.usage.input_tokens, 1_000_000)
        self.assertTrue(response.usage.reported)
        self.assertAlmostEqual(response.estimated_cost_usd, 30.0, places=4)

    def test_missing_usage_is_reported_as_unknown_not_zero(self):
        response_object = _Response([_Block(type="text", text="done")])
        response_object.usage = None
        client = _FakeClient(response_object)
        response = AnthropicProvider(api_key="sk-test", client=client).complete([Message.user("x")])
        self.assertFalse(response.usage.reported)


class ErrorTranslationTests(unittest.TestCase):
    def test_each_sdk_error_maps_to_a_specific_jarvis_error(self):
        cases = {
            "AuthenticationError": ProviderAuthError,
            "PermissionDeniedError": ProviderAuthError,
            "APITimeoutError": ProviderTimeout,
            "RateLimitError": ProviderRateLimited,
            "APIConnectionError": ProviderServerError,
            "BadRequestError": ProviderRequestError,
        }
        for name, expected in cases.items():
            with self.subTest(error=name):
                error = type(name, (Exception,), {})("boom")
                self.assertIsInstance(_translate_error(error), expected)

    def test_status_codes_are_classified(self):
        server = type("X", (Exception,), {})("boom")
        server.status_code = 503
        self.assertIsInstance(_translate_error(server), ProviderServerError)
        bad = type("Y", (Exception,), {})("boom")
        bad.status_code = 400
        self.assertIsInstance(_translate_error(bad), ProviderRequestError)

    def test_retryable_flag_distinguishes_transient_failures(self):
        self.assertTrue(ProviderRateLimited("x").retryable)
        self.assertTrue(ProviderTimeout("x").retryable)
        self.assertFalse(ProviderRequestError("x").retryable)

    def test_provider_never_leaks_the_raw_vendor_exception(self):
        client = _FakeClient(RuntimeError("vendor exploded"))
        provider = AnthropicProvider(api_key="sk-test", client=client)
        with self.assertRaises(ProviderServerError):
            provider.complete([Message.user("x")])


class ScriptedProviderTests(unittest.TestCase):
    def test_responses_are_replayed_in_order(self):
        provider = ScriptedProvider([tool_response("get_time"), text_response("It's noon.")])
        first = provider.complete([Message.user("time?")])
        second = provider.complete([Message.user("time?")])
        self.assertTrue(first.wants_tools)
        self.assertEqual(second.text, "It's noon.")

    def test_running_out_of_script_is_an_explicit_failure(self):
        provider = ScriptedProvider([])
        with self.assertRaises(ProviderUnavailable):
            provider.complete([Message.user("x")])


class UsageTrackingTests(unittest.TestCase):
    def setUp(self):
        self.store = UsageStore(Path(tempfile.mkdtemp()) / "usage.sqlite3")

    def tearDown(self):
        self.store.close()

    def test_successful_calls_are_recorded_with_tokens_and_cost(self):
        provider = ScriptedProvider([text_response("hi", input_tokens=100, output_tokens=20)])
        tracked = TrackedProvider(provider, self.store, task_id="t1", session_id="s1")
        tracked.complete([Message.user("x")])
        summary = self.store.for_task("t1")
        self.assertEqual(summary.calls, 1)
        self.assertEqual(summary.input_tokens, 100)
        self.assertEqual(summary.output_tokens, 20)
        self.assertEqual(self.store.for_session("s1").calls, 1)

    def test_an_unpriced_model_is_reported_as_incomplete_cost_not_as_free(self):
        provider = ScriptedProvider([text_response("hi")], model="mystery-model")
        tracked = TrackedProvider(provider, self.store, task_id="t2")
        tracked.complete([Message.user("x")])
        summary = self.store.for_task("t2")
        self.assertEqual(summary.unpriced_calls, 1)
        self.assertFalse(summary.cost_is_complete)

    def test_a_failed_call_is_still_recorded(self):
        provider = ScriptedProvider([])
        tracked = TrackedProvider(provider, self.store, task_id="t3")
        with self.assertRaises(ProviderUnavailable):
            tracked.complete([Message.user("x")])
        self.assertEqual(self.store.for_task("t3").failures, 1)

    def test_a_failed_call_is_not_counted_as_an_unpriced_call(self):
        """A failure records cost_usd=NULL because nothing was billed --
        that must not make a fully priced model report as unpriced."""
        provider = ScriptedProvider([])
        tracked = TrackedProvider(provider, self.store, task_id="t6")
        with self.assertRaises(ProviderUnavailable):
            tracked.complete([Message.user("x")])
        summary = self.store.for_task("t6")
        self.assertEqual(summary.failures, 1)
        self.assertEqual(summary.unpriced_calls, 0)
        self.assertTrue(summary.cost_is_complete)
        self.assertEqual(tracked.summary().unpriced_calls, 0)

    def test_tracked_provider_is_transparent(self):
        provider = ScriptedProvider([text_response("hi")], model="m")
        tracked = TrackedProvider(provider, self.store)
        self.assertEqual(tracked.name, provider.name)
        self.assertEqual(tracked.model, "m")
        self.assertTrue(tracked.is_available())

    def test_cumulative_total_spans_tasks(self):
        provider = ScriptedProvider([text_response("a"), text_response("b")])
        TrackedProvider(provider, self.store, task_id="t4").complete([Message.user("x")])
        TrackedProvider(provider, self.store, task_id="t5").complete([Message.user("x")])
        self.assertEqual(self.store.total().calls, 2)


if __name__ == "__main__":
    unittest.main()
