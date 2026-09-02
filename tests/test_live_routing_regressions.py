from __future__ import annotations

from unittest.mock import patch

import pytest

from brain.router import route_command
from providers.base import Message, ModelResponse, ProviderRequestError, ToolSpec
from providers.model_registry import ModelRegistry, ModelRoute, infer_capabilities
from providers.pool import MultiModelProvider, classify_capability
from vault.policy import LIGHT, FULL, assess


@pytest.mark.parametrize(
    "command,route_type,tool",
    [
        ("Turn the volume down.", "tool", "volume_down"),
        ("Open YouTube.", "tool", "open_website"),
        ("Go to youtube.com.", "tool", "open_website"),
        ("Open Chrome and go to github.com.", "local_plan", None),
        ("Search Jynxzi on Google.", "tool", "open_website"),
        ("Search for Jynxzi on YouTube.", "tool", "open_website"),
    ],
)
def test_deterministic_commands_finish_routing_without_a_reasoning_provider(command, route_type, tool):
    with patch("providers.pool.MultiModelProvider.complete", side_effect=AssertionError("reasoning provider called")):
        route = route_command(command)
    assert route["type"] == route_type
    if tool:
        assert route["tool"] == tool


@pytest.mark.parametrize(
    "command,expected_url",
    [
        ("Search for Jynxzi on YouTube.", "https://www.youtube.com/results?search_query=jynxzi"),
        ("Search Jynxzi on YouTube.", "https://www.youtube.com/results?search_query=jynxzi"),
        ("Find Jynxzi on YouTube.", "https://www.youtube.com/results?search_query=jynxzi"),
        ("YouTube search Jynxzi.", "https://www.youtube.com/results?search_query=jynxzi"),
        ("Open YouTube and search for Jynxzi.", "https://www.youtube.com/results?search_query=jynxzi"),
        ("Search Jynxzi on Google.", "https://www.google.com/search?q=jynxzi"),
    ],
)
def test_search_phrasings_use_direct_encoded_navigation(command, expected_url):
    route = route_command(command)
    assert route == {"type": "tool", "tool": "open_website", "arguments": {"url": expected_url}}


@pytest.mark.parametrize(
    "command",
    [
        "Explain in one sentence why the sky is blue.",
        "What does Python IndexError mean?",
        "Search for Jynxzi on YouTube.",
    ],
)
def test_non_missions_are_not_eligible_for_job_selection(command):
    policy = assess(command)
    assert policy.mode == LIGHT
    assert not policy.selects_job


def test_real_project_fix_is_a_reusable_execution_mission():
    policy = assess("Open my JARVIS project and fix the IndexError.")
    assert policy.mode == FULL
    assert policy.selects_job


class FakeProvider:
    def __init__(self, name, outcomes):
        self.name, self.outcomes, self.calls = name, list(outcomes), []

    def is_available(self):
        return True

    def complete(self, messages, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return ModelResponse(outcome, model=kwargs["model"], provider=self.name)


def test_tool_capability_failure_is_cached_and_falls_back_without_retrying_bad_route(tmp_path):
    cache = tmp_path / "models.json"
    registry = ModelRegistry(cache)
    registry.add_route(ModelRoute("groq-bad", "mystery-model", "groq", "mystery-model", 1, capabilities=frozenset({"chat", "tool_use"})))
    registry.add_route(ModelRoute("openrouter-good", "tools", "openrouter", "tools", 2, capabilities=frozenset({"chat", "tool_use"})))
    bad = FakeProvider("groq", [ProviderRequestError("`tool calling` is not supported with this model")])
    good = FakeProvider("openrouter", ["done", "done again"])
    pool = MultiModelProvider(registry, {"groq": bad, "openrouter": good})
    tools = [ToolSpec("x", "x", {"type": "object"})]

    assert pool.complete([Message.user("do it")], tools=tools).provider == "openrouter"
    assert "tool_use" not in registry.routes["groq-bad"].capabilities
    assert pool.complete([Message.user("do it again")], tools=tools).provider == "openrouter"
    assert len(bad.calls) == 1
    assert cache.exists()


def test_known_classifier_and_audio_models_are_never_inferred_as_tool_models():
    for model in ("meta-llama/llama-prompt-guard-2-22m", "openai/gpt-oss-safeguard-20b", "whisper-large-v3", "canopylabs/orpheus-v1-english"):
        assert "tool_use" not in infer_capabilities(model)


def test_tool_schemas_override_coding_and_vision_text_classification():
    tools = [ToolSpec("x", "x", {"type": "object"})]
    assert classify_capability([Message.user("debug this screenshot")], tools) == "tool_use"


def test_non_tool_model_never_receives_tool_definitions():
    registry = ModelRegistry()
    registry.add_route(ModelRoute("chat-only", "chat", "groq", "chat", capabilities=frozenset({"chat"})))
    provider = FakeProvider("groq", ["must not be called"])
    backend = MultiModelProvider(registry, {"groq": provider})
    with pytest.raises(Exception, match="All eligible model routes failed"):
        backend.complete([Message.user("use a tool")], tools=[ToolSpec("x", "x", {"type": "object"})])
    assert provider.calls == []


def test_stale_cached_tool_claim_for_prompt_guard_is_sanitized_on_load(tmp_path):
    path = tmp_path / "models.json"
    registry = ModelRegistry(path)
    registry.add_route(ModelRoute("r", "prompt-guard", "groq", "meta-llama/llama-prompt-guard-2-22m", capabilities=frozenset({"chat", "tool_use"})))
    assert "tool_use" not in registry.routes["r"].capabilities
