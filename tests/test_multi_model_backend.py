from __future__ import annotations

import threading

import pytest

from providers.base import Message, ModelResponse, ProviderAuthError, ProviderRateLimited, ProviderRequestError, ProviderServerError, ProviderTimeout
from providers.local_models import LocalModelManager
from providers.model_registry import ModelRegistry, ModelRoute
from providers.pool import MultiModelProvider, TaskCheckpoint


class FakeProvider:
    def __init__(self, name, outcomes, available=True):
        self.name, self.outcomes, self.available = name, list(outcomes), available
        self.calls = []
    def is_available(self): return self.available
    def unavailable_reason(self): return None if self.available else "missing_api_key"
    def complete(self, messages, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception): raise outcome
        return ModelResponse(outcome, model=kwargs["model"], provider=self.name)


def pool(*providers, clock=lambda: 100.0, circuit_failures=1):
    registry = ModelRegistry()
    mapping = {}
    for index, provider in enumerate(providers):
        mapping[provider.name] = provider
        registry.add_route(ModelRoute(f"route-{index}", "shared-model", provider.name, f"model-{index}", index, capabilities=frozenset({"chat", "fast", "tool_use"})))
    return MultiModelProvider(registry, mapping, clock=clock, circuit_failures=circuit_failures, cooldown_seconds=10)


def test_first_provider_succeeds_and_response_is_normalized():
    backend = pool(FakeProvider("one", ["ok"]), FakeProvider("two", ["unused"]))
    response = backend.complete([Message.user("hello")], request_id="abc")
    assert (response.text, response.request_id, response.route_id, response.provider) == ("ok", "abc", "route-0", "one")


@pytest.mark.parametrize("error", [ProviderRateLimited("limited"), ProviderTimeout("slow"), ProviderServerError("503")])
def test_retryable_failure_falls_back_with_same_request_id(error):
    first, second = FakeProvider("one", [error]), FakeProvider("two", ["ok"])
    response = pool(first, second).complete([Message.user("hello")], request_id="same")
    assert response.request_id == "same" and response.provider == "two"


def test_auth_failure_marks_route_unavailable_and_falls_back():
    backend = pool(FakeProvider("one", [ProviderAuthError("bad")]), FakeProvider("two", ["ok"]))
    assert backend.complete([Message.user("hello")]).provider == "two"
    assert backend.health("route-0").state == "auth_error"


def test_retry_after_and_cooldown_are_respected_then_route_recovers():
    now = [100.0]
    first, second = FakeProvider("one", [ProviderRateLimited("quota", 30), "recovered"]), FakeProvider("two", ["fallback", "other"])
    backend = pool(first, second, clock=lambda: now[0])
    assert backend.complete([Message.user("hello")]).provider == "two"
    assert backend.health("route-0").cooldown_until == 130
    assert backend.complete([Message.user("hello")]).provider == "two"
    now[0] = 131
    assert backend.complete([Message.user("hello")]).provider == "one"


def test_same_model_can_have_multiple_provider_routes():
    backend = pool(FakeProvider("one", [ProviderServerError("gone")]), FakeProvider("two", ["ok"]))
    assert {r.model_id for r in backend.registry.routes.values()} == {"shared-model"}
    assert backend.complete([Message.user("hello")]).provider == "two"


def test_malformed_internal_request_does_not_fan_out():
    first, second = FakeProvider("one", [ProviderRequestError("schema")]), FakeProvider("two", ["must not run"])
    with pytest.raises(ProviderRequestError): pool(first, second).complete([Message.user("hello")])
    assert not second.calls


def test_all_providers_fail_cleanly_and_missing_provider_is_skipped():
    with pytest.raises(Exception, match="All eligible model routes failed"):
        pool(FakeProvider("one", [ProviderTimeout("x")]), FakeProvider("two", [], available=False)).complete([Message.user("hello")])


def test_unavailable_local_servers_do_not_break_startup():
    backend = pool(FakeProvider("lmstudio", [], available=False), FakeProvider("ollama", [], available=False))
    assert backend.is_available() is False


def test_checkpoint_is_passed_to_next_model_without_hidden_reasoning():
    first, second = FakeProvider("one", [ProviderTimeout("x")]), FakeProvider("two", ["ok"])
    checkpoint = TaskCheckpoint(original_goal="fix router", completed_steps=["inspected provider"], next_action="run tests")
    pool(first, second).complete([Message.user("work")], system="base", checkpoint=checkpoint)
    assert "inspected provider" in second.calls[0]["system"] and "do not repeat" in second.calls[0]["system"]


def test_two_concurrent_requests_keep_independent_ids():
    barrier = threading.Barrier(2)
    class ConcurrentFake(FakeProvider):
        def complete(self, messages, **kwargs): barrier.wait(); return ModelResponse(messages[0].content, model=kwargs["model"])
    backend = pool(ConcurrentFake("one", []))
    results = {}
    threads = [threading.Thread(target=lambda key=k: results.setdefault(key, backend.complete([Message.user(key)], request_id=key))) for k in ("a", "b")]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert {r.request_id for r in results.values()} == {"a", "b"}


def test_registry_cache_last_known_good(tmp_path):
    path = tmp_path / "models.json"
    original = ModelRegistry(path); original.add_route(ModelRoute("r", "m", "p", "pm")); original.discovered_at["p"] = 12; original.save_cache()
    loaded = ModelRegistry(path)
    assert loaded.load_cache() and loaded.routes["r"].provider_model_name == "pm"
    path.write_text("broken", encoding="utf-8")
    assert loaded.load_cache() is False and "r" in loaded.routes


def test_local_specialist_load_unload_and_active_idle_protection():
    now = [0.0]; unloaded = []
    manager = LocalModelManager({"coding": "qwen-coder"}, unload=lambda model: unloaded.append(model) or True, idle_minutes=1, clock=lambda: now[0])
    assert manager.load_local_model("coding") == "qwen-coder"
    manager.request_started(); now[0] = 120
    assert not manager.unload_if_idle()
    manager.request_finished(); now[0] = 181
    assert manager.unload_if_idle() and unloaded == ["qwen-coder"]
