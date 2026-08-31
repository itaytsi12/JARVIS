"""Capability routing, health/cooldowns, checkpoints and route fallback."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from config import events
from providers.base import (
    Message, ModelResponse, ProviderAuthError, ProviderError,
    ProviderRateLimited, ProviderRequestError, ProviderServerError,
    ProviderTimeout, ProviderUnavailable, ToolSpec,
)
from providers.model_registry import ModelRegistry, ModelRoute

log = logging.getLogger("jarvis.providers.pool")


def classify_capability(messages: list[Message], tools: list[ToolSpec] | None = None) -> str:
    text = " ".join(message.content for message in messages if message.content).lower()
    if any(x in text for x in ("image", "screenshot", "photo", "what do you see")):
        return "vision"
    if any(x in text for x in ("code", "debug", "repository", "pytest", "traceback", "function", "class ")):
        return "coding"
    if tools:
        return "tool_use"
    if any(x in text for x in ("plan", "multi-step", "step by step", "autonomous")):
        return "planning"
    if any(x in text for x in ("reason", "analyze", "prove", "compare", "why")):
        return "reasoning"
    if len(text) < 180:
        return "fast"
    return "chat"


@dataclass
class RouteHealth:
    state: str = "healthy"
    failures: int = 0
    cooldown_until: float = 0.0
    last_error: str | None = None
    successes: int = 0
    total_latency_ms: float = 0.0


@dataclass
class TaskCheckpoint:
    original_goal: str = ""
    capability: str = "chat"
    plan: list[str] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    important_tool_results: list[str] = field(default_factory=list)
    files_inspected: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    validated_outputs: list[str] = field(default_factory=list)
    current_step: str = ""
    next_action: str = ""
    constraints: list[str] = field(default_factory=list)
    continuation_count: int = 0

    def operational_summary(self) -> str:
        fields = []
        for label, value in (("Goal", self.original_goal), ("Completed", self.completed_steps), ("Important tool results", self.important_tool_results), ("Current step", self.current_step), ("Next", self.next_action), ("Constraints", self.constraints)):
            if value:
                fields.append(f"{label}: {value}")
        return "\n".join(fields)


class MultiModelProvider:
    name = "multi_model"

    def __init__(self, registry: ModelRegistry, providers: dict[str, Any], *, circuit_failures: int = 3, cooldown_seconds: float = 60.0, clock: Callable[[], float] = time.time):
        self.registry = registry
        self.providers = providers
        self.model = "dynamic"
        self.circuit_failures = circuit_failures
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self._health: dict[str, RouteHealth] = {}
        self._lock = threading.RLock()
        self.stats: dict[str, dict[str, float]] = {}

    def is_available(self) -> bool:
        # A configured local base URL is not evidence that the server is
        # running. Local providers become eligible only after discovery has
        # produced at least one concrete route (possibly from the cache).
        for route in self.registry.routes.values():
            provider = self.providers.get(route.provider)
            if provider is not None and provider.is_available():
                return True
        return False

    def unavailable_reason(self) -> str | None:
        return None if self.is_available() else "no_eligible_routes"

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "available": self.is_available(), "routes": len(self.registry.routes), "providers": sorted(self.providers)}

    def health(self, route_id: str) -> RouteHealth:
        with self._lock:
            return self._health.setdefault(route_id, RouteHealth())

    def complete(self, messages: list[Message], *, system: str | None = None, tools: list[ToolSpec] | None = None, max_tokens: int | None = None, temperature: float | None = None, timeout: float | None = None, model: str | None = None, effort: str | None = None, cache: bool = True, on_text: Any = None, capability: str | None = None, request_id: str | None = None, checkpoint: TaskCheckpoint | None = None) -> ModelResponse:
        request_id = request_id or uuid.uuid4().hex
        capability = capability or classify_capability(messages, tools)
        checkpoint = checkpoint or TaskCheckpoint(original_goal=messages[0].content if messages else "", capability=capability)
        routes = self.registry.eligible(capability)
        if model:
            routes = [r for r in routes if r.model_id == model or r.provider_model_name == model]
        events.publish("request_started", request_id=request_id, capability=capability)
        errors: list[str] = []
        attempted = 0
        for route in routes:
            provider = self.providers.get(route.provider)
            if provider is None or not provider.is_available():
                continue
            health = self.health(route.route_id)
            now = self.clock()
            with self._lock:
                if health.cooldown_until > now:
                    continue
                if health.cooldown_until and health.cooldown_until <= now:
                    health.state, health.cooldown_until = "degraded", 0.0
            attempted += 1
            events.publish("route_selected", request_id=request_id, capability=capability, provider=route.provider, model=route.model_id, route_id=route.route_id, state="active")
            events.publish("model_active", request_id=request_id, capability=capability, provider=route.provider, model=route.model_id, route_id=route.route_id, state="thinking")
            continuation_system = system
            if checkpoint.continuation_count:
                continuation_system = (system or "") + "\n\nContinue from this operational checkpoint; do not repeat completed actions:\n" + checkpoint.operational_summary()
            started = time.perf_counter()
            try:
                response = provider.complete(messages, system=continuation_system, tools=tools, max_tokens=max_tokens, temperature=temperature, timeout=timeout or route.timeout, model=route.provider_model_name, effort=effort, cache=cache, on_text=on_text)
            except ProviderRequestError:
                events.publish("model_error", request_id=request_id, capability=capability, provider=route.provider, model=route.model_id, route_id=route.route_id, state="invalid_request")
                raise
            except ProviderError as exc:
                errors.append(f"{route.route_id}: {type(exc).__name__}: {exc}")
                self._failed(route, exc)
                checkpoint.continuation_count += 1
                event = "model_rate_limited" if isinstance(exc, ProviderRateLimited) else "model_error"
                events.publish(event, request_id=request_id, capability=capability, provider=route.provider, model=route.model_id, route_id=route.route_id, state=self.health(route.route_id).state)
                events.publish("fallback_started", request_id=request_id, capability=capability, provider=route.provider, model=route.model_id, route_id=route.route_id, state="fallback", reason=type(exc).__name__)
                continue
            latency = (time.perf_counter() - started) * 1000
            response.request_id = request_id
            response.route_id = route.route_id
            response.capability = capability
            response.provider = route.provider
            self._succeeded(route, latency)
            events.publish("request_completed", request_id=request_id, capability=capability, provider=route.provider, model=response.model, route_id=route.route_id, state="completed")
            log.info("model_request request_id=%s capability=%s provider=%s model=%s route=%s attempt=%s latency_ms=%.1f success=true", request_id, capability, route.provider, response.model, route.route_id, attempted, latency)
            return response
        events.publish("request_completed", request_id=request_id, capability=capability, state="failed")
        raise ProviderUnavailable("All eligible model routes failed" + (": " + " | ".join(errors) if errors else ""))

    def _failed(self, route: ModelRoute, exc: ProviderError) -> None:
        health = self.health(route.route_id)
        now = self.clock()
        with self._lock:
            health.failures += 1
            health.last_error = type(exc).__name__
            cooldown = self.cooldown_seconds
            if isinstance(exc, ProviderRateLimited):
                health.state = "quota_exhausted" if "quota" in str(exc).lower() or "free" in str(exc).lower() else "rate_limited"
                cooldown = max(0.0, exc.retry_after or cooldown)
            elif isinstance(exc, ProviderAuthError):
                health.state, cooldown = "auth_error", max(cooldown, 3600.0)
            elif isinstance(exc, ProviderTimeout): health.state = "degraded"
            elif isinstance(exc, ProviderServerError): health.state = "degraded"
            else: health.state = "offline"
            if health.failures >= self.circuit_failures or isinstance(exc, (ProviderRateLimited, ProviderAuthError)):
                health.cooldown_until = now + cooldown
            stats = self.stats.setdefault(route.route_id, {"requests": 0, "successes": 0, "failures": 0, "fallbacks": 0, "latency_ms": 0})
            stats["requests"] += 1
            stats["failures"] += 1
            stats["fallbacks"] += 1

    def _succeeded(self, route: ModelRoute, latency: float) -> None:
        with self._lock:
            health = self.health(route.route_id)
            health.state, health.failures, health.cooldown_until = "healthy", 0, 0.0
            health.successes += 1
            health.total_latency_ms += latency
            stats = self.stats.setdefault(route.route_id, {"requests": 0, "successes": 0, "failures": 0, "fallbacks": 0, "latency_ms": 0})
            stats["requests"] += 1; stats["successes"] += 1; stats["latency_ms"] += latency


__all__ = ["MultiModelProvider", "RouteHealth", "TaskCheckpoint", "classify_capability"]
