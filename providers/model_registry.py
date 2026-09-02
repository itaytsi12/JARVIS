"""Central model/route registry with a last-known-good discovery cache."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


CAPABILITIES = frozenset({"fast", "chat", "reasoning", "coding", "vision", "planning", "tool_use", "structured_output", "local"})


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    display_name: str
    capabilities: frozenset[str] = field(default_factory=lambda: frozenset({"chat"}))
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    context_window: int | None = None
    size_class: str | None = None
    open_weight: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRoute:
    route_id: str
    model_id: str
    provider: str
    provider_model_name: str
    priority: int = 100
    enabled: bool = True
    free_tier: bool = True
    timeout: float = 60.0
    max_retries: int = 0
    base_url: str = ""
    credential_ref: str | None = None
    capabilities: frozenset[str] = field(default_factory=lambda: frozenset({"chat"}))
    metadata: dict[str, Any] = field(default_factory=dict)


def infer_capabilities(model_name: str) -> frozenset[str]:
    name = model_name.lower()
    caps = {"chat"}
    # These are classifiers, audio generators/transcribers, or provider
    # orchestration endpoints—not general function-calling chat models.
    non_tool_roles = ("guard", "safety", "whisper", "orpheus", "lyria", "compound")
    if any(x in name for x in ("flash", "small", "mini", "instant", "8b", "3b", "1b")):
        caps.add("fast")
    if any(x in name for x in ("reason", "deepseek-r1", "qwq", "thinking", "gpt-oss", "kimi")):
        caps.update(("reasoning", "planning"))
    if any(x in name for x in ("code", "coder", "codestral", "devstral", "gpt-oss")):
        caps.add("coding")
    if any(x in name for x in ("vision", "vl", "multimodal", "gemini")):
        caps.add("vision")
    if not any(x in name for x in non_tool_roles) and any(x in name for x in ("llama", "qwen", "mistral", "gemini", "gpt", "claude", "command")):
        caps.add("tool_use")
    return frozenset(caps)


def _known_non_tool_role(model_name: str) -> bool:
    name = model_name.lower()
    return any(x in name for x in ("guard", "safety", "whisper", "orpheus", "lyria", "compound"))


class ModelRegistry:
    def __init__(self, cache_path: Path | str | None = None, cache_ttl_seconds: float = 21600):
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache_ttl_seconds = cache_ttl_seconds
        self.models: dict[str, ModelInfo] = {}
        self.routes: dict[str, ModelRoute] = {}
        self.discovered_at: dict[str, float] = {}
        self._lock = threading.RLock()

    def add_model(self, model: ModelInfo) -> None:
        with self._lock:
            existing = self.models.get(model.model_id)
            if existing:
                model = ModelInfo(
                    model_id=model.model_id,
                    display_name=model.display_name or existing.display_name,
                    capabilities=existing.capabilities | model.capabilities,
                    supports_tools=existing.supports_tools or model.supports_tools,
                    supports_vision=existing.supports_vision or model.supports_vision,
                    supports_streaming=existing.supports_streaming or model.supports_streaming,
                    context_window=model.context_window or existing.context_window,
                    size_class=model.size_class or existing.size_class,
                    open_weight=model.open_weight if model.open_weight is not None else existing.open_weight,
                    metadata={**existing.metadata, **model.metadata},
                )
            self.models[model.model_id] = model

    def add_route(self, route: ModelRoute) -> None:
        caps = route.capabilities or infer_capabilities(route.provider_model_name)
        # Revalidate cached/discovered claims against known model roles so an
        # old heuristic-produced cache cannot keep a safety/audio model alive
        # as a tool route after the inference rule is fixed.
        if _known_non_tool_role(route.provider_model_name):
            caps = frozenset(caps - {"tool_use"})
        if caps != route.capabilities:
            raw = asdict(route)
            raw["capabilities"] = caps
            route = ModelRoute(**raw)
        self.add_model(ModelInfo(
            route.model_id, route.model_id, caps, "tool_use" in caps, "vision" in caps,
            context_window=route.metadata.get("context_window"), metadata=dict(route.metadata),
        ))
        with self._lock:
            self.routes[route.route_id] = route

    def replace_discovered(self, provider: str, routes: Iterable[ModelRoute]) -> None:
        with self._lock:
            self.routes = {k: v for k, v in self.routes.items() if not (v.provider == provider and v.metadata.get("discovered"))}
        for route in routes:
            self.add_route(route)
        self.discovered_at[provider] = time.time()

    def eligible(self, capability: str) -> list[ModelRoute]:
        with self._lock:
            routes = [r for r in self.routes.values() if r.enabled and (capability in r.capabilities or capability == "chat")]
        return sorted(routes, key=lambda r: (r.priority, r.provider, r.provider_model_name))

    def mark_capability_unsupported(self, route_id: str, capability: str) -> None:
        """Persist a provider-confirmed negative capability for a route."""
        with self._lock:
            route = self.routes.get(route_id)
            if route is None or capability not in route.capabilities:
                return
            raw = asdict(route)
            raw["capabilities"] = frozenset(route.capabilities - {capability})
            raw["metadata"] = {**route.metadata, f"unsupported_{capability}": True}
            self.routes[route_id] = ModelRoute(**raw)
        self.save_cache()

    def discovery_stale(self, provider: str, now: float | None = None) -> bool:
        return (now or time.time()) - self.discovered_at.get(provider, 0) >= self.cache_ttl_seconds

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "discovered_at": self.discovered_at,
            "models": [{**asdict(m), "capabilities": sorted(m.capabilities)} for m in self.models.values()],
            "routes": [{**asdict(r), "capabilities": sorted(r.capabilities)} for r in self.routes.values()],
        }
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.cache_path)

    def load_cache(self) -> bool:
        if not self.cache_path or not self.cache_path.exists():
            return False
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            for raw in payload.get("models", []):
                raw["capabilities"] = frozenset(raw.get("capabilities", ()))
                self.add_model(ModelInfo(**raw))
            for raw in payload.get("routes", []):
                raw["capabilities"] = frozenset(raw.get("capabilities", ()))
                self.add_route(ModelRoute(**raw))
            self.discovered_at.update({k: float(v) for k, v in payload.get("discovered_at", {}).items()})
            return True
        except (OSError, ValueError, TypeError):
            return False


__all__ = ["CAPABILITIES", "ModelInfo", "ModelRoute", "ModelRegistry", "infer_capabilities"]
