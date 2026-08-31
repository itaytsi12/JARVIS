"""Build the default route pool without network work on the startup path."""
from __future__ import annotations

import threading
import os
from typing import Any

from config import get_config
from providers.anthropic_provider import AnthropicProvider
from providers.model_registry import ModelRegistry, ModelRoute, infer_capabilities
from providers.openai_compatible import OpenAICompatibleProvider
from providers.pool import MultiModelProvider


_ENDPOINTS = {
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "mistral": "https://api.mistral.ai/v1",
    "huggingface": "https://router.huggingface.co/v1",
    "moonshot": "https://api.moonshot.ai/v1",
    "github": "https://models.github.ai/inference",
    "vercel": "https://ai-gateway.vercel.sh/v1",
}

# Small last-known seed only; discovery replaces/extends these at runtime.
_SEEDS = {
    "openrouter": ("openrouter/free", 40),
    "groq": ("openai/gpt-oss-120b", 20),
    "cerebras": ("gpt-oss-120b", 25),
    "google": ("gemini-2.5-flash", 30),
    "nvidia": ("meta/llama-3.1-70b-instruct", 35),
    "mistral": ("open-mistral-nemo", 45),
    "moonshot": ("kimi-k2-instruct", 45),
    "github": ("openai/gpt-4.1-mini", 60),
    "vercel": ("openai/gpt-oss-120b", 60),
}


def build_multi_model_provider() -> MultiModelProvider:
    config = get_config()
    registry = ModelRegistry(config.data_dir / "model_registry_cache.json", config.model_registry_cache_ttl)
    registry.load_cache()
    keys = {
        "nvidia": config.nvidia_api_key, "openrouter": config.openrouter_api_key,
        "groq": config.groq_api_key, "cerebras": config.cerebras_api_key,
        "google": config.google_api_key, "mistral": config.mistral_api_key,
        "huggingface": config.hf_token, "moonshot": config.moonshot_api_key,
        "github": config.github_token, "vercel": config.vercel_ai_gateway_api_key,
    }
    providers: dict[str, Any] = {
        name: OpenAICompatibleProvider(name=name, base_url=_ENDPOINTS[name], api_key=key, timeout=config.agent_request_timeout)
        for name, key in keys.items()
    }
    providers["lmstudio"] = OpenAICompatibleProvider(name="lmstudio", base_url=config.lmstudio_base_url, api_key=None, credential_required=False, timeout=config.agent_request_timeout)
    providers["ollama"] = OpenAICompatibleProvider(name="ollama", base_url=config.ollama_base_url, api_key=None, credential_required=False, timeout=config.agent_request_timeout)
    if config.enable_anthropic_fallback:
        providers["anthropic"] = AnthropicProvider()
    for provider, (model, priority) in _SEEDS.items():
        caps = infer_capabilities(model)
        registry.add_route(ModelRoute(f"{provider}:{model}", model, provider, model, priority, credential_ref=provider, capabilities=caps))
    if config.enable_anthropic_fallback:
        registry.add_route(ModelRoute(f"anthropic:{config.agent_model}", config.agent_model, "anthropic", config.agent_model, 1000, free_tier=False, credential_ref="ANTHROPIC_API_KEY", capabilities=frozenset({"chat", "reasoning", "coding", "vision", "planning", "tool_use"})))
    pool = MultiModelProvider(registry, providers)
    # Discovery is lazy/background: never hold up JARVIS startup.
    if "PYTEST_CURRENT_TEST" not in os.environ:
        threading.Thread(target=_refresh_discovery, args=(registry, providers, config.provider_discovery_timeout), daemon=True, name="jarvis-model-discovery").start()
    return pool


def _refresh_discovery(registry: ModelRegistry, providers: dict[str, Any], timeout: float) -> None:
    changed = False
    for name, provider in providers.items():
        if not registry.discovery_stale(name) or not hasattr(provider, "discover_models"):
            continue
        routes = provider.discover_models(timeout=timeout, priority=_SEEDS.get(name, ("", 80))[1], free_only=name == "openrouter")
        if routes:
            registry.replace_discovered(name, routes)
            changed = True
    if changed:
        registry.save_cache()


__all__ = ["build_multi_model_provider"]
