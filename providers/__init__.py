"""Swappable model providers.

`providers.base` defines the vendor-neutral interface; every concrete
backend (Anthropic today, a local model later) implements it. Nothing
outside this package imports a vendor SDK.
"""
from providers.base import (
    Message,
    ModelProvider,
    ModelResponse,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimited,
    ProviderRequestError,
    ProviderServerError,
    ProviderTimeout,
    ProviderUnavailable,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    Usage,
)
from providers.registry import (
    available_providers,
    get_agent_provider,
    get_provider,
    provider_status,
    register_provider,
)
from providers.usage import (
    TrackedProvider,
    UsageRecord,
    UsageStore,
    UsageSummary,
    get_usage_store,
)

__all__ = [
    "Message",
    "ModelProvider",
    "ModelResponse",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateLimited",
    "ProviderRequestError",
    "ProviderServerError",
    "ProviderTimeout",
    "ProviderUnavailable",
    "ToolCall",
    "ToolOutcome",
    "ToolSpec",
    "Usage",
    "available_providers",
    "get_agent_provider",
    "get_provider",
    "provider_status",
    "register_provider",
    "TrackedProvider",
    "UsageRecord",
    "UsageStore",
    "UsageSummary",
    "get_usage_store",
]
