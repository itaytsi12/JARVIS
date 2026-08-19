"""Typed, centralized configuration for the JARVIS agent runtime.

Design rules this file follows:

- Secrets are read from the environment only, never written to a file,
  never included in `describe()` / `__repr__` output, and never logged.
- Every setting has a working default so JARVIS runs with an empty `.env`.
- `get_config()` returns a process-wide cached instance (config is read
  once, not on every request -- `#24 responsiveness`); `reload_config()`
  exists for tests and for an explicit runtime reload.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_TRUTHY = {"1", "true", "yes", "on"}


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in _TRUTHY


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _text(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


def _secret(name: str) -> str | None:
    raw = os.getenv(name)
    raw = raw.strip() if isinstance(raw, str) else raw
    return raw or None


@dataclass(frozen=True)
class JarvisConfig:
    """One immutable snapshot of every setting the agent runtime needs."""

    # ---- paths -------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_text("JARVIS_DATA_DIR", str(Path.cwd() / "data"))))
    debug: bool = field(default_factory=lambda: _flag("JARVIS_DEBUG", False))

    # ---- model provider ---------------------------------------------
    anthropic_api_key: str | None = field(default_factory=lambda: _secret("ANTHROPIC_API_KEY"), repr=False)
    agent_model: str = field(default_factory=lambda: _text("JARVIS_AGENT_MODEL", "claude-opus-5"))
    agent_max_tokens: int = field(default_factory=lambda: _int("JARVIS_AGENT_MAX_TOKENS", 4096))
    agent_temperature: float = field(default_factory=lambda: _float("JARVIS_AGENT_TEMPERATURE", 0.0))
    agent_request_timeout: float = field(default_factory=lambda: _float("JARVIS_AGENT_REQUEST_TIMEOUT", 120.0))
    agent_max_provider_retries: int = field(default_factory=lambda: _int("JARVIS_AGENT_PROVIDER_RETRIES", 2))
    provider_order: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            part.strip().lower()
            for part in _text("JARVIS_PROVIDER_ORDER", "anthropic").split(",")
            if part.strip()
        )
    )

    # ---- agent loop safety limits -----------------------------------
    max_agent_steps: int = field(default_factory=lambda: _int("JARVIS_MAX_AGENT_STEPS", 25))
    max_action_retries: int = field(default_factory=lambda: _int("JARVIS_MAX_ACTION_RETRIES", 2))
    max_consecutive_failures: int = field(default_factory=lambda: _int("JARVIS_MAX_CONSECUTIVE_FAILURES", 4))
    agent_task_timeout: float = field(default_factory=lambda: _float("JARVIS_AGENT_TASK_TIMEOUT", 900.0))

    # ---- escalation --------------------------------------------------
    agent_enabled: bool = field(default_factory=lambda: _flag("JARVIS_AGENT_ENABLED", True))
    agent_escalation_enabled: bool = field(default_factory=lambda: _flag("JARVIS_AGENT_ESCALATION", True))

    # ---- context budgeting ------------------------------------------
    context_max_chars: int = field(default_factory=lambda: _int("JARVIS_CONTEXT_MAX_CHARS", 12000))
    context_recent_turns: int = field(default_factory=lambda: _int("JARVIS_CONTEXT_RECENT_TURNS", 6))
    context_max_memories: int = field(default_factory=lambda: _int("JARVIS_CONTEXT_MAX_MEMORIES", 6))
    context_max_episodes: int = field(default_factory=lambda: _int("JARVIS_CONTEXT_MAX_EPISODES", 3))
    context_max_observation_chars: int = field(default_factory=lambda: _int("JARVIS_CONTEXT_MAX_OBSERVATION_CHARS", 4000))

    # ---- concurrency -------------------------------------------------
    max_concurrent_tasks: int = field(default_factory=lambda: _int("JARVIS_MAX_CONCURRENT_TASKS", 4))

    # ---- terminal tool ----------------------------------------------
    terminal_timeout: float = field(default_factory=lambda: _float("JARVIS_TERMINAL_TIMEOUT", 120.0))
    terminal_max_output_chars: int = field(default_factory=lambda: _int("JARVIS_TERMINAL_MAX_OUTPUT", 20000))
    terminal_allow_unrestricted: bool = field(default_factory=lambda: _flag("JARVIS_TERMINAL_UNRESTRICTED", False))

    # ---- cost tracking ----------------------------------------------
    cost_tracking_enabled: bool = field(default_factory=lambda: _flag("JARVIS_COST_TRACKING", True))
    pricing_file: str = field(default_factory=lambda: _text("JARVIS_PRICING_FILE", ""))

    # ---- memory ------------------------------------------------------
    memory_min_importance: int = field(default_factory=lambda: _int("JARVIS_MEMORY_MIN_IMPORTANCE", 2))
    memory_enabled: bool = field(default_factory=lambda: _flag("JARVIS_MEMORY_ENABLED", True))

    # ---- derived paths ----------------------------------------------
    @property
    def agent_db_path(self) -> Path:
        override = _text("JARVIS_AGENT_DB_PATH")
        return Path(override) if override else self.data_dir / "jarvis_agent.sqlite3"

    @property
    def usage_db_path(self) -> Path:
        override = _text("JARVIS_USAGE_DB_PATH")
        return Path(override) if override else self.data_dir / "jarvis_usage.sqlite3"

    @property
    def has_anthropic_credentials(self) -> bool:
        return bool(self.anthropic_api_key)

    def describe(self) -> dict[str, object]:
        """A log-safe view: secrets become a presence flag, never a value."""
        payload: dict[str, object] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name.endswith(("_key", "_token", "_secret", "_password")):
                payload[item.name] = "<set>" if value else "<unset>"
                continue
            payload[item.name] = str(value) if isinstance(value, Path) else value
        payload["agent_db_path"] = str(self.agent_db_path)
        payload["usage_db_path"] = str(self.usage_db_path)
        return payload


_CONFIG: JarvisConfig | None = None
_CONFIG_LOCK = threading.Lock()


def get_config() -> JarvisConfig:
    global _CONFIG
    if _CONFIG is None:
        with _CONFIG_LOCK:
            if _CONFIG is None:
                _CONFIG = JarvisConfig()
    return _CONFIG


def reload_config() -> JarvisConfig:
    """Re-read every setting from the current environment.

    Used by tests (which patch `os.environ`) and by an explicit runtime
    reload; ordinary request handling always uses the cached instance.
    """
    global _CONFIG
    with _CONFIG_LOCK:
        _CONFIG = JarvisConfig()
    return _CONFIG
