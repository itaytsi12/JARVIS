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

#: The repository root, derived from THIS file's location -- never from the
#: process's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Load `.env` from the project root explicitly.
#:
#: A bare `load_dotenv()` searches upward from the CURRENT WORKING
#: DIRECTORY, so JARVIS's configuration silently depended on where the
#: process happened to be started. Launched from anywhere but the
#: repository root -- a Task Scheduler entry, a shortcut, another
#: application importing `config` -- nothing was found, `get_config()`
#: cached a config with no API key and the DEFAULT `agent_model`, and the
#: later explicit `load_dotenv(PROJECT_ROOT / ".env")` in
#: `voice/tray_app.py::run_tray` could not undo it (the config is cached,
#: and `load_dotenv` never overrides an already-set variable). The result
#: was "Complex request, but no agent provider is configured" with a
#: perfectly valid key sitting in `.env`.
#:
#: This is also the ONE place `.env` is loaded. Other modules must not call
#: `load_dotenv` themselves: a second, CWD-relative load re-introduces
#: exactly the ordering hazard above, and which file wins then depends on
#: import order.
load_dotenv(PROJECT_ROOT / ".env")

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


#: Public, empty-tolerant environment readers.
#:
#: A variable that is PRESENT BUT EMPTY in `.env` (`FOO=`) is what
#: `os.getenv("FOO", "1800")` returns as `""`, not as the default -- so a
#: bare `float(os.getenv(...))` at module scope raises
#: `ValueError: could not convert string to float: ''` and takes the whole
#: import down with it. That is not hypothetical: one blank line in `.env`
#: made `brain/context_resolver.py` unimportable and every test in the
#: suite fail to collect. Modules that still read their own environment
#: must use these rather than a bare cast, so "unset" and "set to nothing"
#: both mean "use the default".
env_flag = _flag
env_int = _int
env_float = _float


def env_text(name: str, default: str = "") -> str:
    """A text setting, where a set-but-EMPTY value means "use the default".

    The same hazard as the numeric readers above, and it bites just as
    quietly: `JARVIS_OPENAI_TTS_INSTRUCTIONS=` in `.env` made
    `os.getenv(name, "<a careful voice prompt>")` return `""`, so JARVIS
    shipped an empty instruction string to the TTS API on every spoken
    line and the configured voice character was silently gone.
    """
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else raw


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
    nvidia_api_key: str | None = field(default_factory=lambda: _secret("NVIDIA_API_KEY"), repr=False)
    openrouter_api_key: str | None = field(default_factory=lambda: _secret("OPENROUTER_API_KEY"), repr=False)
    groq_api_key: str | None = field(default_factory=lambda: _secret("GROQ_API_KEY"), repr=False)
    cerebras_api_key: str | None = field(default_factory=lambda: _secret("CEREBRAS_API_KEY"), repr=False)
    google_api_key: str | None = field(default_factory=lambda: _secret("GOOGLE_API_KEY"), repr=False)
    cloudflare_api_token: str | None = field(default_factory=lambda: _secret("CLOUDFLARE_API_TOKEN"), repr=False)
    cloudflare_account_id: str | None = field(default_factory=lambda: _secret("CLOUDFLARE_ACCOUNT_ID"), repr=False)
    mistral_api_key: str | None = field(default_factory=lambda: _secret("MISTRAL_API_KEY"), repr=False)
    cohere_api_key: str | None = field(default_factory=lambda: _secret("COHERE_API_KEY"), repr=False)
    hf_token: str | None = field(default_factory=lambda: _secret("HF_TOKEN"), repr=False)
    moonshot_api_key: str | None = field(default_factory=lambda: _secret("MOONSHOT_API_KEY"), repr=False)
    github_token: str | None = field(default_factory=lambda: _secret("GITHUB_TOKEN"), repr=False)
    vercel_ai_gateway_api_key: str | None = field(default_factory=lambda: _secret("VERCEL_AI_GATEWAY_API_KEY"), repr=False)
    lmstudio_base_url: str = field(default_factory=lambda: _text("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"))
    ollama_base_url: str = field(default_factory=lambda: _text("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    enable_anthropic_fallback: bool = field(default_factory=lambda: _flag("ENABLE_ANTHROPIC_FALLBACK", True))
    local_model_idle_unload_minutes: int = field(default_factory=lambda: _int("LOCAL_MODEL_IDLE_UNLOAD_MINUTES", 15))
    model_registry_cache_ttl: int = field(default_factory=lambda: _int("MODEL_REGISTRY_CACHE_TTL_SECONDS", 21600))
    provider_discovery_timeout: float = field(default_factory=lambda: _float("JARVIS_PROVIDER_DISCOVERY_TIMEOUT", 3.0))
    agent_model: str = field(default_factory=lambda: _text("JARVIS_AGENT_MODEL", "claude-opus-5"))
    agent_max_tokens: int = field(default_factory=lambda: _int("JARVIS_AGENT_MAX_TOKENS", 4096))
    agent_temperature: float = field(default_factory=lambda: _float("JARVIS_AGENT_TEMPERATURE", 0.0))
    agent_request_timeout: float = field(default_factory=lambda: _float("JARVIS_AGENT_REQUEST_TIMEOUT", 120.0))
    agent_max_provider_retries: int = field(default_factory=lambda: _int("JARVIS_AGENT_PROVIDER_RETRIES", 2))
    provider_order: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            part.strip().lower()
            for part in _text("JARVIS_PROVIDER_ORDER", "multi_model,anthropic").split(",")
            if part.strip()
        )
    )

    # ---- agent loop safety limits -----------------------------------
    max_agent_steps: int = field(default_factory=lambda: _int("JARVIS_MAX_AGENT_STEPS", 25))
    #: How many independent READ-ONLY tools the agent loop may run at once
    #: (`brain/agent_loop.py::_parallel_safe`). Only read-only tools with no
    #: exclusive resource are ever eligible, so this bounds concurrency, not
    #: safety. 1 disables batching without changing any other behaviour.
    max_parallel_tools: int = field(default_factory=lambda: _int("JARVIS_MAX_PARALLEL_TOOLS", 4))
    #: `output_config.effort` for an ordinary interactive agent turn. The API
    #: default is "high"; `brain/agent_service.py::select_effort` raises it
    #: back to "high" for tasks that genuinely need the depth.
    agent_effort: str = field(default_factory=lambda: _text("JARVIS_AGENT_EFFORT", "medium"))
    agent_effort_complex: str = field(default_factory=lambda: _text("JARVIS_AGENT_EFFORT_COMPLEX", "high"))
    max_action_retries: int = field(default_factory=lambda: _int("JARVIS_MAX_ACTION_RETRIES", 2))
    max_consecutive_failures: int = field(default_factory=lambda: _int("JARVIS_MAX_CONSECUTIVE_FAILURES", 4))
    agent_task_timeout: float = field(default_factory=lambda: _float("JARVIS_AGENT_TASK_TIMEOUT", 900.0))

    # ---- outbound paid calls ----------------------------------------
    #: The single switch for "may this process make a paid cloud call".
    #:
    #: `openai_api_key` is read here rather than by each call site so
    #: there is one answer to "is a cloud call possible", the same way
    #: `providers/registry.py::agent_escalation_available` is the single
    #: answer for the agent provider. `cloud_calls_enabled` is what the
    #: test suite sets to false: a fake-but-non-empty key is not "absent"
    #: to the OpenAI SDK, so before this existed the suite genuinely
    #: reached `api.openai.com` on the user's real key during collection
    #: (the key was captured at import time, before test isolation ran)
    #: and spent real money on ordinary routing tests.
    openai_api_key: str | None = field(default_factory=lambda: _secret("OPENAI_API_KEY"), repr=False)
    cloud_calls_enabled: bool = field(default_factory=lambda: _flag("JARVIS_ALLOW_CLOUD_CALLS", True))

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

    # ---- desktop startup / UI ---------------------------------------
    #: Whether `main.py --start` (and therefore the Windows logon task)
    #: brings up the graphical interface. `--no-ui` overrides it for one
    #: run; this is the persistent default.
    ui_enabled: bool = field(default_factory=lambda: _flag("JARVIS_UI_ENABLED", True))
    #: Fullscreen vs maximized on launch. Escape always leaves fullscreen,
    #: whatever this says -- see `ui/qml/main.qml`.
    ui_fullscreen: bool = field(default_factory=lambda: _flag("START_UI_FULLSCREEN", False))
    #: Open JARVIS's OWN dedicated Chrome profile at startup if it is not
    #: already running (`startup/chrome.py`). Never the user's personal
    #: profile, and never a second copy of JARVIS's.
    auto_open_chrome: bool = field(default_factory=lambda: _flag("AUTO_OPEN_CHROME", True))
    #: Start the always-on wake-word/voice assistant at startup.
    auto_start_voice: bool = field(default_factory=lambda: _flag("AUTO_START_VOICE", True))
    #: Show the notification-area icon alongside the window. The tray is
    #: how JARVIS is exited and configured, so this defaults on.
    tray_enabled: bool = field(default_factory=lambda: _flag("TRAY_ENABLED", True))

    # ---- the Obsidian knowledge vault --------------------------------
    #: JARVIS's persistent long-term brain: plain Markdown notes JARVIS
    #: reads and writes directly, and the user opens in Obsidian.
    #: `JARVIS_VAULT_PATH` may be absolute (a real Obsidian vault anywhere
    #: on the machine) or relative to the repository root. Resolved by
    #: `vault/paths.py`, which is the only module that knows the layout.
    vault_enabled: bool = field(default_factory=lambda: _flag("JARVIS_VAULT_ENABLED", True))
    #: How many characters of vault knowledge one primed mission may load.
    #: Spent in priority order (identity, Job, Skills, project,
    #: preferences, lessons, continuity), so a tight budget drops the
    #: least important knowledge rather than an arbitrary slice.
    vault_context_chars: int = field(default_factory=lambda: _int("JARVIS_VAULT_CONTEXT_CHARS", 6000))
    #: Learn durable rules from the user's corrections. Turning this off
    #: leaves reading and mission recording intact; only the automatic
    #: WRITING of new rules stops.
    vault_learning_enabled: bool = field(default_factory=lambda: _flag("JARVIS_VAULT_LEARNING", True))

    # ---- memory ------------------------------------------------------
    memory_min_importance: int = field(default_factory=lambda: _int("JARVIS_MEMORY_MIN_IMPORTANCE", 2))
    memory_enabled: bool = field(default_factory=lambda: _flag("JARVIS_MEMORY_ENABLED", True))

    # ---- derived paths ----------------------------------------------
    @property
    def agent_db_path(self) -> Path:
        override = _text("JARVIS_AGENT_DB_PATH")
        return Path(override) if override else self.data_dir / "jarvis_agent.sqlite3"

    @property
    def vault_path(self) -> Path:
        """The vault root. Delegates so there is ONE resolver."""
        from vault.paths import default_vault_path

        return default_vault_path()

    @property
    def usage_db_path(self) -> Path:
        override = _text("JARVIS_USAGE_DB_PATH")
        return Path(override) if override else self.data_dir / "jarvis_usage.sqlite3"

    @property
    def has_anthropic_credentials(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def openai_available(self) -> bool:
        """May an OpenAI call actually be made right now?

        Every OpenAI call site asks this instead of assuming a key that is
        merely present is a key that may be used.
        """
        return bool(self.openai_api_key) and self.cloud_calls_enabled

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
