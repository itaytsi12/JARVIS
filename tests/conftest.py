"""Test-suite isolation from the developer's real `.env`.

Several JARVIS subsystems read provider credentials straight from the
environment (`voice/text_to_speech.py`, `voice/elevenlabs_realtime_stt.py`,
`providers/anthropic_provider.py`, ...), and `python-dotenv` loads the real
`.env` at import time. Without this file the outcome of a test depends on
whether the developer happens to have a key configured -- and, worse, a
test that patches the Whisper fallback can silently reach the REAL
ElevenLabs realtime API instead, opening a live microphone session and
spending money. That was observed happening to
`tests/test_barge_in.py` and `tests/test_chatterbox_service.py` once the
declared audio dependencies were actually installed.

So the whole suite runs with external providers switched off:

- No real credentials. `OPENAI_API_KEY` is set to an obviously-fake value
  rather than cleared, because `brain/agent.py` constructs an `OpenAI`
  client at import time and the SDK raises when the key is missing
  entirely; every other key is cleared.
- ElevenLabs STT and TTS explicitly disabled, STT pinned to the local
  Whisper path that tests already patch.
- `ANTHROPIC_API_KEY` cleared, so `providers.get_agent_provider()`
  returns None and the agent path is exercised only by tests that inject
  a provider deliberately.
- Browser autostart disabled, so no test can launch a real Chrome window
  by asking for the authenticated session.
- Agent databases and the data directory pointed at a temporary
  directory, so a test run never writes to the real `data/`.

A test that specifically needs one of these settings still overrides it
locally with `patch.dict(os.environ, ...)`; this only changes the default.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import tempfile

import pytest

_TEST_ENVIRONMENT = {
    # Credentials -- no real external calls from the test suite.
    "OPENAI_API_KEY": "test-openai-key-not-real",
    # A fake-but-non-empty key is not "absent" to the OpenAI SDK: it
    # builds a client and makes a real request that fails with a 401.
    # Worse, `brain/agent.py` captures `OPENAI_API_KEY` at IMPORT time,
    # which happens during collection -- so before this flag existed the
    # suite reached api.openai.com on the developer's REAL key and spent
    # real money on ordinary routing tests. This is the single switch
    # every OpenAI call site checks (`JarvisConfig.openai_available`).
    "JARVIS_ALLOW_CLOUD_CALLS": "0",
    "ANTHROPIC_API_KEY": "",
    "OPENROUTER_API_KEY": "",
    "GROQ_API_KEY": "",
    "NVIDIA_API_KEY": "",
    "CEREBRAS_API_KEY": "",
    "GOOGLE_API_KEY": "",
    "MISTRAL_API_KEY": "",
    "HF_TOKEN": "",
    "MOONSHOT_API_KEY": "",
    "GITHUB_TOKEN": "",
    "VERCEL_AI_GATEWAY_API_KEY": "",
    # Declared in `config/settings.py` but not yet wired to a provider.
    # Cleared anyway: the guard is meant to be exhaustive, so wiring one up
    # later cannot quietly give the suite a live credential.
    "CLOUDFLARE_API_TOKEN": "",
    "CLOUDFLARE_ACCOUNT_ID": "",
    "JARVIS_PROVIDER_ORDER": "anthropic",
    "ENABLE_ANTHROPIC_FALLBACK": "false",
    "ELEVENLABS_API_KEY": "",
    "ELEVENLABS_VOICE_ID": "",
    # Voice providers: local paths only.
    "STT_PROVIDER": "whisper",
    "ELEVENLABS_STT_ENABLED": "false",
    "ELEVENLABS_TTS_ENABLED": "false",
    # Never let a test start a real browser. `tools/browser_authenticated.py`
    # now launches JARVIS's own Chrome when nothing is listening on the
    # debug port, which is exactly right in production and exactly wrong
    # here: observed live, a full-suite run spawned real chrome.exe
    # processes and slowed the run down by minutes. Tests that exercise
    # the autostart do so with an explicit `patch.dict`.
    "JARVIS_BROWSER_AUTOSTART": "0",
    # Agent runtime: deterministic, quiet, and off unless a test opts in.
    "JARVIS_DEBUG": "false",
    "JARVIS_AGENT_ENABLED": "true",
    "JARVIS_AGENT_ESCALATION": "true",
}


#: Applied at conftest IMPORT time, not inside the fixture.
#:
#: pytest imports every test module during COLLECTION, which happens
#: before any fixture runs -- and several project modules build their
#: state at import time (`brain/agent.py` constructs a `MemoryManager`
#: at module scope). A fixture that redirects the databases afterwards is
#: therefore too late: the real, live databases have already been opened.
#: `tests/test_memory_system.py::test_global_agent_memory_is_isolated_during_pytest`
#: exists precisely to catch this, and was correctly failing.
#: Give pytest's own `tmp_path` factory a temp root we know is writable.
#:
#: pytest builds `<temproot>/pytest-of-<user>` and reuses it forever. On
#: this machine that directory exists with broken ACLs -- `os.scandir` and
#: `mkdir` both raise `PermissionError: [WinError 5]` -- so EVERY test
#: using `tmp_path` errored before its body ran. That silently included
#: two of the model-capability regression tests, which meant the behaviour
#: they cover was never actually being verified.
#:
#: `PYTEST_DEBUG_TEMPROOT` is pytest's own supported override. Redirecting
#: it fixes the suite without deleting anything outside the repository,
#: and makes the suite robust on any machine whose temp root is poisoned.
_PYTEST_TEMPROOT = tempfile.mkdtemp(prefix="jarvis-pytest-temproot-")
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", _PYTEST_TEMPROOT)

_TEST_ROOT = tempfile.mkdtemp(prefix="jarvis-tests-")
_TEST_PATHS = {
    "JARVIS_DATA_DIR": _TEST_ROOT,
    "JARVIS_AGENT_DB_PATH": os.path.join(_TEST_ROOT, "jarvis_agent.sqlite3"),
    "JARVIS_USAGE_DB_PATH": os.path.join(_TEST_ROOT, "jarvis_usage.sqlite3"),
    # CLEARED, not redirected. `memory/memory_manager.py` already has its
    # own isolation: with no `MEMORY_DB_PATH` set and pytest loaded, it
    # creates its own `jarvis-memory-pytest-*` temporary database. The
    # developer's real `.env` sets the variable, which silently disabled
    # that mechanism and pointed the whole suite at the user's live
    # memory database -- exactly what
    # `test_memory_system.py::test_global_agent_memory_is_isolated_during_pytest`
    # asserts must not happen. Clearing it restores the existing design
    # rather than adding a second, competing one.
    "MEMORY_DB_PATH": "",
    # Same story as MEMORY_DB_PATH. `training_data/recorder.py` has its own
    # pytest branch that builds a throwaway database -- but only when this
    # variable is unset, and the real `.env` sets it to the user's live
    # 47MB training dataset. The suite was therefore writing captured
    # events into real training data, and `_repair_sequences()` walked the
    # whole of it on first use, which is what made an unrelated voice test
    # time out waiting two seconds for a spoken line.
    "TRAINING_DATA_DB_PATH": "",
    # The vault is the user's real long-term memory. A test must never
    # read, write or bootstrap into it.
    "JARVIS_VAULT_PATH": os.path.join(_TEST_ROOT, "vault"),
    "JARVIS_VAULT_CACHE_PATH": os.path.join(_TEST_ROOT, "vault_index_cache.json"),
}
# `.env` is loaded into `os.environ` by importing `config.settings` (the
# one place that does it). It has to happen HERE, before the overrides
# below: otherwise the pops and assignments run against an environment
# that does not yet contain the developer's real values, and the first
# project module imported during collection loads `.env` over the top of
# them.
import config.settings as _settings  # noqa: E402,F401  -- imported for its .env load

os.environ.update(_TEST_ENVIRONMENT)
for _key, _value in _TEST_PATHS.items():
    if _value == "":
        os.environ.pop(_key, None)
    else:
        os.environ[_key] = _value
_settings.reload_config()


@pytest.fixture(scope="session", autouse=True)
def isolated_environment():
    previous = {key: os.environ.get(key) for key in _TEST_ENVIRONMENT}
    os.environ.update(_TEST_ENVIRONMENT)

    root = _TEST_ROOT
    paths = dict(_TEST_PATHS)
    previous_paths = {key: os.environ.get(key) for key in paths}
    for key, value in paths.items():
        # Assignment, not `setdefault`: the real `.env` has already been
        # loaded into `os.environ` by `config/settings.py` at import time,
        # so a `setdefault` here silently keeps the developer's real
        # database path and the isolation never happens.
        if value == "":
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    # Config is cached process-wide; rebuild it now that the environment is set.
    from config.settings import reload_config

    reload_config()

    yield root

    for key, value in {**previous, **previous_paths}.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    reload_config()
    shutil.rmtree(root, ignore_errors=True)


@contextlib.contextmanager
def allow_cloud_calls():
    """Temporarily re-enable the outbound-call switch for one test.

    The suite runs with `JARVIS_ALLOW_CLOUD_CALLS=0` so a fake-but-
    non-empty API key can never reach a real endpoint. A handful of tests
    exercise a cloud call site ON PURPOSE, with the client itself patched,
    and need the guard lifted -- this is the one, explicit way to do it,
    so "which tests touch a cloud path" stays greppable.

    Nothing leaves the process: the caller is still responsible for
    patching the client.
    """
    from config.settings import reload_config

    previous = os.environ.get("JARVIS_ALLOW_CLOUD_CALLS")
    os.environ["JARVIS_ALLOW_CLOUD_CALLS"] = "1"
    reload_config()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("JARVIS_ALLOW_CLOUD_CALLS", None)
        else:
            os.environ["JARVIS_ALLOW_CLOUD_CALLS"] = previous
        reload_config()
