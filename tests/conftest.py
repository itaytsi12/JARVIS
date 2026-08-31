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

import os
import shutil
import tempfile

import pytest

_TEST_ENVIRONMENT = {
    # Credentials -- no real external calls from the test suite.
    "OPENAI_API_KEY": "test-openai-key-not-real",
    "ANTHROPIC_API_KEY": "",
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


@pytest.fixture(scope="session", autouse=True)
def isolated_environment():
    previous = {key: os.environ.get(key) for key in _TEST_ENVIRONMENT}
    os.environ.update(_TEST_ENVIRONMENT)

    root = tempfile.mkdtemp(prefix="jarvis-tests-")
    paths = {
        "JARVIS_DATA_DIR": root,
        "JARVIS_AGENT_DB_PATH": os.path.join(root, "jarvis_agent.sqlite3"),
        "JARVIS_USAGE_DB_PATH": os.path.join(root, "jarvis_usage.sqlite3"),
    }
    previous_paths = {key: os.environ.get(key) for key in paths}
    for key, value in paths.items():
        os.environ.setdefault(key, value)

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
