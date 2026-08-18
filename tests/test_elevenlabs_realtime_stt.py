"""Focused tests for voice/elevenlabs_realtime_stt.py.

No real network/websocket connection is ever made: a `FakeWSApp` factory
stands in for `websocket.WebSocketApp`, matching the small
(url, header, on_open, on_message, on_error, on_close) / run_forever / send
/ close surface `ElevenLabsRealtimeSTT` actually uses.
"""
from __future__ import annotations

import json
import os
import threading
import time
import unittest
from unittest.mock import patch

from voice.elevenlabs_realtime_stt import (
    ElevenLabsRealtimeSTT,
    ElevenLabsSTTError,
    TranscriptEvent,
    is_configured,
)

# Tests never need the real (production-default 300ms) post-`on_open`
# auth-error grace window `connect()` now waits out (see module docstring
# on the grace window itself) -- zero it here so this whole file stays
# fast; the grace window's actual behavior is covered by its own dedicated
# tests below, which set it explicitly per-test.
os.environ.setdefault("ELEVENLABS_STT_AUTH_GRACE_MS", "0")


class FakeWSApp:
    """A controllable stand-in for websocket.WebSocketApp. Tests choose
    exactly when `on_open`/`on_message`/`on_error`/`on_close` fire by
    calling helper methods, instead of a real socket handshake."""

    instances: list["FakeWSApp"] = []

    def __init__(self, url, header, on_open, on_message, on_error, on_close, *, fail_immediately=False, never_open=False):
        self.url = url
        self.header = header
        self._on_open = on_open
        self._on_message = on_message
        self._on_error = on_error
        self._on_close = on_close
        self.sent: list[str] = []
        self.closed = False
        self._fail_immediately = fail_immediately
        self._never_open = never_open
        FakeWSApp.instances.append(self)

    def run_forever(self, **kwargs):
        if self._never_open:
            return
        if self._fail_immediately:
            self._on_error(self, RuntimeError("simulated connect failure"))
            return
        self._on_open(self)

    def send(self, payload):
        if self.closed:
            raise RuntimeError("socket already closed")
        self.sent.append(payload)

    def close(self):
        self.closed = True

    def push(self, message_type, **fields):
        self._on_message(self, json.dumps({"message_type": message_type, **fields}))


def _factory(**kwargs):
    return lambda url, header, on_open, on_message, on_error, on_close: FakeWSApp(
        url, header, on_open, on_message, on_error, on_close, **kwargs
    )


class ElevenLabsSTTConfigTests(unittest.TestCase):
    def test_not_configured_without_api_key(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "", "STT_PROVIDER": "elevenlabs"}, clear=False):
            self.assertFalse(is_configured())

    def test_configured_when_key_and_provider_set(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk-test", "STT_PROVIDER": "elevenlabs", "ELEVENLABS_STT_ENABLED": "true"}, clear=False):
            self.assertTrue(is_configured())

    def test_disabled_flag_overrides_key_presence(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk-test", "STT_PROVIDER": "elevenlabs", "ELEVENLABS_STT_ENABLED": "false"}, clear=False):
            self.assertFalse(is_configured())

    def test_missing_key_raises_on_construction(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}, clear=False):
            with self.assertRaises(ElevenLabsSTTError):
                ElevenLabsRealtimeSTT(ws_app_factory=_factory())


class ElevenLabsSTTConnectionTests(unittest.TestCase):
    def setUp(self):
        FakeWSApp.instances.clear()

    def test_connect_opens_only_when_called_never_on_construction(self):
        session = ElevenLabsRealtimeSTT(api_key="sk-test", ws_app_factory=_factory())
        self.assertEqual(FakeWSApp.instances, [], "constructing a session must not open a connection")
        session.connect()
        self.assertEqual(len(FakeWSApp.instances), 1)
        self.assertTrue(session.connected)

    def test_connect_sends_xi_api_key_header_never_in_url(self):
        session = ElevenLabsRealtimeSTT(api_key="sk-secret-value", ws_app_factory=_factory())
        session.connect()
        ws = FakeWSApp.instances[0]
        self.assertIn("xi-api-key: sk-secret-value", ws.header)
        self.assertNotIn("sk-secret-value", ws.url)

    def test_connect_timeout_raises_cleanly(self):
        session = ElevenLabsRealtimeSTT(api_key="sk-test", connect_timeout=0.05, ws_app_factory=_factory(never_open=True))
        with self.assertRaises(ElevenLabsSTTError):
            session.connect()

    def test_auth_error_arriving_shortly_after_on_open_still_fails_connect(self):
        # Confirmed live against the real ElevenLabs realtime endpoint: the
        # WebSocket handshake (on_open) can succeed and THEN a business-
        # logic auth_error message arrives (followed by close) -- on_open
        # alone is not proof of a genuinely usable session. This simulates
        # exactly that ordering: on_open fires synchronously, then a
        # separate thread delivers the auth_error message shortly after,
        # while connect() is still inside its post-open grace window.
        def factory(url, header, on_open, on_message, on_error, on_close):
            app = FakeWSApp(url, header, on_open, on_message, on_error, on_close)

            def deliver_late_auth_error():
                app.push("auth_error", error="You must be authenticated to use this endpoint.")

            app._deliver_late_auth_error = deliver_late_auth_error
            return app

        with patch.dict(os.environ, {"ELEVENLABS_STT_AUTH_GRACE_MS": "500"}, clear=False):
            session = ElevenLabsRealtimeSTT(api_key="sk-test", ws_app_factory=factory)
            timer = threading.Timer(0.05, lambda: FakeWSApp.instances[0]._deliver_late_auth_error())
            timer.start()
            try:
                with self.assertRaises(ElevenLabsSTTError):
                    session.connect()
            finally:
                timer.cancel()

    def test_connect_error_raises_cleanly_never_crashes(self):
        session = ElevenLabsRealtimeSTT(api_key="sk-test", connect_timeout=1.0, ws_app_factory=_factory(fail_immediately=True))
        with self.assertRaises(ElevenLabsSTTError):
            session.connect()

    def test_close_is_idempotent_and_safe_before_connect(self):
        session = ElevenLabsRealtimeSTT(api_key="sk-test", ws_app_factory=_factory())
        session.close()
        session.close()


class ElevenLabsSTTTranscriptTests(unittest.TestCase):
    def setUp(self):
        FakeWSApp.instances.clear()

    def test_partial_transcript_delivered_to_callback(self):
        events = []
        session = ElevenLabsRealtimeSTT(api_key="sk-test", on_event=events.append, ws_app_factory=_factory())
        session.connect()
        FakeWSApp.instances[0].push("partial_transcript", text="open sp")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "partial")
        self.assertEqual(events[0].text, "open sp")

    def test_committed_transcript_unblocks_commit(self):
        session = ElevenLabsRealtimeSTT(api_key="sk-test", ws_app_factory=_factory())
        session.connect()
        ws = FakeWSApp.instances[0]

        def respond_after_commit():
            deadline = time.time() + 2
            while time.time() < deadline:
                if any(json.loads(p).get("commit") for p in ws.sent):
                    ws.push("committed_transcript", text="open spotify")
                    return
                time.sleep(0.01)

        threading.Thread(target=respond_after_commit, daemon=True).start()
        result = session.commit(timeout=2)
        self.assertEqual(result, "open spotify")

    def test_commit_returns_none_on_timeout_never_raises(self):
        session = ElevenLabsRealtimeSTT(api_key="sk-test", ws_app_factory=_factory())
        session.connect()
        result = session.commit(timeout=0.05)
        self.assertIsNone(result)

    def test_error_message_type_recorded_and_reported_as_event(self):
        events = []
        session = ElevenLabsRealtimeSTT(api_key="sk-test", on_event=events.append, ws_app_factory=_factory())
        session.connect()
        FakeWSApp.instances[0].push("auth_error", message="invalid api key")
        self.assertIsNotNone(session.error)
        self.assertEqual(events[-1].kind, "error")

    def test_quota_exceeded_does_not_crash_session(self):
        session = ElevenLabsRealtimeSTT(api_key="sk-test", ws_app_factory=_factory())
        session.connect()
        FakeWSApp.instances[0].push("quota_exceeded", message="no credits left")
        # No exception raised; session remains inspectable.
        self.assertIsNotNone(session.error)

    def test_send_audio_base64_encodes_pcm_and_includes_sample_rate(self):
        session = ElevenLabsRealtimeSTT(api_key="sk-test", sample_rate=16000, ws_app_factory=_factory())
        session.connect()
        session.send_audio(b"\x01\x02\x03\x04")
        ws = FakeWSApp.instances[0]
        payload = json.loads(ws.sent[-1])
        self.assertEqual(payload["message_type"], "input_audio_chunk")
        self.assertEqual(payload["sample_rate"], 16000)
        self.assertFalse(payload["commit"])
        import base64
        self.assertEqual(base64.b64decode(payload["audio_base_64"]), b"\x01\x02\x03\x04")

    def test_close_after_interaction_marks_closed(self):
        session = ElevenLabsRealtimeSTT(api_key="sk-test", ws_app_factory=_factory())
        session.connect()
        session.close()
        self.assertTrue(FakeWSApp.instances[0].closed)
        self.assertFalse(session.connected)


class ElevenLabsSTTVoiceLanguageTests(unittest.TestCase):
    """VOICE_LANGUAGE must reach the SAME `language_code` query parameter
    this client already sent (previously hardcoded to "en") -- no new/
    guessed parameter, no mixed-language transcription."""

    def setUp(self):
        FakeWSApp.instances.clear()

    def test_english_mode_requests_english_language_code(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "en"}, clear=False):
            session = ElevenLabsRealtimeSTT(api_key="sk-test", ws_app_factory=_factory())
            session.connect()
        self.assertIn("language_code=en", FakeWSApp.instances[0].url)

    def test_hebrew_mode_requests_hebrew_language_code_not_english(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "he"}, clear=False):
            session = ElevenLabsRealtimeSTT(api_key="sk-test", ws_app_factory=_factory())
            session.connect()
        url = FakeWSApp.instances[0].url
        self.assertIn("language_code=he", url)
        self.assertNotIn("language_code=en", url)

    def test_missing_voice_language_env_defaults_to_auto(self):
        env = dict(os.environ)
        env.pop("VOICE_LANGUAGE", None)
        with patch.dict(os.environ, env, clear=True):
            session = ElevenLabsRealtimeSTT(api_key="sk-test", ws_app_factory=_factory())
            session.connect()
        url = FakeWSApp.instances[0].url
        self.assertIn("include_language_detection=true", url)
        self.assertNotIn("language_code=", url)

    def test_auto_mode_never_forces_a_language_code(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "auto"}, clear=False):
            session = ElevenLabsRealtimeSTT(api_key="sk-test", ws_app_factory=_factory())
            session.connect()
        url = FakeWSApp.instances[0].url
        self.assertNotIn("language_code=en", url)
        self.assertNotIn("language_code=he", url)
        self.assertIn("include_language_detection=true", url)

    def test_unsupported_voice_language_fails_clearly_not_silently(self):
        from voice.voice_language import UnsupportedVoiceLanguage
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "fr"}, clear=False):
            session = ElevenLabsRealtimeSTT(api_key="sk-test", ws_app_factory=_factory())
            with self.assertRaises(UnsupportedVoiceLanguage):
                session.connect()


if __name__ == "__main__":
    unittest.main()
