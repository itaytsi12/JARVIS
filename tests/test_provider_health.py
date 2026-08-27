"""Regression tests for bug report 5: an ElevenLabs quota/insufficient-
funds failure should degrade the provider once, not be retried on every
subsequent request, while a transient failure must never permanently
disable it. All mocked -- no real network calls."""
import unittest
from unittest.mock import Mock, patch

from voice.provider_health import ProviderHealth, get_provider_health, is_non_transient_error, reset, reset_all


class IsNonTransientErrorTests(unittest.TestCase):
    def test_quota_exceeded_is_non_transient(self):
        self.assertTrue(is_non_transient_error(RuntimeError("quota_exceeded: no credits remaining")))

    def test_insufficient_funds_is_non_transient(self):
        self.assertTrue(is_non_transient_error(RuntimeError("HTTP 401: insufficient funds")))

    def test_ordinary_connect_timeout_is_transient(self):
        self.assertFalse(is_non_transient_error(TimeoutError("timed out connecting to ElevenLabs realtime STT")))

    def test_ordinary_5xx_is_transient(self):
        self.assertFalse(is_non_transient_error(RuntimeError("HTTP 503: Service Unavailable")))


class ProviderHealthTests(unittest.TestCase):
    def setUp(self):
        self.health = ProviderHealth("test_provider")

    def test_available_by_default(self):
        self.assertTrue(self.health.available)
        self.assertIsNone(self.health.reason)

    def test_non_transient_error_marks_unavailable(self):
        self.health.note_result(RuntimeError("quota_exceeded"))
        self.assertFalse(self.health.available)
        self.assertIn("quota_exceeded", self.health.reason)

    def test_transient_error_does_not_mark_unavailable(self):
        self.health.note_result(TimeoutError("connect timed out"))
        self.assertTrue(self.health.available)

    def test_success_does_not_mark_unavailable(self):
        self.health.note_result(None)
        self.assertTrue(self.health.available)

    def test_reset_restores_availability(self):
        self.health.note_result(RuntimeError("quota_exceeded"))
        self.assertFalse(self.health.available)
        self.health.reset()
        self.assertTrue(self.health.available)

    def test_logs_the_reason_exactly_once(self):
        with patch("voice.provider_health.log") as mock_log:
            self.health.note_result(RuntimeError("quota_exceeded: first"))
            self.health.note_result(RuntimeError("quota_exceeded: second"))
            self.health.note_result(RuntimeError("quota_exceeded: third"))
        self.assertEqual(mock_log.warning.call_count, 1)


class RegistryTests(unittest.TestCase):
    def tearDown(self):
        reset_all()

    def test_get_provider_health_returns_the_same_instance(self):
        self.assertIs(get_provider_health("elevenlabs_stt"), get_provider_health("elevenlabs_stt"))

    def test_different_names_are_independent(self):
        get_provider_health("provider_a").mark_unavailable("dead")
        self.assertFalse(get_provider_health("provider_a").available)
        self.assertTrue(get_provider_health("provider_b").available)

    def test_reset_by_name(self):
        get_provider_health("provider_c").mark_unavailable("dead")
        reset("provider_c")
        self.assertTrue(get_provider_health("provider_c").available)


class ThreeLaterRequestsSkipTheDeadProviderTests(unittest.TestCase):
    """Sequence F from the bug report: simulate quota_exceeded, then three
    later STT/TTS requests must not retry the network at all."""

    def tearDown(self):
        reset_all()

    def test_elevenlabs_tts_is_available_becomes_false_after_quota_error_and_stays_false(self):
        from voice.tts import elevenlabs_tts

        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "k", "ELEVENLABS_VOICE_ID": "v", "ELEVENLABS_TTS_ENABLED": "true"}):
            self.assertTrue(elevenlabs_tts.is_available())

            with patch.object(elevenlabs_tts, "_speak_impl", side_effect=RuntimeError("HTTP 401: quota_exceeded")):
                with self.assertRaises(RuntimeError):
                    elevenlabs_tts.speak("hello")

            # No further network attempt is even possible now: is_available()
            # is the gate every caller (voice/text_to_speech.py) checks
            # before trying, and it must report False without touching the
            # network again.
            for _ in range(3):
                self.assertFalse(elevenlabs_tts.is_available())

    def test_a_transient_tts_failure_does_not_disable_the_provider(self):
        from voice.tts import elevenlabs_tts

        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "k", "ELEVENLABS_VOICE_ID": "v", "ELEVENLABS_TTS_ENABLED": "true"}):
            with patch.object(elevenlabs_tts, "_speak_impl", side_effect=RuntimeError("HTTP 503: Service Unavailable")):
                with self.assertRaises(RuntimeError):
                    elevenlabs_tts.speak("hello")
            self.assertTrue(elevenlabs_tts.is_available())

    def test_elevenlabs_realtime_stt_is_configured_becomes_false_after_quota_error(self):
        from voice import elevenlabs_realtime_stt as stt

        with patch.dict("os.environ", {
            "ELEVENLABS_API_KEY": "k", "ELEVENLABS_STT_ENABLED": "true", "STT_PROVIDER": "elevenlabs",
        }), patch.object(stt, "_WEBSOCKET_AVAILABLE", True):
            self.assertTrue(stt.is_configured())
            get_provider_health("elevenlabs_stt").note_result(RuntimeError("quota_exceeded: no credits"))
            for _ in range(3):
                self.assertFalse(stt.is_configured())

    def test_realtime_capture_records_quota_failure_and_skips_reattempt(self):
        from voice.realtime_capture import RealtimeSTTController
        from voice.voice_perf import VoiceInteractionTimer
        from voice.elevenlabs_realtime_stt import ElevenLabsSTTError

        class _FailingFactory:
            def __call__(self, *args, **kwargs):
                session = Mock()
                session.connect.side_effect = ElevenLabsSTTError("quota_exceeded: no credits")
                return session

        controller = RealtimeSTTController(perf=VoiceInteractionTimer(), stt_factory=_FailingFactory())
        controller.start()
        controller._connected.wait(2)

        self.assertFalse(get_provider_health("elevenlabs_stt").available)


if __name__ == "__main__":
    unittest.main()
