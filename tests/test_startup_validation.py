"""Focused tests for voice/startup_validation.py (Part Q): startup provider
validation must never spend API credits -- no real network call."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from voice import startup_validation


class StartupValidationTests(unittest.TestCase):
    def test_never_makes_a_network_request(self):
        with patch("requests.post") as post, patch("requests.get") as get, patch("builtins.print"):
            startup_validation.log_provider_status()
        post.assert_not_called()
        get.assert_not_called()

    def test_reports_elevenlabs_stt_when_configured(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk-test", "STT_PROVIDER": "elevenlabs", "ELEVENLABS_STT_ENABLED": "true"}, clear=False), \
             patch("builtins.print") as printed:
            startup_validation.log_provider_status()
        lines = [call.args[0] for call in printed.call_args_list]
        self.assertTrue(any("STT provider: ElevenLabs realtime" in line for line in lines))

    def test_reports_whisper_when_elevenlabs_not_configured(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}, clear=False), patch("builtins.print") as printed:
            startup_validation.log_provider_status()
        lines = [call.args[0] for call in printed.call_args_list]
        self.assertTrue(any("STT provider: Whisper (local)" in line for line in lines))

    def test_reports_voice_id_configuration_state(self):
        with patch.dict(os.environ, {"ELEVENLABS_VOICE_ID": "voice-123"}, clear=False), patch("builtins.print") as printed:
            startup_validation.log_provider_status()
        lines = [call.args[0] for call in printed.call_args_list]
        self.assertTrue(any("JARVIS voice: configured" in line for line in lines))

    def test_reports_every_expected_line(self):
        with patch("builtins.print") as printed:
            startup_validation.log_provider_status()
        lines = [call.args[0] for call in printed.call_args_list]
        self.assertEqual(len(lines), 6)
        for prefix in (
            "STT provider:",
            "TTS provider:",
            "Voice input language:",
            "JARVIS voice:",
            "Whisper fallback:",
            "TTS fallback:",
        ):
            with self.subTest(prefix=prefix):
                self.assertTrue(any(line.startswith(prefix) for line in lines))

    def test_reports_the_resolved_language_policy_not_just_the_mode(self):
        """The mode name alone said nothing about what the FALLBACK would do,
        which is why forcing Hebrew on English commands was invisible."""
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "he"}, clear=False), patch("builtins.print") as printed:
            startup_validation.log_provider_status()
        line = next(l for l in [c.args[0] for c in printed.call_args_list] if l.startswith("Voice input language:"))
        self.assertIn("mode=he", line)
        self.assertIn("en", line)
        self.assertIn("auto-detect", line)

        with patch.dict(os.environ, {"VOICE_LANGUAGE": "en"}, clear=False), patch("builtins.print") as printed:
            startup_validation.log_provider_status()
        line = next(l for l in [c.args[0] for c in printed.call_args_list] if l.startswith("Voice input language:"))
        self.assertIn("forced_language=en", line)


if __name__ == "__main__":
    unittest.main()
