"""Focused tests for voice/tts/elevenlabs_tts.py and its wiring into
voice/text_to_speech.py's provider chain. No real HTTP/audio-device calls:
`requests.post` and `sounddevice` are mocked throughout.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from voice.tts import elevenlabs_tts
from voice import text_to_speech as tts


def _fake_pcm_response(chunks, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.text = "error detail" if status_code != 200 else ""
    response.iter_content = MagicMock(return_value=iter(chunks))
    response.close = MagicMock()
    return response


class ElevenLabsTTSAvailabilityTests(unittest.TestCase):
    def test_unavailable_without_key_or_voice_id(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "", "ELEVENLABS_VOICE_ID": ""}, clear=False):
            self.assertFalse(elevenlabs_tts.is_available())

    def test_available_with_key_and_voice_id(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk-test", "ELEVENLABS_VOICE_ID": "voice-123", "ELEVENLABS_TTS_ENABLED": "true"}, clear=False):
            self.assertTrue(elevenlabs_tts.is_available())

    def test_disabled_flag_overrides_configuration(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk-test", "ELEVENLABS_VOICE_ID": "voice-123", "ELEVENLABS_TTS_ENABLED": "false"}, clear=False):
            self.assertFalse(elevenlabs_tts.is_available())


class ElevenLabsTTSRequestTests(unittest.TestCase):
    def test_uses_configured_voice_id_never_hardcoded(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk-test", "ELEVENLABS_VOICE_ID": "my-custom-voice-id"}, clear=False):
            with patch("requests.post", return_value=_fake_pcm_response([b"\x00\x01" * 10])) as post:
                list(elevenlabs_tts.stream_pcm_chunks("hello"))
        url = post.call_args.args[0]
        self.assertIn("my-custom-voice-id", url)

    def test_uses_configured_model_id(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk-test", "ELEVENLABS_VOICE_ID": "v1", "ELEVENLABS_TTS_MODEL": "eleven_flash_v2_5"}, clear=False):
            with patch("requests.post", return_value=_fake_pcm_response([b"\x00\x01" * 10])) as post:
                list(elevenlabs_tts.stream_pcm_chunks("hello"))
        self.assertEqual(post.call_args.kwargs["json"]["model_id"], "eleven_flash_v2_5")

    def test_requests_pcm_output_format_no_temp_file_container(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk-test", "ELEVENLABS_VOICE_ID": "v1", "ELEVENLABS_TTS_SAMPLE_RATE": "24000"}, clear=False):
            with patch("requests.post", return_value=_fake_pcm_response([b"\x00\x01" * 10])) as post:
                list(elevenlabs_tts.stream_pcm_chunks("hello"))
        self.assertEqual(post.call_args.kwargs["params"]["output_format"], "pcm_24000")

    def test_http_error_status_raises_with_detail(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk-test", "ELEVENLABS_VOICE_ID": "v1"}, clear=False):
            with patch("requests.post", return_value=_fake_pcm_response([], status_code=401)):
                with self.assertRaises(RuntimeError):
                    list(elevenlabs_tts.stream_pcm_chunks("hello"))

    def test_missing_voice_id_raises_before_any_request(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk-test", "ELEVENLABS_VOICE_ID": ""}, clear=False):
            with patch("requests.post") as post:
                with self.assertRaises(RuntimeError):
                    list(elevenlabs_tts.stream_pcm_chunks("hello"))
            post.assert_not_called()


class ElevenLabsTTSStreamingPlaybackTests(unittest.TestCase):
    def test_first_chunk_plays_before_stream_exhausted(self):
        """Playback must start from the FIRST chunk, not wait for the
        whole response -- verified by observing stream.write() get called
        once per chunk as `stream_pcm_chunks` yields, not once at the end."""
        chunks = [b"\x01\x00" * 4, b"\x02\x00" * 4, b"\x03\x00" * 4]
        write_calls = []
        fake_stream = MagicMock()
        fake_stream.write.side_effect = lambda samples: write_calls.append(len(write_calls))
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk-test", "ELEVENLABS_VOICE_ID": "v1"}, clear=False):
            with patch("requests.post", return_value=_fake_pcm_response(chunks)), \
                 patch("sounddevice.OutputStream", return_value=fake_stream):
                elevenlabs_tts.speak("hello")
        self.assertEqual(len(write_calls), 3, "each network chunk should be written as it arrives")

    def test_first_audio_chunk_callback_fires_once(self):
        chunks = [b"\x01\x00" * 4, b"\x02\x00" * 4]
        fake_stream = MagicMock()
        first_chunk_calls = []
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk-test", "ELEVENLABS_VOICE_ID": "v1"}, clear=False):
            with patch("requests.post", return_value=_fake_pcm_response(chunks)), \
                 patch("sounddevice.OutputStream", return_value=fake_stream):
                elevenlabs_tts.speak("hello", on_first_audio_chunk=lambda: first_chunk_calls.append(1))
        self.assertEqual(len(first_chunk_calls), 1)

    def test_stop_closes_in_flight_response_for_fast_barge_in(self):
        response = _fake_pcm_response([b"\x01\x00" * 4] * 100)
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk-test", "ELEVENLABS_VOICE_ID": "v1"}, clear=False):
            with patch("requests.post", return_value=response):
                gen = elevenlabs_tts.stream_pcm_chunks("hello")
                next(gen)
                elevenlabs_tts.stop()
                remaining = list(gen)
        self.assertEqual(remaining, [])
        elevenlabs_tts._STOP_EVENT.clear()

    def test_empty_text_produces_no_request(self):
        with patch("requests.post") as post:
            elevenlabs_tts.speak("")
        post.assert_not_called()


class TextToSpeechProviderWiringTests(unittest.TestCase):
    def test_elevenlabs_first_when_configured_and_available(self):
        with patch.dict(os.environ, {"TTS_PROVIDER": "elevenlabs"}, clear=False):
            with patch.object(tts, "_elevenlabs_is_available", return_value=True):
                self.assertEqual(tts._provider_order()[0], "elevenlabs")

    def test_falls_back_to_next_provider_on_elevenlabs_failure(self):
        pyttsx3_engine = MagicMock()
        with patch.dict(os.environ, {"TTS_PROVIDER": "elevenlabs"}, clear=False):
            with patch.object(tts, "_elevenlabs_is_available", return_value=True), \
                 patch.object(tts, "_elevenlabs_provider") as fake_provider, \
                 patch.object(tts, "_openai_is_available", return_value=False), \
                 patch.object(tts, "_chatterbox_available", False), \
                 patch.object(tts, "_pyttsx3_available", True), \
                 patch.object(tts, "_init_pyttsx3", return_value=pyttsx3_engine):
                fake_provider.speak.side_effect = RuntimeError("network down")
                result = tts._speak_unlocked("hello")
        self.assertTrue(result["success"])
        self.assertEqual(result["provider"], "pyttsx3")
        self.assertIn("elevenlabs", result["attempted_providers"])
        pyttsx3_engine.say.assert_called_once()

    def test_stop_reaches_elevenlabs_provider(self):
        fake_provider = MagicMock()
        with patch.object(tts, "_elevenlabs_provider", fake_provider):
            tts.stop()
        fake_provider.stop.assert_called_once()

    def test_unknown_provider_value_falls_back_to_auto(self):
        with patch.dict(os.environ, {"TTS_PROVIDER": "not-a-real-provider"}, clear=False), patch("builtins.print"):
            self.assertEqual(tts._configured_provider(), "auto")


if __name__ == "__main__":
    unittest.main()
