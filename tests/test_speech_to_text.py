"""Focused tests for voice/speech_to_text.py's VOICE_LANGUAGE wiring.

No real faster-whisper model is ever loaded: `WhisperModel` is patched
with a fake constructor/`.transcribe()` so these stay fast, offline unit
tests.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import voice.speech_to_text as stt


class FakeSegment:
    def __init__(self, text):
        self.text = text


def _reset_model_singleton():
    stt._MODEL = None


class WhisperModelSelectionTests(unittest.TestCase):
    def setUp(self):
        _reset_model_singleton()

    def tearDown(self):
        _reset_model_singleton()

    def test_english_mode_defaults_to_english_optimized_model(self):
        captured = {}

        def fake_ctor(model_size, **kwargs):
            captured["model_size"] = model_size
            return MagicMock()

        with patch.dict(os.environ, {"VOICE_LANGUAGE": "en"}, clear=False), \
             patch.object(stt, "_AVAILABLE", True), \
             patch.object(stt, "WhisperModel", side_effect=fake_ctor):
            env = dict(os.environ)
            env.pop("WHISPER_MODEL", None)
            with patch.dict(os.environ, env, clear=True):
                stt._get_model()
        self.assertEqual(captured["model_size"], "small.en")

    def test_hebrew_mode_defaults_to_multilingual_model_not_dot_en(self):
        captured = {}

        def fake_ctor(model_size, **kwargs):
            captured["model_size"] = model_size
            return MagicMock()

        env = dict(os.environ)
        env.pop("WHISPER_MODEL", None)
        env["VOICE_LANGUAGE"] = "he"
        with patch.dict(os.environ, env, clear=True), \
             patch.object(stt, "_AVAILABLE", True), \
             patch.object(stt, "WhisperModel", side_effect=fake_ctor):
            stt._get_model()
        self.assertEqual(captured["model_size"], "small")
        self.assertNotIn(".en", captured["model_size"])

    def test_explicit_whisper_model_env_overrides_the_language_default(self):
        captured = {}

        def fake_ctor(model_size, **kwargs):
            captured["model_size"] = model_size
            return MagicMock()

        with patch.dict(os.environ, {"VOICE_LANGUAGE": "he", "WHISPER_MODEL": "medium"}, clear=False), \
             patch.object(stt, "_AVAILABLE", True), \
             patch.object(stt, "WhisperModel", side_effect=fake_ctor):
            stt._get_model()
        self.assertEqual(captured["model_size"], "medium")

    def test_only_one_model_is_ever_loaded_per_process(self):
        calls = []

        def fake_ctor(model_size, **kwargs):
            calls.append(model_size)
            return MagicMock()

        with patch.dict(os.environ, {"VOICE_LANGUAGE": "en"}, clear=False), \
             patch.object(stt, "_AVAILABLE", True), \
             patch.object(stt, "WhisperModel", side_effect=fake_ctor):
            stt._get_model()
            stt._get_model()
            stt._get_model()
        self.assertEqual(len(calls), 1)


class TranscribeLanguageParamTests(unittest.TestCase):
    def setUp(self):
        _reset_model_singleton()

    def tearDown(self):
        _reset_model_singleton()

    def _fake_model(self, captured):
        model = MagicMock()

        def fake_transcribe(path, **kwargs):
            captured.update(kwargs)
            return [FakeSegment("hello")], MagicMock()

        model.transcribe.side_effect = fake_transcribe
        return model

    def test_english_mode_passes_language_en_and_the_english_prompt(self):
        captured = {}
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "en"}, clear=False), \
             patch.object(stt, "_get_model", return_value=self._fake_model(captured)):
            stt.transcribe_audio("fake.wav")
        self.assertEqual(captured["language"], "en")
        self.assertIsNotNone(captured["initial_prompt"])

    def test_hebrew_mode_passes_language_he_never_forces_english(self):
        captured = {}
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "he"}, clear=False), \
             patch.object(stt, "_get_model", return_value=self._fake_model(captured)):
            stt.transcribe_audio("fake.wav")
        self.assertEqual(captured["language"], "he")
        self.assertNotEqual(captured["language"], "en")

    def test_hebrew_mode_omits_the_english_initial_prompt(self):
        # The English initial_prompt is an English-phrasing hint -- keeping
        # it in Hebrew mode would bias transcription back toward English.
        captured = {}
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "he"}, clear=False), \
             patch.object(stt, "_get_model", return_value=self._fake_model(captured)):
            stt.transcribe_audio("fake.wav")
        self.assertIsNone(captured["initial_prompt"])

    def test_unsupported_voice_language_fails_clearly(self):
        from voice.voice_language import UnsupportedVoiceLanguage
        captured = {}
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "fr"}, clear=False), \
             patch.object(stt, "_get_model", return_value=self._fake_model(captured)):
            with self.assertRaises(UnsupportedVoiceLanguage):
                stt.transcribe_audio("fake.wav")


if __name__ == "__main__":
    unittest.main()
