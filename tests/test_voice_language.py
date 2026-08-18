import os
import unittest
from unittest.mock import patch

from voice.voice_language import (
    UnsupportedVoiceLanguage,
    detect_input_language,
    get_tts_language,
    get_voice_language,
    is_auto_mode,
    is_english_mode,
    is_hebrew_mode,
    language_name,
    resolve_utterance_language,
)


class VoiceLanguageTests(unittest.TestCase):
    def test_defaults_to_auto_when_unset(self):
        env = dict(os.environ)
        env.pop("VOICE_LANGUAGE", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_voice_language(), "auto")
            self.assertTrue(is_auto_mode())
            self.assertFalse(is_english_mode())
            self.assertFalse(is_hebrew_mode())

    def test_explicit_english(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "en"}, clear=False):
            self.assertEqual(get_voice_language(), "en")
            self.assertFalse(is_auto_mode())

    def test_explicit_hebrew(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "he"}, clear=False):
            self.assertEqual(get_voice_language(), "he")
            self.assertTrue(is_hebrew_mode())
            self.assertFalse(is_english_mode())

    def test_explicit_auto(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "auto"}, clear=False):
            self.assertEqual(get_voice_language(), "auto")
            self.assertTrue(is_auto_mode())

    def test_case_and_whitespace_insensitive(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": " HE "}, clear=False):
            self.assertEqual(get_voice_language(), "he")

    def test_unsupported_value_fails_clearly_not_silently(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "fr"}, clear=False):
            with self.assertRaises(UnsupportedVoiceLanguage):
                get_voice_language()

    def test_unsupported_value_error_names_the_bad_value_and_supported_ones(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "xx"}, clear=False):
            with self.assertRaisesRegex(UnsupportedVoiceLanguage, "xx"):
                get_voice_language()

    def test_language_name_mapping(self):
        self.assertEqual(language_name("en"), "English")
        self.assertEqual(language_name("he"), "Hebrew")

    def test_language_name_uses_configured_language_when_omitted(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "he"}, clear=False):
            self.assertEqual(language_name(), "Hebrew")

    def test_tts_language_is_always_english_regardless_of_voice_language(self):
        for mode in ("auto", "en", "he"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"VOICE_LANGUAGE": mode}, clear=False):
                self.assertEqual(get_tts_language(), "en")

    def test_tts_language_ignores_tts_language_env_override(self):
        # TTS_LANGUAGE is documented/logged only -- it can never actually
        # change what JARVIS speaks. A misconfigured value must not leak
        # through into real behavior.
        with patch.dict(os.environ, {"TTS_LANGUAGE": "he"}, clear=False):
            self.assertEqual(get_tts_language(), "en")


class InputLanguageDetectionTests(unittest.TestCase):
    def test_detects_english(self):
        self.assertEqual(detect_input_language("open YouTube"), "en")
        self.assertEqual(detect_input_language("what song is playing?"), "en")

    def test_detects_hebrew(self):
        self.assertEqual(detect_input_language("פתח יוטיוב"), "he")
        self.assertEqual(detect_input_language("נגן את השיר האחרון ששמעתי"), "he")

    def test_empty_text_defaults_to_english(self):
        self.assertEqual(detect_input_language(""), "en")
        self.assertEqual(detect_input_language(None), "en")


class ResolveUtteranceLanguageTests(unittest.TestCase):
    def test_auto_mode_detects_per_utterance(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "auto"}, clear=False):
            self.assertEqual(resolve_utterance_language("open YouTube"), "en")
            self.assertEqual(resolve_utterance_language("פתח מוזיקה"), "he")

    def test_auto_mode_alternates_within_same_process(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "auto"}, clear=False):
            sequence = ["open YouTube", "פתח מוזיקה", "what song is playing?", "שיר הבא"]
            self.assertEqual([resolve_utterance_language(t) for t in sequence], ["en", "he", "en", "he"])

    def test_forced_english_mode_ignores_actual_hebrew_text(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "en"}, clear=False):
            self.assertEqual(resolve_utterance_language("פתח מוזיקה"), "en")

    def test_forced_hebrew_mode_ignores_actual_english_text(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "he"}, clear=False):
            self.assertEqual(resolve_utterance_language("open YouTube"), "he")


if __name__ == "__main__":
    unittest.main()
