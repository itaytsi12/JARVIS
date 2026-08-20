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
        self.assertIn("Hebrew", language_name("he"))

    def test_language_name_uses_configured_language_when_omitted(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "he"}, clear=False):
            self.assertIn("Hebrew", language_name())

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

    def test_hebrew_mode_still_recognizes_an_english_utterance_as_english(self):
        """`"he"` is bilingual, not "everything is Hebrew".

        JARVIS's command vocabulary is English, so a Hebrew-mode user still
        says "open YouTube"; treating that as Hebrew would give it the
        entity-free Hebrew acknowledgement instead of the contextual one.
        """
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "he"}, clear=False):
            self.assertEqual(resolve_utterance_language("open YouTube"), "en")
            self.assertEqual(resolve_utterance_language("פתח יוטיוב"), "he")


class SttLanguagePolicyTests(unittest.TestCase):
    """One shared rule for BOTH STT providers: force a language code only
    when the configuration expects exactly one language."""

    def test_only_english_mode_forces_a_language_code(self):
        from voice.voice_language import stt_language_code

        with patch.dict(os.environ, {"VOICE_LANGUAGE": "en"}, clear=False):
            self.assertEqual(stt_language_code(), "en")
        for bilingual in ("auto", "he"):
            with self.subTest(mode=bilingual), patch.dict(os.environ, {"VOICE_LANGUAGE": bilingual}, clear=False):
                self.assertIsNone(stt_language_code())

    def test_every_mode_can_recognize_english(self):
        """The command grammar is English; no mode may exclude it."""
        from voice.voice_language import expected_input_languages

        for mode in ("auto", "en", "he"):
            with self.subTest(mode=mode):
                self.assertIn("en", expected_input_languages(mode))

    def test_hebrew_is_preserved_where_it_is_configured(self):
        from voice.voice_language import expected_input_languages

        self.assertIn("he", expected_input_languages("he"))
        self.assertIn("he", expected_input_languages("auto"))
        self.assertEqual(expected_input_languages("he")[0], "he")


if __name__ == "__main__":
    unittest.main()
