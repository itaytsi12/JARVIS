import os
import unittest
from unittest.mock import patch

from voice.voice_language import (
    UnsupportedVoiceLanguage,
    get_voice_language,
    is_english_mode,
    is_hebrew_mode,
    language_name,
)


class VoiceLanguageTests(unittest.TestCase):
    def test_defaults_to_english_when_unset(self):
        env = dict(os.environ)
        env.pop("VOICE_LANGUAGE", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_voice_language(), "en")
            self.assertTrue(is_english_mode())
            self.assertFalse(is_hebrew_mode())

    def test_explicit_english(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "en"}, clear=False):
            self.assertEqual(get_voice_language(), "en")

    def test_explicit_hebrew(self):
        with patch.dict(os.environ, {"VOICE_LANGUAGE": "he"}, clear=False):
            self.assertEqual(get_voice_language(), "he")
            self.assertTrue(is_hebrew_mode())
            self.assertFalse(is_english_mode())

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


if __name__ == "__main__":
    unittest.main()
