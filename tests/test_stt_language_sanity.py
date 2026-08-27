"""Regression tests for bug report 6: Whisper's unconstrained auto-detect
must not be trusted blindly when it lands outside the languages JARVIS is
actually configured to recognize (he/en) -- a plainly English/Hebrew
utterance mistranscribed as Dutch/Russian must be re-interpreted, not
accepted as-is. All local/offline -- no real model, no cloud LLM call."""
import unittest
from unittest.mock import Mock, patch

from voice import speech_to_text


class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _FakeInfo:
    def __init__(self, language):
        self.language = language


def _fake_model(transcripts_by_language: dict):
    """A fake faster-whisper model: `.transcribe(path, language=..., ...)`
    returns whatever `transcripts_by_language` maps that language (or the
    literal key `None` for unconstrained auto-detect) to -- a
    `(text, detected_language)` pair."""
    model = Mock()

    def transcribe(path, language=None, **kwargs):
        text, detected = transcripts_by_language[language]
        return [_FakeSegment(text)], _FakeInfo(detected)

    model.transcribe.side_effect = transcribe
    return model


class WhisperLanguageSanityTests(unittest.TestCase):
    def _run(self, transcripts_by_language, voice_language="auto"):
        model = _fake_model(transcripts_by_language)
        with patch.object(speech_to_text, "_get_model", return_value=model), \
             patch("voice.voice_language.get_voice_language", return_value=voice_language):
            return speech_to_text.transcribe_audio("fake.wav")

    def test_english_audio_mistranscribed_as_dutch_is_reinterpreted(self):
        # Auto-detect (unconstrained) lands on Dutch for plainly English
        # audio -- confirmed live. Expected one of he/en, not accepted.
        transcripts = {
            None: ("Stop.", "nl"),
            "en": ("Stop.", "en"),
        }
        result = self._run(transcripts)
        self.assertEqual(result, "Stop.")

    def test_stop_mistranscribed_as_russian_is_reinterpreted(self):
        transcripts = {
            None: ("Стоп.", "ru"),
            "en": ("Stop.", "en"),
        }
        result = self._run(transcripts)
        self.assertEqual(result, "Stop.")

    def test_genuine_hebrew_text_is_reinterpreted_toward_hebrew_not_english(self):
        # The mistranscribed text itself is Hebrew-scripted -- the
        # constrained retry must not blindly default to English and
        # destroy genuine Hebrew recognition.
        transcripts = {
            None: ("עצור", "ru"),  # implausible detected language, Hebrew-scripted text
            "he": ("עצור", "he"),
        }
        result = self._run(transcripts, voice_language="he")
        self.assertEqual(result, "עצור")

    def test_plausible_english_detection_is_never_retried(self):
        model = _fake_model({None: ("Open Notepad.", "en")})
        with patch.object(speech_to_text, "_get_model", return_value=model), \
             patch("voice.voice_language.get_voice_language", return_value="auto"):
            result = speech_to_text.transcribe_audio("fake.wav")
        self.assertEqual(result, "Open Notepad.")
        self.assertEqual(model.transcribe.call_count, 1)

    def test_plausible_hebrew_detection_is_never_retried(self):
        model = _fake_model({None: ("פתח יוטיוב", "he")})
        with patch.object(speech_to_text, "_get_model", return_value=model), \
             patch("voice.voice_language.get_voice_language", return_value="auto"):
            result = speech_to_text.transcribe_audio("fake.wav")
        self.assertEqual(result, "פתח יוטיוב")
        self.assertEqual(model.transcribe.call_count, 1)

    def test_forced_single_language_mode_is_never_second_guessed(self):
        # VOICE_LANGUAGE=en forces language="en" outright -- there is no
        # unconstrained detection to sanity-check at all.
        model = _fake_model({"en": ("Open Notepad.", "en")})
        with patch.object(speech_to_text, "_get_model", return_value=model), \
             patch("voice.voice_language.get_voice_language", return_value="en"):
            result = speech_to_text.transcribe_audio("fake.wav")
        self.assertEqual(result, "Open Notepad.")
        self.assertEqual(model.transcribe.call_count, 1)

    def test_empty_transcript_is_never_retried(self):
        model = _fake_model({None: ("", "ru")})
        with patch.object(speech_to_text, "_get_model", return_value=model), \
             patch("voice.voice_language.get_voice_language", return_value="auto"):
            result = speech_to_text.transcribe_audio("fake.wav")
        self.assertEqual(result, "")
        self.assertEqual(model.transcribe.call_count, 1)


if __name__ == "__main__":
    unittest.main()
