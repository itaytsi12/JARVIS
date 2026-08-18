"""Explicit voice INPUT language mode (`VOICE_LANGUAGE` env var).

JARVIS does not attempt mixed-language/auto-detected transcription right
now. Exactly one of two supported modes is active for the whole STT
pipeline at a time:

- `"en"` (default): English speech only -- the historical, already-working
  behavior, unchanged.
- `"he"`: Hebrew speech only.

Every STT provider (`voice/elevenlabs_realtime_stt.py`'s realtime Scribe
session, `voice/speech_to_text.py`'s local Whisper fallback) reads this
SAME value rather than each hardcoding or guessing its own language, so
the two providers can never disagree about which language is expected.

This is deliberately about STT INPUT only. Spoken OUTPUT stays English
regardless of `VOICE_LANGUAGE` (see `voice/language_utils.py`'s
`detect_dominant_language`, unchanged) -- Hebrew TTS is out of scope for
now; JARVIS executes a Hebrew command and answers with a short English
acknowledgement.
"""
from __future__ import annotations

import os

#: code -> human-readable name, also used to validate `VOICE_LANGUAGE`.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "he": "Hebrew",
}
DEFAULT_LANGUAGE = "en"


class UnsupportedVoiceLanguage(ValueError):
    """`VOICE_LANGUAGE` is set to something other than a supported code.
    Raised rather than silently falling back to a guessed language."""


def get_voice_language() -> str:
    """Return the configured voice language code (`"en"` / `"he"`).
    Raises `UnsupportedVoiceLanguage` for any other value -- callers must
    never guess a fallback for a typo'd/unsupported setting."""
    raw = os.getenv("VOICE_LANGUAGE", DEFAULT_LANGUAGE)
    value = (raw or "").strip().lower()
    if value not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise UnsupportedVoiceLanguage(
            f"VOICE_LANGUAGE={raw!r} is not supported. Supported values: {supported}."
        )
    return value


def language_name(code: str | None = None) -> str:
    """Human-readable name for `code` (or the currently configured
    language when `code` is omitted)."""
    return SUPPORTED_LANGUAGES[code if code is not None else get_voice_language()]


def is_hebrew_mode() -> bool:
    return get_voice_language() == "he"


def is_english_mode() -> bool:
    return get_voice_language() == "en"
