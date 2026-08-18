"""Voice INPUT language configuration (`VOICE_LANGUAGE` env var) and
per-utterance input-language detection.

Three supported modes for the whole STT pipeline:

- `"auto"` (recommended default): English OR Hebrew, detected per utterance.
  Neither STT provider is told to force a single language; the actual
  language of each committed transcript is determined locally afterward
  (see `detect_input_language` below).
- `"en"`: English speech only (forced).
- `"he"`: Hebrew speech only (forced).

Every STT provider (`voice/elevenlabs_realtime_stt.py`'s realtime Scribe
session, `voice/speech_to_text.py`'s local Whisper fallback) reads this SAME
value rather than each hardcoding or guessing its own language, so the two
providers can never disagree about which language(s) are expected.

This is deliberately about STT INPUT only. Spoken OUTPUT stays English
regardless of `VOICE_LANGUAGE` -- see `TTS_LANGUAGE`/`get_tts_language`
below, a separate, hard-enforced policy `VOICE_LANGUAGE` can never affect.
"""
from __future__ import annotations

import os
import re

#: code -> human-readable name, also used to validate `VOICE_LANGUAGE`.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "auto": "Auto (English/Hebrew)",
    "en": "English",
    "he": "Hebrew",
}
DEFAULT_LANGUAGE = "auto"

#: TTS is a hard, separate policy from STT input -- see `get_tts_language`.
_SUPPORTED_TTS_LANGUAGES = {"en"}
DEFAULT_TTS_LANGUAGE = "en"


class UnsupportedVoiceLanguage(ValueError):
    """`VOICE_LANGUAGE` is set to something other than a supported code.
    Raised rather than silently falling back to a guessed language."""


def get_voice_language() -> str:
    """Return the configured voice INPUT language mode
    (`"auto"` / `"en"` / `"he"`). Raises `UnsupportedVoiceLanguage` for any
    other value -- callers must never guess a fallback for a typo'd/
    unsupported setting."""
    raw = os.getenv("VOICE_LANGUAGE", DEFAULT_LANGUAGE)
    value = (raw or "").strip().lower()
    if value not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise UnsupportedVoiceLanguage(
            f"VOICE_LANGUAGE={raw!r} is not supported. Supported values: {supported}."
        )
    return value


def get_tts_language() -> str:
    """Hard, internal TTS-output policy -- ALWAYS English, regardless of
    `VOICE_LANGUAGE` (which only ever controls STT input). `TTS_LANGUAGE`
    is read only so a non-"en" value can be reported/logged honestly
    (e.g. at startup); it can never change what this function returns."""
    return DEFAULT_TTS_LANGUAGE


def language_name(code: str | None = None) -> str:
    """Human-readable name for `code` (or the currently configured
    language when `code` is omitted)."""
    return SUPPORTED_LANGUAGES[code if code is not None else get_voice_language()]


def is_auto_mode() -> bool:
    return get_voice_language() == "auto"


def is_hebrew_mode() -> bool:
    """True only when `VOICE_LANGUAGE` is FORCED to Hebrew. In `"auto"`
    mode, use `detect_input_language(transcript) == "he"` per utterance
    instead -- this function alone can't answer "is this Hebrew" in auto
    mode, since the configured mode and the actual utterance language are
    deliberately different questions (see module docstring)."""
    return get_voice_language() == "he"


def is_english_mode() -> bool:
    return get_voice_language() == "en"


#: Hebrew block (U+0590-U+05FF), same range already used by
#: `brain/music_intent.py`'s `_HEBREW_CHAR` -- one canonical detector
#: rather than two independently-maintained regexes.
_HEBREW_CHAR = re.compile(r"[֐-׿]")


def detect_input_language(text: str) -> str:
    """Local, script-based (no LLM) per-utterance language detection: `"he"`
    if `text` contains any Hebrew-block character, else `"en"`. This is
    deliberately independent of the configured `VOICE_LANGUAGE` mode -- a
    forced `"en"`/`"he"` mode still uses this for the code that needs the
    ACTUAL utterance language (e.g. TTS acknowledgement wording), while
    `"auto"` mode uses it as the only source of truth for that language."""
    return "he" if _HEBREW_CHAR.search(text or "") else "en"


def resolve_utterance_language(text: str) -> str:
    """The actual input language for one committed utterance: the forced
    language in `"en"`/`"he"` mode, or the detected language in `"auto"`
    mode. This is the value callers (ack composition, routing diagnostics)
    should store per-interaction -- never the raw configured mode itself."""
    mode = get_voice_language()
    if mode == "auto":
        return detect_input_language(text)
    return mode
