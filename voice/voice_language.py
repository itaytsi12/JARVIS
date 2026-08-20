"""Voice INPUT language configuration (`VOICE_LANGUAGE` env var) and
per-utterance input-language detection.

Three supported modes for the whole STT pipeline:

- `"auto"` (recommended default): English OR Hebrew, detected per utterance.
  Neither STT provider is told to force a single language; the actual
  language of each committed transcript is determined locally afterward
  (see `detect_input_language` below).
- `"en"`: English speech only (forced) -- the ONLY single-language mode.
- `"he"`: Hebrew expected, English still recognized. Not "force Hebrew":
  JARVIS's own command vocabulary is English, so a Hebrew-speaking user
  still says "open Spotify" (see `EXPECTED_INPUT_LANGUAGES`).

Every STT provider (`voice/elevenlabs_realtime_stt.py`'s realtime Scribe
session, `voice/speech_to_text.py`'s local Whisper fallback) reads this SAME
value -- through `expected_input_languages`/`stt_language_code`, never by
interpreting the mode name itself -- so the two providers can never disagree
about which language(s) are expected, and the fallback provider can never be
stricter than the primary one.

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
    "he": "Hebrew (English also recognized)",
}
DEFAULT_LANGUAGE = "auto"

#: Which languages each mode must be ABLE to recognize, as opposed to which
#: one it EXPECTS. The distinction matters because JARVIS's own command
#: vocabulary is English: the wake phrase is "Hey Jarvis", spoken output is
#: hard-policy English (`get_tts_language`), and `brain/router.py`'s command
#: grammar is English with only a small Hebrew subset. A Hebrew-speaking
#: user therefore issues English commands routinely -- confirmed live, where
#: `VOICE_LANGUAGE=he` forced the Whisper fallback to `language="he"` and
#: turned ordinary English commands into Hebrew transliterations that no
#: route could match.
#:
#: So `"he"` means "Hebrew is expected, English is still recognized", not
#: "transcribe every sound as Hebrew". Only `"en"` names exactly one
#: language, and only a single-language mode is ever allowed to FORCE a
#: language code on a provider (see `stt_language_code`).
EXPECTED_INPUT_LANGUAGES: dict[str, tuple[str, ...]] = {
    "auto": ("en", "he"),
    "en": ("en",),
    "he": ("he", "en"),
}

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


def expected_input_languages(mode: str | None = None) -> tuple[str, ...]:
    """Every language the STT layer must be able to recognize in `mode`.

    See `EXPECTED_INPUT_LANGUAGES` for why forced `"he"` still includes
    English. Callers use this instead of comparing against the mode name,
    so a new mode never has to be special-cased in two providers.
    """
    return EXPECTED_INPUT_LANGUAGES[mode if mode is not None else get_voice_language()]


def stt_language_code(mode: str | None = None) -> str | None:
    """The language code to FORCE on an STT provider, or None for
    per-utterance detection.

    One rule, shared by every provider: force a language only when the
    configuration expects exactly one. Anything else uses the provider's
    own detection (ElevenLabs `include_language_detection=true`, Whisper
    `language=None`), because forcing one of several possible languages is
    precisely what corrupts the others.

    This also guarantees the two providers can never disagree about which
    language(s) are expected -- they ask the same function rather than each
    interpreting `VOICE_LANGUAGE` themselves.
    """
    languages = expected_input_languages(mode)
    return languages[0] if len(languages) == 1 else None


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
    """The actual input language for one committed utterance.

    Uses the same single-language rule as `stt_language_code`: the
    configured language is only assumed when the mode expects exactly one
    (`"en"`), and is otherwise detected from the text. That keeps this
    honest in `"he"` mode, where an English command genuinely can be
    recognized (see `EXPECTED_INPUT_LANGUAGES`) and must not be treated as
    Hebrew by the acknowledgement/response policy.

    This is the value callers (ack composition, routing diagnostics) should
    store per-interaction -- never the raw configured mode itself."""
    forced = stt_language_code()
    if forced is not None:
        return forced
    return detect_input_language(text)
