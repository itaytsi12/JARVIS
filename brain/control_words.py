"""Deliberately bounded multilingual normalization for JARVIS's
highest-priority control words (stop / cancel / pause).

This is NOT a translation system -- it is the small, fixed set of
equivalents STT has been observed, LIVE, to substitute for the English
control word when it mis-detects the utterance's language entirely (e.g.
English "Stop!" committed by Whisper/ElevenLabs as Russian "Стоп!", which
is simply the Russian word for "stop"). `brain/router.py`'s cancel/pause
routes must never require a paid model call to recognize these, so this
stays a plain dict lookup. Anything not in the table is returned unchanged
(punctuation-trimmed and lowercased, matching what the router already does
at every other control-command check)."""
from __future__ import annotations

_EQUIVALENTS: dict[str, str] = {
    # Russian -- confirmed live: English "stop" transcribed as this.
    "стоп": "stop",
    "отмена": "cancel",
    "отменить": "cancel",
    # Hebrew -- JARVIS already has first-class bilingual Hebrew voice
    # support (see brain/music_intent.py's Hebrew classifier), so a
    # Hebrew-speaking user saying these words is a real, expected case,
    # not just an STT misfire.
    "עצור": "stop",
    "בטל": "cancel",
    "ביטול": "cancel",
    "תבטל": "cancel",
}


def normalize_control_word(text: str) -> str:
    """`text`, lowercased and stripped of trailing punctuation, with a
    known-equivalent control word swapped for its English form."""
    stripped = (text or "").strip().rstrip("!.?,;:").lower()
    return _EQUIVALENTS.get(stripped, stripped)


__all__ = ["normalize_control_word"]
