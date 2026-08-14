"""Bilingual transcript normalizer.

Performs small, local cleanups on STT output:
- removes common wake prefixes (jarvis / היי jarvis / ג'רוויס)
- normalizes common Hebrew aliases to their English equivalents
- avoids calling any external services
"""
from __future__ import annotations

import re
from typing import Tuple

# Map common Hebrew words to their English aliases (do not transliterate names)
ALIASES = {
    r"יוטיוב": "youtube",
    r"ביוטיוב": "youtube",
    r"גוגל": "google",
    r"בגוגל": "google",
    r"ספוטיפיי": "spotify",
    r"בספוטיפיי": "spotify",
    r"כרום": "chrome",
    r"בכרום": "chrome",
    r"דיסקורד": "discord",
    r"בדיסקורד": "discord",
    r"וי אס קוד": "vscode",
    r"ויזואל סטודיו קוד": "vscode",
}

# Variants for Jarvis wake words
WAKE_WORDS = [
    r"^\s*jarvis\b",
    r"^\s*hey\s+jarvis\b",
    r"^\s*היי\s+jarvis\b",
    r"^\s*ג'[ר׳]?רוויס\b",
    r"^\s*ג׳רוויס\b",
    r"^\s*היי\s+ג'[ר׳]?רוויס\b",
]


def normalize_transcript(text: str) -> Tuple[str, bool]:
    """Normalize `text` and remove wake prefix if present.

    Returns (normalized_text, wake_removed_flag).
    """
    if not text:
        return "", False

    orig = text
    t = text.strip()

    # Basic cleanup: collapse whitespace
    t = re.sub(r"\s+", " ", t)

    wake_removed = False
    # Remove wake words at start
    for pattern in WAKE_WORDS:
        m = re.match(pattern, t, flags=re.IGNORECASE)
        if m:
            t = t[m.end():].strip()
            wake_removed = True
            break

    # Normalize apostrophe variants for Jarvis within the sentence
    t = re.sub(r"[גg]'?רוויס|ג׳רוויס", "jarvis", t, flags=re.IGNORECASE)

    # Apply aliases (word boundaries)
    for heb, eng in ALIASES.items():
        t = re.sub(rf"\b{heb}\b", eng, t, flags=re.IGNORECASE)

    # Small verb/phrase mappings to support Hebrew command forms
    # e.g. 'תפתח notepad' -> 'open notepad', 'תחפש ביוטיוב Jude' -> 'search youtube for Jude'
    # Replace 'תפתח X' and 'פתח X'
    t = re.sub(r"\b(?:תפתח|פתח)\s+([\w\s\.]+)$", r"open \1", t, flags=re.IGNORECASE)
    # Replace 'תחפש ביוטיוב <query>' -> 'search youtube for <query>'
    t = re.sub(r"\b(?:תחפש|חפש)\s+(?:ביוטיוב|יוטיוב|youtube)\s+(.+)$", r"search youtube for \1", t, flags=re.IGNORECASE)
    # Replace 'תנמיך volume' -> 'volume down'
    t = re.sub(r"\bתנמיך\s+volume\b", "volume down", t, flags=re.IGNORECASE)
    # Replace 'תגביר' or 'תעלה' -> 'volume up'
    t = re.sub(r"\b(?:תגביר|תעלה)\b", "volume up", t, flags=re.IGNORECASE)

    return t.strip(), wake_removed
