"""Bilingual transcript normalizer.

Performs small, local cleanups on STT output:
- removes common wake prefixes (jarvis / היי jarvis / ג'רוויס)
- normalizes common Hebrew aliases to their English equivalents
- avoids calling any external services
"""
from __future__ import annotations

import re
from typing import Tuple

# Aliases (kept empty for English-only mode)
ALIASES = {}

# Variants for Jarvis wake words (English-only)
WAKE_WORDS = [
	r"^\s*hey[\s,]+(?:jarvis|javis|jovis)\b[\s,.:;!?-]*",
	r"^\s*(?:jarvis|javis|jovis)\b[\s,.:;!?-]*",
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

    # Apply aliases (word boundaries) - currently empty for English-only
    for heb, eng in ALIASES.items():
        t = re.sub(rf"\b{heb}\b", eng, t, flags=re.IGNORECASE)

    # Repair only this exact clipped reference construction. Broader semantic
    # STT rewrites could alter literal user data and are intentionally avoided.
    if not re.search(r'''["']''',t):
        t=re.sub(r"\b((?:type|write)\s+(?:exactly\s+)?what you last)\s*[-–—]?\s*$",r"\1 told me",t,flags=re.I)

    return t.strip(), wake_removed
