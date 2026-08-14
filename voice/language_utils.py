"""Simple language utilities for Hebrew/English dominance detection.

Uses a Unicode-character heuristic: counts Hebrew letters vs Latin letters.
"""
from __future__ import annotations

import re


def detect_dominant_language(text: str) -> str:
    """Return 'he' if text is mostly Hebrew letters, otherwise 'en'.

This is a simple heuristic sufficient for choosing spoken response language.
"""
    if not text:
        return 'en'

    heb = len(re.findall(r'[\u0590-\u05FF]', text))
    lat = len(re.findall(r'[A-Za-z]', text))

    # If more Hebrew characters than Latin, consider Hebrew dominant.
    if heb > lat:
        return 'he'
    return 'en'
