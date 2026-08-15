"""Simple language utilities for Hebrew/English dominance detection.

Uses a Unicode-character heuristic: counts Hebrew letters vs Latin letters.
"""
from __future__ import annotations

import re


def detect_dominant_language(text: str) -> str:
    """Force English-only language detection: always returns 'en'.

This keeps downstream TTS and response formatting consistent with
an English-only STT pipeline.
"""
    return 'en'
