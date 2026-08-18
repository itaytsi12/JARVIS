"""Deterministic coding-task eligibility classification (Part A, Phase A1).

Decides whether a natural-language JARVIS command is a genuine coding/
self-improvement task that should go through the student-first ->
Claude-teacher pipeline (`brain/improvement_student_teacher.py`), as
opposed to an ordinary JARVIS command (open app, volume, calculator,
browser, search, ...) that must never reach a coding agent.

No LLM call -- the same small, deterministic, evidence-driven design as
`brain/improvement_classifier.py` and `brain/request_intent.py`. A bare
keyword match is never sufficient on its own (that would false-positive on
things like "fix this printer issue"); eligibility requires either a
strong, specific phrase, or a coding VERB co-occurring with a coding-domain
NOUN in the same utterance.
"""
from __future__ import annotations

import re

_CODING_VERB_RE = re.compile(r"\b(fix|debug|repair|refactor|implement|patch|resolve)\b", re.I)
_CODING_CONTEXT_RE = re.compile(
    r"\b(bug|bugs|code|codebase|repo|repository|function|functions|class|classes|module|modules|"
    r"implementation|failing test|failing tests|regression|exception|feature|self[- ]?improve)\b",
    re.I,
)

# Specific enough on their own that no separate context word is required.
_STRONG_PHRASES = (
    re.compile(r"\bfailing test", re.I),
    re.compile(r"\binspect\b.{0,60}\b(repo|repository|codebase)\b", re.I),
    re.compile(r"\badd\b.{0,40}\bfeature\b", re.I),
    re.compile(r"\bwhy\b.{0,60}\b(class|function|module)\b.{0,60}\b(behav|incorrect|wrong|fail|broken|error)", re.I),
    re.compile(r"\bself[- ]?improv", re.I),
    re.compile(r"\bwrite\b.{0,20}\bcode\b", re.I),
    re.compile(r"\bimprove\b.{0,30}\b(code|codebase|implementation)\b", re.I),
)


def is_coding_task(command: str) -> bool:
    """Pure, offline, deterministic. `True` only for genuine coding/
    self-improvement requests."""
    text = (command or "").strip()
    if not text:
        return False
    if any(pattern.search(text) for pattern in _STRONG_PHRASES):
        return True
    return bool(_CODING_VERB_RE.search(text) and _CODING_CONTEXT_RE.search(text))
