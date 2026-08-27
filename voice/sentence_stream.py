"""Turn a stream of model text into chunks that are safe to SPEAK.

Streaming exists so the first sentence of a long answer can be spoken while
the rest is still being generated. That only helps if what gets spoken is
never wrong, so this module is deliberately conservative about two separate
risks:

1. **Half-formed speech.** A raw token stream produces "The proj", "ect
   contains" -- unspeakable. Text is released only at a sentence boundary,
   and a boundary is only a boundary when the next character confirms it
   (so "main.py contains" does not split after "main.").

2. **Speaking the wrong thing entirely.** A turn that ends in a tool call
   often begins with a short preamble ("Let me look at the files."). That is
   not the answer, and speaking it would both pre-empt the real answer and
   narrate the agent's approach. `HOLD_BACK_CHARS` is the guard: nothing is
   released until enough text has accumulated that this cannot be a
   preamble. `providers/anthropic_provider.py` independently stops feeding
   text the moment a `tool_use` block starts, so the two mechanisms cover
   the case from both ends.

Markdown is stripped rather than spoken -- "**brain**" must not become
"star star brain star star" -- and fenced code blocks are dropped entirely,
since reading source aloud is never what the user wanted.

Nothing here is model-specific: it takes text in and gives speakable text
out, so it is equally usable for any provider.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

#: Characters of assistant text that must accumulate before ANY of it may be
#: spoken. A tool-call preamble is a short phrase; a real answer is not. Set
#: to 0 to release from the first complete sentence (used by tests, and safe
#: when the caller already knows the turn is final).
HOLD_BACK_CHARS = 180

#: A sentence ends at . ! ? or a newline -- but only when what follows is
#: whitespace or the end of the text, so decimals, "main.py" and "e.g." do
#: not split a sentence in half.
_SENTENCE_END = re.compile(r"(?<=[.!?])(?=\s)|\n+")

#: Speech is not writing. These are removed, not read out.
_CODE_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1", re.S)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_BULLET = re.compile(r"^\s{0,4}[-*+]\s+", re.M)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_LEFTOVER_MARKS = re.compile(r"[*_`>#]+")
_WHITESPACE = re.compile(r"[ \t]+")


def speakable(text: str) -> str:
    """Strip markup that must never be pronounced, keeping the words."""
    cleaned = _CODE_FENCE.sub(" ", text)
    cleaned = _LINK.sub(r"\1", cleaned)
    cleaned = _BOLD_ITALIC.sub(r"\2", cleaned)
    cleaned = _INLINE_CODE.sub(r"\1", cleaned)
    cleaned = _HEADING.sub("", cleaned)
    cleaned = _BULLET.sub("", cleaned)
    cleaned = _LEFTOVER_MARKS.sub("", cleaned)
    return _WHITESPACE.sub(" ", cleaned).strip()


def split_sentences(text: str) -> tuple[list[str], str]:
    """Split `text` into complete sentences and the incomplete remainder.

    The remainder is whatever follows the last confirmed boundary; the caller
    keeps it and prepends the next chunk to it.
    """
    if not text:
        return [], ""
    pieces = [piece for piece in _SENTENCE_END.split(text)]
    if not pieces:
        return [], text
    remainder = pieces[-1]
    complete = [piece.strip() for piece in pieces[:-1] if piece.strip()]
    return complete, remainder


@dataclass
class SentenceStream:
    """Accumulate streamed text; emit only whole, speakable sentences.

    `emit` is called with each released chunk. It is never called with a
    partial word, a code fence, a tool-call payload (those never reach here)
    or an empty string.
    """

    emit: Callable[[str], None]
    hold_back_chars: int = HOLD_BACK_CHARS
    _buffer: str = ""
    _seen_chars: int = 0
    _released: bool = False
    spoken_chunks: list[str] = field(default_factory=list)

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        self._buffer += chunk
        self._seen_chars += len(chunk)
        if not self._released and self._seen_chars < self.hold_back_chars:
            # Still possibly a tool preamble -- hold everything.
            return
        # An unterminated code fence would be released mid-block; wait for
        # its closing fence rather than speaking half of it.
        if self._buffer.count("```") % 2:
            return
        # Remove COMPLETE fenced blocks from the buffer before splitting.
        # Doing it later, per released sentence, is too late: the sentence
        # splitter breaks on newlines and would hand `speakable` fragments
        # with no fence left to recognise, so the code got spoken.
        self._buffer = _CODE_FENCE.sub(" ", self._buffer)
        complete, remainder = split_sentences(self._buffer)
        if not complete:
            return
        self._buffer = remainder
        for sentence in complete:
            self._release(sentence)

    def flush(self) -> None:
        """Release whatever is left. Called when generation has finished, so
        an answer with no trailing punctuation is still spoken."""
        remaining, self._buffer = self._buffer, ""
        if remaining.strip():
            self._release(remaining)

    def discard(self) -> None:
        """Throw away buffered text without speaking it -- used when the turn
        turned out not to be the final answer after all."""
        self._buffer = ""

    @property
    def has_spoken(self) -> bool:
        return self._released

    def _release(self, text: str) -> None:
        cleaned = speakable(text)
        if not cleaned:
            return
        self._released = True
        self.spoken_chunks.append(cleaned)
        self.emit(cleaned)


__all__ = ["HOLD_BACK_CHARS", "SentenceStream", "speakable", "split_sentences"]
