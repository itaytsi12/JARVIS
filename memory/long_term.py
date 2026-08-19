"""Long-term memory: the small set of facts genuinely worth keeping.

The hard part is not storage, it is deciding what deserves to be stored.
"open YouTube" must never become a permanent memory; "my main project is
at C:/dev/jarvis" must. `extract_memories` implements that judgement
locally, with rules, so it costs nothing and runs on every interaction:

- An explicit instruction ("remember that ...", "from now on ...") is
  always kept, at high importance and with `source="explicit"`.
- A stated preference ("I prefer ...", "always use ...", "don't ...")
  is kept.
- A stated identity/location fact ("my project is at ...", "my name
  is ...", "I use ...") is kept.
- A correction ("no, I meant ...", "that's wrong, ...") is kept, because
  corrections are exactly the signal that should not be repeated.
- Everything else is NOT promoted. Ordinary commands stay in raw
  conversation history and in episodes, where they are still available.

Storage is idempotent on (kind, text): repeating a fact updates its
importance and recency instead of creating duplicates.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from memory.agent_store import AgentDatabase, get_agent_database, utc_now
from memory.memory_manager import redact

# Memory kinds.
PREFERENCE = "preference"
FACT = "fact"
PROJECT = "project"
CORRECTION = "correction"
WORKFLOW = "workflow"

EXPLICIT = "explicit"
INFERRED = "inferred"

_EXPLICIT_PATTERNS = (
    re.compile(r"^(?:please\s+)?remember(?:\s+that|\s+this)?[:,]?\s+(?P<text>.+)$", re.I),
    re.compile(r"^(?:please\s+)?(?:keep|make a note|note) (?:in mind|of)(?:\s+that)?[:,]?\s+(?P<text>.+)$", re.I),
    re.compile(r"^from now on[,]?\s+(?P<text>.+)$", re.I),
    re.compile(r"^don'?t forget(?:\s+that)?[:,]?\s+(?P<text>.+)$", re.I),
)

_PREFERENCE_PATTERNS = (
    re.compile(r"^i (?:prefer|like|want|always want)\b.+$", re.I),
    re.compile(r"^(?:always|never)\s+(?!mind\b).+$", re.I),
    re.compile(r"^i (?:don'?t|do not) (?:like|want)\b.+$", re.I),
    re.compile(r"^(?:use|call me|address me as)\b.+$", re.I),
)

_FACT_PATTERNS = (
    re.compile(r"^my\s+\w[\w \-]{0,40}\s+(?:is|are|lives?|sits?)\b.+$", re.I),
    re.compile(r"^i (?:am|work|use|run|own|have)\b.+$", re.I),
    re.compile(r"^the\s+\w[\w \-]{0,40}\s+(?:is at|lives at|is located)\b.+$", re.I),
)

_CORRECTION_PATTERNS = (
    re.compile(r"^(?:no,?|actually,?)\s+i (?:meant|said)\b.+$", re.I),
    re.compile(r"^that'?s (?:wrong|not right|not what i)\b.+$", re.I),
    re.compile(r"^i said\b.+$", re.I),
)

_PROJECT_HINT = re.compile(r"\b(?:project|repo|repository|codebase|solution)\b", re.I)

# Phrasings that look declarative but are ordinary one-off commands.
_NEVER_REMEMBER = re.compile(
    r"^(?:open|close|launch|start|stop|play|pause|resume|mute|unmute|type|write|press|click|"
    r"search|google|calculate|screenshot|take a screenshot|volume|next|previous|show|minimize|maximize)\b",
    re.I,
)


@dataclass
class MemoryRecord:
    memory_id: str
    kind: str
    text: str
    subject: str | None = None
    importance: int = 1
    confidence: float = 1.0
    source: str = INFERRED
    session_id: str | None = None
    task_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    last_used_at: str | None = None
    use_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryCandidate:
    """A fact `extract_memories` believes is worth keeping, before storage."""

    kind: str
    text: str
    importance: int
    source: str
    subject: str | None = None
    reason: str = ""


def _subject_of(text: str) -> str | None:
    if _PROJECT_HINT.search(text):
        return "project"
    match = re.match(r"^my\s+([\w \-]{1,40}?)\s+(?:is|are)\b", text, re.I)
    return match.group(1).strip().lower() if match else None


def extract_memories(user_text: str, *, assistant_text: str | None = None) -> list[MemoryCandidate]:
    """Decide what, if anything, in this exchange deserves to be remembered.

    Returns an empty list for the overwhelming majority of utterances --
    that is the point. Never calls a model.
    """
    text = (user_text or "").strip().rstrip(".!")
    if not text or len(text) < 4:
        return []

    for pattern in _EXPLICIT_PATTERNS:
        match = pattern.match(text)
        if match:
            payload = match.group("text").strip()
            if not payload:
                return []
            kind = PROJECT if _PROJECT_HINT.search(payload) else FACT
            return [
                MemoryCandidate(
                    kind=kind,
                    text=payload,
                    importance=5,
                    source=EXPLICIT,
                    subject=_subject_of(payload),
                    reason="explicit_instruction",
                )
            ]

    if _NEVER_REMEMBER.match(text):
        return []

    for pattern in _CORRECTION_PATTERNS:
        if pattern.match(text):
            return [MemoryCandidate(CORRECTION, text, 4, INFERRED, _subject_of(text), "user_correction")]

    for pattern in _PREFERENCE_PATTERNS:
        if pattern.match(text):
            return [MemoryCandidate(PREFERENCE, text, 3, INFERRED, _subject_of(text), "stated_preference")]

    for pattern in _FACT_PATTERNS:
        if pattern.match(text):
            kind = PROJECT if _PROJECT_HINT.search(text) else FACT
            return [MemoryCandidate(kind, text, 3, INFERRED, _subject_of(text), "stated_fact")]

    return []


class LongTermMemoryStore:
    def __init__(self, database: AgentDatabase | None = None):
        self.db = database or get_agent_database()

    def remember(
        self,
        text: str,
        *,
        kind: str = FACT,
        importance: int = 2,
        source: str = INFERRED,
        subject: str | None = None,
        confidence: float = 1.0,
        session_id: str | None = None,
        task_id: str | None = None,
        **metadata: Any,
    ) -> MemoryRecord:
        """Store (or refresh) one long-term memory.

        Idempotent on (kind, text): the same fact stated twice keeps the
        HIGHER importance and refreshes `updated_at`, so repetition
        strengthens a memory instead of duplicating it.
        """
        safe_text = redact((text or "").strip())
        if not safe_text:
            raise ValueError("A memory must have text")
        now = utc_now()
        existing = self.db.query(
            "SELECT * FROM long_term_memories WHERE kind=? AND text=?", (kind, safe_text)
        )
        if existing:
            row = existing[0]
            merged_importance = max(int(row["importance"]), int(importance))
            self.db.execute(
                "UPDATE long_term_memories SET importance=?, updated_at=?, confidence=?, source=? WHERE memory_id=?",
                (merged_importance, now, max(float(row["confidence"]), confidence), source if source == EXPLICIT else row["source"], row["memory_id"]),
            )
            return self.get(row["memory_id"])  # type: ignore[return-value]

        record = MemoryRecord(
            memory_id=uuid.uuid4().hex,
            kind=kind,
            text=safe_text,
            subject=subject,
            importance=int(importance),
            confidence=float(confidence),
            source=source,
            session_id=session_id,
            task_id=task_id,
            created_at=now,
            updated_at=now,
            metadata=redact(metadata),
        )
        self.db.execute(
            "INSERT INTO long_term_memories VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.memory_id,
                record.kind,
                record.subject,
                record.text,
                record.importance,
                record.confidence,
                record.source,
                record.session_id,
                record.task_id,
                record.created_at,
                record.updated_at,
                record.last_used_at,
                record.use_count,
                None,
                json.dumps(record.metadata),
            ),
        )
        return record

    def remember_candidates(
        self,
        candidates: Iterable[MemoryCandidate],
        *,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> list[MemoryRecord]:
        return [
            self.remember(
                candidate.text,
                kind=candidate.kind,
                importance=candidate.importance,
                source=candidate.source,
                subject=candidate.subject,
                session_id=session_id,
                task_id=task_id,
                reason=candidate.reason,
            )
            for candidate in candidates
        ]

    def get(self, memory_id: str) -> MemoryRecord | None:
        rows = self.db.query("SELECT * FROM long_term_memories WHERE memory_id=?", (memory_id,))
        return _decode(rows[0]) if rows else None

    def all(self, limit: int = 500) -> list[MemoryRecord]:
        rows = self.db.query(
            "SELECT * FROM long_term_memories WHERE superseded_by IS NULL ORDER BY importance DESC, updated_at DESC LIMIT ?",
            (limit,),
        )
        return [_decode(row) for row in rows]

    def by_kind(self, kind: str, limit: int = 100) -> list[MemoryRecord]:
        rows = self.db.query(
            "SELECT * FROM long_term_memories WHERE kind=? AND superseded_by IS NULL ORDER BY importance DESC, updated_at DESC LIMIT ?",
            (kind, limit),
        )
        return [_decode(row) for row in rows]

    def mark_used(self, memory_id: str) -> None:
        self.db.execute(
            "UPDATE long_term_memories SET use_count=use_count+1, last_used_at=? WHERE memory_id=?",
            (utc_now(), memory_id),
        )

    def forget(self, memory_id: str, superseded_by: str | None = None) -> None:
        """Retire a memory. Kept as a row (marked superseded) rather than
        deleted, so the record of what was once believed survives."""
        self.db.execute(
            "UPDATE long_term_memories SET superseded_by=?, updated_at=? WHERE memory_id=?",
            (superseded_by or "retired", utc_now(), memory_id),
        )

    def count(self) -> int:
        rows = self.db.query("SELECT COUNT(*) AS n FROM long_term_memories WHERE superseded_by IS NULL")
        return int(rows[0]["n"]) if rows else 0


def _decode(row) -> MemoryRecord:
    return MemoryRecord(
        memory_id=row["memory_id"],
        kind=row["kind"],
        text=row["text"],
        subject=row["subject"],
        importance=int(row["importance"]),
        confidence=float(row["confidence"]),
        source=row["source"],
        session_id=row["session_id"],
        task_id=row["task_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_used_at=row["last_used_at"],
        use_count=int(row["use_count"]),
        metadata=json.loads(row["metadata_json"] or "{}"),
    )
