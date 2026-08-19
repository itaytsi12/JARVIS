"""Persistent conversation history.

Raw history, deliberately kept separate from useful memory
(`memory/long_term.py`). Everything said is stored here; only the small
subset worth keeping forever is promoted to a long-term memory. That
separation is what keeps the model's context small: a request retrieves a
handful of recent turns plus a handful of RELEVANT memories, never the
entire transcript.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from memory.agent_store import AgentDatabase, get_agent_database, utc_now
from memory.memory_manager import redact

USER = "user"
ASSISTANT = "assistant"
SYSTEM = "system"


@dataclass
class ConversationTurn:
    turn_id: str
    session_id: str
    sequence: int
    role: str
    text: str
    created_at: str
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "role": self.role,
            "text": self.text,
            "created_at": self.created_at,
            "task_id": self.task_id,
            "metadata": self.metadata,
        }


class ConversationStore:
    def __init__(self, database: AgentDatabase | None = None):
        self.db = database or get_agent_database()

    def start_session(self, session_id: str | None = None, title: str | None = None, **metadata: Any) -> str:
        session_id = session_id or uuid.uuid4().hex
        self.db.execute(
            "INSERT OR IGNORE INTO conversations(session_id, started_at, title, metadata_json) VALUES(?,?,?,?)",
            (session_id, utc_now(), title, json.dumps(redact(metadata))),
        )
        return session_id

    def end_session(self, session_id: str) -> None:
        self.db.execute("UPDATE conversations SET ended_at=? WHERE session_id=?", (utc_now(), session_id))

    def add_turn(
        self,
        session_id: str,
        role: str,
        text: str,
        *,
        task_id: str | None = None,
        **metadata: Any,
    ) -> ConversationTurn:
        self.start_session(session_id)
        row = self.db.query(
            "SELECT COALESCE(MAX(sequence),0) AS last FROM conversation_turns WHERE session_id=?", (session_id,)
        )
        sequence = int(row[0]["last"]) + 1 if row else 1
        turn = ConversationTurn(
            turn_id=uuid.uuid4().hex,
            session_id=session_id,
            sequence=sequence,
            role=role,
            text=redact(text or ""),
            created_at=utc_now(),
            task_id=task_id,
            metadata=redact(metadata),
        )
        self.db.execute(
            "INSERT INTO conversation_turns VALUES(?,?,?,?,?,?,?,?)",
            (
                turn.turn_id,
                turn.session_id,
                turn.task_id,
                turn.sequence,
                turn.role,
                turn.text,
                turn.created_at,
                json.dumps(turn.metadata),
            ),
        )
        return turn

    def recent_turns(self, session_id: str, limit: int = 10) -> list[ConversationTurn]:
        rows = self.db.query(
            "SELECT * FROM conversation_turns WHERE session_id=? ORDER BY sequence DESC LIMIT ?",
            (session_id, limit),
        )
        return [_decode(row) for row in reversed(rows)]

    def search(self, query: str, limit: int = 10) -> list[ConversationTurn]:
        """Substring search across every stored turn, newest first.

        Deliberately a simple LIKE: this is raw history, used to answer
        "did we talk about X", not the primary relevance path (that is
        `memory/retrieval.py` over long-term memories and episodes).
        """
        rows = self.db.query(
            "SELECT * FROM conversation_turns WHERE text LIKE ? ESCAPE '\\' ORDER BY created_at DESC LIMIT ?",
            (f"%{_escape_like(query)}%", limit),
        )
        return [_decode(row) for row in rows]

    def sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM conversations ORDER BY started_at DESC LIMIT ?", (limit,))
        return [
            {
                "session_id": row["session_id"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "title": row["title"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
            }
            for row in rows
        ]

    def turn_count(self, session_id: str | None = None) -> int:
        if session_id:
            rows = self.db.query("SELECT COUNT(*) AS n FROM conversation_turns WHERE session_id=?", (session_id,))
        else:
            rows = self.db.query("SELECT COUNT(*) AS n FROM conversation_turns")
        return int(rows[0]["n"]) if rows else 0


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _decode(row) -> ConversationTurn:
    return ConversationTurn(
        turn_id=row["turn_id"],
        session_id=row["session_id"],
        sequence=int(row["sequence"]),
        role=row["role"],
        text=row["text"],
        created_at=row["created_at"],
        task_id=row["task_id"],
        metadata=json.loads(row["metadata_json"] or "{}"),
    )
