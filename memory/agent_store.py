"""The shared SQLite database behind conversation, long-term and episodic memory.

One file, one connection, one schema, so a restart restores all three
together and a single WAL-mode connection serves every reader. This is
separate from `memory/database.py` (the pre-existing entity/session store,
which is still used by `MemoryManager` and is untouched) because the
agent-memory tables are a new, independent concern with their own
lifecycle -- merging them would have meant rewriting a working schema.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from config import get_config

AGENT_MEMORY_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_schema_version(version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS conversations(
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    title TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS conversation_turns(
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    task_id TEXT,
    sequence INTEGER NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON conversation_turns(session_id, sequence);
CREATE INDEX IF NOT EXISTS idx_turns_created ON conversation_turns(created_at DESC);

CREATE TABLE IF NOT EXISTS long_term_memories(
    memory_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    subject TEXT,
    text TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 1,
    confidence REAL NOT NULL DEFAULT 1.0,
    source TEXT NOT NULL DEFAULT 'inferred',
    session_id TEXT,
    task_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_used_at TEXT,
    use_count INTEGER NOT NULL DEFAULT 0,
    superseded_by TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON long_term_memories(kind, importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_subject ON long_term_memories(subject);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_unique ON long_term_memories(kind, text);

CREATE TABLE IF NOT EXISTS episodes(
    episode_id TEXT PRIMARY KEY,
    session_id TEXT,
    task_id TEXT,
    created_at TEXT NOT NULL,
    user_request TEXT NOT NULL,
    route TEXT,
    model_used TEXT,
    success INTEGER NOT NULL DEFAULT 0,
    verified INTEGER NOT NULL DEFAULT 0,
    duration_ms REAL NOT NULL DEFAULT 0,
    step_count INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL,
    final_result TEXT,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task_id);
CREATE INDEX IF NOT EXISTS idx_episodes_success ON episodes(success, created_at DESC);
"""


class AgentDatabase:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else get_config().agent_db_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock, self.connection:
            self.connection.executescript(SCHEMA)
            row = self.connection.execute("SELECT COUNT(*) FROM agent_schema_version").fetchone()
            if not row[0]:
                self.connection.execute("INSERT INTO agent_schema_version VALUES(?)", (AGENT_MEMORY_SCHEMA_VERSION,))

    def execute(self, sql: str, parameters: tuple = ()):
        with self._lock, self.connection:
            return self.connection.execute(sql, parameters)

    def executescript(self, script: str) -> None:
        """Run a multi-statement DDL script under this database's own lock.

        Exposed so a module that owns its own table (`tasks/store.py`) can
        create it without reaching past `self._lock` into `.connection`,
        which would race with concurrent readers.
        """
        with self._lock, self.connection:
            self.connection.executescript(script)

    def query(self, sql: str, parameters: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.connection.execute(sql, parameters))

    def close(self) -> None:
        with self._lock:
            self.connection.close()


_DATABASE: AgentDatabase | None = None
_DATABASE_LOCK = threading.Lock()


def get_agent_database() -> AgentDatabase:
    global _DATABASE
    if _DATABASE is None:
        with _DATABASE_LOCK:
            if _DATABASE is None:
                path = None
                if "pytest" in sys.modules and not os.getenv("JARVIS_AGENT_DB_PATH"):
                    path = Path(tempfile.mkdtemp(prefix="jarvis-agentmem-pytest-")) / "jarvis_agent.sqlite3"
                _DATABASE = AgentDatabase(path)
    return _DATABASE


def reset_agent_database_for_tests(path: str | Path | None = None) -> AgentDatabase:
    global _DATABASE
    with _DATABASE_LOCK:
        if _DATABASE is not None:
            _DATABASE.close()
        _DATABASE = AgentDatabase(path or Path(tempfile.mkdtemp(prefix="jarvis-agentmem-pytest-")) / "jarvis_agent.sqlite3")
    return _DATABASE
