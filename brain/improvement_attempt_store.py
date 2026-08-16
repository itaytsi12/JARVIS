"""Persistence for ImprovementAttempt records.

Deliberately a second, small SQLite store rather than folding attempts into
`brain/improvement_store.py`'s `improvement_candidates` table: a candidate
is a deduplicated, upsert-by-fingerprint record of a *problem*; an attempt
is an append-only history of *tries* at fixing one, and a single candidate
can accumulate multiple attempts over time. Mixing those write patterns into
one table would make both harder to reason about. This otherwise follows
`ImprovementStore`'s exact shape (thread-safe SQLite, WAL, one small
wrapper) on purpose, for the same reasons documented there.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from brain.improvement_attempt_models import TERMINAL_ATTEMPT_STATUSES, AttemptStatus, ImprovementAttempt
from brain.improvement_models import canonical_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ImprovementAttemptStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("IMPROVEMENT_ATTEMPT_DB_PATH") or Path.cwd() / "data" / "jarvis_improvement_attempts.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock, self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS improvement_attempts(
                    attempt_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attempts_candidate ON improvement_attempts(candidate_id);
                CREATE INDEX IF NOT EXISTS idx_attempts_status ON improvement_attempts(status);
                """
            )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def create(self, attempt: ImprovementAttempt) -> ImprovementAttempt:
        """Insert a brand-new attempt row. Raises sqlite3.IntegrityError if
        `attempt_id` already exists -- attempts are never silently
        overwritten; use `update` for an in-progress attempt's own lifecycle."""
        timestamp = now()
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT INTO improvement_attempts VALUES(?,?,?,?,?,?,?)",
                (attempt.attempt_id, attempt.candidate_id, attempt.status, attempt.created_at,
                 timestamp, canonical_json(attempt.to_dict()), attempt.schema_version),
            )
        return attempt

    def update(self, attempt: ImprovementAttempt) -> ImprovementAttempt:
        """Replace the stored snapshot for an existing attempt (its own
        `attempt_id`) with the latest state. Never touches any other
        attempt's row -- this is not an upsert-by-fingerprint like the
        candidate store, since two attempts for the same candidate are
        always distinct history, not duplicates of one another."""
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "UPDATE improvement_attempts SET status=?, updated_at=?, payload_json=? WHERE attempt_id=?",
                (attempt.status, now(), canonical_json(attempt.to_dict()), attempt.attempt_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"no such attempt: {attempt.attempt_id!r}")
        return attempt

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, attempt_id: str) -> ImprovementAttempt | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT payload_json FROM improvement_attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
        return ImprovementAttempt.from_dict(json.loads(row["payload_json"])) if row else None

    def list_for_candidate(self, candidate_id: str) -> list[ImprovementAttempt]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT payload_json FROM improvement_attempts WHERE candidate_id=? ORDER BY created_at DESC", (candidate_id,)
            ).fetchall()
        return [ImprovementAttempt.from_dict(json.loads(row["payload_json"])) for row in rows]

    def has_active_attempt(self, candidate_id: str) -> bool:
        """True if any non-terminal attempt exists for this candidate --
        the exact signal brain/improvement_triage.py's `has_active_attempt`
        parameter expects, so a candidate is never picked up twice while an
        attempt is still in flight."""
        return any(a.status not in TERMINAL_ATTEMPT_STATUSES for a in self.list_for_candidate(candidate_id))

    def has_ready_for_review_attempt(self, candidate_id: str) -> bool:
        return any(a.status == AttemptStatus.READY_FOR_REVIEW.value for a in self.list_for_candidate(candidate_id))

    def query(self, status: str | None = None, limit: int = 100) -> list[ImprovementAttempt]:
        with self._lock:
            if status is not None:
                rows = self.connection.execute(
                    "SELECT payload_json FROM improvement_attempts WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit)
                ).fetchall()
            else:
                rows = self.connection.execute(
                    "SELECT payload_json FROM improvement_attempts ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [ImprovementAttempt.from_dict(json.loads(row["payload_json"])) for row in rows]

    def count(self) -> int:
        with self._lock:
            return self.connection.execute("SELECT COUNT(*) FROM improvement_attempts").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self.connection.close()


_STORE: ImprovementAttemptStore | None = None
_STORE_LOCK = threading.Lock()


def get_improvement_attempt_store() -> ImprovementAttemptStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                import sys
                import tempfile
                test_path = None
                if "pytest" in sys.modules and not os.getenv("IMPROVEMENT_ATTEMPT_DB_PATH"):
                    test_path = Path(tempfile.mkdtemp(prefix="jarvis-improvement-attempt-pytest-")) / "jarvis_improvement_attempts.sqlite3"
                _STORE = ImprovementAttemptStore(test_path)
    return _STORE


def reset_improvement_attempt_store_for_tests(path: str | Path | None = None) -> ImprovementAttemptStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            _STORE.close()
        _STORE = ImprovementAttemptStore(path)
    return _STORE
