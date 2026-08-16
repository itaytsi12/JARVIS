"""Persistence for ImprovementCandidate records.

Modeled directly on the existing `memory.database.MemoryDatabase` pattern
(thread-safe SQLite, WAL mode, simple execute/query wrapper) rather than
reusing that class as-is: the schema here is a distinct concern (a
deduplicated, status-tracked candidate queue) and mixing it into the
general-purpose memory database would create unrelated tables in a file
that already has its own well-tested schema. This is the "small storage
abstraction" the design explicitly allows when nothing existing fits.

Deduplication is upsert-by-fingerprint: a repeat observation of the same
underlying root cause increments `occurrence_count` and refreshes
`last_seen`/the representative payload on the *existing* row rather than
ever creating a second row for the same fingerprint.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from brain.improvement_models import CandidateStatus, ImprovementCandidate, SCHEMA_VERSION, canonical_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ImprovementStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("IMPROVEMENT_DB_PATH") or Path.cwd() / "data" / "jarvis_improvements.sqlite3")
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
                CREATE TABLE IF NOT EXISTS improvement_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS improvement_candidates(
                    candidate_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    occurrence_count INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL,
                    gap_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    subsystem TEXT,
                    tool_involved TEXT,
                    payload_json TEXT NOT NULL,
                    schema_version INTEGER NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_candidates_fingerprint ON improvement_candidates(fingerprint);
                CREATE INDEX IF NOT EXISTS idx_candidates_status ON improvement_candidates(status);
                CREATE INDEX IF NOT EXISTS idx_candidates_gap_type ON improvement_candidates(gap_type);
                CREATE INDEX IF NOT EXISTS idx_candidates_subsystem ON improvement_candidates(subsystem);
                """
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO improvement_meta VALUES(?,?)", ("schema_version", str(SCHEMA_VERSION))
            )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def record(self, candidate: ImprovementCandidate) -> tuple[ImprovementCandidate, bool]:
        """Insert a new candidate, or -- if its fingerprint already exists --
        increment the existing row's occurrence_count/last_seen instead of
        creating a duplicate. Returns (stored_candidate, is_new)."""
        with self._lock, self.connection:
            existing = self.connection.execute(
                "SELECT * FROM improvement_candidates WHERE fingerprint=?", (candidate.fingerprint,)
            ).fetchone()
            if existing is None:
                self.connection.execute(
                    "INSERT INTO improvement_candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        candidate.candidate_id, candidate.fingerprint, candidate.created_at,
                        candidate.first_seen, candidate.last_seen, candidate.occurrence_count,
                        candidate.status, candidate.gap_type, candidate.confidence,
                        candidate.subsystem, candidate.tool_involved,
                        canonical_json(candidate.to_dict()), candidate.schema_version,
                    ),
                )
                return candidate, True
            stored = ImprovementCandidate.from_dict(json.loads(existing["payload_json"]))
            stored.occurrence_count = int(existing["occurrence_count"]) + 1
            stored.last_seen = candidate.last_seen
            # The representative payload is refreshed to the newest
            # occurrence's evidence (most actionable for a future agent),
            # but identity/first-seen/status/occurrence bookkeeping is
            # preserved -- a repeat observation must never silently
            # overwrite a status a human or future agent already set.
            stored.candidate_id = existing["candidate_id"]
            stored.fingerprint = existing["fingerprint"]
            stored.first_seen = existing["first_seen"]
            stored.created_at = existing["created_at"]
            stored.status = existing["status"]
            self.connection.execute(
                "UPDATE improvement_candidates SET last_seen=?, occurrence_count=?, confidence=?, "
                "gap_type=?, subsystem=?, tool_involved=?, payload_json=? WHERE fingerprint=?",
                (
                    stored.last_seen, stored.occurrence_count, candidate.confidence,
                    candidate.gap_type, candidate.subsystem, candidate.tool_involved,
                    canonical_json(stored.to_dict()), candidate.fingerprint,
                ),
            )
            return stored, False

    def set_status(self, candidate_id: str, status: str) -> bool:
        if status not in {s.value for s in CandidateStatus}:
            raise ValueError(f"unknown status: {status!r}")
        with self._lock, self.connection:
            row = self.connection.execute(
                "SELECT payload_json FROM improvement_candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
            if row is None:
                return False
            payload = json.loads(row["payload_json"])
            payload["status"] = status
            self.connection.execute(
                "UPDATE improvement_candidates SET status=?, payload_json=? WHERE candidate_id=?",
                (status, canonical_json(payload), candidate_id),
            )
            return True

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, candidate_id: str) -> ImprovementCandidate | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT payload_json FROM improvement_candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
        return ImprovementCandidate.from_dict(json.loads(row["payload_json"])) if row else None

    def get_by_fingerprint(self, fingerprint: str) -> ImprovementCandidate | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT payload_json FROM improvement_candidates WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
        return ImprovementCandidate.from_dict(json.loads(row["payload_json"])) if row else None

    def query(self, status: str | None = None, gap_type: str | None = None,
              subsystem: str | None = None, limit: int = 100) -> list[ImprovementCandidate]:
        clauses, params = [], []
        if status is not None:
            clauses.append("status=?"); params.append(status)
        if gap_type is not None:
            clauses.append("gap_type=?"); params.append(gap_type)
        if subsystem is not None:
            clauses.append("subsystem=?"); params.append(subsystem)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self.connection.execute(
                f"SELECT payload_json FROM improvement_candidates {where} ORDER BY last_seen DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [ImprovementCandidate.from_dict(json.loads(row["payload_json"])) for row in rows]

    def count(self) -> int:
        with self._lock:
            return self.connection.execute("SELECT COUNT(*) FROM improvement_candidates").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self.connection.close()


_STORE: ImprovementStore | None = None
_STORE_LOCK = threading.Lock()


def get_improvement_store() -> ImprovementStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                import sys
                import tempfile
                test_path = None
                if "pytest" in sys.modules and not os.getenv("IMPROVEMENT_DB_PATH"):
                    test_path = Path(tempfile.mkdtemp(prefix="jarvis-improvement-pytest-")) / "jarvis_improvements.sqlite3"
                _STORE = ImprovementStore(test_path)
    return _STORE


def reset_improvement_store_for_tests(path: str | Path | None = None) -> ImprovementStore:
    """Test-only helper: force a fresh store instance (e.g. pointed at a
    temp file) instead of the process-wide singleton."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            _STORE.close()
        _STORE = ImprovementStore(path)
    return _STORE
