"""Persistence for LearningJob records.

Same shape as `brain/improvement_attempt_store.py` on purpose (thread-safe
SQLite, WAL mode, one small wrapper) -- a third bespoke storage
abstraction is not needed; this is the same well-tested pattern applied to
a new, distinct table. Deduplication is upsert-by-fingerprint, exactly like
`brain/improvement_store.py`'s candidate table: a new approval offer for an
underlying problem that already has a non-terminal learning job must never
create a second row (see `brain/learning_trigger.py`).

Everything here survives a restart by construction (SQLite file on disk) --
Phase 25's persistence requirement is a property of this store, not
something a caller has to remember to implement.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from brain.learning_models import (
    LearningJob, LearningJobStatus, TERMINAL_LEARNING_JOB_STATUSES, TRAINABLE_LEARNING_JOB_STATUSES,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LearningJobStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("LEARNING_JOB_DB_PATH") or Path.cwd() / "data" / "jarvis_learning_jobs.sqlite3")
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
                CREATE TABLE IF NOT EXISTS learning_jobs(
                    learning_job_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    improvement_attempt_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    learning_status TEXT NOT NULL,
                    approval_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_learning_jobs_fingerprint ON learning_jobs(fingerprint);
                CREATE INDEX IF NOT EXISTS idx_learning_jobs_status ON learning_jobs(learning_status);
                CREATE INDEX IF NOT EXISTS idx_learning_jobs_candidate ON learning_jobs(candidate_id);
                """
            )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def create(self, job: LearningJob) -> LearningJob:
        """Insert a brand-new job row. Raises sqlite3.IntegrityError if
        `learning_job_id` already exists -- never silently overwritten."""
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT INTO learning_jobs VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    job.learning_job_id, job.candidate_id, job.improvement_attempt_id, job.fingerprint,
                    job.learning_status, job.approval_status, job.created_at, job.updated_at,
                    json.dumps(job.to_dict()), job.schema_version,
                ),
            )
        return job

    def update(self, job: LearningJob) -> LearningJob:
        """Replace the stored snapshot for an existing job (its own
        `learning_job_id`) with the latest state."""
        job.updated_at = now()
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "UPDATE learning_jobs SET fingerprint=?, learning_status=?, approval_status=?, updated_at=?, payload_json=? "
                "WHERE learning_job_id=?",
                (job.fingerprint, job.learning_status, job.approval_status, job.updated_at,
                 json.dumps(job.to_dict()), job.learning_job_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"no such learning job: {job.learning_job_id!r}")
        return job

    def upsert(self, job: LearningJob) -> LearningJob:
        if self.get(job.learning_job_id) is None:
            return self.create(job)
        return self.update(job)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, learning_job_id: str) -> LearningJob | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT payload_json FROM learning_jobs WHERE learning_job_id=?", (learning_job_id,)
            ).fetchone()
        return LearningJob.from_dict(json.loads(row["payload_json"])) if row else None

    def get_by_candidate(self, candidate_id: str) -> list[LearningJob]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT payload_json FROM learning_jobs WHERE candidate_id=? ORDER BY created_at DESC", (candidate_id,)
            ).fetchall()
        return [LearningJob.from_dict(json.loads(row["payload_json"])) for row in rows]

    def find_active_by_fingerprint(self, fingerprint: str) -> LearningJob | None:
        """The dedup check Phase 5 requires: a non-terminal (still
        pending/approved/in-flight) job for the same underlying fingerprint
        means "don't ask again" -- a DECLINED or TIMED_OUT job does NOT
        suppress a future offer (the user may reasonably say yes next time),
        but every other status does."""
        with self._lock:
            rows = self.connection.execute(
                "SELECT payload_json FROM learning_jobs WHERE fingerprint=? ORDER BY created_at DESC", (fingerprint,)
            ).fetchall()
        for row in rows:
            job = LearningJob.from_dict(json.loads(row["payload_json"]))
            if job.learning_status not in TERMINAL_LEARNING_JOB_STATUSES:
                return job
            if job.learning_status == LearningJobStatus.TRAINED.value:
                return job
        return None

    def query(self, learning_status: str | None = None, limit: int = 200) -> list[LearningJob]:
        with self._lock:
            if learning_status is not None:
                rows = self.connection.execute(
                    "SELECT payload_json FROM learning_jobs WHERE learning_status=? ORDER BY created_at DESC LIMIT ?",
                    (learning_status, limit),
                ).fetchall()
            else:
                rows = self.connection.execute(
                    "SELECT payload_json FROM learning_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [LearningJob.from_dict(json.loads(row["payload_json"])) for row in rows]

    def query_trainable(self, limit: int = 500) -> list[LearningJob]:
        """All approved-but-not-yet-successfully-trained jobs -- exactly the
        batch `brain/learning_orchestrator.py`'s "start learning" gathers."""
        with self._lock:
            rows = self.connection.execute(
                "SELECT payload_json FROM learning_jobs ORDER BY created_at ASC LIMIT ?", (limit,)
            ).fetchall()
        jobs = [LearningJob.from_dict(json.loads(row["payload_json"])) for row in rows]
        return [j for j in jobs if j.learning_status in TRAINABLE_LEARNING_JOB_STATUSES]

    def count(self) -> int:
        with self._lock:
            return self.connection.execute("SELECT COUNT(*) FROM learning_jobs").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self.connection.close()


_STORE: LearningJobStore | None = None
_STORE_LOCK = threading.Lock()


def get_learning_job_store() -> LearningJobStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                import sys
                import tempfile
                test_path = None
                if "pytest" in sys.modules and not os.getenv("LEARNING_JOB_DB_PATH"):
                    test_path = Path(tempfile.mkdtemp(prefix="jarvis-learning-job-pytest-")) / "jarvis_learning_jobs.sqlite3"
                _STORE = LearningJobStore(test_path)
    return _STORE


def reset_learning_job_store_for_tests(path: str | Path | None = None) -> LearningJobStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            _STORE.close()
        _STORE = LearningJobStore(path)
    return _STORE
