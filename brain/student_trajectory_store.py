"""Student/teacher trajectory records (Part A, Phases A6/A9/A10).

A durable, queryable log of every coding-task attempt's outcome --
independent of the `LearningJob`/voice-approval flow, since a successful
STUDENT attempt is never gated behind voice approval (Phase A6: "do NOT
ask... for an ordinary successful student task"), but must still be
observable and usable as future training signal (Phase A9). Reuses the same
SQLite-store shape as every other store in this codebase
(`brain/learning_store.py`, `brain/experience_store.py`) -- a new,
small, genuinely-needed table, not a competing system.

Quality labels are exactly the `REAL_VERIFIED_STUDENT`/`REAL_FAILED_STUDENT`/
`REAL_FAILED_TEACHER` values already defined in
`brain.learning_models.DataQualityLabel` (added in a prior session but never
wired to a store until now).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRAJECTORY_SCHEMA_VERSION = 1


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StudentTrajectory:
    trajectory_id: str
    created_at: str
    candidate_id: str
    task: str
    quality_label: str  # REAL_VERIFIED_STUDENT | REAL_FAILED_STUDENT | REAL_FAILED_TEACHER
    student_model_version: str | None = None
    student_attempt_id: str | None = None
    teacher_attempt_id: str | None = None
    solved_by: str = "none"  # "student" | "teacher" | "none"
    high_value: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)
    schema_version: int = TRAJECTORY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StudentTrajectory":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


class StudentTrajectoryStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("STUDENT_TRAJECTORY_DB_PATH") or Path.cwd() / "data" / "jarvis_student_trajectories.sqlite3")
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
                CREATE TABLE IF NOT EXISTS student_trajectories(
                    trajectory_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    quality_label TEXT NOT NULL,
                    student_model_version TEXT,
                    high_value INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trajectories_label ON student_trajectories(quality_label);
                CREATE INDEX IF NOT EXISTS idx_trajectories_high_value ON student_trajectories(high_value);
                """
            )

    def record(self, trajectory: StudentTrajectory) -> StudentTrajectory:
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO student_trajectories VALUES(?,?,?,?,?,?,?)",
                (
                    trajectory.trajectory_id, trajectory.candidate_id, trajectory.quality_label,
                    trajectory.student_model_version, int(trajectory.high_value), trajectory.created_at,
                    json.dumps(trajectory.to_dict()),
                ),
            )
        return trajectory

    def get(self, trajectory_id: str) -> StudentTrajectory | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT payload_json FROM student_trajectories WHERE trajectory_id=?", (trajectory_id,)
            ).fetchone()
        return StudentTrajectory.from_dict(json.loads(row["payload_json"])) if row else None

    def query(self, quality_label: str | None = None, *, high_value_only: bool = False, limit: int = 200) -> list[StudentTrajectory]:
        clauses, params = [], []
        if quality_label is not None:
            clauses.append("quality_label=?")
            params.append(quality_label)
        if high_value_only:
            clauses.append("high_value=1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self.connection.execute(
                f"SELECT payload_json FROM student_trajectories {where} ORDER BY created_at DESC LIMIT ?", (*params, limit)
            ).fetchall()
        return [StudentTrajectory.from_dict(json.loads(row["payload_json"])) for row in rows]

    def count(self) -> int:
        with self._lock:
            return self.connection.execute("SELECT COUNT(*) FROM student_trajectories").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self.connection.close()


_STORE: StudentTrajectoryStore | None = None
_STORE_LOCK = threading.Lock()


def get_student_trajectory_store() -> StudentTrajectoryStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                import sys
                import tempfile
                test_path = None
                if "pytest" in sys.modules and not os.getenv("STUDENT_TRAJECTORY_DB_PATH"):
                    test_path = Path(tempfile.mkdtemp(prefix="jarvis-student-trajectory-pytest-")) / "jarvis_student_trajectories.sqlite3"
                _STORE = StudentTrajectoryStore(test_path)
    return _STORE


def reset_student_trajectory_store_for_tests(path: str | Path | None = None) -> StudentTrajectoryStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            _STORE.close()
        _STORE = StudentTrajectoryStore(path)
    return _STORE
