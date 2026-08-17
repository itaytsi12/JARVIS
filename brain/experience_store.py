"""Immediate experience memory (Phase 7): as soon as a learning job is
approved, its `LearningPackage` becomes an `ExperienceRecord` that a future
coding task can retrieve RIGHT AWAY, before any model retraining happens.

Same SQLite-store shape as the rest of this codebase's persistence
(`brain/improvement_store.py`, `brain/learning_store.py`). Retrieval is a
small, local, dependency-free token-overlap scorer -- not a neural
embedding index. That's a deliberate scope choice: Phase 7 only requires
"local similarity retrieval" and "a small relevant set," not a particular
retrieval algorithm, and this repository has no existing vector-index
infrastructure to reuse (the sentence-transformers dependency that exists
is training/-scoped, for the intent classifier, not for this). A future
embedding-based retriever can replace `_score` without changing this
module's public interface.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain.learning_package import LearningPackage

EXPERIENCE_SCHEMA_VERSION = 1

_TOKEN = re.compile(r"[a-z0-9_]+")


def _tokenize(*texts: str | None) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        if text:
            tokens.update(_TOKEN.findall(text.lower()))
    return tokens


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperienceRecord:
    experience_id: str
    learning_job_id: str
    problem_family: str
    subsystem: str | None
    gap_type: str
    original_task: str
    reusable_strategy: str
    applicability_conditions: list[str] = field(default_factory=list)
    do_not_generalize: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    created_at: str = ""
    schema_version: int = EXPERIENCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperienceRecord":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})

    @classmethod
    def from_package(cls, package: LearningPackage, *, experience_id: str) -> "ExperienceRecord":
        return cls(
            experience_id=experience_id,
            learning_job_id=package.learning_job_id,
            problem_family=package.problem_family,
            subsystem=package.subsystem,
            gap_type=package.gap_type,
            original_task=package.original_task,
            reusable_strategy=package.reusable_strategy,
            applicability_conditions=list(package.applicability_conditions),
            do_not_generalize=list(package.do_not_generalize),
            files_changed=list(package.files_changed),
            created_at=now(),
        )


@dataclass
class ScoredExperience:
    experience: ExperienceRecord
    score: float


class ExperienceStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("EXPERIENCE_DB_PATH") or Path.cwd() / "data" / "jarvis_experiences.sqlite3")
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
                CREATE TABLE IF NOT EXISTS experiences(
                    experience_id TEXT PRIMARY KEY,
                    problem_family TEXT NOT NULL,
                    subsystem TEXT,
                    gap_type TEXT,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_experiences_family ON experiences(problem_family);
                CREATE INDEX IF NOT EXISTS idx_experiences_subsystem ON experiences(subsystem);
                """
            )

    def store(self, record: ExperienceRecord) -> ExperienceRecord:
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO experiences VALUES(?,?,?,?,?,?)",
                (record.experience_id, record.problem_family, record.subsystem, record.gap_type,
                 record.created_at, json.dumps(record.to_dict())),
            )
        return record

    def get(self, experience_id: str) -> ExperienceRecord | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT payload_json FROM experiences WHERE experience_id=?", (experience_id,)
            ).fetchone()
        return ExperienceRecord.from_dict(json.loads(row["payload_json"])) if row else None

    def all(self, limit: int = 5000) -> list[ExperienceRecord]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT payload_json FROM experiences ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [ExperienceRecord.from_dict(json.loads(row["payload_json"])) for row in rows]

    def count(self) -> int:
        with self._lock:
            return self.connection.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self.connection.close()


def _score(query_tokens: set[str], record: ExperienceRecord, *, subsystem: str | None, gap_type: str | None) -> float:
    record_tokens = _tokenize(record.original_task, record.reusable_strategy, " ".join(record.applicability_conditions))
    if not query_tokens or not record_tokens:
        overlap = 0.0
    else:
        overlap = len(query_tokens & record_tokens) / len(query_tokens | record_tokens)
    score = overlap
    if subsystem and record.subsystem == subsystem:
        score += 0.5
    if gap_type and record.gap_type == gap_type:
        score += 0.3
    return score


def retrieve_relevant_experiences(
    query_text: str,
    *,
    subsystem: str | None = None,
    gap_type: str | None = None,
    top_k: int = 3,
    min_score: float = 0.05,
    store: ExperienceStore | None = None,
) -> list[ScoredExperience]:
    """Local similarity retrieval only (Phase 7): scans this process's own
    experience table and returns at most `top_k` records above `min_score`.
    Never returns the whole database -- a future student/local agent is
    meant to receive only this small, ranked slice, not every stored
    experience."""
    store = store or get_experience_store()
    query_tokens = _tokenize(query_text)
    scored = [
        ScoredExperience(record, _score(query_tokens, record, subsystem=subsystem, gap_type=gap_type))
        for record in store.all()
    ]
    scored = [s for s in scored if s.score >= min_score]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:max(0, top_k)]


_STORE: ExperienceStore | None = None
_STORE_LOCK = threading.Lock()


def get_experience_store() -> ExperienceStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                import sys
                import tempfile
                test_path = None
                if "pytest" in sys.modules and not os.getenv("EXPERIENCE_DB_PATH"):
                    test_path = Path(tempfile.mkdtemp(prefix="jarvis-experience-pytest-")) / "jarvis_experiences.sqlite3"
                _STORE = ExperienceStore(test_path)
    return _STORE


def reset_experience_store_for_tests(path: str | Path | None = None) -> ExperienceStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            _STORE.close()
        _STORE = ExperienceStore(path)
    return _STORE
