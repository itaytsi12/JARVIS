"""Token / cost observability for every model call JARVIS makes.

Two pieces:

- `UsageStore`: a small SQLite table of one row per model call, with the
  tokens the provider actually reported and the cost estimate derived
  from `config/pricing.py`. Same store shape as the rest of this
  codebase's persistence (`brain/experience_store.py` et al).
- `TrackedProvider`: a transparent `ModelProvider` wrapper that records
  each call, so cost tracking is not something a call site can forget.

An unpriced model records `cost_usd = NULL`, never 0.0 -- "unknown" and
"free" must stay distinguishable. A FAILED call also records NULL, but is
not counted as unpriced: no call was billed, so it says nothing about
whether the model's price is known.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import get_config
from providers.base import Message, ModelResponse

log = logging.getLogger("jarvis.usage")

USAGE_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class UsageRecord:
    usage_id: str
    created_at: str
    provider: str
    model: str
    operation: str
    session_id: str | None = None
    task_id: str | None = None
    interaction_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    tokens_reported: bool = True
    cost_usd: float | None = None
    latency_ms: float = 0.0
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UsageSummary:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0
    failures: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_is_complete(self) -> bool:
        """False when at least one call used a model with no known price,
        so the reported total is a floor, not the real figure."""
        return self.unpriced_calls == 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_tokens"] = self.total_tokens
        payload["cost_is_complete"] = self.cost_is_complete
        return payload


class UsageStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else get_config().usage_db_path
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
                CREATE TABLE IF NOT EXISTS model_usage(
                    usage_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    session_id TEXT,
                    task_id TEXT,
                    interaction_id TEXT,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                    tokens_reported INTEGER NOT NULL DEFAULT 1,
                    cost_usd REAL,
                    latency_ms REAL NOT NULL DEFAULT 0,
                    success INTEGER NOT NULL DEFAULT 1,
                    error TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_usage_task ON model_usage(task_id);
                CREATE INDEX IF NOT EXISTS idx_usage_session ON model_usage(session_id);
                CREATE INDEX IF NOT EXISTS idx_usage_created ON model_usage(created_at DESC);
                """
            )

    def record(self, record: UsageRecord) -> UsageRecord:
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO model_usage VALUES(" + ",".join("?" * 18) + ")",
                (
                    record.usage_id,
                    record.created_at,
                    record.provider,
                    record.model,
                    record.operation,
                    record.session_id,
                    record.task_id,
                    record.interaction_id,
                    record.input_tokens,
                    record.output_tokens,
                    record.cache_creation_tokens,
                    record.cache_read_tokens,
                    int(record.tokens_reported),
                    record.cost_usd,
                    record.latency_ms,
                    int(record.success),
                    record.error,
                    json.dumps(record.metadata),
                ),
            )
        return record

    def _summarize(self, where: str = "", args: tuple = ()) -> UsageSummary:
        sql = (
            "SELECT COUNT(*) AS calls,"
            " COALESCE(SUM(input_tokens),0) AS input_tokens,"
            " COALESCE(SUM(output_tokens),0) AS output_tokens,"
            " COALESCE(SUM(cache_creation_tokens),0) AS cache_creation_tokens,"
            " COALESCE(SUM(cache_read_tokens),0) AS cache_read_tokens,"
            " COALESCE(SUM(cost_usd),0) AS cost_usd,"
            " SUM(CASE WHEN cost_usd IS NULL AND success=1 THEN 1 ELSE 0 END) AS unpriced_calls,"
            " SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) AS failures"
            " FROM model_usage"
        ) + (f" WHERE {where}" if where else "")
        with self._lock:
            row = self.connection.execute(sql, args).fetchone()
        return UsageSummary(
            calls=int(row["calls"] or 0),
            input_tokens=int(row["input_tokens"] or 0),
            output_tokens=int(row["output_tokens"] or 0),
            cache_creation_tokens=int(row["cache_creation_tokens"] or 0),
            cache_read_tokens=int(row["cache_read_tokens"] or 0),
            cost_usd=round(float(row["cost_usd"] or 0.0), 8),
            unpriced_calls=int(row["unpriced_calls"] or 0),
            failures=int(row["failures"] or 0),
        )

    def total(self) -> UsageSummary:
        return self._summarize()

    def for_task(self, task_id: str) -> UsageSummary:
        return self._summarize("task_id=?", (task_id,))

    def for_session(self, session_id: str) -> UsageSummary:
        return self._summarize("session_id=?", (session_id,))

    def recent(self, limit: int = 50) -> list[UsageRecord]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM model_usage ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_decode(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self.connection.close()


def _decode(row: sqlite3.Row) -> UsageRecord:
    return UsageRecord(
        usage_id=row["usage_id"],
        created_at=row["created_at"],
        provider=row["provider"],
        model=row["model"],
        operation=row["operation"],
        session_id=row["session_id"],
        task_id=row["task_id"],
        interaction_id=row["interaction_id"],
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        cache_creation_tokens=int(row["cache_creation_tokens"]),
        cache_read_tokens=int(row["cache_read_tokens"]),
        tokens_reported=bool(row["tokens_reported"]),
        cost_usd=row["cost_usd"],
        latency_ms=float(row["latency_ms"]),
        success=bool(row["success"]),
        error=row["error"],
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


class TrackedProvider:
    """Wraps any provider so every call is measured and persisted.

    Delegates `name` / `model` / `is_available` / `describe` so callers
    cannot tell the difference from the underlying provider.
    """

    def __init__(
        self,
        provider: Any,
        store: UsageStore | None = None,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
        interaction_id: str | None = None,
        operation: str = "agent_loop",
    ):
        self._provider = provider
        self._store = store
        self.session_id = session_id
        self.task_id = task_id
        self.interaction_id = interaction_id
        self.operation = operation
        self.records: list[UsageRecord] = []

    @property
    def name(self) -> str:
        return getattr(self._provider, "name", "unknown")

    @property
    def model(self) -> str:
        return getattr(self._provider, "model", "")

    @property
    def inner(self) -> Any:
        return self._provider

    def is_available(self) -> bool:
        return bool(self._provider.is_available())

    def unavailable_reason(self) -> str | None:
        reason = getattr(self._provider, "unavailable_reason", None)
        return reason() if callable(reason) else None

    def describe(self) -> dict[str, Any]:
        return dict(self._provider.describe())

    def summary(self) -> UsageSummary:
        """Usage for THIS wrapper's own calls, independent of the store --
        available even when persistence is disabled."""
        summary = UsageSummary()
        for record in self.records:
            summary.calls += 1
            summary.input_tokens += record.input_tokens
            summary.output_tokens += record.output_tokens
            summary.cache_creation_tokens += record.cache_creation_tokens
            summary.cache_read_tokens += record.cache_read_tokens
            if record.cost_usd is None:
                # A FAILED call has no cost because it never happened, not
                # because the model has no known price. Counting it as
                # unpriced made `cost_is_complete` false -- and a fully
                # priced model look unpriced -- on any run that errored.
                if record.success:
                    summary.unpriced_calls += 1
            else:
                summary.cost_usd = round(summary.cost_usd + record.cost_usd, 8)
            if not record.success:
                summary.failures += 1
        return summary

    def complete(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        import time

        started = time.perf_counter()
        try:
            response = self._provider.complete(messages, **kwargs)
        except Exception as exc:
            self._write(
                UsageRecord(
                    usage_id=uuid.uuid4().hex,
                    created_at=_now(),
                    provider=self.name,
                    model=str(kwargs.get("model") or self.model),
                    operation=self.operation,
                    session_id=self.session_id,
                    task_id=self.task_id,
                    interaction_id=self.interaction_id,
                    latency_ms=round((time.perf_counter() - started) * 1000, 3),
                    success=False,
                    error=f"{type(exc).__name__}: {exc}"[:500],
                    tokens_reported=False,
                )
            )
            raise
        self._write(
            UsageRecord(
                usage_id=uuid.uuid4().hex,
                created_at=_now(),
                provider=response.provider or self.name,
                model=response.model or self.model,
                operation=self.operation,
                session_id=self.session_id,
                task_id=self.task_id,
                interaction_id=self.interaction_id,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_creation_tokens=response.usage.cache_creation_tokens,
                cache_read_tokens=response.usage.cache_read_tokens,
                tokens_reported=response.usage.reported,
                cost_usd=response.estimated_cost_usd,
                latency_ms=response.latency_ms,
                success=True,
                metadata={"stop_reason": response.stop_reason, "tool_calls": len(response.tool_calls)},
            )
        )
        return response

    def _write(self, record: UsageRecord) -> None:
        self.records.append(record)
        store = self._store
        if store is None and get_config().cost_tracking_enabled:
            store = get_usage_store()
        if store is None:
            return
        try:
            store.record(record)
        except Exception:
            # Cost bookkeeping must never break a working model call.
            log.exception("Could not persist model usage record")


_STORE: UsageStore | None = None
_STORE_LOCK = threading.Lock()


def get_usage_store() -> UsageStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                path = None
                if "pytest" in sys.modules and not os.getenv("JARVIS_USAGE_DB_PATH"):
                    path = Path(tempfile.mkdtemp(prefix="jarvis-usage-pytest-")) / "jarvis_usage.sqlite3"
                _STORE = UsageStore(path)
    return _STORE


def reset_usage_store_for_tests(path: str | Path | None = None) -> UsageStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            _STORE.close()
        _STORE = UsageStore(path or Path(tempfile.mkdtemp(prefix="jarvis-usage-pytest-")) / "jarvis_usage.sqlite3")
    return _STORE
