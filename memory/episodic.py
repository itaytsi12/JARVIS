"""Episodic memory: the full, structured record of what actually happened.

One `Episode` per handled request. It is both:

- the memory an agent retrieves when the user says "continue fixing the
  voice bug" -- what was tried, what the error was, what worked;
- and the raw training record described in the project brief: request,
  context, route, model, plan, actions, tool calls, observations, errors,
  retries, result, success, duration, token usage, cost.

Nothing is summarized away on write. Episodes are stored twice on
purpose: a normalized SQLite row (queryable, indexed) and an append-only
JSONL file (the untouched raw payload, easy to ship to a training
pipeline later). Processing happens downstream; capture stays lossless.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config import get_config
from memory.agent_store import AgentDatabase, get_agent_database, utc_now
from memory.memory_manager import redact

log = logging.getLogger("jarvis.episodes")


@dataclass
class StepRecord:
    """One action -> observation pair inside an episode."""

    index: int
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    verified: bool = False
    observation: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    attempt: int = 1
    thought: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Episode:
    episode_id: str
    user_request: str
    created_at: str = ""
    session_id: str | None = None
    task_id: str | None = None
    route: str | None = None
    model_used: str | None = None
    provider: str | None = None
    context_summary: str = ""
    plan: list[str] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    retries: int = 0
    final_result: str = ""
    success: bool = False
    verified: bool = False
    stop_reason: str | None = None
    user_correction: str | None = None
    duration_ms: float = 0.0
    token_usage: dict[str, Any] = field(default_factory=dict)
    estimated_cost_usd: float | None = None
    memories_written: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- derived reward-relevant signals (see brief section 16) -------
    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def error_count(self) -> int:
        return len(self.errors) + sum(1 for step in self.steps if step.error)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        payload["step_count"] = self.step_count
        payload["error_count"] = self.error_count
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Episode":
        data = dict(payload)
        data.pop("step_count", None)
        data.pop("error_count", None)
        steps = [StepRecord(**step) for step in data.pop("steps", [])]
        known = set(cls.__dataclass_fields__)
        return cls(steps=steps, **{k: v for k, v in data.items() if k in known})

    def searchable_text(self) -> str:
        parts = [self.user_request, self.final_result, self.context_summary]
        parts.extend(step.tool for step in self.steps)
        parts.extend(step.observation[:200] for step in self.steps)
        parts.extend(self.errors)
        return " ".join(part for part in parts if part)


class EpisodeStore:
    def __init__(self, database: AgentDatabase | None = None, jsonl_path: Path | None = None):
        self.db = database or get_agent_database()
        self.jsonl_path = Path(jsonl_path) if jsonl_path else get_config().data_dir / "episodes" / "episodes.jsonl"

    def record(self, episode: Episode) -> Episode:
        if not episode.episode_id:
            episode.episode_id = uuid.uuid4().hex
        if not episode.created_at:
            episode.created_at = utc_now()
        payload = redact(episode.to_dict())
        self.db.execute(
            "INSERT OR REPLACE INTO episodes VALUES(" + ",".join("?" * 16) + ")",
            (
                episode.episode_id,
                episode.session_id,
                episode.task_id,
                episode.created_at,
                payload["user_request"],
                episode.route,
                episode.model_used,
                int(episode.success),
                int(episode.verified),
                float(episode.duration_ms),
                episode.step_count,
                int(episode.retries),
                episode.error_count,
                episode.estimated_cost_usd,
                str(payload.get("final_result") or "")[:4000],
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        self._append_jsonl(payload)
        return episode

    def _append_jsonl(self, payload: dict[str, Any]) -> None:
        """Raw, append-only capture.

        A failure here must never lose the episode -- the SQLite row is
        already committed by this point -- so it is logged, not raised.
        """
        try:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            log.exception("Could not append episode to %s", self.jsonl_path)

    def get(self, episode_id: str) -> Episode | None:
        rows = self.db.query("SELECT payload_json FROM episodes WHERE episode_id=?", (episode_id,))
        return Episode.from_dict(json.loads(rows[0]["payload_json"])) if rows else None

    def recent(self, limit: int = 20, *, task_id: str | None = None, successful_only: bool = False) -> list[Episode]:
        clauses, args = [], []
        if task_id:
            clauses.append("task_id=?")
            args.append(task_id)
        if successful_only:
            clauses.append("success=1")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.query(
            f"SELECT payload_json FROM episodes{where} ORDER BY created_at DESC LIMIT ?", (*args, limit)
        )
        return [Episode.from_dict(json.loads(row["payload_json"])) for row in rows]

    def count(self) -> int:
        rows = self.db.query("SELECT COUNT(*) AS n FROM episodes")
        return int(rows[0]["n"]) if rows else 0

    def statistics(self) -> dict[str, Any]:
        rows = self.db.query(
            "SELECT COUNT(*) AS total,"
            " SUM(success) AS successes,"
            " SUM(verified) AS verified,"
            " COALESCE(SUM(retry_count),0) AS retries,"
            " COALESCE(SUM(error_count),0) AS errors,"
            " COALESCE(AVG(duration_ms),0) AS average_duration_ms,"
            " COALESCE(SUM(cost_usd),0) AS cost_usd"
            " FROM episodes"
        )
        row = rows[0] if rows else None
        total = int(row["total"]) if row else 0
        return {
            "episodes": total,
            "successes": int(row["successes"] or 0) if row else 0,
            "verified": int(row["verified"] or 0) if row else 0,
            "success_rate": round((int(row["successes"] or 0) / total), 4) if total else 0.0,
            "total_retries": int(row["retries"] or 0) if row else 0,
            "total_errors": int(row["errors"] or 0) if row else 0,
            "average_duration_ms": round(float(row["average_duration_ms"] or 0.0), 3) if row else 0.0,
            "total_cost_usd": round(float(row["cost_usd"] or 0.0), 8) if row else 0.0,
            "jsonl_path": str(self.jsonl_path),
        }
