"""Persistence for tasks, so a restart does not lose what JARVIS was doing.

The `TaskManager` keeps live tasks in memory (they own threads and
cancellation tokens, which cannot be persisted); this store keeps their
observable state on disk. A task that was RUNNING when the process died
is reported honestly as `interrupted` on the next start rather than
resurrected as if still running.
"""
from __future__ import annotations

import json
from typing import Any

from memory.agent_store import AgentDatabase, get_agent_database, utc_now
from memory.memory_manager import redact
from tasks.models import Task, TaskStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks_v2(
    task_id TEXT PRIMARY KEY,
    parent_task_id TEXT,
    goal TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT NOT NULL,
    session_id TEXT,
    result TEXT,
    error TEXT,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_v2_status ON tasks_v2(status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_v2_parent ON tasks_v2(parent_task_id);
"""


class TaskStore:
    def __init__(self, database: AgentDatabase | None = None):
        self.db = database or get_agent_database()
        self.db.executescript(SCHEMA)

    def save(self, task: Task) -> Task:
        payload = redact(task.to_dict())
        self.db.execute(
            "INSERT OR REPLACE INTO tasks_v2 VALUES(" + ",".join("?" * 13) + ")",
            (
                task.task_id,
                task.parent_task_id,
                payload["goal"],
                task.kind.value,
                task.status.value,
                task.created_at,
                task.started_at,
                task.finished_at,
                utc_now(),
                task.session_id,
                str(payload.get("result") or "")[:4000] or None,
                task.error,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        return task

    def get(self, task_id: str) -> dict[str, Any] | None:
        rows = self.db.query("SELECT payload_json FROM tasks_v2 WHERE task_id=?", (task_id,))
        return json.loads(rows[0]["payload_json"]) if rows else None

    def list(self, status: TaskStatus | str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if status is not None:
            value = status.value if isinstance(status, TaskStatus) else str(status)
            rows = self.db.query(
                "SELECT payload_json FROM tasks_v2 WHERE status=? ORDER BY updated_at DESC LIMIT ?", (value, limit)
            )
        else:
            rows = self.db.query("SELECT payload_json FROM tasks_v2 ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [json.loads(row["payload_json"]) for row in rows]

    def mark_interrupted_tasks(self) -> list[str]:
        """Reconcile after a restart.

        Any task still marked RUNNING or PENDING on disk cannot actually
        be running -- its thread died with the process -- so it is marked
        FAILED with an explicit `process_interrupted` error instead of
        being left as a phantom running task.
        """
        rows = self.db.query(
            "SELECT task_id, payload_json FROM tasks_v2 WHERE status IN (?,?)",
            (TaskStatus.RUNNING.value, TaskStatus.PENDING.value),
        )
        interrupted = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["status"] = TaskStatus.FAILED.value
            payload["error"] = "process_interrupted"
            payload["finished_at"] = utc_now()
            self.db.execute(
                "UPDATE tasks_v2 SET status=?, error=?, finished_at=?, updated_at=?, payload_json=? WHERE task_id=?",
                (
                    TaskStatus.FAILED.value,
                    "process_interrupted",
                    payload["finished_at"],
                    utc_now(),
                    json.dumps(payload, ensure_ascii=False),
                    row["task_id"],
                ),
            )
            interrupted.append(row["task_id"])
        return interrupted

    def count(self, status: TaskStatus | str | None = None) -> int:
        if status is not None:
            value = status.value if isinstance(status, TaskStatus) else str(status)
            rows = self.db.query("SELECT COUNT(*) AS n FROM tasks_v2 WHERE status=?", (value,))
        else:
            rows = self.db.query("SELECT COUNT(*) AS n FROM tasks_v2")
        return int(rows[0]["n"]) if rows else 0
