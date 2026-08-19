"""Task state.

A task is the unit of work JARVIS tracks: it has a goal, a lifecycle, a
plan, observations, a result and a cancellation state. Several tasks can
exist at once, which is why every mutable field lives on the record
rather than in a module-level global.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED})


class TaskKind(str, Enum):
    """How a task must be scheduled.

    `EXCLUSIVE_UI` tasks drive the real keyboard, mouse or foreground
    window; only one may run at a time, because two of them interleaved
    would type into each other's windows. `CONCURRENT` tasks (research,
    reading files, running a test suite) are safe to run in parallel with
    anything else.
    """

    CONCURRENT = "concurrent"
    EXCLUSIVE_UI = "exclusive_ui"


class CancellationToken:
    """Cooperative cancellation, shaped like `brain.task_supervisor`'s.

    Deliberately the same interface as the existing token so a task's
    token can be handed to `AgentRuntime.execute`, `Executor` and every
    tool that already understands cancellation.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason: str | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise TaskCancelled(self.reason or "cancelled")


class TaskCancelled(RuntimeError):
    """Raised inside a task body when its token has been cancelled."""


@dataclass
class TaskObservation:
    """One thing the task learned, in order."""

    index: int
    source: str
    text: str
    success: bool = True
    created_at: str = field(default_factory=utc_now)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Task:
    goal: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_task_id: str | None = None
    kind: TaskKind = TaskKind.CONCURRENT
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    plan: list[str] = field(default_factory=list)
    current_step: int = 0
    observations: list[TaskObservation] = field(default_factory=list)
    result: str | None = None
    error: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    token: CancellationToken = field(default_factory=CancellationToken, repr=False)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def cancelled(self) -> bool:
        return self.token.cancelled

    @property
    def duration_ms(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or utc_now()
        try:
            return max(
                0.0,
                (datetime.fromisoformat(end) - datetime.fromisoformat(self.started_at)).total_seconds() * 1000,
            )
        except ValueError:
            return 0.0

    def observe(self, source: str, text: str, *, success: bool = True, **data: Any) -> TaskObservation:
        observation = TaskObservation(
            index=len(self.observations), source=source, text=text, success=success, data=data
        )
        self.observations.append(observation)
        return observation

    def to_dict(self, include_observations: bool = True) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "goal": self.goal,
            "kind": self.kind.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "plan": list(self.plan),
            "current_step": self.current_step,
            "result": self.result,
            "error": self.error,
            "session_id": self.session_id,
            "cancelled": self.cancelled,
            "duration_ms": round(self.duration_ms, 3),
            "observation_count": len(self.observations),
            "metadata": dict(self.metadata),
        }
        if include_observations:
            payload["observations"] = [observation.to_dict() for observation in self.observations]
        return payload

    def summary(self) -> str:
        """A one-line, speech-friendly description of this task's state."""
        if self.status is TaskStatus.RUNNING:
            step = f" (step {self.current_step + 1} of {len(self.plan)})" if self.plan else ""
            return f"running{step}"
        if self.status is TaskStatus.COMPLETED:
            return "completed"
        if self.status is TaskStatus.FAILED:
            return f"failed: {self.error}" if self.error else "failed"
        if self.status is TaskStatus.CANCELLED:
            return "cancelled"
        return self.status.value
