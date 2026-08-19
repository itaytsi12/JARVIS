"""The task manager: several things at once, safely.

Concurrency model
-----------------
Threads, not asyncio. Every capability JARVIS actually drives -- pywinauto,
Playwright's sync API, `subprocess`, Windows message pumps, the audio
stream -- is a blocking, synchronous API, and the existing runtime
(`brain/executor.py`, `brain/agent_runtime.py`, `brain/task_supervisor.py`)
is built around a process-wide RLock that is thread-affine. A thread pool
composes with all of that; an event loop would have to push every one of
those calls into a thread pool anyway, and would have made the existing
locking incorrect. `run_async` is provided so an asyncio caller can await
a task without either side blocking the other.

Safety model
------------
`TaskKind.EXCLUSIVE_UI` tasks are serialized behind a single semaphore:
only one task may drive the real keyboard, mouse or foreground window at
a time, because two interleaved UI tasks would type into each other's
windows. `TaskKind.CONCURRENT` tasks (research, file reading, a test run)
run in parallel up to `JARVIS_MAX_CONCURRENT_TASKS`. This is enforced
here, at scheduling time, on top of -- not instead of -- the per-tool
resource locks in `brain/resource_locks.py`.

Interruption
------------
Every task carries a `CancellationToken`. Cancelling sets the token and
returns immediately; the task body notices cooperatively. Submitting a
new task never blocks on an old one, so voice input is never held hostage
by a long-running agent task.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Iterable

from config import get_config
from tasks.models import (
    CancellationToken,
    Task,
    TaskCancelled,
    TaskKind,
    TaskStatus,
    utc_now,
)
from tasks.store import TaskStore

log = logging.getLogger("jarvis.tasks")

TaskBody = Callable[[Task], Any]


class TaskHandle:
    """A live reference to a submitted task."""

    def __init__(self, task: Task, future: Future, manager: "TaskManager"):
        self.task = task
        self.future = future
        self._manager = manager

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def status(self) -> TaskStatus:
        return self.task.status

    @property
    def done(self) -> bool:
        return self.future.done()

    def cancel(self, reason: str = "user_cancelled") -> bool:
        return self._manager.cancel(self.task.task_id, reason)

    def result(self, timeout: float | None = None) -> Any:
        return self.future.result(timeout=timeout)

    def wait(self, timeout: float | None = None) -> bool:
        try:
            self.future.result(timeout=timeout)
        except TaskCancelled:
            return True
        except Exception:
            return True
        except TimeoutError:
            return False
        return True


class TaskManager:
    def __init__(
        self,
        max_workers: int | None = None,
        store: TaskStore | None = None,
        *,
        persist: bool = True,
    ):
        config = get_config()
        self._max_workers = max_workers or config.max_concurrent_tasks
        # +1 so an EXCLUSIVE_UI task queued behind another one can never
        # starve the pool of every worker a CONCURRENT task might need.
        self._pool = ThreadPoolExecutor(max_workers=max(2, self._max_workers + 1), thread_name_prefix="jarvis-task")
        self._ui_slot = threading.Semaphore(1)
        self._lock = threading.RLock()
        self._tasks: dict[str, Task] = {}
        self._handles: dict[str, TaskHandle] = {}
        self._store = store if store is not None else (TaskStore() if persist else None)
        self._closed = False

    # -- submission ----------------------------------------------------
    def submit(
        self,
        goal: str,
        body: TaskBody,
        *,
        kind: TaskKind = TaskKind.CONCURRENT,
        parent_task_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        plan: Iterable[str] = (),
        **metadata: Any,
    ) -> TaskHandle:
        """Queue a task and return immediately.

        The caller is never blocked, whatever the task does or how long
        it takes.
        """
        if self._closed:
            raise RuntimeError("The task manager has been shut down")
        task = Task(
            goal=goal,
            task_id=task_id or uuid.uuid4().hex,
            parent_task_id=parent_task_id,
            kind=kind,
            session_id=session_id,
            plan=list(plan),
            metadata=dict(metadata),
        )
        with self._lock:
            self._tasks[task.task_id] = task
        self._persist(task)
        future = self._pool.submit(self._run, task, body)
        handle = TaskHandle(task, future, self)
        with self._lock:
            self._handles[task.task_id] = handle
        log.info("Task submitted: id=%s kind=%s goal=%r", task.task_id, task.kind.value, goal[:120])
        return handle

    async def run_async(self, goal: str, body: TaskBody, **kwargs: Any) -> Any:
        """Await a task from asyncio without blocking the event loop."""
        import asyncio

        handle = self.submit(goal, body, **kwargs)
        return await asyncio.wrap_future(handle.future)

    # -- execution -----------------------------------------------------
    def _run(self, task: Task, body: TaskBody) -> Any:
        exclusive = task.kind is TaskKind.EXCLUSIVE_UI
        if exclusive:
            # Wait for the single UI slot, but stay cancellable while waiting:
            # a user who says "cancel" should not have to wait for a queue.
            while not self._ui_slot.acquire(timeout=0.05):
                if task.token.cancelled:
                    return self._finish(task, TaskStatus.CANCELLED, error="cancelled_while_queued")
        try:
            if task.token.cancelled:
                return self._finish(task, TaskStatus.CANCELLED, error="cancelled_before_start")
            with self._lock:
                task.status = TaskStatus.RUNNING
                task.started_at = utc_now()
            self._persist(task)
            started = time.perf_counter()
            try:
                result = body(task)
            except TaskCancelled as exc:
                return self._finish(task, TaskStatus.CANCELLED, error=str(exc))
            except Exception as exc:
                log.exception("Task %s failed", task.task_id)
                return self._finish(task, TaskStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            if task.token.cancelled:
                return self._finish(task, TaskStatus.CANCELLED, error="cancelled_during_execution", result=result)
            duration_ms = (time.perf_counter() - started) * 1000
            log.info("Task completed: id=%s duration_ms=%.1f", task.task_id, duration_ms)
            return self._finish(task, TaskStatus.COMPLETED, result=result)
        finally:
            if exclusive:
                self._ui_slot.release()

    def _finish(self, task: Task, status: TaskStatus, *, result: Any = None, error: str | None = None) -> Any:
        with self._lock:
            task.status = status
            task.finished_at = utc_now()
            task.error = error
            if result is not None:
                task.result = result if isinstance(result, str) else str(result)
        self._persist(task)
        if status is TaskStatus.FAILED and error:
            raise RuntimeError(error)
        if status is TaskStatus.CANCELLED:
            raise TaskCancelled(error or "cancelled")
        return result

    # -- control -------------------------------------------------------
    def cancel(self, task_id: str, reason: str = "user_cancelled") -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            handle = self._handles.get(task_id)
        if task is None or task.is_terminal:
            return False
        task.token.cancel(reason)
        if handle is not None:
            # Only succeeds if it never started; a running task stops
            # cooperatively via its token.
            handle.future.cancel()
        log.info("Task cancellation requested: id=%s reason=%s", task_id, reason)
        return True

    def cancel_all(self, reason: str = "user_cancelled") -> int:
        with self._lock:
            ids = [task_id for task_id, task in self._tasks.items() if not task.is_terminal]
        return sum(1 for task_id in ids if self.cancel(task_id, reason))

    # -- inspection ----------------------------------------------------
    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def handle(self, task_id: str) -> TaskHandle | None:
        with self._lock:
            return self._handles.get(task_id)

    def active(self) -> list[Task]:
        with self._lock:
            return [task for task in self._tasks.values() if not task.is_terminal]

    def all(self) -> list[Task]:
        with self._lock:
            return list(self._tasks.values())

    def snapshot(self) -> dict[str, Any]:
        active = self.active()
        return {
            "active": [task.to_dict(include_observations=False) for task in active],
            "active_count": len(active),
            "running_ui_task": next(
                (task.task_id for task in active if task.kind is TaskKind.EXCLUSIVE_UI and task.status is TaskStatus.RUNNING),
                None,
            ),
            "max_concurrent": self._max_workers,
            "total_tracked": len(self.all()),
        }

    def describe_active(self) -> str:
        """A short, speech-friendly answer to 'what are you doing?'."""
        active = [task for task in self.active() if task.status is TaskStatus.RUNNING]
        if not active:
            return "Nothing is running right now, sir."
        if len(active) == 1:
            return f"I'm working on: {active[0].goal}."
        listed = "; ".join(task.goal for task in active[:3])
        more = f", and {len(active) - 3} more" if len(active) > 3 else ""
        return f"I'm working on {len(active)} things: {listed}{more}."

    # -- persistence ---------------------------------------------------
    def _persist(self, task: Task) -> None:
        if self._store is None:
            return
        try:
            self._store.save(task)
        except Exception:
            # Bookkeeping must never take down real work.
            log.exception("Could not persist task %s", task.task_id)

    def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        self._closed = True
        self.cancel_all("shutdown")
        self._pool.shutdown(wait=wait, cancel_futures=True)


_MANAGER: TaskManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_task_manager() -> TaskManager:
    global _MANAGER
    if _MANAGER is None:
        with _MANAGER_LOCK:
            if _MANAGER is None:
                _MANAGER = TaskManager()
    return _MANAGER


def reset_task_manager_for_tests(manager: TaskManager | None = None) -> TaskManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is not None:
            _MANAGER.shutdown(wait=False)
        _MANAGER = manager or TaskManager(persist=False)
    return _MANAGER


__all__ = [
    "TaskManager",
    "TaskHandle",
    "Task",
    "TaskKind",
    "TaskStatus",
    "TaskCancelled",
    "CancellationToken",
    "get_task_manager",
    "reset_task_manager_for_tests",
]
