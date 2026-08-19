"""Tasks: lifecycle, persistence, concurrency, UI exclusivity and cancellation."""
import tempfile
import threading
import time
import unittest
from pathlib import Path

from memory.agent_store import AgentDatabase
from tasks.manager import TaskManager
from tasks.models import Task, TaskCancelled, TaskKind, TaskStatus
from tasks.store import TaskStore


class TaskStateTests(unittest.TestCase):
    def test_a_new_task_starts_pending_and_is_not_terminal(self):
        task = Task(goal="do a thing")
        self.assertIs(task.status, TaskStatus.PENDING)
        self.assertFalse(task.is_terminal)
        self.assertFalse(task.cancelled)

    def test_observations_are_ordered(self):
        task = Task(goal="x")
        task.observe("run_command", "exit code 0")
        task.observe("read_code", "10 lines")
        self.assertEqual([item.index for item in task.observations], [0, 1])
        self.assertEqual(task.observations[1].source, "read_code")

    def test_cancellation_token_is_cooperative(self):
        task = Task(goal="x")
        task.token.cancel("user_cancelled")
        self.assertTrue(task.cancelled)
        with self.assertRaises(TaskCancelled):
            task.token.raise_if_cancelled()

    def test_serialization_includes_the_lifecycle(self):
        payload = Task(goal="x", plan=["coding"]).to_dict()
        for key in ("task_id", "goal", "status", "created_at", "plan", "observations", "cancelled"):
            self.assertIn(key, payload)


class TaskManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = TaskManager(persist=False)

    def tearDown(self):
        self.manager.shutdown(wait=False)

    def test_a_task_runs_and_completes(self):
        handle = self.manager.submit("add numbers", lambda task: "42")
        self.assertEqual(handle.result(timeout=5), "42")
        self.assertIs(self.manager.get(handle.task_id).status, TaskStatus.COMPLETED)

    def test_a_failing_task_is_recorded_as_failed(self):
        def explode(task):
            raise ValueError("nope")

        handle = self.manager.submit("explode", explode)
        with self.assertRaises(RuntimeError):
            handle.result(timeout=5)
        task = self.manager.get(handle.task_id)
        self.assertIs(task.status, TaskStatus.FAILED)
        self.assertIn("nope", task.error)

    def test_submitting_never_blocks_the_caller(self):
        release = threading.Event()
        started = time.perf_counter()
        handle = self.manager.submit("slow", lambda task: release.wait(5))
        self.assertLess(time.perf_counter() - started, 1.0)
        release.set()
        handle.wait(timeout=5)

    def test_independent_tasks_run_concurrently(self):
        both_running = threading.Barrier(2, timeout=5)

        def body(task):
            both_running.wait()
            return "ok"

        first = self.manager.submit("research", body)
        second = self.manager.submit("run tests", body)
        # If these were serialized the barrier would time out.
        self.assertEqual(first.result(timeout=5), "ok")
        self.assertEqual(second.result(timeout=5), "ok")

    def test_ui_tasks_are_serialized_so_they_never_fight_over_the_keyboard(self):
        concurrent = []
        peak = []
        lock = threading.Lock()

        def ui(task):
            with lock:
                concurrent.append(1)
                peak.append(len(concurrent))
            time.sleep(0.05)
            with lock:
                concurrent.pop()
            return "done"

        handles = [self.manager.submit(f"ui {index}", ui, kind=TaskKind.EXCLUSIVE_UI) for index in range(4)]
        for handle in handles:
            handle.result(timeout=15)
        self.assertEqual(max(peak), 1)

    def test_a_ui_task_does_not_block_a_concurrent_task(self):
        release = threading.Event()
        ui_handle = self.manager.submit("ui", lambda task: release.wait(5), kind=TaskKind.EXCLUSIVE_UI)
        concurrent_handle = self.manager.submit("research", lambda task: "answer")
        self.assertEqual(concurrent_handle.result(timeout=5), "answer")
        release.set()
        ui_handle.wait(timeout=5)

    def test_a_running_task_can_be_cancelled_cooperatively(self):
        def body(task):
            for _ in range(200):
                task.token.raise_if_cancelled()
                time.sleep(0.01)
            return "finished"

        handle = self.manager.submit("long", body)
        time.sleep(0.05)
        self.assertTrue(handle.cancel())
        with self.assertRaises(TaskCancelled):
            handle.result(timeout=5)
        self.assertIs(self.manager.get(handle.task_id).status, TaskStatus.CANCELLED)

    def test_a_queued_ui_task_can_be_cancelled_before_it_starts(self):
        release = threading.Event()
        blocker = self.manager.submit("ui blocker", lambda task: release.wait(5), kind=TaskKind.EXCLUSIVE_UI)
        queued = self.manager.submit("ui queued", lambda task: "ran", kind=TaskKind.EXCLUSIVE_UI)
        time.sleep(0.05)
        self.assertTrue(queued.cancel())
        release.set()
        blocker.wait(timeout=5)
        with self.assertRaises(TaskCancelled):
            queued.result(timeout=5)

    def test_cancel_all_stops_every_active_task(self):
        def body(task):
            while not task.token.cancelled:
                time.sleep(0.01)
            task.token.raise_if_cancelled()

        handles = [self.manager.submit(f"t{index}", body) for index in range(3)]
        time.sleep(0.05)
        self.assertEqual(self.manager.cancel_all(), 3)
        for handle in handles:
            handle.wait(timeout=5)
        self.assertEqual(self.manager.active(), [])

    def test_cancelling_a_finished_task_is_a_no_op(self):
        handle = self.manager.submit("quick", lambda task: "done")
        handle.result(timeout=5)
        self.assertFalse(handle.cancel())

    def test_snapshot_describes_what_is_running(self):
        release = threading.Event()
        handle = self.manager.submit("research something", lambda task: release.wait(5))
        time.sleep(0.05)
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["active_count"], 1)
        self.assertIn("research something", self.manager.describe_active())
        release.set()
        handle.wait(timeout=5)

    def test_describe_active_when_idle(self):
        self.assertIn("Nothing is running", self.manager.describe_active())

    def test_async_wrapper_awaits_a_task(self):
        import asyncio

        result = asyncio.run(self.manager.run_async("async task", lambda task: "async result"))
        self.assertEqual(result, "async result")


class TaskPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.database = AgentDatabase(Path(tempfile.mkdtemp()) / "agent.sqlite3")
        self.store = TaskStore(self.database)
        self.manager = TaskManager(store=self.store)

    def tearDown(self):
        self.manager.shutdown(wait=False)

    def test_task_state_is_persisted(self):
        handle = self.manager.submit("persisted goal", lambda task: "done")
        handle.result(timeout=5)
        stored = self.store.get(handle.task_id)
        self.assertEqual(stored["goal"], "persisted goal")
        self.assertEqual(stored["status"], TaskStatus.COMPLETED.value)

    def test_interrupted_tasks_are_reconciled_rather_than_left_running(self):
        task = Task(goal="was running", status=TaskStatus.RUNNING)
        self.store.save(task)
        interrupted = self.store.mark_interrupted_tasks()
        self.assertIn(task.task_id, interrupted)
        self.assertEqual(self.store.get(task.task_id)["status"], TaskStatus.FAILED.value)
        self.assertEqual(self.store.get(task.task_id)["error"], "process_interrupted")

    def test_tasks_can_be_listed_by_status(self):
        handle = self.manager.submit("listed", lambda task: "done")
        handle.result(timeout=5)
        self.assertTrue(self.store.list(TaskStatus.COMPLETED))


if __name__ == "__main__":
    unittest.main()
