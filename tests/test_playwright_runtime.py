"""Regression tests for the "Playwright Sync API inside the asyncio loop" bug.

Live symptom: "Open Music." and "Play Israeli playlist." failed with

    It looks like you are using Playwright Sync API inside the asyncio loop.
    Please use the Async API instead.

Root cause (reproduced by `SyncPlaywrightLoopBehaviourTests` below against the
REAL installed Playwright, not a mock): `sync_playwright().start()` leaves a
RUNNING asyncio loop on the calling thread, so a second sync-Playwright
session on that same thread cannot start. JARVIS has two such sessions
(`tools/browser_agent.py` and `tools/browser_authenticated.py`), each a
process-wide singleton started on whichever thread called first, and it reuses
threads -- so they collided.

The tests here execute the real tool-dispatch entry points while an asyncio
loop is already running on the calling thread, which is the condition the
error checks for.
"""
from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import patch

from tools.playwright_runtime import (
    AUTHENTICATED,
    BROWSER,
    PlaywrightWorker,
    run_for_tool,
    worker_for_tool,
)

try:  # The optional automation dependency; these tests adapt rather than fail.
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT_INSTALLED = True
except Exception:  # pragma: no cover - environment without playwright
    PLAYWRIGHT_INSTALLED = False


def call_with_a_running_loop(fn, *args, **kwargs):
    """Run `fn` on a thread that has a RUNNING asyncio event loop.

    This is exactly the situation Playwright's sync API refuses to start in,
    and the situation the live failure happened in.
    """
    box = {}

    async def main():
        assert asyncio.get_running_loop().is_running()
        try:
            box["result"] = fn(*args, **kwargs)
        except BaseException as exc:  # recorded, re-raised on the caller
            box["error"] = exc

    asyncio.run(main())
    if "error" in box:
        raise box["error"]
    return box["result"]


class WorkerMappingTests(unittest.TestCase):
    def test_browser_and_authenticated_tools_get_separate_workers(self):
        """One shared worker would re-serialize ordinary browsing behind
        authenticated playback, which `brain/resource_locks.py` deliberately
        keeps apart."""
        self.assertIs(worker_for_tool("browser_open_url"), BROWSER)
        self.assertIs(worker_for_tool("music_play"), AUTHENTICATED)
        self.assertIsNot(BROWSER, AUTHENTICATED)

    def test_non_browser_tools_have_no_worker(self):
        for tool in ("get_time", "volume_up", "read_text_file", "run_command"):
            with self.subTest(tool=tool):
                self.assertIsNone(worker_for_tool(tool))

    def test_a_non_browser_tool_runs_on_the_calling_thread(self):
        """No thread hop for the fast path: it would add latency for nothing."""
        seen = {}
        run_for_tool("get_time", lambda: seen.setdefault("thread", threading.current_thread()))
        self.assertIs(seen["thread"], threading.current_thread())

    def test_a_browser_tool_runs_off_the_calling_thread(self):
        seen = {}
        run_for_tool("music_play", lambda: seen.setdefault("thread", threading.current_thread()))
        self.assertIsNot(seen["thread"], threading.current_thread())
        self.assertIn("jarvis-playwright", seen["thread"].name)

    def test_every_authenticated_browser_tool_maps_to_the_authenticated_worker(self):
        from brain.resource_locks import AUTHENTICATED_BROWSER_TOOLS, BROWSER_TOOLS

        for tool in AUTHENTICATED_BROWSER_TOOLS:
            with self.subTest(tool=tool):
                self.assertIs(worker_for_tool(tool), AUTHENTICATED)
        for tool in BROWSER_TOOLS:
            with self.subTest(tool=tool):
                self.assertIs(worker_for_tool(tool), BROWSER)


class WorkerBehaviourTests(unittest.TestCase):
    def setUp(self):
        self.worker = PlaywrightWorker("test")
        self.addCleanup(self.worker.shutdown)

    def test_the_result_comes_back_to_the_caller(self):
        self.assertEqual(self.worker.submit(lambda a, b: a + b, 2, 3), 5)

    def test_an_exception_is_reraised_unchanged_not_suppressed(self):
        """"Do not suppress the exception": a real Playwright failure must
        still reach the caller with its own type and message."""
        class SpecificFailure(RuntimeError):
            pass

        def boom():
            raise SpecificFailure("the page was closed")

        with self.assertRaises(SpecificFailure) as caught:
            self.worker.submit(boom)
        self.assertEqual(str(caught.exception), "the page was closed")

    def test_calls_all_land_on_one_thread(self):
        threads = {self.worker.submit(threading.current_thread) for _ in range(5)}
        self.assertEqual(len(threads), 1)
        self.assertIsNot(threads.pop(), threading.current_thread())

    def test_a_nested_call_from_the_worker_thread_does_not_deadlock(self):
        def outer():
            return self.worker.submit(lambda: "inner")

        self.assertEqual(self.worker.submit(outer), "inner")

    def test_the_worker_survives_a_failing_call(self):
        with self.assertRaises(ValueError):
            self.worker.submit(lambda: (_ for _ in ()).throw(ValueError("x")))
        self.assertEqual(self.worker.submit(lambda: "still alive"), "still alive")


class RunningLoopTests(unittest.TestCase):
    """The dispatchers must work when the CALLING thread has a running loop."""

    def test_execute_tool_reaches_a_browser_tool_from_inside_a_running_loop(self):
        from brain import tool_router

        called = {}

        def fake_impl(tool_name, arguments):
            called["thread"] = threading.current_thread()
            return {"success": True, "message": "ok"}

        with patch.object(tool_router, "_execute_tool_impl", fake_impl):
            result = call_with_a_running_loop(tool_router.execute_tool, "music_play", {"song": "x"})

        self.assertTrue(result["success"], "the call did not survive a running loop on the caller's thread")
        self.assertIsNot(called["thread"], threading.current_thread())
        self.assertIn("jarvis-playwright", called["thread"].name)

    def test_a_worker_thread_starts_with_no_running_loop(self):
        """The property that makes the sync API usable there. Checked on a
        FRESH worker: a worker whose session already holds a live Playwright
        instance legitimately has that instance's loop running on it."""
        worker = PlaywrightWorker("clean")
        self.addCleanup(worker.shutdown)

        def probe():
            try:
                asyncio.get_running_loop()
                return True
            except RuntimeError:
                return False

        self.assertFalse(call_with_a_running_loop(worker.submit, probe))

    def test_a_non_browser_tool_still_works_from_inside_a_running_loop(self):
        from brain.tool_router import execute_tool

        result = call_with_a_running_loop(execute_tool, "get_time", {})
        self.assertTrue(str(result))

    def test_agent_runtime_browser_actions_are_isolated_too(self):
        """`browser_*` never reaches execute_tool -- it has its own dispatch."""
        from brain.agent_runtime import AgentRuntime
        from brain.models import ToolResult

        runtime = AgentRuntime()
        called = {}

        def fake_impl(tool, args):
            called["thread"] = threading.current_thread()
            return ToolResult(True, tool, "ok")

        with patch.object(AgentRuntime, "_browser_action_impl", staticmethod(fake_impl)):
            result = call_with_a_running_loop(runtime._browser_action, "browser_open_url", {"url": "https://x"})

        self.assertTrue(result.success)
        self.assertIsNot(called["thread"], threading.current_thread())
        self.assertIn("jarvis-playwright", called["thread"].name)


@unittest.skipUnless(PLAYWRIGHT_INSTALLED, "playwright is not installed")
class SyncPlaywrightLoopBehaviourTests(unittest.TestCase):
    """The REAL Playwright behaviour this module exists for. No browser is
    launched -- only the Playwright driver process, which is cheap."""

    def test_starting_sync_playwright_leaves_a_running_loop_on_that_thread(self):
        box = {}

        def probe():
            instance = sync_playwright().start()
            try:
                box["running"] = asyncio.get_running_loop().is_running()
                try:
                    sync_playwright().start()
                    box["second_start_failed"] = False
                except Exception as exc:
                    box["second_start_failed"] = "asyncio loop" in str(exc)
            finally:
                instance.stop()

        thread = threading.Thread(target=probe, name="probe")
        thread.start()
        thread.join(120)
        self.assertTrue(box.get("running"), "sync_playwright().start() no longer leaves a running loop")
        self.assertTrue(
            box.get("second_start_failed"),
            "a second sync_playwright().start() on the same thread should still fail",
        )

    def test_two_sessions_can_start_concurrently_on_separate_workers(self):
        """The actual fix: the two JARVIS sessions each start their own sync
        Playwright instance, which is impossible on one shared thread."""
        one = PlaywrightWorker("one")
        two = PlaywrightWorker("two")
        self.addCleanup(one.shutdown)
        self.addCleanup(two.shutdown)
        first = one.submit(lambda: sync_playwright().start())
        second = two.submit(lambda: sync_playwright().start())
        try:
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertIsNot(first, second)
        finally:
            one.submit(first.stop)
            two.submit(second.stop)

    def test_starting_both_sessions_on_one_thread_would_still_fail(self):
        """Documents WHY there are two workers rather than one shared one."""
        worker = PlaywrightWorker("shared")
        self.addCleanup(worker.shutdown)
        first = worker.submit(lambda: sync_playwright().start())
        try:
            with self.assertRaises(Exception) as caught:
                worker.submit(lambda: sync_playwright().start())
            self.assertIn("asyncio loop", str(caught.exception))
        finally:
            worker.submit(first.stop)


if __name__ == "__main__":
    unittest.main()
