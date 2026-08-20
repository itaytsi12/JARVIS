"""Dedicated worker threads for Playwright's SYNCHRONOUS API.

Why this exists (confirmed live, and reproduced here in three lines):

    >>> pw = sync_playwright().start()
    >>> asyncio.get_running_loop().is_running()
    True

`sync_playwright().start()` creates an asyncio event loop, starts it inside
a greenlet, and leaves it RUNNING on the calling thread for the whole life
of that Playwright instance. Two consequences follow, and JARVIS was hitting
both:

1. A second `sync_playwright().start()` on that same thread raises
   ``Error: It looks like you are using Playwright Sync API inside the
   asyncio loop. Please use the Async API instead.`` -- the exact live
   failure seen for "Open Music." / "Play Israeli playlist.". JARVIS has two
   independent sync-Playwright sessions (``tools/browser_agent.py``'s
   ephemeral unauthenticated browser and ``tools/browser_authenticated.py``'s
   CDP attachment to the user's signed-in Chrome), each a process-wide
   singleton started lazily on whichever thread happened to call first. As
   soon as one of them started on a thread, the other could not start there
   -- and JARVIS reuses threads (the task manager's pool, the audio thread,
   the parallel-action executor), so the two collided in practice.
2. Sync Playwright objects are bound to the loop and greenlet of the thread
   that created them. Sharing a session singleton across threads was a
   latent correctness bug regardless of the error message above.

The fix is the second option the brief allows -- keep the synchronous API
(JARVIS is thread-based, not asyncio-based: see ``tasks/manager.py``, "Threads,
not asyncio", and every capability it drives -- pywinauto, pyautogui, the
Windows APIs -- is blocking and synchronous) and isolate it from any running
loop. Each logical session gets its OWN worker thread that owns exactly one
Playwright instance:

- the calling thread's loop state becomes irrelevant, because the sync API is
  never invoked there;
- each session's Playwright instance is the only one on its thread, so the two
  sessions can never conflict;
- every page object is created and used on one thread, satisfying Playwright's
  thread affinity.

The sessions still OWN their Playwright instances and their whole lifecycle
(start, connect, discard, stop) exactly as before -- this module only supplies
the thread they run on. Moving ownership here would have meant one instance
shared across every reconnect and relaunch, with no honest place to stop it.

Deliberately one worker PER SESSION rather than one shared worker: the two
sessions have separate resource locks in ``brain/resource_locks.py`` precisely
so ordinary web browsing and authenticated playback do not serialize behind
each other, and funnelling both through a single thread would have quietly
undone that.

The exception a submitted call raises is re-raised on the CALLING thread with
its original type and message -- nothing is suppressed or translated, so a
real Playwright failure still reaches the caller exactly as before.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

log = logging.getLogger("jarvis.playwright")

#: How long a submitted call may run before the caller gives up waiting. Long
#: enough for a real page load plus Playwright's own timeouts, short enough
#: that a wedged worker cannot hang a voice interaction forever.
DEFAULT_CALL_TIMEOUT = 300.0


class PlaywrightWorkerError(RuntimeError):
    """The worker thread itself could not run the call (it never started, or
    the call did not finish in time). Distinct from an exception raised BY the
    submitted function, which is re-raised unchanged."""


class _Job:
    __slots__ = ("fn", "args", "kwargs", "done", "result", "error")

    def __init__(self, fn, args, kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.done = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None


class PlaywrightWorker:
    """A single daemon thread that owns one sync Playwright instance.

    Calls are executed one at a time, in submission order. That is not a new
    constraint: everything routed here already serializes on a
    ``brain/resource_locks.py`` lock for the same session.
    """

    def __init__(self, name: str):
        self.name = name
        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    # -- lifecycle -----------------------------------------------------
    def _ensure_thread(self) -> threading.Thread:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._thread
            thread = threading.Thread(
                target=self._run, name=f"jarvis-playwright-{self.name}", daemon=True
            )
            self._thread = thread
            thread.start()
            return thread

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                break
            try:
                job.result = job.fn(*job.args, **job.kwargs)
            except BaseException as exc:  # re-raised on the caller's thread
                job.error = exc
            finally:
                job.done.set()

    def is_current_thread(self) -> bool:
        return threading.current_thread() is self._thread

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Run `fn` on this worker's thread and return its result.

        Called FROM the worker thread (a nested dispatch), `fn` runs inline --
        queueing it would deadlock, and it is already on the right thread.

        `**kwargs` is forwarded to `fn` untouched, deliberately: this takes no
        options of its own, so a wrapped function is free to have a parameter
        called `timeout` (several browser helpers do).
        """
        if self.is_current_thread():
            return fn(*args, **kwargs)
        self._ensure_thread()
        job = _Job(fn, args, kwargs)
        self._queue.put(job)
        if not job.done.wait(DEFAULT_CALL_TIMEOUT):
            raise PlaywrightWorkerError(
                f"The {self.name} browser worker did not finish within the time limit."
            )
        if job.error is not None:
            raise job.error
        return job.result

    def shutdown(self) -> None:
        """Stop the worker thread. For tests and process teardown; production
        never needs to call this. The session that owns a Playwright instance
        is responsible for stopping it (on this thread) before shutdown."""
        with self._lock:
            thread, self._thread = self._thread, None
        if thread is None or not thread.is_alive():
            return
        self._queue.put(None)
        thread.join(timeout=30)


#: The two logical sessions. `browser` is `tools/browser_agent.py`'s ephemeral
#: unauthenticated browser; `authenticated` is
#: `tools/browser_authenticated.py`'s CDP attachment to the user's signed-in
#: Chrome. A future authenticated-session consumer (WhatsApp Web, Gmail, ...)
#: reuses `AUTHENTICATED` -- it shares the session, so it must share the thread.
BROWSER = PlaywrightWorker("browser")
AUTHENTICATED = PlaywrightWorker("authenticated")

#: Which worker a tool's Playwright work belongs on, keyed by the resource
#: name `brain/resource_locks.py::resource_for_tool` already assigns. One
#: mapping, derived from the existing one, rather than a second list of tool
#: names to keep in sync.
_WORKER_BY_RESOURCE = {
    "browser_session": BROWSER,
    "authenticated_browser": AUTHENTICATED,
}


def worker_for_tool(tool_name: str) -> PlaywrightWorker | None:
    """The worker a tool must run on, or None if it does not drive Playwright."""
    from brain.resource_locks import resource_for_tool

    return _WORKER_BY_RESOURCE.get(resource_for_tool(tool_name))


def run_for_tool(tool_name: str, fn: Callable[..., Any], *args, **kwargs) -> Any:
    """Run `fn` on the worker thread `tool_name` belongs to, or inline when the
    tool drives no browser at all.

    This is the ONLY place the hop happens. The session classes below it
    (`tools/browser_agent.py`, `tools/browser_authenticated.py`) deliberately
    do NOT hop themselves: several of their methods hold a per-session RLock
    across calls to each other, and RLock reentrancy is per-thread, so a
    second hop from inside the lock would deadlock against the caller still
    holding it. Hop once, at the dispatch boundary
    (`brain/tool_router.py::execute_tool`,
    `brain/agent_runtime.py::_browser_action`), and everything underneath
    runs on the right thread for free."""
    worker = worker_for_tool(tool_name)
    if worker is None:
        return fn(*args, **kwargs)
    return worker.submit(fn, *args, **kwargs)


def shutdown_all() -> None:
    BROWSER.shutdown()
    AUTHENTICATED.shutdown()


__all__ = [
    "AUTHENTICATED",
    "BROWSER",
    "PlaywrightWorker",
    "PlaywrightWorkerError",
    "run_for_tool",
    "shutdown_all",
    "worker_for_tool",
]
