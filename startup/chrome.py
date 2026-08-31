"""Bring up JARVIS's OWN Chrome at startup -- and only if it isn't already.

JARVIS already has a dedicated, signed-in Chrome profile: the one
`tools/browser_authenticated.py` owns
(`data/browser_profiles/authenticated_chrome/` by default, overridable
with `JARVIS_AUTH_CHROME_PROFILE_DIR`). This module does not invent a
second profile, a second launcher or a second detection scheme -- it
reuses that module's `resolved_auth_profile_dir`,
`jarvis_chrome_is_running`, `is_cdp_available` and
`launch_chrome_for_jarvis` unchanged.

**How JARVIS's Chrome is identified.** Never "is chrome.exe running" --
the user's personal Chrome is running most of the time and must never be
mistaken for this one. Two independent, specific indicators are used, in
order:

1. **The debugger endpoint.** `is_cdp_available(127.0.0.1, JARVIS_CDP_PORT)`
   -- JARVIS's Chrome is the only browser on this machine launched with
   `--remote-debugging-port`, and a reachable endpoint means the session
   JARVIS actually attaches to is live. This is the strongest signal
   because it proves the thing that matters, not merely that a process
   exists.
2. **The profile directory on the process command line.**
   `jarvis_chrome_is_running()` inspects each chrome.exe's own
   `--user-data-dir` argument and compares it to JARVIS's resolved,
   absolute profile path. The user's personal Chrome has a different
   `--user-data-dir` (or none at all), so it never matches.

If either says JARVIS's Chrome is already up, nothing is launched -- a
second process against the same profile would not add remote debugging
anyway; Chrome's process-singleton lock just forwards the command line to
the running instance and exits.

**Failure is never fatal.** Every outcome is returned as data and logged;
a Chrome that will not start must not stop JARVIS from starting.
"""
from __future__ import annotations

import contextlib
import io
import logging
from typing import Any

log = logging.getLogger("jarvis.startup")

#: Returned in the `action` field of `ensure_jarvis_chrome`.
ACTION_DISABLED = "disabled"
ACTION_ALREADY_DEBUGGABLE = "already_running_debuggable"
ACTION_ALREADY_RUNNING = "already_running_no_debugger"
ACTION_LAUNCHED = "launched"
ACTION_FAILED = "failed"


def describe_jarvis_chrome() -> dict[str, Any]:
    """A log-safe report of whether JARVIS's own Chrome is up, and how we
    know. Makes no changes -- safe to call any time."""
    from tools.browser_authenticated import (
        DEFAULT_HOST,
        DEFAULT_PORT,
        ProfileRefusal,
        is_cdp_available,
        jarvis_chrome_is_running,
        resolved_auth_profile_dir,
    )

    report: dict[str, Any] = {"host": DEFAULT_HOST, "port": DEFAULT_PORT}
    try:
        report["profile_dir"] = str(resolved_auth_profile_dir())
        report["profile_refusal"] = None
    except ProfileRefusal as refusal:
        report["profile_dir"] = None
        report["profile_refusal"] = str(refusal)
    report["cdp_reachable"] = is_cdp_available(DEFAULT_HOST, DEFAULT_PORT)
    report["process_using_jarvis_profile"] = jarvis_chrome_is_running()
    return report


def ensure_jarvis_chrome(enabled: bool = True, verify_timeout: float = 12.0) -> dict[str, Any]:
    """Start JARVIS's dedicated Chrome if, and only if, it is not already
    running. Never raises."""
    if not enabled:
        log.info("JARVIS Chrome startup is disabled (AUTO_OPEN_CHROME=false)")
        return {"action": ACTION_DISABLED, "launched": False}

    try:
        from tools.browser_authenticated import DEFAULT_HOST, DEFAULT_PORT, launch_chrome_for_jarvis

        status = describe_jarvis_chrome()

        if status.get("profile_refusal"):
            log.error("Not starting JARVIS Chrome: %s", status["profile_refusal"])
            return {"action": ACTION_FAILED, "launched": False, "reason": status["profile_refusal"], **status}

        if status["cdp_reachable"]:
            log.info(
                "JARVIS Chrome is already running with its debugger reachable at %s:%s -- not launching another.",
                DEFAULT_HOST,
                DEFAULT_PORT,
            )
            return {"action": ACTION_ALREADY_DEBUGGABLE, "launched": False, **status}

        if status["process_using_jarvis_profile"]:
            # A chrome.exe is using JARVIS's exact profile but the debugger
            # is not answering. Launching again would NOT fix that (the
            # singleton lock forwards and exits) and would look like a
            # duplicate session, so report it honestly instead.
            log.warning(
                "A Chrome process is already using JARVIS's profile (%s) but the debugger at %s:%s is not "
                "answering. Not launching a second one -- close that window (check Task Manager for a "
                "background-only chrome.exe) and restart JARVIS to get an attachable session.",
                status["profile_dir"],
                DEFAULT_HOST,
                DEFAULT_PORT,
            )
            return {"action": ACTION_ALREADY_RUNNING, "launched": False, **status}

        # `launch_chrome_for_jarvis` reports through print() -- it was
        # written as an interactive command. Capture that into the log
        # rather than reimplementing a launcher that would then be a
        # second, drifting copy of its hard-won behaviour (absolute path,
        # forward-and-exit detection, real CDP verification).
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            pid = launch_chrome_for_jarvis(verify_timeout=verify_timeout)
        for line in buffer.getvalue().splitlines():
            if line.strip():
                log.info("[chrome] %s", line)

        if pid == -1:
            log.error("JARVIS Chrome could not be started; continuing without it.")
            return {"action": ACTION_FAILED, "launched": False, "reason": "launch_refused_or_failed", **status}

        log.info("JARVIS Chrome launched (pid=%s).", pid)
        return {"action": ACTION_LAUNCHED, "launched": True, "pid": pid, **status}

    except Exception as exc:
        # Requirement: log the error, keep starting JARVIS.
        log.exception("JARVIS Chrome startup failed; continuing without it")
        return {"action": ACTION_FAILED, "launched": False, "reason": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "ensure_jarvis_chrome",
    "describe_jarvis_chrome",
    "ACTION_DISABLED",
    "ACTION_ALREADY_DEBUGGABLE",
    "ACTION_ALREADY_RUNNING",
    "ACTION_LAUNCHED",
    "ACTION_FAILED",
]
