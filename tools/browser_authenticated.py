"""Shared, reusable authenticated-browser session manager (CDP attach).

## Why this exists

Some JARVIS capabilities need a browser that's already signed into the
user's REAL accounts (Apple Music today; WhatsApp Web/Gmail/Calendar are
natural future consumers -- see `get_authenticated_browser_session`). Two
approaches were tried and rejected before this one:

1. Attaching to the user's already-running, everyday Chrome window: not
   technically available -- Chrome's profile-singleton lock means a second
   automated process pointed at the same `user-data-dir` while that Chrome
   is already open can't get a distinct, debuggable process out of it.
2. A separate, JARVIS-owned persistent Playwright profile
   (`launch_persistent_context` against its own `user_data_dir`): the
   profile itself worked, but Apple's interactive sign-in flow
   (idmsa.apple.com/appleid.apple.com) hangs indefinitely after the
   password step specifically when the browser doing the signing-in is
   Playwright-driven (`navigator.webdriver`, `--enable-automation`, a live
   CDP connection) -- confirmed by the SAME account signing in fine in an
   ordinary Chrome window using the exact same profile mechanism.

## The approach used here

The USER launches their own real Chrome profile -- the one they already
use every day, already signed into Apple Music/whatever else -- with a
remote-debugging port enabled (`launch_chrome_for_jarvis`, exposed as
`python -m tools.browser_authenticated --launch`). That Chrome window is
completely ordinary and human-driven; nothing here scripts the sign-in
flow inside it. JARVIS then ATTACHES to that already-running browser via
`chromium.connect_over_cdp(...)` -- the same protocol Chrome DevTools
itself uses -- and reuses its existing signed-in tabs/pages exactly as a
human would with a mouse and keyboard.

## Safety

- The debugging port is bound to `127.0.0.1` only
  (`--remote-debugging-address=127.0.0.1`) -- `launch_chrome_for_jarvis`
  never exposes it to the LAN, and nothing in this module accepts a
  non-loopback host.
- JARVIS never reads Chrome's on-disk cookie database, never extracts or
  logs cookie values, authorization headers, tokens, or passwords. It only
  asks the browser (over CDP) for its current pages and drives them like a
  user would.
- Every method here fails honestly (`AuthenticatedBrowserUnavailable`) when
  the debuggable Chrome isn't reachable -- callers must never silently
  substitute a separate, unauthenticated browser for a capability that
  needs real account state (see `tools/music/apple_music_provider.py`'s
  use of this).
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.browser_authenticated")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = int(os.getenv("JARVIS_CDP_PORT", "9222"))
# Safe, non-secret diagnostic logging (URL transitions, popups, console/page
# errors, failed-request status codes -- never headers/cookies/tokens/
# request or response bodies). Off by default; see `attach_diagnostics`.
DEBUG_ENV_VAR = "JARVIS_AUTH_BROWSER_DEBUG"


def _debug_enabled() -> bool:
    return os.getenv(DEBUG_ENV_VAR, "false").lower() in {"1", "true", "yes", "on"}


def _redact_url(url: str | None) -> str:
    """Host + path only, for logging -- strips the query string and
    fragment, since a redirect URL (e.g. mid-auth-flow) can carry
    state/nonce parameters that shouldn't end up in a log file even
    though they aren't themselves credentials."""
    if not url:
        return "<no url>"
    try:
        parsed = urllib.parse.urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except Exception:
        return "<unparseable url>"


def attach_diagnostics(context, highlight_hosts: tuple[str, ...] = ()) -> None:
    """Wire safe, non-secret event logging onto every current AND future
    page in `context` -- URL transitions, popup/new-page events,
    console/page errors, and failed-request status codes. Never logs
    headers, cookies, tokens, or request/response bodies (see
    `_redact_url`). Only called when `JARVIS_AUTH_BROWSER_DEBUG=1` -- a
    manual diagnostic aid, not part of normal control flow.

    `highlight_hosts` optionally annotates navigation log lines when the
    URL matches one of them (e.g. an auth provider's domains) -- purely
    cosmetic, never used to gate behavior."""

    def _wire(page) -> None:
        try:
            page.on("framenavigated", lambda frame: log.info(
                "Authenticated browser nav: %s%s", _redact_url(frame.url),
                " [highlighted host]" if any(host in (frame.url or "") for host in highlight_hosts) else "",
            ) if frame == page.main_frame else None)
            page.on("console", lambda msg: log.info("Authenticated browser console[%s]: %s", msg.type, (msg.text or "")[:300]))
            page.on("pageerror", lambda exc: log.info("Authenticated browser page error: %s", exc))
            page.on("requestfailed", lambda req: log.info(
                "Authenticated browser request failed: %s %s (%s)", req.method, _redact_url(req.url),
                (req.failure or {}).get("errorText") if isinstance(req.failure, dict) else req.failure,
            ))
            page.on("response", lambda res: log.info("Authenticated browser response %s: %s", res.status, _redact_url(res.url)) if res.status >= 400 else None)
            page.on("close", lambda: log.info("Authenticated browser page closed: %s", _redact_url(getattr(page, "url", None))))
        except Exception:
            log.exception("Authenticated browser diagnostics: failed to wire a page")

    for page in context.pages:
        _wire(page)

    def _on_new_page(new_page) -> None:
        log.info("Authenticated browser: new page/popup opened: %s", _redact_url(getattr(new_page, "url", None)))
        _wire(new_page)

    context.on("page", _on_new_page)


class AuthenticatedBrowserUnavailable(RuntimeError):
    """The CDP-debuggable Chrome this session needs isn't running/reachable.
    Callers must report this honestly (see module docstring) rather than
    falling back to a separate, unauthenticated browser."""


def cdp_endpoint(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{port}"


def is_cdp_available(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 1.0) -> bool:
    """Cheap liveness probe: a raw HTTP GET to Chrome's own `/json/version`
    endpoint -- the standard way to check a CDP debugger is up without
    paying for a full Playwright connection. Reads only Chrome's own
    version/target metadata, never cookies or page content."""
    try:
        with urllib.request.urlopen(f"{cdp_endpoint(host, port)}/json/version", timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


class AuthenticatedBrowserSession:
    """One reused connection to the user's own remote-debugging-enabled
    Chrome. Never launches a browser itself -- only attaches to one
    that's already running (see `launch_chrome_for_jarvis` for the
    separate, explicit, human-driven launch step)."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self._playwright = None
        self._browser = None
        self._lock = threading.RLock()

    def is_connected(self) -> bool:
        with self._lock:
            if self._browser is None:
                return False
            try:
                return bool(self._browser.is_connected())
            except Exception:
                return False

    def ensure_connected(self):
        """Return a live Playwright `Browser` attached over CDP, reusing
        the existing connection if it's still alive; reconnecting
        (Part: "reconnect if browser restarts") when it isn't. Raises
        `AuthenticatedBrowserUnavailable` with an actionable message if no
        debuggable Chrome is reachable -- never launches or substitutes a
        different browser."""
        with self._lock:
            if self.is_connected():
                return self._browser
            self._discard()
            if not is_cdp_available(self.host, self.port):
                log.info(
                    "Authenticated Chrome not reachable at %s -- run: "
                    "python -m tools.browser_authenticated --launch",
                    cdp_endpoint(self.host, self.port),
                )
                raise AuthenticatedBrowserUnavailable(
                    "Authenticated Chrome is not running. Start the JARVIS browser session first."
                )
            try:
                from playwright.sync_api import sync_playwright
            except ImportError as exc:
                raise AuthenticatedBrowserUnavailable(
                    "Browser automation requires the optional 'playwright' package."
                ) from exc
            self._playwright = sync_playwright().start()
            try:
                self._browser = self._playwright.chromium.connect_over_cdp(cdp_endpoint(self.host, self.port))
            except Exception as exc:
                try:
                    self._playwright.stop()
                finally:
                    self._playwright = None
                raise AuthenticatedBrowserUnavailable(
                    "Authenticated Chrome is not running. Start the JARVIS browser session first."
                ) from exc
            log.info(
                "Authenticated Chrome CDP connect succeeded at %s; contexts=%d",
                cdp_endpoint(self.host, self.port), len(self._browser.contexts),
            )
            if _debug_enabled() and self._browser.contexts:
                attach_diagnostics(self._browser.contexts[0])
            return self._browser

    def _discard(self) -> None:
        browser, playwright = self._browser, self._playwright
        self._browser = self._playwright = None
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        if playwright is not None:
            try:
                playwright.stop()
            except Exception:
                pass

    def close(self) -> None:
        """Drop JARVIS's own connection handle. Never closes the user's
        actual Chrome window/process -- this only ends the CDP attachment,
        exactly like closing Chrome DevTools does not close the tab."""
        with self._lock:
            self._discard()

    # ------------------------------------------------------------------
    # Tab/page access -- shared by every authenticated-session consumer
    # (Apple Music today; a future WhatsApp Web/Gmail/Calendar tool reuses
    # the exact same session/lock rather than inventing a second one).
    # ------------------------------------------------------------------

    def default_context(self):
        browser = self.ensure_connected()
        contexts = browser.contexts
        if not contexts:
            raise AuthenticatedBrowserUnavailable(
                "Attached to Chrome, but it reported no open profile/context."
            )
        return contexts[0]

    def list_pages(self) -> list[Any]:
        context = self.default_context()
        pages = []
        for page in context.pages:
            try:
                if not page.is_closed():
                    pages.append(page)
            except Exception:
                continue
        return pages

    def find_page(self, hostname_substring: str):
        """Return the first open tab whose URL contains
        `hostname_substring`, or None. Never opens a new tab."""
        for page in self.list_pages():
            try:
                if hostname_substring in (page.url or ""):
                    return page
            except Exception:
                continue
        return None

    def ensure_page(self, hostname_substring: str, url_if_missing: str, focus: bool = True):
        """Reuse an existing matching tab in the user's real session;
        open exactly one new tab IN THAT SAME session/context if none
        exists. Never opens a duplicate."""
        with self._lock:
            pages = self.list_pages()
            log.info("Authenticated browser: %d open page(s) while looking for %r", len(pages), hostname_substring)
            page = next((p for p in pages if hostname_substring in (getattr(p, "url", "") or "")), None)
            if page is None:
                log.info("Authenticated browser: no existing %r tab found -- opening one", hostname_substring)
                context = self.default_context()
                page = context.new_page()
                page.goto(url_if_missing, wait_until="domcontentloaded", timeout=20000)
                log.info("Authenticated browser: opened new tab -> %s", _redact_url(page.url))
            else:
                log.info("Authenticated browser: reusing existing tab -> %s", _redact_url(page.url))
            if focus:
                try:
                    page.bring_to_front()
                except Exception:
                    pass
            return page

    # ------------------------------------------------------------------
    # Safe diagnostics -- counts only, never values (Part: "never log
    # cookies/tokens").
    # ------------------------------------------------------------------

    def cookie_counts(self, urls: list[str] | None = None) -> dict[str, Any]:
        """Report per-domain cookie COUNTS only -- never values -- as a
        safe diagnostic for confirming a sign-in actually landed. When
        `urls` is given, only cookies that would be sent to those URLs are
        counted (Playwright's own `context.cookies(urls)` filter) -- this
        matters because the attached context is the user's REAL profile,
        which may also hold Gmail/WhatsApp/etc. session state a
        feature-specific diagnostic has no reason to enumerate."""
        context = self.default_context()
        try:
            cookies = context.cookies(urls) if urls else context.cookies()
        except Exception:
            return {"cookie_counts": {}, "error": "could_not_read_cookies"}
        counts: dict[str, int] = {}
        for cookie in cookies:
            domain = (cookie.get("domain") or "").lstrip(".")
            counts[domain] = counts.get(domain, 0) + 1
        return {"cookie_counts": counts, "total_cookies": len(cookies)}


_SHARED_SESSION: AuthenticatedBrowserSession | None = None
_SHARED_LOCK = threading.Lock()


def get_authenticated_browser_session() -> AuthenticatedBrowserSession:
    global _SHARED_SESSION
    with _SHARED_LOCK:
        if _SHARED_SESSION is None:
            _SHARED_SESSION = AuthenticatedBrowserSession()
        return _SHARED_SESSION


def reset_authenticated_browser_session_for_tests(session: AuthenticatedBrowserSession | None = None) -> None:
    """Test-only helper: swap the process-wide singleton for an isolated
    instance (or None, to force recreation) so tests never share state
    with a real run."""
    global _SHARED_SESSION
    with _SHARED_LOCK:
        _SHARED_SESSION = session


# ---------------------------------------------------------------------
# Launcher: an explicit, human-driven step -- never invoked automatically
# by any JARVIS automation path.
# ---------------------------------------------------------------------

#: The dedicated, non-default profile `--launch` uses unless overridden.
#: Live-confirmed why this must NOT be the OS-default profile directory:
#: launching Chrome with `--remote-debugging-port` against a
#: `--user-data-dir` that ANOTHER already-running chrome.exe process is
#: also using (which, for the true default directory, is almost always
#: true -- that's what the user's ordinary daily Chrome uses, including a
#: background/tray process left running after all windows are closed,
#: which many Chrome installs default to) does NOT start a new,
#: debug-enabled browser process at all. Chrome's per-profile "process
#: singleton" lock just forwards the new command line to the process
#: that's already running (stdout literally reads "Opening in existing
#: browser session.") and the new process exits immediately -- the
#: `--remote-debugging-port` flag is never applied to anything. A
#: DEDICATED directory nothing else ever opens can't collide with the
#: user's regular browsing this way, confirmed live: a fresh launch
#: against an unused directory got a working CDP endpoint immediately
#: even with 20+ unrelated chrome.exe processes already running.
DEFAULT_AUTH_PROFILE_DIR = Path(os.getenv("JARVIS_AUTH_CHROME_PROFILE_DIR", "data/browser_profiles/authenticated_chrome"))


def default_user_data_dir() -> Path | None:
    """The real Chrome profile root the user's own daily-driver Chrome
    uses (override with `JARVIS_CHROME_USER_DATA_DIR`, the same env var
    `tools/browser.py::open_website` already honors). NOT what `--launch`
    targets by default any more -- see `DEFAULT_AUTH_PROFILE_DIR`'s
    docstring for why pointing remote debugging at this exact directory
    reliably fails while your regular Chrome is running (which is
    virtually always). Still exposed for diagnostics/detection (e.g.
    warning a user who explicitly overrides `JARVIS_CHROME_USER_DATA_DIR`
    to point at it anyway) and because other tools
    (`tools/browser.py::open_website`) use it for an unrelated purpose
    (opening an ordinary, non-debugged window)."""
    override = os.getenv("JARVIS_CHROME_USER_DATA_DIR")
    if override:
        return Path(override)
    local_app_data = os.getenv("LOCALAPPDATA")
    if not local_app_data:
        return None
    return Path(local_app_data) / "Google" / "Chrome" / "User Data"


def _normalize_dir(path: str) -> str:
    return path.rstrip("\\/").lower()


def _chrome_running_with_profile(resolved_dir: Path) -> bool:
    """True if a chrome.exe process is already running against this EXACT
    `--user-data-dir` (inspected via each process's own command line, not
    merely "some chrome.exe exists somewhere" -- the user's regular,
    unrelated daily browsing under a DIFFERENT profile must never block a
    launch against our own dedicated directory). See
    `DEFAULT_AUTH_PROFILE_DIR` for why a collision here silently defeats
    the debug flags rather than erroring."""
    try:
        import psutil
    except ImportError:
        return False
    target = _normalize_dir(str(resolved_dir))
    try:
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                if (proc.info.get("name") or "").lower() != "chrome.exe":
                    continue
                cmdline = proc.info.get("cmdline") or []
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            for arg in cmdline:
                if arg.lower().startswith("--user-data-dir=") and _normalize_dir(arg.split("=", 1)[1]) == target:
                    return True
    except Exception:
        return False
    return False


def launch_chrome_for_jarvis(
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    user_data_dir: Path | str | None = None,
    profile_directory: str | None = None,
    url: str | None = None,
    verify_timeout: float = 10.0,
) -> int:
    """Launch a Chrome profile with remote debugging enabled, bound to
    localhost only. This is a genuinely ordinary, human-driven Chrome
    window -- JARVIS attaches to it afterwards over CDP; it never
    automates this launch or whatever the user signs into inside it.

    Defaults to a DEDICATED, non-default profile directory
    (`DEFAULT_AUTH_PROFILE_DIR`) rather than the user's regular Chrome
    profile -- confirmed live that launching against the directory the
    user's ordinary, already-running Chrome also uses does not actually
    enable remote debugging (Chrome's process-singleton lock forwards the
    command to the already-running process and the new one exits
    immediately, flags never applied; see `_chrome_running_with_profile`).
    That dedicated profile starts out signed out of everything -- sign in
    manually once, directly in the window this opens (this launch adds no
    `--enable-automation`/testing flags, so it looks like an entirely
    ordinary Chrome window to any site's fraud/bot detection).

    Prints exactly which profile directory (and, if set,
    `--profile-directory`) and command line it's about to use -- never
    guesses silently. Refuses to launch (returns -1) if a chrome.exe is
    already running against that SAME directory (see
    `_chrome_running_with_profile`), or if an EXPLICITLY given directory
    (`user_data_dir` param / `JARVIS_CHROME_USER_DATA_DIR`) doesn't exist
    on disk, since launching against a nonexistent path would silently
    create a brand-new, signed-out profile there instead of reusing the
    real one the caller presumably meant. Does NOT require the dedicated
    default directory to pre-exist -- that one is JARVIS's own and is
    expected to be empty on first use.

    Never claims success by "`Popen` didn't raise" alone: polls
    `/json/version` for up to `verify_timeout` seconds and, if the launched
    process exits almost immediately (the process-singleton "forwarded to
    an existing session" signature) or the debugger never becomes
    reachable, reports that honestly instead.

    Returns the launched process's PID, or -1 on an outright refusal.
    """
    if host != "127.0.0.1" and host != "localhost":
        print(f"Refusing to bind the debugger to {host!r} -- only 127.0.0.1/localhost is allowed.")
        return -1
    from tools.browser import _resolve_chrome
    executable = os.getenv("JARVIS_CHROME_EXECUTABLE") or _resolve_chrome()
    if not executable:
        print("Could not locate a real installed Chrome executable.")
        return -1

    explicit_override = user_data_dir or os.getenv("JARVIS_CHROME_USER_DATA_DIR")
    if explicit_override:
        resolved_dir = Path(explicit_override)
        if not resolved_dir.exists():
            print(f"Profile directory not found: {resolved_dir}")
            print("Refusing to launch -- that would silently create a brand-new, signed-out")
            print("profile there instead of reusing the one you meant.")
            return -1
    else:
        resolved_dir = DEFAULT_AUTH_PROFILE_DIR
        resolved_dir.mkdir(parents=True, exist_ok=True)
    # MUST be absolute: live-confirmed on this machine that passing a
    # RELATIVE --user-data-dir makes Chrome behave as if the profile is
    # already in use -- it prints "Opening in existing browser session."
    # and exits immediately (code 0) without ever starting a new process
    # or applying the remote-debugging flags, even against a directory
    # nothing else has ever touched. An absolute path (`.resolve()`) does
    # not exhibit this at all.
    resolved_dir = resolved_dir.resolve()

    true_default = default_user_data_dir()
    if true_default is not None and _normalize_dir(str(resolved_dir)) == _normalize_dir(str(true_default)):
        print(f"Refusing to enable remote debugging on your REGULAR Chrome profile ({resolved_dir}).")
        print("Chrome does not actually turn on the debugger there while your normal Chrome")
        print("is running (confirmed live -- it just forwards to that existing process and")
        print("does not apply the flag), so this would silently fail. Use a dedicated profile")
        print(f"instead -- unset JARVIS_CHROME_USER_DATA_DIR to use the default dedicated one")
        print(f"({DEFAULT_AUTH_PROFILE_DIR}), sign in there once, and JARVIS will reuse it from then on.")
        return -1

    if _chrome_running_with_profile(resolved_dir):
        print(f"Chrome is already running against this exact profile ({resolved_dir}).")
        print("A second process against the SAME profile does not add remote debugging to")
        print("the one already running -- Chrome just forwards the command to it instead.")
        print("Close that Chrome window (check Task Manager too -- Chrome can keep a")
        print("background process alive after all windows are closed) and try again.")
        return -1

    resolved_profile_directory = profile_directory or os.getenv("JARVIS_CHROME_PROFILE_DIRECTORY")
    command = [
        executable,
        f"--remote-debugging-port={port}",
        f"--remote-debugging-address={host}",
        f"--user-data-dir={resolved_dir}",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    if resolved_profile_directory:
        command.append(f"--profile-directory={resolved_profile_directory}")
    if url:
        command.append(url)
    print(f"Chrome executable: {executable}")
    print(f"Profile directory (--user-data-dir): {resolved_dir}")
    print(f"Profile within it: {resolved_profile_directory or '(Chrome default)'}")
    print(f"Remote debugging: {host}:{port} (bound to localhost only)")
    print(f"Command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # A process-singleton "forward to the existing session" exits almost
    # immediately with code 0 -- distinguish that (unambiguous, actionable)
    # from "still starting" / "blocked by something else" before polling.
    time.sleep(0.6)
    early_exit_code = process.poll()
    if early_exit_code is not None:
        output = ""
        try:
            output = (process.stdout.read() or b"").decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        print(f"Chrome process exited immediately (code={early_exit_code}) instead of staying open.")
        if output:
            print(f"Chrome's own output: {output!r}")
        if "existing browser session" in output.lower():
            print("This is the process-singleton forward-and-exit signature: another chrome.exe")
            print("was already using this profile, so the debug flags were never applied to it.")
        print("Treating this launch as FAILED -- run --diagnose to confirm, and check Task Manager")
        print("for a lingering chrome.exe (including a background-only one) using this profile.")
        return -1

    # Confirm the debug port actually came up before declaring success --
    # the process staying alive only means Chrome is running, not that
    # nothing (a firewall/AV product, a slow first launch) kept the
    # debugger from ever binding.
    deadline = time.monotonic() + verify_timeout
    while time.monotonic() < deadline:
        if is_cdp_available(host, port):
            print(f"Chrome launched (pid={process.pid}) and the debugger at {cdp_endpoint(host, port)} is reachable.")
            print("Leave this window open -- JARVIS will attach to it.")
            return process.pid
        time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    print(f"Chrome process started (pid={process.pid}) and is still running, but the debugger at")
    print(f"{cdp_endpoint(host, port)} never became reachable within {verify_timeout:.0f} seconds.")
    print("Not claiming success -- run 'python -m tools.browser_authenticated --diagnose' to")
    print("check again, and check whether a firewall/antivirus product is blocking 127.0.0.1.")
    return process.pid


def diagnose(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> dict[str, Any]:
    """Direct, no-voice-needed diagnostic (Part 10 of the live-path debug
    request): reports CDP reachability, then context/page counts and each
    page's hostname/title -- never a full URL with query string, never
    cookies/tokens. Safe to run any time; makes no changes."""
    report: dict[str, Any] = {"cdp_endpoint": cdp_endpoint(host, port), "cdp_reachable": is_cdp_available(host, port)}
    print(f"CDP endpoint: {report['cdp_endpoint']}")
    print(f"CDP reachable: {'yes' if report['cdp_reachable'] else 'no'}")
    if not report["cdp_reachable"]:
        print("Run: python -m tools.browser_authenticated --launch")
        return report
    session = AuthenticatedBrowserSession(host, port)
    try:
        browser = session.ensure_connected()
        report["contexts"] = len(browser.contexts)
        print(f"Contexts: {report['contexts']}")
        pages = session.list_pages()
        report["pages"] = len(pages)
        print(f"Pages: {report['pages']}")
        page_info = []
        for page in pages:
            try:
                info = {"host": _redact_url(page.url), "title": page.title()}
            except Exception:
                info = {"host": _redact_url(getattr(page, "url", None)), "title": None}
            page_info.append(info)
            print(f"  - {info['host']}  (title: {info['title']!r})")
        report["page_info"] = page_info
    finally:
        session.close()
    return report


if __name__ == "__main__":
    import sys
    if "--launch" in sys.argv:
        launch_chrome_for_jarvis()
    elif "--diagnose" in sys.argv:
        diagnose()
    else:
        print("Usage: python -m tools.browser_authenticated --launch | --diagnose")
        print(f"Checking for an already-running debuggable Chrome at {cdp_endpoint()} ...")
        print("Available:" if is_cdp_available() else "Not available.")
