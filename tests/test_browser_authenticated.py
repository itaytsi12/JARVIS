"""Tests for tools/browser_authenticated.py -- the shared, reusable CDP
"attach to the user's own already-signed-in Chrome" session manager that
replaced the earlier dedicated-persistent-profile approach (Apple sign-in
hangs indefinitely inside a Playwright-driven browser; see the module
docstring)."""
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from tools import browser_authenticated as ba
from tools.browser_authenticated import (
    AuthenticatedBrowserSession,
    AuthenticatedBrowserUnavailable,
    cdp_endpoint,
    default_user_data_dir,
    is_cdp_available,
    launch_chrome_for_jarvis,
)


def _live_session():
    session = AuthenticatedBrowserSession()
    fake_browser = Mock()
    fake_browser.is_connected.return_value = True
    fake_browser.contexts = []
    session._browser = fake_browser
    session._playwright = Mock()
    return session, fake_browser


class CdpAvailabilityTests(unittest.TestCase):
    def test_available_when_endpoint_responds(self):
        cm = Mock()
        cm.__enter__ = Mock(return_value=Mock())
        cm.__exit__ = Mock(return_value=False)
        with patch("tools.browser_authenticated.urllib.request.urlopen", return_value=cm):
            self.assertTrue(is_cdp_available())

    def test_unavailable_on_connection_refused(self):
        with patch("tools.browser_authenticated.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            self.assertFalse(is_cdp_available())

    def test_unavailable_on_timeout(self):
        with patch("tools.browser_authenticated.urllib.request.urlopen", side_effect=TimeoutError()):
            self.assertFalse(is_cdp_available())

    def test_endpoint_is_localhost_by_default(self):
        self.assertEqual(cdp_endpoint(), "http://127.0.0.1:9222")


class EnsureConnectedTests(unittest.TestCase):
    def test_fails_honestly_when_cdp_unavailable(self):
        session = AuthenticatedBrowserSession()
        with patch.object(ba, "is_cdp_available", return_value=False):
            with self.assertRaisesRegex(AuthenticatedBrowserUnavailable, "Authenticated Chrome is not running"):
                session.ensure_connected()

    def test_connects_over_cdp_when_available(self):
        session = AuthenticatedBrowserSession()
        fake_browser = Mock()
        fake_browser.contexts = []
        fake_playwright_instance = Mock()
        fake_playwright_instance.chromium.connect_over_cdp.return_value = fake_browser
        fake_sync_playwright_factory = Mock()
        fake_sync_playwright_factory.return_value.start.return_value = fake_playwright_instance
        with patch.object(ba, "is_cdp_available", return_value=True), \
             patch("playwright.sync_api.sync_playwright", fake_sync_playwright_factory):
            browser = session.ensure_connected()
        self.assertIs(browser, fake_browser)
        fake_playwright_instance.chromium.connect_over_cdp.assert_called_once_with(cdp_endpoint())

    def test_reuses_existing_live_connection_without_reconnecting(self):
        session, fake_browser = _live_session()
        with patch.object(ba, "is_cdp_available") as probe:
            browser = session.ensure_connected()
        probe.assert_not_called()
        self.assertIs(browser, fake_browser)

    def test_reconnects_when_previous_connection_died(self):
        session, fake_browser = _live_session()
        fake_browser.is_connected.return_value = False
        new_browser = Mock()
        new_browser.contexts = []
        fake_playwright_instance = Mock()
        fake_playwright_instance.chromium.connect_over_cdp.return_value = new_browser
        fake_sync_playwright_factory = Mock()
        fake_sync_playwright_factory.return_value.start.return_value = fake_playwright_instance
        with patch.object(ba, "is_cdp_available", return_value=True), \
             patch("playwright.sync_api.sync_playwright", fake_sync_playwright_factory):
            browser = session.ensure_connected()
        self.assertIs(browser, new_browser)

    def test_is_connected_false_before_any_connection(self):
        session = AuthenticatedBrowserSession()
        self.assertFalse(session.is_connected())


class TabReuseTests(unittest.TestCase):
    def test_reuses_existing_matching_tab(self):
        session, fake_browser = _live_session()
        existing_page = Mock()
        existing_page.is_closed.return_value = False
        existing_page.url = "https://music.apple.com/listen-now"
        fake_context = Mock()
        fake_context.pages = [existing_page]
        fake_browser.contexts = [fake_context]
        page = session.ensure_page("music.apple.com", "https://music.apple.com")
        self.assertIs(page, existing_page)
        fake_context.new_page.assert_not_called()

    def test_opens_new_tab_in_the_same_context_when_missing(self):
        session, fake_browser = _live_session()
        fake_context = Mock()
        fake_context.pages = []
        new_page = Mock()
        fake_context.new_page.return_value = new_page
        fake_browser.contexts = [fake_context]
        page = session.ensure_page("music.apple.com", "https://music.apple.com")
        self.assertIs(page, new_page)
        new_page.goto.assert_called_once()
        # The new tab was opened on the SAME (authenticated) context/session,
        # never a second, separate browser/context.
        fake_context.new_page.assert_called_once()

    def test_no_duplicate_tab_across_repeated_calls(self):
        session, fake_browser = _live_session()
        existing_page = Mock()
        existing_page.is_closed.return_value = False
        existing_page.url = "https://music.apple.com/listen-now"
        fake_context = Mock()
        fake_context.pages = [existing_page]
        fake_browser.contexts = [fake_context]
        session.ensure_page("music.apple.com", "https://music.apple.com")
        session.ensure_page("music.apple.com", "https://music.apple.com")
        fake_context.new_page.assert_not_called()

    def test_default_context_raises_when_browser_reports_none(self):
        session, fake_browser = _live_session()
        fake_browser.contexts = []
        with self.assertRaises(AuthenticatedBrowserUnavailable):
            session.default_context()


class CookieCountsTests(unittest.TestCase):
    def test_counts_only_no_values_exposed_in_the_report(self):
        session, fake_browser = _live_session()
        fake_context = Mock()
        fake_context.cookies.return_value = [
            {"name": "a", "value": "SECRET-TOKEN-1", "domain": ".apple.com"},
            {"name": "b", "value": "SECRET-TOKEN-2", "domain": ".apple.com"},
            {"name": "c", "value": "SECRET-TOKEN-3", "domain": ".music.apple.com"},
        ]
        fake_browser.contexts = [fake_context]
        report = session.cookie_counts(urls=["https://apple.com"])
        self.assertEqual(report["cookie_counts"]["apple.com"], 2)
        self.assertEqual(report["cookie_counts"]["music.apple.com"], 1)
        self.assertEqual(report["total_cookies"], 3)
        self.assertNotIn("SECRET-TOKEN", str(report))
        fake_context.cookies.assert_called_once_with(["https://apple.com"])

    def test_raises_honestly_when_not_connected(self):
        session = AuthenticatedBrowserSession()
        with patch.object(ba, "is_cdp_available", return_value=False):
            with self.assertRaises(AuthenticatedBrowserUnavailable):
                session.cookie_counts()


class RedactUrlAndDiagnosticsTests(unittest.TestCase):
    def test_strips_query_string_and_fragment(self):
        redacted = ba._redact_url("https://idmsa.apple.com/appleauth/auth?token=secret&state=abc#frag")
        self.assertEqual(redacted, "https://idmsa.apple.com/appleauth/auth")
        self.assertNotIn("secret", redacted)

    def test_none_and_empty_url(self):
        self.assertEqual(ba._redact_url(None), "<no url>")
        self.assertEqual(ba._redact_url(""), "<no url>")

    def test_attach_diagnostics_wires_existing_and_future_pages(self):
        context = Mock()
        page = Mock()
        context.pages = [page]
        ba.attach_diagnostics(context)
        self.assertTrue(page.on.called)
        self.assertTrue(context.on.called)
        self.assertEqual(context.on.call_args.args[0], "page")


class DefaultUserDataDirTests(unittest.TestCase):
    def test_honors_explicit_override(self):
        with patch.dict(os.environ, {"JARVIS_CHROME_USER_DATA_DIR": "D:/custom/profile"}, clear=False):
            self.assertEqual(default_user_data_dir(), Path("D:/custom/profile"))

    def test_falls_back_to_local_app_data(self):
        env = dict(os.environ)
        env.pop("JARVIS_CHROME_USER_DATA_DIR", None)
        env["LOCALAPPDATA"] = "C:/Users/test/AppData/Local"
        with patch.dict(os.environ, env, clear=True):
            result = default_user_data_dir()
        self.assertEqual(result, Path("C:/Users/test/AppData/Local") / "Google" / "Chrome" / "User Data")

    def test_none_when_undeterminable_never_guesses(self):
        env = dict(os.environ)
        env.pop("JARVIS_CHROME_USER_DATA_DIR", None)
        env.pop("LOCALAPPDATA", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertIsNone(default_user_data_dir())


def _fake_process(pid, poll_result=None, stdout=b""):
    process = Mock()
    process.pid = pid
    process.poll.return_value = poll_result
    process.stdout.read.return_value = stdout
    return process


class LaunchChromeForJarvisTests(unittest.TestCase):
    def test_refuses_non_localhost_bind(self):
        for host in ("0.0.0.0", "192.168.1.5", "example.com"):
            with self.subTest(host=host):
                self.assertEqual(launch_chrome_for_jarvis(host=host), -1)

    def test_refuses_when_no_chrome_executable(self):
        with patch("tools.browser._resolve_chrome", return_value=None):
            self.assertEqual(launch_chrome_for_jarvis(user_data_dir="somewhere"), -1)

    def test_refuses_when_explicit_profile_directory_does_not_exist(self):
        with patch("tools.browser._resolve_chrome", return_value="C:/chrome.exe"):
            code = launch_chrome_for_jarvis(user_data_dir="Z:/definitely/does/not/exist/anywhere")
        self.assertEqual(code, -1)

    def test_default_dedicated_profile_dir_is_created_if_missing(self):
        fake_process = _fake_process(4242)
        with tempfile.TemporaryDirectory() as tmp:
            dedicated = Path(tmp) / "does" / "not" / "exist" / "yet"
            with patch("tools.browser._resolve_chrome", return_value="C:/chrome.exe"), \
                 patch.object(ba, "DEFAULT_AUTH_PROFILE_DIR", dedicated), \
                 patch.object(ba, "_chrome_running_with_profile", return_value=False), \
                 patch.object(ba, "is_cdp_available", return_value=True), \
                 patch.object(ba.subprocess, "Popen", return_value=fake_process):
                pid = launch_chrome_for_jarvis()
            self.assertEqual(pid, 4242)
            self.assertTrue(dedicated.exists())

    def test_refuses_the_true_default_profile_directory_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            real_default = Path(tmp) / "Google" / "Chrome" / "User Data"
            real_default.mkdir(parents=True)
            with patch("tools.browser._resolve_chrome", return_value="C:/chrome.exe"), \
                 patch.object(ba, "default_user_data_dir", return_value=real_default):
                code = launch_chrome_for_jarvis(user_data_dir=real_default)
        self.assertEqual(code, -1)

    def test_refuses_when_chrome_already_running_on_this_exact_profile(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch("tools.browser._resolve_chrome", return_value="C:/chrome.exe"), \
             patch.object(ba, "_chrome_running_with_profile", return_value=True):
            code = launch_chrome_for_jarvis(user_data_dir=tmp)
        self.assertEqual(code, -1)

    def test_launches_with_localhost_only_debug_flags_and_no_other_automation_flags(self):
        fake_process = _fake_process(4242)
        with tempfile.TemporaryDirectory() as tmp, \
             patch("tools.browser._resolve_chrome", return_value="C:/chrome.exe"), \
             patch.object(ba, "_chrome_running_with_profile", return_value=False), \
             patch.object(ba, "is_cdp_available", return_value=True), \
             patch.object(ba.subprocess, "Popen", return_value=fake_process) as popen:
            pid = launch_chrome_for_jarvis(user_data_dir=tmp, port=9222, host="127.0.0.1")
        self.assertEqual(pid, 4242)
        command = popen.call_args.args[0]
        self.assertIn("--remote-debugging-port=9222", command)
        self.assertIn("--remote-debugging-address=127.0.0.1", command)
        self.assertTrue(any(arg.startswith("--user-data-dir=") for arg in command))
        joined = " ".join(command)
        self.assertNotIn("--enable-automation", joined)
        self.assertNotIn("--headless", joined)

    def test_user_data_dir_flag_is_always_an_absolute_path(self):
        # Live-confirmed root cause of a real failure: a RELATIVE
        # --user-data-dir makes Chrome behave as though the profile is
        # already in use (prints "Opening in existing browser session."
        # and exits immediately) even against a directory nothing else
        # has ever touched. Always resolving to an absolute path fixes it.
        fake_process = _fake_process(7)
        with tempfile.TemporaryDirectory() as tmp:
            # A real relative path (from the actual test-process cwd) to
            # the temp dir -- exercises the exact same resolution
            # `launch_chrome_for_jarvis` performs, without mocking cwd
            # itself (Path.resolve() on Windows asks the OS directly and
            # ignores a mocked os.getcwd()).
            relative_marker = os.path.relpath(tmp, os.getcwd())
            with patch("tools.browser._resolve_chrome", return_value="C:/chrome.exe"), \
                 patch.object(ba, "_chrome_running_with_profile", return_value=False), \
                 patch.object(ba, "is_cdp_available", return_value=True), \
                 patch.object(ba.subprocess, "Popen", return_value=fake_process) as popen:
                launch_chrome_for_jarvis(user_data_dir=Path(relative_marker))
        command = popen.call_args.args[0]
        flag = next(arg for arg in command if arg.startswith("--user-data-dir="))
        passed_dir = flag.split("=", 1)[1]
        self.assertTrue(Path(passed_dir).is_absolute(), f"--user-data-dir was not absolute: {passed_dir!r}")

    def test_default_dedicated_profile_flag_is_absolute(self):
        fake_process = _fake_process(8)
        with tempfile.TemporaryDirectory() as tmp:
            dedicated = Path(tmp) / "authenticated_chrome"
            with patch("tools.browser._resolve_chrome", return_value="C:/chrome.exe"), \
                 patch.object(ba, "DEFAULT_AUTH_PROFILE_DIR", dedicated), \
                 patch.object(ba, "_chrome_running_with_profile", return_value=False), \
                 patch.object(ba, "is_cdp_available", return_value=True), \
                 patch.object(ba.subprocess, "Popen", return_value=fake_process) as popen:
                launch_chrome_for_jarvis()
        command = popen.call_args.args[0]
        flag = next(arg for arg in command if arg.startswith("--user-data-dir="))
        self.assertTrue(Path(flag.split("=", 1)[1]).is_absolute())

    def test_passes_profile_directory_when_given(self):
        fake_process = _fake_process(1)
        with tempfile.TemporaryDirectory() as tmp, \
             patch("tools.browser._resolve_chrome", return_value="C:/chrome.exe"), \
             patch.object(ba, "_chrome_running_with_profile", return_value=False), \
             patch.object(ba, "is_cdp_available", return_value=True), \
             patch.object(ba.subprocess, "Popen", return_value=fake_process) as popen:
            launch_chrome_for_jarvis(user_data_dir=tmp, profile_directory="Profile 3")
        command = popen.call_args.args[0]
        self.assertIn("--profile-directory=Profile 3", command)

    def test_reports_honestly_when_debug_port_never_comes_up(self):
        fake_process = _fake_process(99)
        with tempfile.TemporaryDirectory() as tmp, \
             patch("tools.browser._resolve_chrome", return_value="C:/chrome.exe"), \
             patch.object(ba, "_chrome_running_with_profile", return_value=False), \
             patch.object(ba, "is_cdp_available", return_value=False), \
             patch.object(ba.subprocess, "Popen", return_value=fake_process):
            pid = launch_chrome_for_jarvis(user_data_dir=tmp, verify_timeout=0.05)
        # Still returns the real pid (the process DID start and stay
        # alive) -- but the caller can tell from stdout that the debugger
        # never came up rather than being told a bare "success".
        self.assertEqual(pid, 99)

    def test_detects_process_singleton_forward_and_exit_as_failure(self):
        # The exact live signature confirmed on this machine: a second
        # chrome.exe against an already-in-use profile prints "Opening in
        # existing browser session." and exits almost immediately with
        # code 0 -- the debug flags were never applied to anything.
        fake_process = _fake_process(555, poll_result=0, stdout=b"Opening in existing browser session.\r\n")
        with tempfile.TemporaryDirectory() as tmp, \
             patch("tools.browser._resolve_chrome", return_value="C:/chrome.exe"), \
             patch.object(ba, "_chrome_running_with_profile", return_value=False), \
             patch.object(ba.subprocess, "Popen", return_value=fake_process):
            code = launch_chrome_for_jarvis(user_data_dir=tmp)
        self.assertEqual(code, -1)


class DiagnoseCliTests(unittest.TestCase):
    """python -m tools.browser_authenticated --diagnose (Part 10 of the
    live-path debug request): a direct, no-voice-needed way to check
    exactly what's reachable, with no secrets in the output."""

    def test_reports_not_reachable_honestly_without_raising(self):
        with patch.object(ba, "is_cdp_available", return_value=False):
            report = ba.diagnose()
        self.assertFalse(report["cdp_reachable"])
        self.assertNotIn("contexts", report)

    def test_reports_contexts_and_pages_when_reachable(self):
        fake_page = Mock()
        fake_page.is_closed.return_value = False
        fake_page.url = "https://music.apple.com/listen-now?foo=secret"
        fake_page.title.return_value = "Apple Music"
        fake_context = Mock()
        fake_context.pages = [fake_page]
        fake_browser = Mock()
        fake_browser.contexts = [fake_context]
        fake_browser.is_connected.return_value = True
        with patch.object(ba, "is_cdp_available", return_value=True), \
             patch.object(ba.AuthenticatedBrowserSession, "ensure_connected", return_value=fake_browser):
            report = ba.diagnose()
        self.assertTrue(report["cdp_reachable"])
        self.assertEqual(report["contexts"], 1)
        self.assertEqual(report["pages"], 1)
        self.assertNotIn("secret", str(report))  # query string redacted


if __name__ == "__main__":
    unittest.main()
