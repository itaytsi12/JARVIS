"""The startup sequence, the single-instance guard and JARVIS's Chrome.

Everything here runs offline: no Qt window, no microphone, no Chrome and
no scheduled task is really created. What is asserted is the wiring the
requirements are actually about -- that a second launch starts nothing,
that the assistant is created exactly once, that a failing stage does not
take the rest of JARVIS down, and that Chrome detection keys off JARVIS's
OWN profile rather than "is chrome.exe running".
"""
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from startup import chrome as startup_chrome
from startup.launcher import MUTEX_NAME, JarvisLauncher, StartupOptions, start_jarvis


class StartupOptionsTests(unittest.TestCase):
    def test_configuration_supplies_the_defaults(self):
        settings = Mock(
            ui_enabled=True,
            ui_fullscreen=False,
            auto_open_chrome=True,
            auto_start_voice=True,
            tray_enabled=True,
        )
        with patch("config.get_config", return_value=settings):
            options = StartupOptions.from_config()
        self.assertTrue(options.ui and options.chrome and options.voice and options.tray)
        self.assertFalse(options.fullscreen)

    def test_an_unset_flag_never_clobbers_a_configured_value(self):
        """`--no-chrome` must turn Chrome off without also turning the
        window, voice or the tray off. An argparse flag that was not given
        is None, and None means "leave the setting alone"."""
        configured = StartupOptions(ui=True, fullscreen=True, chrome=True, voice=True, tray=True)
        overridden = configured.with_overrides(ui=None, fullscreen=None, chrome=False, voice=None, tray=None)
        self.assertFalse(overridden.chrome)
        self.assertTrue(overridden.ui)
        self.assertTrue(overridden.fullscreen)
        self.assertTrue(overridden.voice)
        self.assertTrue(overridden.tray)

    def test_no_overrides_returns_the_configured_options_unchanged(self):
        configured = StartupOptions(ui=False, fullscreen=True, chrome=False, voice=True, tray=False)
        self.assertEqual(configured.with_overrides(ui=None, chrome=None), configured)


class CommandLineTests(unittest.TestCase):
    """`main.py --start` must pass "not specified" for every flag the user
    did not type, so the configured settings are what actually decide."""

    def _overrides(self, argv):
        import main as main_module

        seen = {}

        def fake_start(**kwargs):
            seen.update(kwargs)
            return 0

        with patch("sys.argv", ["main.py", *argv]), patch("startup.launcher.start_jarvis", fake_start):
            with self.assertRaises(SystemExit) as exit_info:
                main_module.main()
        self.assertEqual(exit_info.exception.code, 0)
        return seen

    def test_a_bare_start_overrides_nothing(self):
        """Regression: `--no-voice`/`--no-tray` originally reused the dests
        of the pre-existing `--voice`/`--tray` flags. argparse allows that
        silently, and a plain `--start` inherited `--voice`'s store_true
        default of False -- so JARVIS came up with no voice and no tray
        while the log reported it as configured. Confirmed live."""
        self.assertEqual(
            self._overrides(["--start"]),
            {"ui": None, "fullscreen": None, "chrome": None, "voice": None, "tray": None},
        )

    def test_each_negative_flag_turns_off_only_its_own_stage(self):
        self.assertEqual(
            self._overrides(["--start", "--no-voice"]),
            {"ui": None, "fullscreen": None, "chrome": None, "voice": False, "tray": None},
        )
        self.assertEqual(
            self._overrides(["--start", "--no-ui", "--no-chrome"]),
            {"ui": False, "fullscreen": None, "chrome": False, "voice": None, "tray": None},
        )
        self.assertEqual(
            self._overrides(["--start", "--fullscreen"]),
            {"ui": None, "fullscreen": True, "chrome": None, "voice": None, "tray": None},
        )


class SingleInstanceTests(unittest.TestCase):
    def test_the_launcher_and_the_tray_share_one_mutex_name(self):
        """If these ever diverged, `main.py --start` and `main.py --tray`
        would each believe it was the only instance, and two processes
        would fight over the one microphone."""
        from voice.single_instance import SingleInstance

        self.assertEqual(SingleInstance().name, MUTEX_NAME)

    def test_a_second_launch_starts_nothing_and_exits_zero(self):
        taken = Mock()
        taken.acquire.return_value = False
        with patch("voice.single_instance.SingleInstance", return_value=taken), patch(
            "config.logging_setup.configure_file_logging", return_value=Path("logs/x.log")
        ), patch("config.logging_setup.configure_logging"), patch(
            "config.logging_setup.log_startup_status"
        ), patch(
            "startup.launcher.JarvisLauncher"
        ) as launcher:
            self.assertEqual(start_jarvis(), 0)
        # No window, no backend, no browser -- the launcher is never even
        # constructed.
        launcher.assert_not_called()
        taken.release.assert_not_called()

    def test_the_mutex_is_released_even_when_startup_raises(self):
        free = Mock()
        free.acquire.return_value = True
        with patch("voice.single_instance.SingleInstance", return_value=free), patch(
            "config.logging_setup.configure_file_logging", return_value=Path("logs/x.log")
        ), patch("config.logging_setup.configure_logging"), patch(
            "config.logging_setup.log_startup_status"
        ), patch(
            "startup.launcher.JarvisLauncher", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                start_jarvis()
        free.release.assert_called_once()


class BackendStartupTests(unittest.TestCase):
    def _launcher(self, **options):
        return JarvisLauncher(StartupOptions(**options))

    def test_the_tray_owns_the_assistant_lifecycle_so_it_starts_once(self):
        """`TrayApplication.run()` starts and stops the assistant itself.
        Starting it here as well would start the one microphone owner
        twice."""
        launcher = self._launcher(voice=True, tray=True)
        assistant = Mock()
        tray = Mock()
        with patch.object(launcher, "build_assistant", return_value=assistant), patch(
            "voice.tray_app.TrayApplication", return_value=tray
        ) as tray_class:
            launcher.start_backend()
            for thread in launcher._threads:
                thread.join(timeout=5)
        assistant.start.assert_not_called()
        tray.run.assert_called_once()
        self.assertIs(tray_class.call_args.kwargs["assistant"], assistant)

    def test_a_failing_tray_still_leaves_voice_running(self):
        launcher = self._launcher(voice=True, tray=True)
        assistant = Mock()
        with patch.object(launcher, "build_assistant", return_value=assistant), patch(
            "voice.tray_app.TrayApplication", side_effect=RuntimeError("no shell notification area")
        ):
            launcher.start_backend()
        assistant.start.assert_called_once()

    def test_no_tray_means_the_launcher_starts_the_assistant_itself(self):
        launcher = self._launcher(voice=True, tray=False)
        assistant = Mock()
        with patch.object(launcher, "build_assistant", return_value=assistant):
            launcher.start_backend()
        assistant.start.assert_called_once()

    def test_voice_disabled_creates_no_assistant_at_all(self):
        launcher = self._launcher(voice=False, tray=True)
        with patch.object(launcher, "build_assistant") as build:
            launcher.start_backend()
        build.assert_not_called()

    def test_a_broken_voice_stack_is_reported_not_raised(self):
        """Requirement 6: a startup failure is logged and survived. An
        ImportError from the audio stack must not stop the window."""
        launcher = self._launcher(voice=True, tray=True)
        with patch.dict(
            "sys.modules", {"voice.background_assistant": None}
        ), patch("startup.launcher.log") as logger:
            self.assertIsNone(launcher.build_assistant())
        self.assertTrue(logger.exception.called)

    def test_shutdown_stops_the_tray_which_stops_the_assistant(self):
        launcher = self._launcher()
        launcher.tray = Mock()
        launcher.assistant = Mock()
        launcher.shutdown()
        launcher.tray.icon.stop.assert_called_once()
        # Stopping it here too would be a double stop: TrayApplication.run
        # already stops the assistant in its own `finally`.
        launcher.assistant.stop.assert_not_called()

    def test_shutdown_stops_the_assistant_directly_when_there_is_no_tray(self):
        launcher = self._launcher()
        launcher.assistant = Mock()
        launcher.shutdown()
        launcher.assistant.stop.assert_called_once()

    def test_the_tray_exit_item_closes_the_window_on_the_gui_thread(self):
        """The tray's Exit runs on the tray's thread; Qt may only be
        touched on the GUI thread, so the close has to be marshalled."""
        launcher = self._launcher()
        launcher.ui = Mock()
        launcher.request_shutdown()
        launcher.ui.bridge.run_on_gui_thread.assert_called_once_with(launcher.ui.quit)
        # Idempotent: a second Exit must not queue a second quit.
        launcher.request_shutdown()
        self.assertEqual(launcher.ui.bridge.run_on_gui_thread.call_count, 1)


class UiStartupTests(unittest.TestCase):
    def test_a_missing_pyside6_degrades_to_headless_rather_than_crashing(self):
        launcher = JarvisLauncher(StartupOptions(ui=True, voice=True, tray=False))
        with patch("ui.app.is_available", return_value=(False, "pyside6_unavailable:ImportError")), patch.object(
            launcher, "_run_headless", return_value=7
        ) as headless, patch("startup.launcher.log") as logger:
            self.assertEqual(launcher.run(), 7)
        headless.assert_called_once()
        self.assertTrue(logger.error.called)

    def test_the_window_is_created_before_chrome_and_the_backend_start(self):
        """The core has to be on screen while the slow parts are still
        loading, not after them -- so both are dispatched from the Qt
        `on_started` hook onto worker threads."""
        launcher = JarvisLauncher(StartupOptions(ui=True))
        order = []

        def fake_run_ui(fullscreen, on_started):
            order.append("window_created")
            on_started(Mock())
            for thread in launcher._threads:
                thread.join(timeout=5)
            return 0

        with patch.object(launcher, "start_chrome", side_effect=lambda: order.append("chrome")), patch.object(
            launcher, "start_backend", side_effect=lambda: order.append("backend")
        ):
            launcher._run_with_ui(fake_run_ui)
        self.assertEqual(order[0], "window_created")
        self.assertEqual(sorted(order[1:]), ["backend", "chrome"])


class ChromeStartupTests(unittest.TestCase):
    """Requirement 4: identify JARVIS's OWN Chrome, never "is chrome.exe
    running" -- the user's personal Chrome is up most of the time."""

    def _status(self, **overrides):
        status = {
            "host": "127.0.0.1",
            "port": 9222,
            "profile_dir": r"C:\jarvis\data\browser_profiles\authenticated_chrome",
            "profile_refusal": None,
            "cdp_reachable": False,
            "process_using_jarvis_profile": False,
        }
        status.update(overrides)
        return status

    def test_a_reachable_debugger_means_nothing_is_launched(self):
        with patch.object(startup_chrome, "describe_jarvis_chrome", return_value=self._status(cdp_reachable=True)), patch(
            "tools.browser_authenticated.launch_chrome_for_jarvis"
        ) as launch:
            result = startup_chrome.ensure_jarvis_chrome()
        launch.assert_not_called()
        self.assertFalse(result["launched"])
        self.assertEqual(result["action"], startup_chrome.ACTION_ALREADY_DEBUGGABLE)

    def test_a_process_on_jarvis_own_profile_means_nothing_is_launched(self):
        with patch.object(
            startup_chrome, "describe_jarvis_chrome", return_value=self._status(process_using_jarvis_profile=True)
        ), patch("tools.browser_authenticated.launch_chrome_for_jarvis") as launch:
            result = startup_chrome.ensure_jarvis_chrome()
        launch.assert_not_called()
        self.assertEqual(result["action"], startup_chrome.ACTION_ALREADY_RUNNING)

    def test_the_personal_chrome_running_does_not_stop_the_launch(self):
        """Both JARVIS-specific indicators are false -- which is exactly
        the state when only the user's own, different-profile Chrome is
        running. JARVIS's Chrome must still be started."""
        with patch.object(startup_chrome, "describe_jarvis_chrome", return_value=self._status()), patch(
            "tools.browser_authenticated.launch_chrome_for_jarvis", return_value=4242
        ) as launch:
            result = startup_chrome.ensure_jarvis_chrome()
        launch.assert_called_once()
        self.assertTrue(result["launched"])
        self.assertEqual(result["pid"], 4242)

    def test_detection_matches_on_the_profile_directory_not_on_chrome_exe(self):
        """`jarvis_chrome_is_running` inspects each process's own
        `--user-data-dir`; a chrome.exe on any other profile is not a
        match."""
        from tools import browser_authenticated

        jarvis_profile = Path(r"C:\jarvis\data\browser_profiles\authenticated_chrome")
        with patch.object(browser_authenticated, "resolved_auth_profile_dir", return_value=jarvis_profile), patch.object(
            browser_authenticated, "_chrome_running_with_profile"
        ) as matcher:
            matcher.return_value = False
            self.assertFalse(browser_authenticated.jarvis_chrome_is_running())
            matcher.assert_called_once_with(jarvis_profile)
            matcher.return_value = True
            self.assertTrue(browser_authenticated.jarvis_chrome_is_running())

    def test_a_failed_launch_is_reported_and_never_raises(self):
        with patch.object(startup_chrome, "describe_jarvis_chrome", return_value=self._status()), patch(
            "tools.browser_authenticated.launch_chrome_for_jarvis", return_value=-1
        ):
            result = startup_chrome.ensure_jarvis_chrome()
        self.assertEqual(result["action"], startup_chrome.ACTION_FAILED)
        self.assertFalse(result["launched"])

    def test_an_exception_anywhere_still_lets_jarvis_start(self):
        with patch.object(startup_chrome, "describe_jarvis_chrome", side_effect=OSError("no chrome")):
            result = startup_chrome.ensure_jarvis_chrome()
        self.assertEqual(result["action"], startup_chrome.ACTION_FAILED)
        self.assertIn("OSError", result["reason"])

    def test_disabling_chrome_launches_nothing(self):
        with patch("tools.browser_authenticated.launch_chrome_for_jarvis") as launch:
            result = startup_chrome.ensure_jarvis_chrome(enabled=False)
        launch.assert_not_called()
        self.assertEqual(result["action"], startup_chrome.ACTION_DISABLED)


if __name__ == "__main__":
    unittest.main()
