import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import autostart
from voice.background_assistant import AssistantState
from voice.single_instance import SingleInstance
from voice.tray_app import TrayApplication, tray_title
from tools.windows_process import hidden_process_kwargs


class AutostartTests(unittest.TestCase):
    def test_background_helpers_use_no_console_by_default(self):
        with patch.dict("os.environ", {"DEBUG_BACKGROUND_CONSOLE": "false"}):
            kwargs = hidden_process_kwargs()
        self.assertTrue(kwargs["creationflags"])
        self.assertIsNotNone(kwargs["startupinfo"])

    def test_xml_uses_background_python_working_directory_and_single_task_policy(self):
        xml = autostart.task_xml()
        self.assertIn("pythonw.exe", xml.lower())
        self.assertIn(str(autostart.PROJECT_ROOT), xml)
        self.assertIn("--tray", xml)
        self.assertIn("IgnoreNew", xml)
        self.assertIn("LeastPrivilege", xml)

    def test_install_and_remove_use_same_task_name(self):
        completed = Mock(returncode=0, stderr="")
        with patch.object(autostart, "validate_installation"), patch.object(autostart.subprocess, "run", return_value=completed) as run:
            autostart.install_autostart()
            create_args = run.call_args.args[0]
            self.assertIn(autostart.TASK_NAME, create_args)
            autostart.remove_autostart()
            delete_args = run.call_args.args[0]
            self.assertIn(autostart.TASK_NAME, delete_args)


class TrayActionTests(unittest.TestCase):
    def test_windows_tooltip_is_bounded(self):
        self.assertLessEqual(len(tray_title(AssistantState.ERROR, "x" * 500)), 127)

    def test_menu_actions_control_existing_assistant(self):
        tray = TrayApplication()
        tray.assistant.request_listen = Mock()
        tray.assistant.restart = Mock()
        tray._listen(None, None)
        tray._restart(None, None)
        tray.assistant.request_listen.assert_called_once()
        tray.assistant.restart.assert_called_once()
        tray._toggle_wake(None, None)
        tray._toggle_mute(None, None)
        self.assertFalse(tray.assistant.wake_enabled)
        self.assertTrue(tray.assistant.muted)

    def test_state_refresh_does_not_query_task_scheduler(self):
        tray = TrayApplication()
        tray.icon = Mock()
        with patch("scripts.autostart.is_autostart_enabled") as query:
            tray._state_changed(AssistantState.LISTENING, "Listening")
            tray._state_changed(AssistantState.PROCESSING, "Processing")
        query.assert_not_called()


class SingleInstanceTests(unittest.TestCase):
    def test_named_mutex_rejects_second_instance(self):
        name = r"Local\JARVIS.Test.BackgroundAssistant"
        first, second = SingleInstance(name), SingleInstance(name)
        try:
            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())
        finally:
            second.release()
            first.release()


if __name__ == "__main__":
    unittest.main()
