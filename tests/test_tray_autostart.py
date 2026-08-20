import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import autostart
from voice.background_assistant import AssistantState
from voice.single_instance import SingleInstance
from voice.tray_app import STATE_COLORS,TrayApplication,make_icon,state_color,tray_title
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
    def test_every_tray_state_has_transparent_corners_colored_circle_and_white_j(self):
        from PIL import ImageColor
        for state,color in STATE_COLORS.items():
            with self.subTest(state=state):
                image=make_icon(state)
                self.assertEqual(image.getpixel((0,0))[3],0)
                self.assertIn(ImageColor.getrgb(color),[pixel[:3] for pixel in image.get_flattened_data() if pixel[3]])
                center=list(image.crop((18,8,46,56)).get_flattened_data())
                self.assertGreater(sum(pixel[:3]==(255,255,255) and pixel[3]>0 for pixel in center),40)
        disabled=make_icon(AssistantState.IDLE,disabled=True)
        self.assertIn((119,119,119),[pixel[:3] for pixel in disabled.get_flattened_data() if pixel[3]])

    def test_every_assistant_state_renders_without_raising(self):
        """A state that reached the runtime but not STATE_COLORS used to
        raise KeyError from the tray -- confirmed live for
        INTERRUPTED_LISTENING during barge-in, where only the tray broke.
        Every enum member must render, enabled and disabled."""
        for state in AssistantState:
            with self.subTest(state=state):
                self.assertIn(state, STATE_COLORS)
                self.assertIsNotNone(make_icon(state))
                self.assertIsNotNone(make_icon(state, disabled=True))
                self.assertLessEqual(len(tray_title(state, state.value)), 127)

    def test_barge_in_state_is_supported(self):
        self.assertIn(AssistantState.INTERRUPTED_LISTENING, STATE_COLORS)
        self.assertNotEqual(
            STATE_COLORS[AssistantState.INTERRUPTED_LISTENING],
            STATE_COLORS[AssistantState.IDLE],
        )

    def test_the_tray_callback_survives_a_barge_in_state_change(self):
        """The live crash path: the runtime entered INTERRUPTED_LISTENING
        during barge-in and `TrayApplication._state_changed` raised KeyError
        while re-rendering the icon."""
        tray = TrayApplication()
        tray.icon = Mock()
        for state in AssistantState:
            with self.subTest(state=state):
                tray._state_changed(state, "Speech interrupted")
        self.assertTrue(tray.icon.update_menu.called)

    def test_an_unmapped_state_falls_back_instead_of_crashing(self):
        """A future enum member must degrade to a default colour: the tray
        icon is cosmetic and must never be able to kill the icon thread."""
        class FutureState:
            value = "SOME_FUTURE_STATE"

        self.assertIsNotNone(state_color(FutureState()))
        self.assertIsNotNone(make_icon(FutureState()))
        self.assertEqual(state_color(FutureState(), disabled=True), "#777777")

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
