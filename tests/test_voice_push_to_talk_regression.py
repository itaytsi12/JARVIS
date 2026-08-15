import unittest
from unittest.mock import patch

from voice import voice_controller


class PushToTalkRegressionTests(unittest.TestCase):
    def test_existing_pipeline_still_records_transcribes_executes_and_speaks(self):
        with patch.object(voice_controller, "listener_available", return_value=True), patch.object(
            voice_controller, "stt_available", return_value=True
        ), patch.object(voice_controller, "listen_push_to_talk", return_value="command.wav"), patch.object(
            voice_controller, "_run_with_interruptible_thread", side_effect=[("Hey Jarvis open YouTube", None), (None, None)]
        ), patch.object(voice_controller, "run_agent", return_value="Opened YouTube."), patch.object(
            voice_controller, "route_command", return_value={"type": "tool", "tool": "open_website", "arguments": {"url": "https://youtube.com"}}
        ), patch.object(voice_controller.os, "unlink"):
            voice_controller.one_round_push_to_talk()


if __name__ == "__main__":
    unittest.main()
