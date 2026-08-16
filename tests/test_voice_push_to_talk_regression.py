import unittest
from unittest.mock import Mock,patch

from voice import voice_controller


class PushToTalkRegressionTests(unittest.TestCase):
    def test_voice_loop_starts_stt_warmup_without_blocking_startup(self):
        with patch.object(voice_controller,"start_background") as tts,patch.object(voice_controller,"_start_stt_warmup") as stt,patch.object(voice_controller,"one_round_push_to_talk",side_effect=KeyboardInterrupt),patch("builtins.print"):
            voice_controller.run_voice_loop()
        tts.assert_called_once_with();stt.assert_called_once_with()

    def test_existing_pipeline_still_records_transcribes_executes_and_speaks(self):
        recorder=Mock();recorder.begin.return_value="voice-iid"
        with patch.object(voice_controller, "listener_available", return_value=True), patch.object(
            voice_controller, "stt_available", return_value=True
        ), patch.object(voice_controller, "listen_push_to_talk", return_value="command.wav"), patch.object(
            voice_controller, "_run_with_interruptible_thread", side_effect=[("Hey Jarvis open YouTube", None), (None, None)]
        ), patch.object(voice_controller, "run_agent", return_value="Opened YouTube.") as run_agent, patch.object(
            voice_controller, "route_command", return_value={"type": "tool", "tool": "open_website", "arguments": {"url": "https://youtube.com"}}
        ), patch.object(voice_controller.os, "unlink"),patch.object(voice_controller,"get_recorder",return_value=recorder):
            voice_controller.one_round_push_to_talk()
        run_agent.assert_called_once_with("open YouTube",route={"type":"tool","tool":"open_website","arguments":{"url":"https://youtube.com"}},interaction_id="voice-iid",original_user_text="Hey Jarvis open YouTube")
        recorded_types=[call.args[0] for call in recorder.record.call_args_list]
        self.assertEqual(recorded_types,[voice_controller.EventType.STT_TRANSCRIPT,voice_controller.EventType.NORMALIZED_REQUEST])
        self.assertEqual(recorder.record.call_args_list[0].args[1]["raw_stt_transcript"],"Hey Jarvis open YouTube")

    def test_ctrl_c_cancels_interactive_question_task(self):
        route = {"type": "question", "message": "who is the CEO of Nvidia"}
        recorder=Mock();recorder.begin.return_value="voice-iid"
        with patch.object(voice_controller, "listener_available", return_value=True), patch.object(
            voice_controller, "stt_available", return_value=True
        ), patch.object(voice_controller, "listen_push_to_talk", return_value="command.wav"), patch.object(
            voice_controller, "_run_with_interruptible_thread", side_effect=[("who is the CEO of Nvidia", None), (None, KeyboardInterrupt())]
        ), patch.object(voice_controller, "route_command", return_value=route), patch.object(
            voice_controller.os, "unlink"
        ),patch.object(voice_controller,"get_recorder",return_value=recorder
        ), patch("brain.task_supervisor.cancel_read_only_tasks") as cancel:
            voice_controller.one_round_push_to_talk()
        cancel.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
