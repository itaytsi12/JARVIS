"""Focused tests for the overlapping speech/planning/execution behavior
added to voice/background_assistant.py (Parts C, E, F, J): immediate
acknowledgement concurrency, duplicate-action prevention against a fired
speculative action, the fast-path-skips-the-planner-ack rule, and using an
ElevenLabs committed transcript instead of re-running Whisper.
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from brain.speculative_execution import PartialActionLedger
from voice.background_assistant import AlwaysOnAssistant, AssistantState
from voice.voice_perf import VoiceInteractionTimer


class FakeWakeEngine:
    sample_rate = 16000
    frame_samples = 1280

    def load(self): pass
    def reset(self): pass
    def process(self, _frame): return False, 0.0


class NeedsPlanningTests(unittest.TestCase):
    def test_simple_known_action_does_not_need_planning(self):
        route = {"type": "tool", "tool": "open_application", "arguments": {"app_name": "notepad"}}
        self.assertFalse(AlwaysOnAssistant._command_needs_planning("open notepad", route))

    def test_multi_clause_command_needs_planning(self):
        route = {"type": "plan", "message": "open chrome and then search for cats"}
        self.assertTrue(AlwaysOnAssistant._command_needs_planning("open chrome and then search for cats", route))

    def test_local_plan_route_type_alone_does_not_force_planning(self):
        # A deterministic local_plan is already fully resolved -- it must
        # not also trigger the "I'll check that, sir." cloud-planning ack.
        route = {"type": "local_plan", "actions": []}
        self.assertFalse(AlwaysOnAssistant._command_needs_planning("open chrome and lower the volume", route))


class SpeculativeAckConcurrencyTests(unittest.TestCase):
    def test_ack_speech_and_execution_are_dispatched_on_separate_threads(self):
        assistant = AlwaysOnAssistant(wake_engine=FakeWakeEngine())
        spoken = []
        exec_started = threading.Event()

        def fake_run_agent(command, route=None, **kwargs):
            exec_started.set()
            time.sleep(0.2)
            return "opened spotify"

        action = PartialActionLedger(min_stable=1).observe_partial("open spotify")
        with patch.object(assistant, "_start_speech_task", side_effect=lambda text, lang, iid=None: spoken.append(text)), \
             patch("brain.agent.run_agent", side_effect=fake_run_agent):
            started = time.perf_counter()
            assistant._on_speculative_action(action)
            # The ack dispatch call itself must return almost immediately --
            # it must not block on the (separately threaded) 0.2s execution.
            self.assertLess(time.perf_counter() - started, 0.15)
            self.assertTrue(exec_started.wait(1))
        self.assertEqual(spoken, ["Opening it now, sir."])

    def test_speculative_ack_text_varies_by_tool_but_is_always_short(self):
        for tool, expected in [
            ("open_application", "Opening it now, sir."),
            ("open_website", "Opening it now, sir."),
            ("volume_down", "Right away, sir."),
            ("mute_volume", "Right away, sir."),
            ("take_screenshot", "On it, sir."),
        ]:
            with self.subTest(tool=tool):
                text = AlwaysOnAssistant._speculative_ack_text({"tool": tool})
                self.assertEqual(text, expected)
                self.assertLess(len(text), 30)


class ProcessCaptureReconciliationTests(unittest.TestCase):
    def _assistant(self):
        return AlwaysOnAssistant(wake_engine=FakeWakeEngine())

    def test_final_tool_command_matching_fired_action_never_calls_run_agent(self):
        assistant = self._assistant()
        ledger = PartialActionLedger(min_stable=1)
        ledger.observe_partial("open spotify")
        with patch("voice.speech_to_text.transcribe_audio", return_value="open spotify"), \
             patch("brain.router.route_command", return_value={"type": "tool", "tool": "open_application", "arguments": {"app_name": "spotify"}}), \
             patch("brain.agent.run_agent") as run_agent, \
             patch.object(assistant, "_start_speech_task") as speech:
            assistant._process_capture([np.zeros(1280, dtype="int16")], ledger=ledger)
        run_agent.assert_not_called()
        speech.assert_not_called()
        self.assertEqual(assistant.state, AssistantState.IDLE)

    def test_final_command_that_differs_from_fired_action_still_executes(self):
        assistant = self._assistant()
        ledger = PartialActionLedger(min_stable=1)
        ledger.observe_partial("open spotify")
        with patch("voice.speech_to_text.transcribe_audio", return_value="open chrome"), \
             patch("brain.router.route_command", return_value={"type": "tool", "tool": "open_application", "arguments": {"app_name": "chrome"}}), \
             patch("brain.agent.run_agent", return_value="Opened chrome.") as run_agent, \
             patch.object(assistant, "_start_speech_task"):
            assistant._process_capture([np.zeros(1280, dtype="int16")], ledger=ledger)
        run_agent.assert_called_once()

    def test_elevenlabs_committed_transcript_skips_whisper(self):
        assistant = self._assistant()
        with patch("voice.speech_to_text.transcribe_audio") as whisper, \
             patch("brain.router.route_command", return_value={"type": "tool", "tool": "open_application", "arguments": {"app_name": "notepad"}}), \
             patch("brain.agent.run_agent", return_value="Opened notepad.") as run_agent, \
             patch.object(assistant, "_start_speech_task"):
            assistant._process_capture([np.zeros(1280, dtype="int16")], elevenlabs_transcript="open notepad")
        whisper.assert_not_called()
        run_agent.assert_called_once()

    def test_no_elevenlabs_transcript_falls_back_to_whisper(self):
        assistant = self._assistant()
        with patch("voice.speech_to_text.transcribe_audio", return_value="open notepad") as whisper, \
             patch("brain.router.route_command", return_value={"type": "tool", "tool": "open_application", "arguments": {"app_name": "notepad"}}), \
             patch("brain.agent.run_agent", return_value="Opened notepad.") as run_agent, \
             patch.object(assistant, "_start_speech_task"):
            assistant._process_capture([np.zeros(1280, dtype="int16")], elevenlabs_transcript=None)
        whisper.assert_called_once()
        run_agent.assert_called_once()

    def test_complex_command_speaks_ack_before_run_agent_returns(self):
        assistant = self._assistant()
        call_order = []

        def fake_run_agent(command, **kwargs):
            call_order.append("run_agent")
            return "done"

        def fake_speech(text, lang, iid=None):
            call_order.append(("speech", text))

        with patch.dict("os.environ", {"VOICE_LANGUAGE": "en"}), \
             patch("voice.speech_to_text.transcribe_audio", return_value="open chrome and then search for cats"), \
             patch("brain.router.route_command", return_value={"type": "plan", "message": "open chrome and then search for cats"}), \
             patch("brain.agent.run_agent", side_effect=fake_run_agent), \
             patch.object(assistant, "_start_speech_task", side_effect=fake_speech):
            assistant._process_capture([np.zeros(1280, dtype="int16")])
        self.assertEqual(call_order[0], ("speech", "Understood."))
        self.assertIn("run_agent", call_order)


if __name__ == "__main__":
    unittest.main()
