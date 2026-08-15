import unittest
from unittest.mock import Mock

import numpy as np

from voice.background_assistant import AlwaysOnAssistant, AssistantState
from voice.text_normalizer import normalize_transcript
from brain.router import route_command


class StepClock:
    def __init__(self, step=0.08):
        self.value = 0.0
        self.step = step

    def __call__(self):
        self.value += self.step
        return self.value


class FakeWakeEngine:
    sample_rate = 16000
    frame_samples = 1280

    def __init__(self, detects=True):
        self.detects = detects
        self.calls = 0
        self.reset_calls = 0

    def load(self): pass

    def reset(self): self.reset_calls += 1

    def process(self, _frame):
        self.calls += 1
        return (self.detects and self.calls == 1, 0.9)


class FakeStream:
    def __init__(self, frames):
        self.frames = list(frames)
        self.active = False

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *_):
        self.active = False

    def read(self, samples):
        frame = self.frames.pop(0) if self.frames else np.zeros(samples, dtype=np.int16)
        return frame.tobytes(), False


class CaptureAssistant(AlwaysOnAssistant):
    def __init__(self, stream, *args, **kwargs):
        super().__init__(*args, stream_factory=lambda: stream, **kwargs)
        self.stream = stream
        self.processed = False
        self.mic_released_before_processing = False

    def _process_capture(self, frames):
        self.processed = bool(frames)
        self.mic_released_before_processing = not self.stream.active
        self._set_state(AssistantState.PROCESSING)
        self._set_state(AssistantState.EXECUTING)
        self._set_state(AssistantState.SPEAKING)
        self._stop.set()


class BackgroundAssistantTests(unittest.TestCase):
    def test_wake_capture_handoff_self_wake_gate_and_return_idle(self):
        loud = np.full(1280, 2000, dtype=np.int16)
        quiet = np.zeros(1280, dtype=np.int16)
        stream = FakeStream([quiet, loud] + [quiet] * 15)
        wake = FakeWakeEngine()
        states = []
        assistant = CaptureAssistant(stream, wake_engine=wake, clock=StepClock(), state_callback=lambda state, detail: states.append(state))
        assistant.cooldown_seconds = 0
        assistant._audio_session()
        self.assertTrue(assistant.processed)
        self.assertTrue(assistant.mic_released_before_processing)
        self.assertEqual(wake.calls, 1, "wake inference must stop after detection and while speaking")
        self.assertEqual(states[:3], [AssistantState.IDLE, AssistantState.WAKE_DETECTED, AssistantState.LISTENING])
        self.assertIn(AssistantState.PROCESSING, states)
        self.assertIn(AssistantState.SPEAKING, states)
        self.assertEqual(assistant.state, AssistantState.IDLE)

    def test_no_speech_timeout_returns_idle_without_processing(self):
        quiet = np.zeros(1280, dtype=np.int16)
        stream = FakeStream([quiet] * 60)
        assistant = CaptureAssistant(stream, wake_engine=FakeWakeEngine(detects=False), clock=StepClock())
        assistant._listen_now.set()
        assistant.no_speech_seconds = 0.4
        assistant._audio_session()
        self.assertFalse(assistant.processed)
        self.assertEqual(assistant.state, AssistantState.IDLE)

    def test_maximum_command_duration_is_bounded(self):
        loud = np.full(1280, 2000, dtype=np.int16)
        stream = FakeStream([loud] * 20)
        assistant = CaptureAssistant(stream, wake_engine=FakeWakeEngine(), clock=StepClock())
        assistant.max_seconds = 0.32
        assistant.silence_seconds = 5
        assistant.cooldown_seconds = 0
        assistant._audio_session()
        self.assertTrue(assistant.processed)

    def test_same_phrase_prefixes_are_stripped(self):
        self.assertEqual(normalize_transcript("Hey Jarvis open YouTube"), ("open YouTube", True))
        self.assertEqual(normalize_transcript("Jarvis search YouTube for Jude Law"), ("search YouTube for Jude Law", True))
        self.assertEqual(normalize_transcript("Hey, Jarvis, open Notepad"), ("open Notepad", True))
        self.assertEqual(normalize_transcript("Jarvis. Open Calculator"), ("Open Calculator", True))

    def test_punctuated_wake_commands_reach_existing_tools(self):
        cases = {
            "Hey, Jarvis, open Notepad": ("open_application", "notepad"),
            "Hey, Jarvis, open Calculator": ("open_application", "calculator"),
            "Hey, Jarvis, open YouTube": ("open_website", "https://www.youtube.com"),
        }
        for transcript, (tool, target) in cases.items():
            with self.subTest(transcript=transcript):
                command, removed = normalize_transcript(transcript)
                route = route_command(command)
                self.assertTrue(removed)
                self.assertEqual(route["tool"], tool)
                self.assertIn(target, route["arguments"].values())

    def test_controls_do_not_start_duplicate_thread(self):
        assistant = AlwaysOnAssistant(wake_engine=FakeWakeEngine())
        assistant._thread = Mock(is_alive=Mock(return_value=True))
        assistant.start()
        self.assertIs(assistant._thread, assistant._thread)
        assistant.set_wake_enabled(False)
        assistant.set_muted(True)
        self.assertFalse(assistant.wake_enabled)
        self.assertTrue(assistant.muted)

    def test_state_display_failure_does_not_kill_audio_state_machine(self):
        assistant = AlwaysOnAssistant(wake_engine=FakeWakeEngine(), state_callback=Mock(side_effect=ValueError("display failed")))
        assistant._set_state(AssistantState.ERROR, "microphone unavailable")
        self.assertEqual(assistant.state, AssistantState.ERROR)
