"""Real production wiring for the voice learning-approval flow (Phase 4):
`AlwaysOnAssistant.request_learning_approval` / `_capture_utterance_via_audio_thread`
/ `_run_pending_capture_session`. Uses the same FakeStream/FakeWakeEngine
style test doubles as tests/test_background_assistant.py -- no physical
microphone or real STT/TTS involved anywhere here.
"""
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from voice.background_assistant import AlwaysOnAssistant, AssistantState
from voice.learning_approval import LearningApprovalOutcome


class FakeWakeEngine:
    sample_rate = 16000
    frame_samples = 1280

    def load(self): pass
    def reset(self): pass
    def process(self, _frame): return (False, 0.0)


class FakeStream:
    """An effectively-infinite stream of frames so the reading side is only
    ever bounded by the code under test's own timeout logic, not by running
    out of fake frames."""

    def __init__(self, frame_fn):
        self.frame_fn = frame_fn
        self.active = False
        self.reads = 0

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *_):
        self.active = False

    def read(self, samples):
        self.reads += 1
        return self.frame_fn(self.reads).tobytes(), False


def _quiet(_n):
    return np.zeros(1280, dtype=np.int16)


def _loud_after(threshold):
    def frame_fn(n):
        return np.full(1280, 2000, dtype=np.int16) if n >= threshold else np.zeros(1280, dtype=np.int16)
    return frame_fn


def _make_assistant(frame_fn, **kwargs):
    stream = FakeStream(frame_fn)
    assistant = AlwaysOnAssistant(wake_engine=FakeWakeEngine(), stream_factory=lambda: stream, clock=time.monotonic, **kwargs)
    assistant.silence_seconds = 0.05
    assistant.speech_rms = 350
    assistant._stop.clear()
    return assistant, stream


def _run_audio_loop_until_stopped(assistant) -> threading.Thread:
    """Mirrors `_run()`'s real loop: keep re-entering `_audio_session()`
    until stop is requested. A `_restart` signal only makes ONE
    `_audio_session()` call yield early -- a pending capture request is
    picked up by the NEXT call, exactly like production."""
    def loop():
        while not assistant._stop.is_set():
            assistant._audio_session()
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    time.sleep(0.03)  # let the audio thread reach its normal idle session first
    return thread


class CaptureUtteranceViaAudioThreadTests(unittest.TestCase):
    def test_raises_if_called_from_the_audio_thread_itself(self):
        assistant, _ = _make_assistant(_quiet)
        assistant._thread = threading.current_thread()
        with self.assertRaises(RuntimeError):
            assistant._capture_utterance_via_audio_thread(1.0)

    def test_second_concurrent_request_is_rejected(self):
        assistant, _ = _make_assistant(_quiet)
        assistant._pending_capture = object()
        with self.assertRaises(RuntimeError):
            assistant._capture_utterance_via_audio_thread(1.0)

    def test_transcribes_captured_speech_using_the_single_mic_consumer(self):
        assistant, stream = _make_assistant(_loud_after(2))
        with patch.object(assistant, "_transcribe_frames", return_value="yes jarvis") as transcribe:
            worker = _run_audio_loop_until_stopped(assistant)
            transcript = assistant._capture_utterance_via_audio_thread(2.0)
            assistant._stop.set()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(transcript, "yes jarvis")
        transcribe.assert_called_once()
        self.assertFalse(stream.active, "mic must be released after the capture completes")
        self.assertIsNone(assistant._pending_capture)

    def test_times_out_and_releases_pending_state_when_nothing_is_heard(self):
        assistant, stream = _make_assistant(_quiet)
        worker = _run_audio_loop_until_stopped(assistant)
        transcript = assistant._capture_utterance_via_audio_thread(0.2)
        assistant._stop.set()
        worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertIsNone(transcript)
        self.assertIsNone(assistant._pending_capture)
        self.assertEqual(assistant.state, AssistantState.IDLE)


class RequestLearningApprovalTests(unittest.TestCase):
    def test_full_approval_flow_speaks_then_listens_then_returns_approved(self):
        assistant, stream = _make_assistant(_loud_after(2))
        spoken = []
        with patch("voice.text_to_speech.speak", side_effect=lambda text, **_: spoken.append(text)), \
             patch.object(assistant, "_transcribe_frames", return_value="yes jarvis"):
            worker = _run_audio_loop_until_stopped(assistant)
            result = assistant.request_learning_approval(timeout_seconds=2.0)
            assistant._stop.set()
            worker.join(timeout=2)

        self.assertEqual(result.outcome, LearningApprovalOutcome.APPROVED)
        self.assertEqual(len(spoken), 1)
        self.assertIn("learn", spoken[0].lower())

    def test_decline_flow_returns_declined(self):
        assistant, stream = _make_assistant(_loud_after(2))
        with patch("voice.text_to_speech.speak", side_effect=lambda text, **_: None), \
             patch.object(assistant, "_transcribe_frames", return_value="no jarvis"):
            worker = _run_audio_loop_until_stopped(assistant)
            result = assistant.request_learning_approval(timeout_seconds=2.0)
            assistant._stop.set()
            worker.join(timeout=2)

        self.assertEqual(result.outcome, LearningApprovalOutcome.DECLINED)

    def test_bare_yes_does_not_approve_and_keeps_listening_until_timeout(self):
        assistant, stream = _make_assistant(_loud_after(2))
        with patch("voice.text_to_speech.speak", side_effect=lambda text, **_: None), \
             patch.object(assistant, "_transcribe_frames", return_value="yeah"):
            worker = _run_audio_loop_until_stopped(assistant)
            result = assistant.request_learning_approval(timeout_seconds=0.3)
            assistant._stop.set()
            worker.join(timeout=2)

        self.assertEqual(result.outcome, LearningApprovalOutcome.TIMED_OUT)

    def test_state_returns_to_idle_after_timeout(self):
        assistant, stream = _make_assistant(_quiet)
        with patch("voice.text_to_speech.speak", side_effect=lambda text, **_: None):
            worker = _run_audio_loop_until_stopped(assistant)
            result = assistant.request_learning_approval(timeout_seconds=0.2)
            state_after_timeout = assistant.state
            assistant._stop.set()
            worker.join(timeout=2)

        self.assertEqual(result.outcome, LearningApprovalOutcome.TIMED_OUT)
        self.assertEqual(state_after_timeout, AssistantState.IDLE)
        self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
