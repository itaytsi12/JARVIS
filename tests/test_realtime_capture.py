"""Focused tests for voice/realtime_capture.py: the bridge between JARVIS's
single audio-owning thread and an ElevenLabs realtime STT session. No real
websocket/network activity -- a fake `stt_factory` stands in for
`ElevenLabsRealtimeSTT`.
"""
from __future__ import annotations

import threading
import time
import unittest

from voice.elevenlabs_realtime_stt import TranscriptEvent
from voice.realtime_capture import RealtimeSTTController
from voice.voice_perf import VoiceInteractionTimer


class FakeSession:
    """Stands in for ElevenLabsRealtimeSTT: no sockets, fully synchronous
    and controllable from the test."""

    instances: list["FakeSession"] = []

    def __init__(self, *, sample_rate, on_event, fail_connect=False, connect_delay=0.0):
        self.sample_rate = sample_rate
        self.on_event = on_event
        self.sent = []
        self.closed = False
        self.commit_calls = 0
        self._fail_connect = fail_connect
        self._connect_delay = connect_delay
        FakeSession.instances.append(self)

    def connect(self):
        if self._connect_delay:
            time.sleep(self._connect_delay)
        if self._fail_connect:
            raise RuntimeError("simulated connect failure")

    def send_audio(self, pcm_bytes, commit=False):
        self.sent.append((pcm_bytes, commit))

    def commit(self, timeout=5.0):
        self.commit_calls += 1
        return "open spotify"

    def close(self):
        self.closed = True

    def push_partial(self, text):
        self.on_event(TranscriptEvent("partial", text, {"message_type": "partial_transcript", "text": text}))


def _factory(**kwargs):
    def factory(*, sample_rate, on_event):
        return FakeSession(sample_rate=sample_rate, on_event=on_event, **kwargs)
    return factory


class RealtimeSTTControllerTests(unittest.TestCase):
    def setUp(self):
        FakeSession.instances.clear()

    def test_never_opens_an_audio_device_itself(self):
        """This module must only ever be FED frames -- it must not import
        or touch sounddevice/microphone APIs directly."""
        import voice.realtime_capture as module
        source = open(module.__file__, encoding="utf-8").read()
        self.assertNotIn("sounddevice", source)
        self.assertNotIn("RawInputStream", source)

    def test_frames_before_connect_are_buffered_then_flushed(self):
        perf = VoiceInteractionTimer()
        controller = RealtimeSTTController(perf=perf, stt_factory=_factory(connect_delay=0.1))
        controller.start()
        controller.feed(b"frame1")
        controller.feed(b"frame2")
        deadline = time.time() + 2
        while not controller.available and time.time() < deadline:
            time.sleep(0.01)
        self.assertTrue(controller.available)
        sent = FakeSession.instances[0].sent
        self.assertEqual([chunk for chunk, _ in sent], [b"frame1", b"frame2"])

    def test_connect_failure_leaves_controller_unavailable_never_raises(self):
        perf = VoiceInteractionTimer()
        controller = RealtimeSTTController(perf=perf, stt_factory=_factory(fail_connect=True))
        controller.start()
        deadline = time.time() + 2
        while controller._connect_error is None and time.time() < deadline:
            time.sleep(0.01)
        self.assertFalse(controller.available)
        # feed() after a failed connect must be a safe no-op.
        controller.feed(b"more audio")

    def test_partial_transcript_stabilizes_into_speculative_action(self):
        fired = []
        perf = VoiceInteractionTimer()
        controller = RealtimeSTTController(perf=perf, on_speculative_action=fired.append, stt_factory=_factory(), min_stable_partials=2)
        controller.start()
        deadline = time.time() + 2
        while not controller.available and time.time() < deadline:
            time.sleep(0.01)
        FakeSession.instances[0].push_partial("open spotify")
        FakeSession.instances[0].push_partial("open spotify")
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].route["tool"], "open_application")

    def test_destructive_partial_never_triggers_speculative_action(self):
        fired = []
        perf = VoiceInteractionTimer()
        controller = RealtimeSTTController(perf=perf, on_speculative_action=fired.append, stt_factory=_factory(), min_stable_partials=1)
        controller.start()
        deadline = time.time() + 2
        while not controller.available and time.time() < deadline:
            time.sleep(0.01)
        FakeSession.instances[0].push_partial("send a message to john")
        self.assertEqual(fired, [])

    def test_commit_and_close_closes_session_and_returns_transcript(self):
        perf = VoiceInteractionTimer()
        controller = RealtimeSTTController(perf=perf, stt_factory=_factory())
        controller.start()
        transcript = controller.commit_and_close(timeout=2)
        self.assertEqual(transcript, "open spotify")
        self.assertTrue(FakeSession.instances[0].closed)

    def test_commit_and_close_before_connect_completes_returns_none_and_closes(self):
        perf = VoiceInteractionTimer()
        controller = RealtimeSTTController(perf=perf, stt_factory=_factory(connect_delay=5.0))
        controller.start()
        transcript = controller.commit_and_close(timeout=0.05)
        self.assertIsNone(transcript)

    def test_close_is_safe_to_call_multiple_times(self):
        perf = VoiceInteractionTimer()
        controller = RealtimeSTTController(perf=perf, stt_factory=_factory())
        controller.close()
        controller.close()


if __name__ == "__main__":
    unittest.main()
