"""Concurrency regression tests for bug report 4: the speaker resource
must not deadlock or time out because a lower-priority utterance (stale
progress) is still holding it when a higher-priority one (the final
answer) wants to speak. `voice.speech_coordinator.SpeechCoordinator` is
the fix -- these tests exercise it directly with fake speak/stop
implementations (no real audio hardware, no network)."""
import threading
import time
import unittest
from unittest.mock import Mock, patch

from voice.speech_coordinator import PRIORITY_FINAL, PRIORITY_STATUS, SpeechCoordinator, get_speech_coordinator


class _FakeSpeechModule:
    """A fake `voice.text_to_speech`-shaped object: `speak` blocks until
    `stop()` is called or its (per-call) duration elapses, recording every
    call so tests can assert on ordering without real audio. Also mimics
    `voice.text_to_speech`'s own `speak_response` resource lock (a single
    real audio device -- exactly one call is ever inside the "playing"
    section at a time), so tests exercise the SAME "one owner" guarantee
    production gets from that lock, not a weaker one invented for the fake.
    """

    def __init__(self, duration: float = 0.2, durations: dict | None = None):
        self.duration = duration
        self.durations = durations or {}
        self.calls = []
        self._stop_events: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self._device_lock = threading.Lock()
        self._next_id = 0

    def speak(self, text, lang=None, **kwargs):
        with self._lock:
            call_id = self._next_id
            self._next_id += 1
            stop_event = threading.Event()
            self._stop_events[call_id] = stop_event
        duration = self.durations.get(text, self.duration)
        with self._device_lock:
            started = time.monotonic()
            self.calls.append(("start", text, call_id))
            interrupted = stop_event.wait(duration)
            elapsed = time.monotonic() - started
            self.calls.append(("end", text, call_id, "interrupted" if interrupted else "completed"))
        with self._lock:
            self._stop_events.pop(call_id, None)
        return {"success": True, "interrupted": interrupted, "elapsed": elapsed}

    def stop(self):
        with self._lock:
            events = list(self._stop_events.values())
        for event in events:
            event.set()


class SpeechCoordinatorTests(unittest.TestCase):
    def test_single_owner_plays_audio_at_a_time(self):
        """Two STATUS-priority calls never overlap -- the second waits for
        the first to actually finish before its own `speak` call starts."""
        fake = _FakeSpeechModule(duration=0.15)
        coordinator = SpeechCoordinator(fake)
        results = []

        def run(text):
            results.append(coordinator.speak(text, priority=PRIORITY_STATUS))

        t1 = threading.Thread(target=run, args=("first",))
        t2 = threading.Thread(target=run, args=("second",))
        t1.start()
        time.sleep(0.02)
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)

        starts = [c for c in fake.calls if c[0] == "start"]
        ends = [c for c in fake.calls if c[0] == "end"]
        self.assertEqual(len(starts), 2)
        # The second call's start must come after the first call's end --
        # i.e. they never overlap.
        first_end_index = fake.calls.index(ends[0])
        second_start_index = fake.calls.index(starts[1])
        self.assertLess(first_end_index, second_start_index)
        # Neither was preempted -- both are equal priority.
        self.assertFalse(results[0]["interrupted"])

    def test_final_priority_preempts_in_flight_lower_priority(self):
        """A FINAL utterance interrupts an in-flight STATUS one instead of
        waiting it out -- this is the actual fix for the reported
        `resource_timeout:speaker` bug."""
        # The stale STATUS utterance would hang for 5s if not preempted; the
        # FINAL one is a normal-length utterance that finishes on its own.
        fake = _FakeSpeechModule(duration=5.0, durations={"final answer": 0.1})
        coordinator = SpeechCoordinator(fake)
        status_result = {}

        def run_status():
            status_result.update(coordinator.speak("stale progress", priority=PRIORITY_STATUS))

        t_status = threading.Thread(target=run_status)
        t_status.start()
        time.sleep(0.05)  # ensure the status call is genuinely in flight

        started = time.monotonic()
        final_result = coordinator.speak("final answer", priority=PRIORITY_FINAL)
        elapsed = time.monotonic() - started

        t_status.join(timeout=2)

        # No normal follow-up should wait until the speaker-resource
        # timeout (or anywhere near the stale utterance's 5s duration).
        self.assertLess(elapsed, 1.0)
        self.assertTrue(status_result["interrupted"])
        self.assertFalse(final_result["interrupted"])

    def test_equal_priority_never_preempts_a_sibling(self):
        """An ordinary ack followed immediately by an ordinary result (both
        PRIORITY_STATUS) must NOT cut each other off -- only a strictly
        higher priority preempts."""
        fake = _FakeSpeechModule(duration=0.1)
        coordinator = SpeechCoordinator(fake)
        first_result = {}

        def run_first():
            first_result.update(coordinator.speak("ack", priority=PRIORITY_STATUS))

        t1 = threading.Thread(target=run_first)
        t1.start()
        time.sleep(0.02)
        coordinator.speak("result", priority=PRIORITY_STATUS)
        t1.join(timeout=2)

        self.assertFalse(first_result["interrupted"])

    def test_stop_failure_does_not_prevent_the_higher_priority_call(self):
        """Even if preemption itself raises, the higher-priority call must
        still go through -- a broken stop() must not block the answer."""
        fake = _FakeSpeechModule(duration=0.05)
        original_stop = fake.stop
        fake.stop = Mock(side_effect=RuntimeError("boom"))
        coordinator = SpeechCoordinator(fake)

        def run_status():
            coordinator.speak("progress", priority=PRIORITY_STATUS)

        t = threading.Thread(target=run_status)
        t.start()
        time.sleep(0.01)
        result = coordinator.speak("final", priority=PRIORITY_FINAL)
        t.join(timeout=2)
        self.assertTrue(result["success"])
        original_stop()  # cleanup, not asserted


class GetSpeechCoordinatorFreshResolutionTests(unittest.TestCase):
    """The coordinator singleton must always call through to whatever
    `voice.text_to_speech.speak`/`.stop` currently ARE, never a stale
    reference captured once -- this is what makes `unittest.mock.patch`
    on the module attribute reliable everywhere the coordinator is used,
    and it's the exact bug (a leaked stale-mock reference silently
    surviving across a whole test session) this design was fixed to
    avoid."""

    def test_repeated_calls_see_a_patched_speak_function(self):
        from voice import text_to_speech

        coordinator = get_speech_coordinator()
        with patch.object(text_to_speech, "speak", return_value={"success": True}) as mocked:
            coordinator.speak("hello", priority=PRIORITY_STATUS)
        mocked.assert_called_once()

        # A second, independent patch is picked up too -- proves the
        # coordinator never cached the first Mock.
        with patch.object(text_to_speech, "speak", return_value={"success": True}) as mocked_again:
            coordinator.speak("world", priority=PRIORITY_STATUS)
        mocked_again.assert_called_once()


if __name__ == "__main__":
    unittest.main()
