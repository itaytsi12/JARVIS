import threading
import time
import unittest
from unittest.mock import patch

from voice.background_assistant import AlwaysOnAssistant
from voice.response_formatter import (
    generic_acknowledgement,
    generic_failure_message,
    _GENERIC_ACKNOWLEDGEMENTS,
)


class FakeWakeEngine:
    sample_rate = 16000
    frame_samples = 1280


def _wait_for(predicate, timeout=2.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class GenericPhraseHelperTests(unittest.TestCase):
    def test_acknowledgement_is_always_english_and_generic(self):
        for _ in range(20):
            phrase = generic_acknowledgement()
            self.assertIn(phrase, _GENERIC_ACKNOWLEDGEMENTS)
            self.assertTrue(phrase.isascii())

    def test_failure_message_is_fixed_and_english(self):
        self.assertEqual(generic_failure_message(), "I couldn't complete that action, sir.")
        self.assertTrue(generic_failure_message().isascii())


class HebrewModeTtsTests(unittest.TestCase):
    """Language UX rule: TTS speaks English only, always. Hebrew-mode
    commands must get an immediate generic English ack (started before/
    during action execution, never waiting on it), no further speech on
    success, and exactly one generic English failure message on failure --
    never the raw response text, which may contain the recognized Hebrew
    entities the action itself must still receive untouched."""

    def _run(self, transcript, route, run_agent_outcome, run_agent_message):
        assistant = AlwaysOnAssistant(wake_engine=FakeWakeEngine())
        spoken_calls = []
        speak_lock = threading.Lock()

        def fake_speak(text, **kwargs):
            with speak_lock:
                spoken_calls.append(text)

        def fake_run_agent(command, **kwargs):
            outcome = kwargs.get("execution_outcome")
            if outcome is not None:
                outcome.update(run_agent_outcome)
            return {"message": run_agent_message}

        with patch.dict("os.environ", {"VOICE_LANGUAGE": "he"}), \
             patch("brain.router.route_command", return_value=route), \
             patch("brain.agent.run_agent", side_effect=fake_run_agent), \
             patch("voice.text_to_speech.speak", side_effect=fake_speak):
            assistant._process_capture([], elevenlabs_transcript=transcript)
            _wait_for(lambda: len(spoken_calls) >= 1)
            # Give any (incorrect) second speech task a chance to fire too.
            time.sleep(0.15)
        return spoken_calls

    def test_success_speaks_only_one_generic_english_ack_no_hebrew(self):
        spoken = self._run(
            transcript="נגן שני משוגעים",
            route={"type": "tool", "tool": "play_music", "arguments": {"song": "שני משוגעים"}},
            run_agent_outcome={"success": True, "verified": True},
            run_agent_message="מנגן עכשיו שני משוגעים",
        )
        self.assertEqual(len(spoken), 1, spoken)
        self.assertIn(spoken[0], _GENERIC_ACKNOWLEDGEMENTS)
        for text in spoken:
            self.assertTrue(text.isascii(), f"non-ASCII (possibly Hebrew) text reached TTS: {text!r}")

    def test_failure_speaks_ack_then_exactly_one_generic_failure_message(self):
        spoken = self._run(
            transcript="נגן את הפלייליסט ישראלי",
            route={"type": "tool", "tool": "play_music", "arguments": {"playlist": "ישראלי"}},
            run_agent_outcome={"success": False, "verified": False},
            run_agent_message="לא הצלחתי למצוא את הפלייליסט ישראלי",
        )
        self.assertEqual(len(spoken), 2, spoken)
        self.assertIn(spoken[0], _GENERIC_ACKNOWLEDGEMENTS)
        self.assertEqual(spoken[1], "I couldn't complete that action, sir.")
        for text in spoken:
            self.assertTrue(text.isascii(), f"non-ASCII (possibly Hebrew) text reached TTS: {text!r}")

    def test_english_mode_still_uses_contextual_acknowledgement_and_response(self):
        """Regression: VOICE_LANGUAGE=en must be completely unaffected by
        the Hebrew-mode branch -- same contextual ack + contextual final
        response as before this change."""
        assistant = AlwaysOnAssistant(wake_engine=FakeWakeEngine())
        spoken_calls = []

        def fake_speak(text, **kwargs):
            spoken_calls.append(text)

        def fake_run_agent(command, **kwargs):
            outcome = kwargs.get("execution_outcome")
            if outcome is not None:
                outcome.update({"success": True, "verified": True})
            return {"message": "Opened https://www.youtube.com"}

        with patch.dict("os.environ", {"VOICE_LANGUAGE": "en"}), \
             patch("brain.router.route_command", return_value={"type": "tool", "tool": "open_website", "arguments": {"url": "https://www.youtube.com"}}), \
             patch("brain.agent.run_agent", side_effect=fake_run_agent), \
             patch("voice.text_to_speech.speak", side_effect=fake_speak):
            assistant._process_capture([], elevenlabs_transcript="open YouTube")
            _wait_for(lambda: len(spoken_calls) >= 1)
            time.sleep(0.15)
        self.assertNotIn("Okay, on it, sir.", spoken_calls)
        self.assertNotIn("I couldn't complete that action, sir.", spoken_calls)
        self.assertTrue(any("YouTube" in text or "youtube" in text for text in spoken_calls), spoken_calls)

    def test_ack_starts_before_run_agent_returns(self):
        release = threading.Event()
        started = threading.Event()
        spoken_calls = []

        def fake_speak(text, **kwargs):
            spoken_calls.append(text)

        def slow_run_agent(command, **kwargs):
            started.set()
            release.wait(2)
            outcome = kwargs.get("execution_outcome")
            if outcome is not None:
                outcome.update({"success": True, "verified": True})
            return {"message": "פתחתי את יוטיוב"}

        assistant = AlwaysOnAssistant(wake_engine=FakeWakeEngine())
        with patch.dict("os.environ", {"VOICE_LANGUAGE": "he"}), \
             patch("brain.router.route_command", return_value={"type": "tool", "tool": "open_website", "arguments": {"url": "https://www.youtube.com"}}), \
             patch("brain.agent.run_agent", side_effect=slow_run_agent), \
             patch("voice.text_to_speech.speak", side_effect=fake_speak):
            worker = threading.Thread(target=lambda: assistant._process_capture([], elevenlabs_transcript="פתח יוטיוב"))
            worker.start()
            self.assertTrue(started.wait(1), "run_agent should have started")
            self.assertTrue(_wait_for(lambda: len(spoken_calls) >= 1, timeout=1), "ack should speak before run_agent returns")
            self.assertIn(spoken_calls[0], _GENERIC_ACKNOWLEDGEMENTS)
            release.set()
            worker.join(2)


if __name__ == "__main__":
    unittest.main()
