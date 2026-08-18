# -*- coding: utf-8 -*-
"""Bilingual (VOICE_LANGUAGE=auto) voice UX coverage: per-utterance input
language detection, deterministic contextual English acknowledgements,
generic English acknowledgements for Hebrew input, ack/action overlap
timing, and duplicate-speech elimination -- all with TTS always English."""
import threading
import time
import unittest
from unittest.mock import patch

from voice.background_assistant import AlwaysOnAssistant
from voice.response_formatter import compose_contextual_ack, _GENERIC_ACKNOWLEDGEMENTS


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


class ContextualAckComposerTests(unittest.TestCase):
    """Deterministic, no-LLM contextual acknowledgements (item 7)."""

    def test_open_website(self):
        route = {"type": "tool", "tool": "open_website", "arguments": {"url": "https://www.youtube.com"}}
        self.assertEqual(compose_contextual_ack(route), "Opening YouTube, sir.")

    def test_open_application(self):
        route = {"type": "tool", "tool": "open_application", "arguments": {"app_name": "VS Code"}}
        self.assertEqual(compose_contextual_ack(route), "Opening VS Code, sir.")

    def test_close_application(self):
        route = {"type": "tool", "tool": "close_application", "arguments": {"app_name": "Notepad"}}
        self.assertEqual(compose_contextual_ack(route), "Closing Notepad, sir.")

    def test_play_song(self):
        route = {"type": "tool", "tool": "music_play", "arguments": {"intent": "PLAY_QUERY", "song": "Starboy"}}
        self.assertEqual(compose_contextual_ack(route), "Playing Starboy, sir.")

    def test_play_playlist(self):
        route = {"type": "tool", "tool": "music_play", "arguments": {"intent": "PLAY_PLAYLIST", "playlist": "Gym"}}
        self.assertEqual(compose_contextual_ack(route), "Okay, playing your Gym playlist, sir.")

    def test_search_query(self):
        route = {"type": "tool", "tool": "open_website", "arguments": {"url": "https://www.google.com/search?q=RTX+5090"}}
        self.assertEqual(compose_contextual_ack(route), "Okay, I'm searching for RTX 5090, sir.")

    def test_cloud_plan_type_gets_generic_check_phrase(self):
        route = {"type": "plan", "message": "what's the weather tomorrow?"}
        self.assertEqual(compose_contextual_ack(route), "I'll check that, sir.")

    def test_unknown_tool_falls_back_to_generic_on_it(self):
        route = {"type": "tool", "tool": "some_future_tool", "arguments": {}}
        self.assertEqual(compose_contextual_ack(route), "On it, sir.")

    def test_never_claims_completion(self):
        # Good: "Opening YouTube, sir." Bad: "YouTube is open, sir."
        route = {"type": "tool", "tool": "open_website", "arguments": {"url": "https://www.youtube.com"}}
        ack = compose_contextual_ack(route)
        self.assertNotIn("is open", ack)
        self.assertNotIn("successfully", ack.lower())
        self.assertNotIn("done", ack.lower())


class AutoModeBilingualDispatchTests(unittest.TestCase):
    """VOICE_LANGUAGE=auto: same running process handles English then
    Hebrew then English again, per utterance, with no restart and no
    manual config change."""

    def _run(self, transcript, route, outcome, message):
        assistant = AlwaysOnAssistant(wake_engine=FakeWakeEngine())
        spoken = []

        def fake_speak(text, **kwargs):
            spoken.append(text)

        def fake_run_agent(command, **kwargs):
            out = kwargs.get("execution_outcome")
            if out is not None:
                out.update(outcome)
            return {"message": message}

        with patch.dict("os.environ", {"VOICE_LANGUAGE": "auto"}), \
             patch("brain.router.route_command", return_value=route), \
             patch("brain.agent.run_agent", side_effect=fake_run_agent), \
             patch("voice.text_to_speech.speak", side_effect=fake_speak):
            assistant._process_capture([], elevenlabs_transcript=transcript)
            _wait_for(lambda: len(spoken) >= 1)
            time.sleep(0.15)
        return spoken

    def test_auto_mode_english_utterance_gets_contextual_ack(self):
        spoken = self._run(
            "open YouTube",
            {"type": "tool", "tool": "open_website", "arguments": {"url": "https://www.youtube.com"}},
            {"success": True, "verified": True},
            "Opened https://www.youtube.com",
        )
        self.assertEqual(spoken, ["Opening YouTube, sir."])

    def test_auto_mode_hebrew_utterance_gets_generic_ack_no_hebrew_text(self):
        spoken = self._run(
            "פתח מוזיקה",
            {"type": "tool", "tool": "open_music", "arguments": {}},
            {"success": True, "verified": True},
            "פתחתי את מוזיקה",
        )
        self.assertEqual(len(spoken), 1)
        self.assertIn(spoken[0], _GENERIC_ACKNOWLEDGEMENTS)
        self.assertTrue(spoken[0].isascii())

    def test_auto_mode_alternates_english_hebrew_english_hebrew_same_process(self):
        assistant = AlwaysOnAssistant(wake_engine=FakeWakeEngine())
        spoken_sequence = []

        def fake_run_agent(command, **kwargs):
            out = kwargs.get("execution_outcome")
            if out is not None:
                out.update({"success": True, "verified": True})
            return {"message": "done"}

        def fake_speak(text, **kwargs):
            spoken_sequence.append(text)

        turns = [
            ("open YouTube", {"type": "tool", "tool": "open_website", "arguments": {"url": "https://www.youtube.com"}}),
            ("פתח מוזיקה", {"type": "tool", "tool": "open_music", "arguments": {}}),
            ("what song is playing?", {"type": "tool", "tool": "music_now_playing", "arguments": {"aspect": "song"}}),
            ("שיר הבא", {"type": "tool", "tool": "music_next", "arguments": {}}),
        ]
        with patch.dict("os.environ", {"VOICE_LANGUAGE": "auto"}), \
             patch("brain.agent.run_agent", side_effect=fake_run_agent), \
             patch("voice.text_to_speech.speak", side_effect=fake_speak):
            for transcript, route in turns:
                with patch("brain.router.route_command", return_value=route):
                    assistant._process_capture([], elevenlabs_transcript=transcript)
                    _wait_for(lambda: len(spoken_sequence) >= len(turns) and False, timeout=0.3)
            _wait_for(lambda: len(spoken_sequence) >= 4, timeout=2)
        self.assertEqual(len(spoken_sequence), 4)
        self.assertEqual(spoken_sequence[0], "Opening YouTube, sir.")
        self.assertIn(spoken_sequence[1], _GENERIC_ACKNOWLEDGEMENTS)
        self.assertEqual(spoken_sequence[2], "Okay, checking, sir.")
        self.assertIn(spoken_sequence[3], _GENERIC_ACKNOWLEDGEMENTS)
        for text in spoken_sequence:
            self.assertTrue(text.isascii(), f"non-English text reached TTS: {text!r}")

    def test_english_failure_preserves_specific_tool_message_not_generic(self):
        spoken = self._run(
            "play Starboy",
            {"type": "tool", "tool": "music_play", "arguments": {"intent": "PLAY_QUERY", "song": "Starboy"}},
            {"success": False, "verified": False},
            "I couldn't confirm playback, sir.",
        )
        self.assertEqual(spoken[0], "Playing Starboy, sir.")
        self.assertEqual(spoken[-1], "I couldn't confirm playback, sir.")
        self.assertEqual(len(spoken), 2)

    def test_hebrew_failure_uses_generic_message_never_specific_tool_text(self):
        spoken = self._run(
            "נגן את הפלייליסט ישראלי",
            {"type": "tool", "tool": "music_play", "arguments": {"intent": "PLAY_PLAYLIST", "playlist": "ישראלי"}},
            {"success": False, "verified": False},
            "I couldn't find the playlist 'ישראלי', sir.",
        )
        self.assertEqual(spoken[-1], "I couldn't complete that action, sir.")
        for text in spoken:
            self.assertTrue(text.isascii())


class AckTimingTests(unittest.TestCase):
    def test_ack_starts_before_action_completes_and_action_does_not_wait_for_tts(self):
        release = threading.Event()
        started = threading.Event()
        spoken = []

        def fake_speak(text, **kwargs):
            spoken.append(text)

        def slow_run_agent(command, **kwargs):
            started.set()
            release.wait(2)
            out = kwargs.get("execution_outcome")
            if out is not None:
                out.update({"success": True, "verified": True})
            return {"message": "Opened notepad"}

        assistant = AlwaysOnAssistant(wake_engine=FakeWakeEngine())
        with patch.dict("os.environ", {"VOICE_LANGUAGE": "en"}), \
             patch("brain.router.route_command", return_value={"type": "tool", "tool": "open_application", "arguments": {"app_name": "notepad"}}), \
             patch("brain.agent.run_agent", side_effect=slow_run_agent), \
             patch("voice.text_to_speech.speak", side_effect=fake_speak):
            worker = threading.Thread(target=lambda: assistant._process_capture([], elevenlabs_transcript="open notepad"))
            worker.start()
            self.assertTrue(started.wait(1), "action should start promptly")
            self.assertTrue(_wait_for(lambda: len(spoken) >= 1, timeout=1), "ack should speak before the (still-running) action finishes")
            self.assertEqual(spoken[0], "Opening Notepad, sir.")
            release.set()
            worker.join(2)


if __name__ == "__main__":
    unittest.main()
