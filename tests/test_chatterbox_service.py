import json
import os
import io
import sys
import threading
import time
import unittest
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from urllib.request import Request, urlopen

from voice import chatterbox_service
from voice.tts import chatterbox_tts
from voice.tts import openai_tts
from voice import text_to_speech
from voice.response_formatter import format_spoken_response
from brain.local_planner import create_local_plan


class ChatterboxServiceTests(unittest.TestCase):
    def setUp(self):
        chatterbox_service._SERVICE_MODEL = object()
        chatterbox_service._SERVICE_STATE = "ready"
        chatterbox_service._SERVICE_ERROR = None
        chatterbox_service._SERVICE_DEVICE = "cpu"
        self.calls = []
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), chatterbox_service.Handler
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_health_reports_ready_process(self):
        with urlopen(f"{self.base_url}/health", timeout=2) as response:
            payload = json.load(response)
        self.assertEqual(payload["status"], "ready")
        self.assertIsInstance(payload["pid"], int)
        self.assertEqual(payload["device"], "cpu")

    def test_synthesize_forwards_english_and_hebrew(self):
        def record(text, language_id=None):
            self.calls.append((text, language_id))

        with patch.object(chatterbox_service, "synth_and_play", side_effect=record):
            for text, language_id in (("Hello", "en"), ("שלום", "he")):
                body = json.dumps(
                    {"text": text, "language_id": language_id}
                ).encode("utf-8")
                request = Request(
                    f"{self.base_url}/synthesize",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=2) as response:
                    self.assertEqual(json.load(response), {"status": "ok"})

        self.assertEqual(self.calls, [("Hello", "en"), ("שלום", "he")])


class ChatterboxClientTests(unittest.TestCase):
    def test_cpu_service_uses_low_latency_fallback(self):
        with patch.object(
            chatterbox_tts, "_get_health", return_value={"status": "ready", "device": "cpu"}
        ):
            self.assertFalse(chatterbox_tts.is_low_latency_ready())

    def test_accelerated_service_is_low_latency_ready(self):
        with patch.object(
            chatterbox_tts, "_get_health", return_value={"status": "ready", "device": "cuda"}
        ):
            self.assertTrue(chatterbox_tts.is_low_latency_ready())


class NaturalVoiceTests(unittest.TestCase):
    def test_openai_wav_playback_uses_soundfile_and_sounddevice(self):
        audio = unittest.mock.Mock()
        audio.shape = (50400,)
        audio.min.return_value = -0.4
        audio.max.return_value = 0.5
        soundfile = unittest.mock.Mock()
        soundfile.read.return_value = (audio, 24000)
        sounddevice = unittest.mock.Mock()

        with patch.dict(
            sys.modules,
            {"soundfile": soundfile, "sounddevice": sounddevice},
        ):
            openai_tts.play_wav("speech.wav")

        soundfile.read.assert_called_once_with("speech.wav", dtype="float32")
        sounddevice.play.assert_called_once_with(audio, 24000)
        sounddevice.wait.assert_called_once()

    def test_single_youtube_search_confirms_completed_action(self):
        spoken = format_spoken_response(
            "search youtube for Jude Law",
            {
                "type": "tool",
                "tool": "open_website",
                "arguments": {
                    "url": "https://www.youtube.com/results?search_query=Jude+Law"
                },
            },
            "Opened YouTube search in Google Chrome.",
            lang="en",
        )
        self.assertEqual(spoken, "I opened YouTube and searched for Jude Law.")

    def test_multistep_reply_says_what_was_done(self):
        actions = create_local_plan("open youtube and search youtube for Jude Law")
        spoken = format_spoken_response(
            "open youtube and search youtube for Jude Law",
            {"type": "local_plan", "actions": actions},
            "Opened YouTube.\nOpened a YouTube search.",
            lang="en",
        )
        self.assertEqual(spoken, "I opened YouTube and searched YouTube for Jude Law.")

    def test_neural_tts_request_uses_agent_voice_prompt_and_wav(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"RIFF-test"

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch.object(
            openai_tts, "urlopen", return_value=Response()
        ) as mocked:
            self.assertEqual(openai_tts.synthesize_wav("I opened YouTube."), b"RIFF-test")

        request = mocked.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["response_format"], "wav")
        self.assertIn("natural pacing", payload["instructions"])
        self.assertEqual(payload["voice"], "cedar")

    def test_ready_service_is_reused_without_spawning(self):
        with patch.object(chatterbox_tts, "_get_health_status", return_value="ready"), patch.object(
            chatterbox_tts.subprocess, "Popen"
        ) as popen:
            self.assertTrue(chatterbox_tts._start_service(timeout=0.1))
        popen.assert_not_called()

    def test_background_start_returns_without_waiting_for_model(self):
        started = threading.Event()
        release = threading.Event()

        def slow_start():
            started.set()
            release.wait(timeout=2)
            return True

        with patch.object(chatterbox_tts, "_get_health_status", return_value=None), patch.object(
            chatterbox_tts, "_start_service", side_effect=slow_start
        ):
            before = time.perf_counter()
            chatterbox_tts.start_service_background()
            elapsed = time.perf_counter() - before
            self.assertTrue(started.wait(timeout=1))
            self.assertLess(elapsed, 0.25)
            release.set()

    def test_client_sends_explicit_language_id(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        response = Response()
        response.read = lambda: b'{"status":"ok"}'

        with patch.object(chatterbox_tts, "urlopen", return_value=response) as mocked:
            chatterbox_tts._call_service_synthesize("שלום", lang="he")

        sent = json.loads(mocked.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(sent, {"text": "שלום", "language_id": "he"})


class TTSProviderSelectionTests(unittest.TestCase):
    def setUp(self):
        self.chatterbox = unittest.mock.Mock()
        self.openai = unittest.mock.Mock()
        self.openai.is_available.return_value = True
        self.engine = unittest.mock.Mock()

    def provider_patches(self):
        return (
            patch.object(text_to_speech, "_chatterbox_provider", self.chatterbox),
            patch.object(text_to_speech, "_chatterbox_available", True),
            patch.object(text_to_speech, "_openai_provider", self.openai),
            patch.object(text_to_speech, "_pyttsx3_available", True),
            patch.object(text_to_speech, "_init_pyttsx3", return_value=self.engine),
        )

    def run_with_patches(self, provider, callback):
        patches = self.provider_patches()
        with patch.dict(os.environ, {"TTS_PROVIDER": provider}, clear=False):
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                callback()

    def test_explicit_chatterbox_never_calls_openai(self):
        self.run_with_patches(
            "chatterbox", lambda: text_to_speech.speak("Local only", lang="en")
        )
        self.chatterbox.speak.assert_called_once_with("Local only", lang="en")
        self.openai.speak.assert_not_called()

    def test_explicit_openai_uses_cedar_provider(self):
        self.run_with_patches("openai", lambda: text_to_speech.speak("Natural voice"))
        self.openai.speak.assert_called_once_with("Natural voice")
        self.chatterbox.speak.assert_not_called()

    def test_explicit_pyttsx3_uses_only_fallback_engine(self):
        self.run_with_patches("pyttsx3", lambda: text_to_speech.speak("Offline"))
        self.engine.say.assert_called_once_with("Offline")
        self.engine.runAndWait.assert_called_once()
        self.openai.speak.assert_not_called()
        self.chatterbox.speak.assert_not_called()

    def test_auto_prefers_fast_ready_chatterbox(self):
        self.chatterbox.is_low_latency_ready.return_value = True
        self.run_with_patches("auto", lambda: text_to_speech.speak("Fast local"))
        self.chatterbox.speak.assert_called_once()
        self.openai.speak.assert_not_called()

    def test_auto_uses_openai_when_chatterbox_is_not_fast_ready(self):
        self.chatterbox.is_low_latency_ready.return_value = False
        self.run_with_patches("auto", lambda: text_to_speech.speak("Natural fallback"))
        self.openai.speak.assert_called_once_with("Natural fallback")
        self.chatterbox.speak.assert_not_called()

    def test_startup_labels_are_exact(self):
        expected = {
            "openai": "TTS provider: OpenAI cedar",
            "chatterbox": "TTS provider: Chatterbox local",
            "pyttsx3": "TTS provider: pyttsx3 fallback",
        }
        for provider, label in expected.items():
            with self.subTest(provider=provider):
                output = io.StringIO()
                patches = self.provider_patches()
                with patch.dict(os.environ, {"TTS_PROVIDER": provider}, clear=False):
                    with patches[0], patches[1], patches[2], patches[3], patches[4]:
                        with redirect_stdout(output):
                            text_to_speech.start_background()
                self.assertIn(label, output.getvalue())


if __name__ == "__main__":
    unittest.main()
