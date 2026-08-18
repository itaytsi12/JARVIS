import threading,time,unittest
from unittest.mock import Mock,patch

import numpy as np

from voice.background_assistant import AlwaysOnAssistant, AssistantState,_redact_for_log
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

    def _start_command_warmup(self):
        pass

    def _process_capture(self, frames, **kwargs):
        self.processed = bool(frames)
        self.mic_released_before_processing = not self.stream.active
        self._set_state(AssistantState.PROCESSING)
        self._set_state(AssistantState.EXECUTING)
        self._set_state(AssistantState.SPEAKING)
        self._stop.set()


class BackgroundAssistantTests(unittest.TestCase):
    def test_multiword_spoken_credentials_are_redacted_but_later_command_remains(self):
        redacted=_redact_for_log("password is my very secret phrase then open Notepad")
        self.assertNotIn("very secret phrase",redacted);self.assertIn("open Notepad",redacted)

    def test_content_bearing_voice_commands_are_summarized_in_default_logs(self):
        for command in ("type private note","send Alex my private message","Hey Jarvis, please send Alex my private message"):
            with self.subTest(command=command),patch.dict("os.environ",{"DEBUG_REFERENCE_RESOLUTION":"false"}):
                redacted=_redact_for_log(command)
                self.assertNotIn("private",redacted);self.assertIn("content redacted",redacted)

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
        self.assertEqual(normalize_transcript("Hey Javis, open Notepad"), ("open Notepad", True))
        self.assertEqual(normalize_transcript("Hey Jovis, open YouTube"), ("open YouTube", True))

    def test_punctuated_wake_commands_reach_existing_tools(self):
        cases = {
            "Hey, Jarvis, open Notepad": ("open_application", "notepad"),
            "Hey, Jarvis, open Calculator": ("open_application", "calculator"),
            "Hey, Jarvis, open YouTube": ("open_website", "https://www.youtube.com"),
            "Hey, Jarvis, could you open YouTube for me": ("open_website", "https://www.youtube.com"),
            "Hey, Jarvis, please open TikTok": ("open_website", "https://www.tiktok.com"),
            "Hey, Jarvis, can you open Notepad please": ("open_application", "notepad"),
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

    def test_tray_start_warms_command_pipeline_without_joining_worker(self):
        assistant=AlwaysOnAssistant(wake_engine=FakeWakeEngine())
        worker=Mock()
        with patch("voice.background_assistant.threading.Thread",return_value=worker),patch.object(assistant,"_start_command_warmup") as warm:
            assistant.start()
        worker.start.assert_called_once_with();warm.assert_called_once_with()

    def test_speech_task_persists_spoken_response_for_reference_resolution(self):
        from brain.agent import agent_runtime
        assistant=AlwaysOnAssistant(wake_engine=FakeWakeEngine());spoken=threading.Event()
        old=agent_runtime.context.last_spoken_response
        try:
            with patch("voice.text_to_speech.speak",side_effect=lambda *args,**kwargs:spoken.set()):
                assistant._start_speech_task("The exact spoken response.","en")
                self.assertTrue(spoken.wait(1))
            self.assertEqual(agent_runtime.context.last_spoken_response,"The exact spoken response.")
        finally:
            agent_runtime.context.last_spoken_response=old

    def test_state_display_failure_does_not_kill_audio_state_machine(self):
        assistant = AlwaysOnAssistant(wake_engine=FakeWakeEngine(), state_callback=Mock(side_effect=ValueError("display failed")))
        assistant._set_state(AssistantState.ERROR, "microphone unavailable")
        self.assertEqual(assistant.state, AssistantState.ERROR)

    def test_question_worker_does_not_block_wake_loop_and_cancel_discards_answer(self):
        assistant=AlwaysOnAssistant(wake_engine=FakeWakeEngine());assistant.state=AssistantState.EXECUTING
        started=threading.Event();release=threading.Event();spoken=[]
        def slow_agent(*args,**kwargs):started.set();release.wait(2);return "I stopped that question."
        route={"type":"question","message":"who created Minecraft","intent_classification_ms":.1}
        with patch("brain.agent.run_agent",side_effect=slow_agent),patch("voice.text_to_speech.speak",side_effect=lambda text,**_:spoken.append(text)):
            before=time.perf_counter();assistant._start_question_task(route["message"],route,"iid","who created Minecraft",before)
            self.assertLess(time.perf_counter()-before,.2);self.assertTrue(started.wait(1))
            assistant.state=AssistantState.IDLE;release.set()
            deadline=time.perf_counter()+1
            while assistant._question_threads and time.perf_counter()<deadline:time.sleep(.01)
        self.assertEqual(spoken,[])

    def test_screen_analysis_uses_cancellable_background_action_path(self):
        assistant=AlwaysOnAssistant(wake_engine=FakeWakeEngine());started=threading.Event()
        with patch.object(assistant,"_start_cancellable_action_task",side_effect=lambda *args:started.set()),patch("voice.speech_to_text.transcribe_audio",return_value="what is on my screen"),patch("brain.router.route_command",return_value={"type":"tool","tool":"analyze_screen","arguments":{"question":"what is on my screen"}}):
            import tempfile,soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as stream:path=stream.name
            try:
                sf.write(path,__import__("numpy").zeros(1280,dtype="float32"),16000)
                with patch("voice.background_assistant.tempfile.NamedTemporaryFile") as temporary:
                    temporary.return_value.name=path;temporary.return_value.close.return_value=None
                    assistant._process_capture([__import__("numpy").zeros(1280,dtype="int16")])
            finally:
                from pathlib import Path;Path(path).unlink(missing_ok=True)
        self.assertTrue(started.is_set())
