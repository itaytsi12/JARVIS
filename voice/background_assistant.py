"""Always-on voice orchestration around JARVIS's existing voice pipeline."""
from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Callable

from .text_normalizer import normalize_transcript
from .wake_word import OpenWakeWordEngine
from training_data.sanitizer import sanitize_text, sanitize_user_request


def _redact_for_log(text: str) -> str:
    value=str(text)
    if os.getenv("DEBUG_REFERENCE_RESOLUTION","false").lower() not in {"1","true","yes","on"}:
        content_command=re.search(r"\b(?:send|message|tell|type|write)\b",value,re.I)
        if content_command:return f"{content_command.group(0).strip().lower()} request <content redacted; length={len(value)}>"
    return sanitize_text(value)


def _route_for_log(route) -> str:
    if not isinstance(route, dict):
        return type(route).__name__
    route_type = route.get("type")
    if route_type == "tool":
        return f"type=tool tool={route.get('tool')}"
    actions = route.get("actions") or []
    tools = [getattr(action, "tool", None) or (action.get("tool") if isinstance(action, dict) else None) for action in actions]
    return f"type={route_type} tools={[tool for tool in tools if tool]}"


class AssistantState(str, Enum):
    IDLE = "IDLE"
    WAKE_DETECTED = "WAKE_DETECTED"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    EXECUTING = "EXECUTING"
    SPEAKING = "SPEAKING"
    INTERRUPTED_LISTENING = "INTERRUPTED_LISTENING"
    ERROR = "ERROR"


class AlwaysOnAssistant:
    def __init__(self, wake_engine=None, state_callback: Callable | None = None, stream_factory=None, clock=time.monotonic):
        self.wake_engine = wake_engine or OpenWakeWordEngine()
        self.state_callback = state_callback
        self.stream_factory = stream_factory
        self.clock = clock
        self.state = AssistantState.IDLE
        self.status_detail = "Starting"
        self.wake_enabled = os.getenv("WAKE_WORD_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.muted = False
        self.silence_seconds = float(os.getenv("COMMAND_SILENCE_SECONDS", "1.0"))
        self.max_seconds = float(os.getenv("COMMAND_MAX_SECONDS", "15"))
        self.no_speech_seconds = float(os.getenv("COMMAND_NO_SPEECH_SECONDS", "4"))
        self.cooldown_seconds = float(os.getenv("WAKE_COOLDOWN_SECONDS", "1.25"))
        self.speech_rms = float(os.getenv("COMMAND_SPEECH_RMS", "350"))
        self.debug_audio = os.getenv("JARVIS_DEBUG_AUDIO", "false").lower() in {"1", "true", "yes"}
        self.performance_debug = os.getenv("PERFORMANCE_DEBUG", "false").lower() in {"1", "true", "yes", "on"}
        self._perf_started = None
        self._stop = threading.Event()
        self._listen_now = threading.Event()
        self._restart = threading.Event()
        self._thread = None
        self._mic_lock = threading.Lock()
        self._warmup_started = threading.Event()
        self._question_threads = set()
        self._question_threads_lock = threading.Lock()
        self.log = logging.getLogger("jarvis.background")

    def _perf(self, stage: str, when: float | None = None) -> None:
        if not self.performance_debug:
            return
        timestamp = self.clock() if when is None else when
        if self._perf_started is None:
            self._perf_started = timestamp
        self.log.info("PERF stage=%s elapsed_ms=%.1f", stage, (timestamp - self._perf_started) * 1000)

    def _set_state(self, state: AssistantState, detail: str | None = None) -> None:
        self.state = state
        self.status_detail = detail or state.value.title()
        self.log.info("State: %s%s", state.value, f" - {detail}" if detail else "")
        if self.state_callback:
            try:
                self.state_callback(state, detail)
            except Exception:
                self.log.exception("State display callback failed")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="jarvis-audio", daemon=True)
        self._thread.start()
        self._start_command_warmup()

    def stop(self) -> None:
        self._stop.set()
        self._listen_now.set()
        try:
            from brain.task_supervisor import cancel_read_only_tasks
            cancel_read_only_tasks()
        except Exception:
            self.log.exception("Could not cancel active read-only tasks")
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)

    def restart(self) -> None:
        self._restart.set()

    def request_listen(self) -> None:
        if self.state in {AssistantState.IDLE, AssistantState.ERROR}:
            self._listen_now.set()

    def set_wake_enabled(self, enabled: bool) -> None:
        self.wake_enabled = bool(enabled)
        if enabled and self.state is AssistantState.ERROR:
            self._restart.set()

    def set_muted(self, muted: bool) -> None:
        self.muted = bool(muted)

    def _default_stream(self):
        from .audio_input import open_jarvis_microphone
        return open_jarvis_microphone(self.wake_engine.sample_rate,self.wake_engine.frame_samples)

    def _start_command_warmup(self) -> None:
        if self._warmup_started.is_set():
            return
        self._warmup_started.set()

        def warm() -> None:
            try:
                from .speech_to_text import warm_up
                warm_up()
                # Import the lazy command pipeline while the user is speaking.
                from brain.agent import run_agent  # noqa: F401
                from brain.router import route_command  # noqa: F401
            except Exception:
                self.log.exception("Command pipeline warm-up failed")

        threading.Thread(target=warm, name="jarvis-command-warmup", daemon=True).start()

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                self.wake_engine.load()
                self._audio_session()
                failures = 0
            except Exception as exc:
                failures += 1
                self.log.exception("Audio loop failed")
                self._set_state(AssistantState.ERROR, str(exc))
                if failures >= 3:
                    self._restart.wait()
                    self._restart.clear()
                    failures = 0
                else:
                    self._stop.wait(min(2 ** (failures - 1), 4))

    def _audio_session(self) -> None:
        import numpy as np

        factory = self.stream_factory or self._default_stream
        frame_seconds = self.wake_engine.frame_samples / self.wake_engine.sample_rate
        ring = deque(maxlen=max(1, int(1.2 / frame_seconds)))
        capture = []
        listen_started = last_speech = None
        speech_after_wake = False
        self.wake_engine.reset()
        if self.state is not AssistantState.SPEAKING or self._stop.is_set():
            self._set_state(AssistantState.IDLE, "Wake word ready" if self.wake_enabled else "Wake word disabled")

        with self._mic_lock, factory() as stream:
            while not self._stop.is_set() and not self._restart.is_set():
                raw, overflowed = stream.read(self.wake_engine.frame_samples)
                if overflowed:
                    self.log.warning("Microphone input overflow")
                frame = np.frombuffer(raw, dtype=np.int16).copy()
                now = self.clock()
                if self.state is AssistantState.SPEAKING:
                    detected,score=self.wake_engine.process(frame) if self.wake_enabled else (False,0.0)
                    speaking_threshold=float(os.getenv("JARVIS_TTS_WAKE_THRESHOLD","0.8"))
                    if detected and score>=speaking_threshold:
                        from .text_to_speech import stop as stop_speech
                        stop_started=time.perf_counter();stop_speech()
                        try:
                            from brain.agent import agent_runtime
                            agent_runtime.context.speech_interrupted=True
                        except Exception:pass
                        self.log.info("Barge-in: wake_to_tts_stop_ms=%.1f",(time.perf_counter()-stop_started)*1000)
                        self._set_state(AssistantState.INTERRUPTED_LISTENING,"Speech interrupted")
                        # The ring belongs to the earlier command/listening phase.
                        # Reusing it here can feed stale user/TTS audio into STT.
                        capture=[];listen_started,last_speech,speech_after_wake=now,None,False
                        self._set_state(AssistantState.LISTENING,"Waiting for correction or command")
                    continue
                if self.state is AssistantState.IDLE:
                    ring.append(frame)
                    manual = self._listen_now.is_set()
                    if manual:
                        self._listen_now.clear()
                    detected = False
                    if self.wake_enabled and not manual:
                        detected, _ = self.wake_engine.process(frame)
                    if manual or detected:
                        self._perf_started = None
                        self._perf("wake_detected", now)
                        self._set_state(AssistantState.WAKE_DETECTED, "Manual listen" if manual else "Jarvis detected")
                        self.log.info("Wake detected")
                        self._start_command_warmup()
                        capture = list(ring)
                        listen_started, last_speech, speech_after_wake = now, None, False
                        self._set_state(AssistantState.LISTENING)
                        self._perf("recording_started", now)
                    continue
                if self.state is not AssistantState.LISTENING:
                    continue
                capture.append(frame)
                rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
                if rms >= self.speech_rms:
                    speech_after_wake, last_speech = True, now
                elapsed = now - listen_started
                finished = speech_after_wake and last_speech is not None and now - last_speech >= self.silence_seconds
                no_speech = not speech_after_wake and elapsed >= self.no_speech_seconds
                if finished or no_speech or elapsed >= self.max_seconds:
                    if last_speech is not None:
                        self._perf("speech_ended", last_speech)
                    self._perf("recording_stopped", now)
                    break

        self._restart.clear()
        if self._stop.is_set():
            return
        if not capture or not speech_after_wake:
            self._set_state(AssistantState.IDLE, "No command heard")
            return
        self._process_capture(capture)
        self._stop.wait(self.cooldown_seconds)
        self.wake_engine.reset()
        if self.state is not AssistantState.SPEAKING or self._stop.is_set():
            self._set_state(AssistantState.IDLE, "Wake word ready" if self.wake_enabled else "Wake word disabled")

    def _process_capture(self, frames) -> None:
        import numpy as np
        import soundfile as sf

        audio = np.concatenate(frames).astype("float32") / 32768.0
        temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        path = Path(temp.name)
        temp.close()
        try:
            sf.write(path, audio, self.wake_engine.sample_rate)
            self._set_state(AssistantState.PROCESSING, "Transcribing")
            from .speech_to_text import transcribe_audio
            self._perf("stt_started")
            transcript = transcribe_audio(str(path))
            self._perf("stt_completed")
            self.log.info("Transcribed command: %s", _redact_for_log(transcript))
            command, _ = normalize_transcript(transcript)
            self._perf("normalization_completed")
            self.log.info("Normalized command: %s", _redact_for_log(command))
            if not command:
                self._set_state(AssistantState.IDLE, "No command after wake phrase")
                return
            self._set_state(AssistantState.EXECUTING)
            from brain.agent import run_agent
            from brain.router import route_command
            from .language_utils import detect_dominant_language
            from .response_formatter import format_spoken_response
            from training_data import EventType, get_recorder
            recorder = get_recorder()
            interaction_id = recorder.begin(transcript, metadata={"input_mode": "voice", "audio_captured": False})
            recorder.record(EventType.STT_TRANSCRIPT, {"raw_stt_transcript": sanitize_user_request(transcript), "audio_path": None}, interaction_id)
            safe_command=sanitize_user_request(command)
            recorder.record(EventType.NORMALIZED_REQUEST, {"normalized_text": safe_command, "final_interpreted_command": safe_command}, interaction_id)
            self._perf("intent_started")
            question_pipeline_started = time.perf_counter()
            route = route_command(command)
            from brain.request_intent import classify_request_kind
            request_kind=classify_request_kind(command)
            self.log.info("Request kind: kind=%s confidence=%.3f reason=%s",request_kind.kind.value,request_kind.confidence,request_kind.reason)
            self._perf("intent_completed")
            self.log.info("Router result: %s",_route_for_log(route))
            self.log.info("Selected route: %s", _route_for_log(route))
            self._perf("planning_completed")
            if route.get("type") == "question":
                self._start_question_task(command, route, interaction_id, transcript, question_pipeline_started)
                return
            if (re.match(r"^(?:send|message|tell)\s+",command,re.I) and route.get("type")!="revise_whatsapp_recipient") or (route.get("type")=="tool" and route.get("tool")=="analyze_screen"):
                self._start_cancellable_action_task(command,route,interaction_id,transcript)
                return
            self._perf("execution_started")
            execution_outcome = {}
            response = run_agent(command, route=route, interaction_id=interaction_id, original_user_text=transcript, execution_outcome=execution_outcome)
            self._perf("execution_completed")
            if self._stop.is_set():
                return
            response_text = response.get("message") if isinstance(response, dict) else str(response)
            self.log.info("Final action result: %s", _redact_for_log(response_text))
            lang = detect_dominant_language(transcript)
            spoken = format_spoken_response(command, route, response_text, lang=lang, execution=execution_outcome)
            self.log.info("Formatter result / final spoken response: %s",_redact_for_log(spoken))
            if not self.muted and spoken:
                self._start_speech_task(spoken,lang,interaction_id)
        finally:
            if not self.debug_audio:
                path.unlink(missing_ok=True)

    def _start_question_task(self, command, route, interaction_id, transcript, question_pipeline_started) -> None:
        """Run only read-only QUESTION work off the wake/microphone loop."""
        def work() -> None:
            try:
                from brain.agent import run_agent
                from .language_utils import detect_dominant_language
                from .response_formatter import format_spoken_response
                execution_outcome = {}
                response = run_agent(command, route=route, interaction_id=interaction_id, original_user_text=transcript, execution_outcome=execution_outcome)
                response_text = response.get("message") if isinstance(response, dict) else str(response)
                self.log.info("Final question result: %s", _redact_for_log(response_text))
                if self._stop.is_set() or response_text == "I stopped that question.":
                    return
                # Do not interrupt capture of a newer wake command. Wait until
                # the microphone state machine is idle before speaking.
                while self.state in {AssistantState.WAKE_DETECTED,AssistantState.LISTENING,AssistantState.PROCESSING,AssistantState.EXECUTING} and not self._stop.wait(.02):
                    pass
                if self._stop.is_set():
                    return
                lang = detect_dominant_language(transcript)
                spoken = format_spoken_response(command, route, response_text, lang=lang, execution=execution_outcome)
                self.log.info("Formatter result / final spoken response: %s",_redact_for_log(spoken))
                if not self.muted and spoken:
                    self._set_state(AssistantState.SPEAKING)
                    from brain.agent import agent_runtime
                    agent_runtime.context.interrupted_response=spoken
                    from .text_to_speech import speak
                    from .speech_journal import begin_speech,finish_speech
                    speech_journal=begin_speech(lang,len(spoken),interaction_id)
                    tts_start_ms=(time.perf_counter()-question_pipeline_started)*1000
                    self.log.info("Question performance: intent_classification_ms=%.1f tts_start_ms=%.1f",route.get("intent_classification_ms",0.0),tts_start_ms)
                    try:speech_result=speak(spoken,lang=lang)
                    except Exception as exc:finish_speech(speech_journal,error=exc);raise
                    finish_speech(speech_journal,speech_result)
                    if not isinstance(speech_result,dict) or speech_result.get("success"):
                        agent_runtime.context.last_spoken_response=spoken
                    self.log.info("Question performance: total_question_to_speech_ms=%.1f",(time.perf_counter()-question_pipeline_started)*1000)
                    if self.state is AssistantState.SPEAKING:self._set_state(AssistantState.IDLE,"Wake word ready" if self.wake_enabled else "Wake word disabled")
            except Exception:
                self.log.exception("Asynchronous question task failed")
            finally:
                with self._question_threads_lock:self._question_threads.discard(threading.current_thread())
        thread=threading.Thread(target=work,name="jarvis-interactive-question",daemon=True)
        with self._question_threads_lock:self._question_threads.add(thread)
        thread.start()

    def _start_speech_task(self,spoken,lang,interaction_id=None) -> None:
        def work():
            try:
                from brain.agent import agent_runtime
                agent_runtime.context.interrupted_response=spoken
                self._set_state(AssistantState.SPEAKING)
                from .text_to_speech import speak
                from .speech_journal import begin_speech,finish_speech
                speech_journal=begin_speech(lang,len(spoken),interaction_id)
                try:speech_result=speak(spoken,lang=lang)
                except Exception as exc:finish_speech(speech_journal,error=exc);raise
                finish_speech(speech_journal,speech_result)
                if not isinstance(speech_result,dict) or speech_result.get("success"):
                    agent_runtime.context.last_spoken_response=spoken
            except Exception:self.log.exception("Speech task failed")
            finally:
                if self.state is AssistantState.SPEAKING:self._set_state(AssistantState.IDLE,"Wake word ready" if self.wake_enabled else "Wake word disabled")
        threading.Thread(target=work,name="jarvis-speech",daemon=True).start()

    def _start_cancellable_action_task(self,command,route,interaction_id,transcript):
        def work():
            from brain.task_supervisor import CancellationToken,register_interactive_task,unregister_interactive_task
            token=CancellationToken();task_id=register_interactive_task(token)
            try:
                from brain.agent import run_agent
                execution_outcome = {}
                response=run_agent(command,route=route,interaction_id=interaction_id,original_user_text=transcript,cancellation_token=token,execution_outcome=execution_outcome)
                if token.cancelled or self._stop.is_set():return
                text=str(response);lang="en"
                from .response_formatter import format_spoken_response
                spoken=format_spoken_response(command,route,text,lang=lang,execution=execution_outcome)
                self.log.info("Formatter result / final spoken response: %s",_redact_for_log(spoken))
                if spoken:self._start_speech_task(spoken,lang,interaction_id)
            except Exception:self.log.exception("Cancellable action task failed")
            finally:unregister_interactive_task(task_id)
        threading.Thread(target=work,name="jarvis-interactive-action",daemon=True).start()
