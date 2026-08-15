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


def _redact_for_log(text: str) -> str:
    return re.sub(
        r"(?i)\b(password|passcode|token|api[ -]?key)\b\s*(?:is|=|:)?\s*\S+",
        lambda match: f"{match.group(1)}=<REDACTED>",
        text,
    )


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
        self._stop = threading.Event()
        self._listen_now = threading.Event()
        self._restart = threading.Event()
        self._thread = None
        self._mic_lock = threading.Lock()
        self.log = logging.getLogger("jarvis.background")

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

    def stop(self) -> None:
        self._stop.set()
        self._listen_now.set()
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
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice is not installed") from exc
        return sd.RawInputStream(samplerate=self.wake_engine.sample_rate, blocksize=self.wake_engine.frame_samples, channels=1, dtype="int16")

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
        self._set_state(AssistantState.IDLE, "Wake word ready" if self.wake_enabled else "Wake word disabled")

        with self._mic_lock, factory() as stream:
            while not self._stop.is_set() and not self._restart.is_set():
                raw, overflowed = stream.read(self.wake_engine.frame_samples)
                if overflowed:
                    self.log.warning("Microphone input overflow")
                frame = np.frombuffer(raw, dtype=np.int16).copy()
                now = self.clock()
                if self.state is AssistantState.IDLE:
                    ring.append(frame)
                    manual = self._listen_now.is_set()
                    if manual:
                        self._listen_now.clear()
                    detected = False
                    if self.wake_enabled and not manual:
                        detected, _ = self.wake_engine.process(frame)
                    if manual or detected:
                        self._set_state(AssistantState.WAKE_DETECTED, "Manual listen" if manual else "Jarvis detected")
                        self.log.info("Wake detected")
                        capture = list(ring)
                        listen_started, last_speech, speech_after_wake = now, None, False
                        self._set_state(AssistantState.LISTENING)
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
            transcript = transcribe_audio(str(path))
            self.log.info("Transcribed command: %s", _redact_for_log(transcript))
            command, _ = normalize_transcript(transcript)
            self.log.info("Normalized command: %s", _redact_for_log(command))
            if not command:
                self._set_state(AssistantState.IDLE, "No command after wake phrase")
                return
            self._set_state(AssistantState.EXECUTING)
            from brain.agent import run_agent
            from brain.router import route_command
            from .language_utils import detect_dominant_language
            from .response_formatter import format_spoken_response
            route = route_command(command)
            self.log.info("Selected route: %s", _route_for_log(route))
            response = run_agent(command)
            response_text = response.get("message") if isinstance(response, dict) else str(response)
            self.log.info("Final action result: %s", _redact_for_log(response_text))
            lang = detect_dominant_language(transcript)
            spoken = format_spoken_response(command, route, response_text, lang=lang)
            if not self.muted and spoken:
                self._set_state(AssistantState.SPEAKING)
                from .text_to_speech import speak
                speak(spoken, lang=lang)
        finally:
            if not self.debug_audio:
                path.unlink(missing_ok=True)
