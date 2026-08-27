"""Always-on voice orchestration around JARVIS's existing voice pipeline."""
from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from .learning_approval import (
    DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    DEFAULT_LEARNING_QUESTION,
    LearningApprovalResult,
    request_learning_approval as _run_learning_approval,
)
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
    WAITING_FOR_LEARNING_APPROVAL = "WAITING_FOR_LEARNING_APPROVAL"
    ERROR = "ERROR"


@dataclass
class _PendingCapture:
    """A cross-thread request for the single audio-owning thread to capture
    and transcribe the next utterance on the requester's behalf, instead of
    a second thread opening a conflicting microphone stream (Phase 4 of the
    voice-approved continual learning pipeline). `done` is set exactly once,
    by the audio thread, after `transcript` has been filled in (None if
    nothing usable was captured within `timeout_seconds`)."""
    timeout_seconds: float
    done: threading.Event
    transcript: str | None = None


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
        self._pending_capture: _PendingCapture | None = None
        self._pending_capture_lock = threading.Lock()
        self._learning_token = None
        self._learning_lock = threading.Lock()
        self._learning_last_status: tuple[str, str] | None = None
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
        from brain.activity_state import set_speaking
        set_speaking(state is AssistantState.SPEAKING)
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

        with self._pending_capture_lock:
            pending = self._pending_capture
        if pending is not None:
            self._run_pending_capture_session(pending, np)
            return

        factory = self.stream_factory or self._default_stream
        frame_seconds = self.wake_engine.frame_samples / self.wake_engine.sample_rate
        ring = deque(maxlen=max(1, int(1.2 / frame_seconds)))
        capture = []
        listen_started = last_speech = None
        speech_after_wake = False
        from .voice_perf import VoiceInteractionTimer
        perf = VoiceInteractionTimer(clock=self.clock)
        realtime = None
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
                        perf=VoiceInteractionTimer(clock=self.clock);perf.mark("wake_detected",now)
                        realtime=self._start_realtime_stt(perf)
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
                        perf = VoiceInteractionTimer(clock=self.clock)
                        perf.mark("wake_detected", now)
                        realtime = self._start_realtime_stt(perf)
                        self._set_state(AssistantState.LISTENING)
                        self._perf("recording_started", now)
                    continue
                if self.state is not AssistantState.LISTENING:
                    continue
                capture.append(frame)
                if realtime is not None:
                    realtime.feed(frame.tobytes())
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
            if realtime is not None:
                realtime.close()
            return
        if not capture or not speech_after_wake:
            if realtime is not None:
                realtime.close()
            self._set_state(AssistantState.IDLE, "No command heard")
            return
        elevenlabs_transcript = None
        ledger = realtime.ledger if realtime is not None else None
        if realtime is not None:
            commit_timeout = float(os.getenv("ELEVENLABS_STT_COMMIT_TIMEOUT", "5.0"))
            elevenlabs_transcript = realtime.commit_and_close(timeout=commit_timeout)
        self._process_capture(capture, elevenlabs_transcript=elevenlabs_transcript, ledger=ledger, perf=perf)
        self._stop.wait(self.cooldown_seconds)
        self.wake_engine.reset()
        if self.state is not AssistantState.SPEAKING or self._stop.is_set():
            self._set_state(AssistantState.IDLE, "Wake word ready" if self.wake_enabled else "Wake word disabled")

    def _run_pending_capture_session(self, pending: "_PendingCapture", np) -> None:
        """Serve one cross-thread `_PendingCapture` request (see
        `_capture_utterance_via_audio_thread`) using the SAME single audio
        stream this whole class ever opens -- never a second, conflicting
        consumer. No wake-word detection here: the caller has already
        decided a capture is wanted (e.g. listening for "yes jarvis"/"no
        jarvis" after a verified teacher fix), so this captures until
        silence-after-speech or the request's own timeout, exactly like the
        normal LISTENING phase's VAD, then transcribes and hands the result
        back through `pending`."""
        self._restart.clear()
        self._set_state(AssistantState.WAITING_FOR_LEARNING_APPROVAL, "Listening for approval")
        factory = self.stream_factory or self._default_stream
        capture: list = []
        listen_started = self.clock()
        last_speech = None
        speech_after_wake = False
        try:
            with self._mic_lock, factory() as stream:
                while not self._stop.is_set():
                    raw, overflowed = stream.read(self.wake_engine.frame_samples)
                    if overflowed:
                        self.log.warning("Microphone input overflow")
                    frame = np.frombuffer(raw, dtype=np.int16).copy()
                    now = self.clock()
                    capture.append(frame)
                    rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
                    if rms >= self.speech_rms:
                        speech_after_wake, last_speech = True, now
                    elapsed = now - listen_started
                    finished = speech_after_wake and last_speech is not None and now - last_speech >= self.silence_seconds
                    if finished or elapsed >= pending.timeout_seconds:
                        break
        except Exception:
            self.log.exception("Learning-approval capture failed")
        transcript = None
        if capture and speech_after_wake and not self._stop.is_set():
            try:
                transcript = self._transcribe_frames(capture)
            except Exception:
                self.log.exception("Learning-approval transcription failed")
        pending.transcript = transcript
        with self._pending_capture_lock:
            if self._pending_capture is pending:
                self._pending_capture = None
        if not self._stop.is_set():
            self._set_state(AssistantState.IDLE, "Wake word ready" if self.wake_enabled else "Wake word disabled")
        # State/pending-slot must be fully settled BEFORE waking the
        # waiting caller, so it never observes a stale WAITING_FOR_LEARNING_
        # APPROVAL state or a still-set _pending_capture race.
        pending.done.set()

    def _transcribe_frames(self, frames) -> str:
        import numpy as np
        import soundfile as sf

        audio = np.concatenate(frames).astype("float32") / 32768.0
        temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        path = Path(temp.name)
        temp.close()
        try:
            sf.write(path, audio, self.wake_engine.sample_rate)
            from .speech_to_text import transcribe_audio
            return transcribe_audio(str(path))
        finally:
            if not self.debug_audio:
                path.unlink(missing_ok=True)

    def _capture_utterance_via_audio_thread(self, timeout_seconds: float) -> str | None:
        """Block the CALLING thread (never the audio thread) until the
        single audio-owning thread has captured and transcribed one
        utterance, or `timeout_seconds` elapses. Raises RuntimeError if
        called from the audio thread itself (that would deadlock) or while
        another capture request is already pending (one at a time, by
        design -- this is not a queue)."""
        if self._thread is not None and threading.current_thread() is self._thread:
            raise RuntimeError("_capture_utterance_via_audio_thread must not be called from the audio thread")
        done = threading.Event()
        request = _PendingCapture(timeout_seconds=max(0.1, timeout_seconds), done=done)
        with self._pending_capture_lock:
            if self._pending_capture is not None:
                raise RuntimeError("a learning-approval capture request is already pending")
            self._pending_capture = request
        # The audio thread may currently be idly wake-word-listening inside
        # its own long-lived mic session; `_restart` is the existing signal
        # that makes it yield that session promptly (checked every frame,
        # ~80ms) so it can pick up this pending request on its next pass.
        self._restart.set()
        completed = done.wait(request.timeout_seconds + 5.0)
        with self._pending_capture_lock:
            if self._pending_capture is request:
                self._pending_capture = None
        return request.transcript if completed else None

    def request_learning_approval(
        self,
        question: str = DEFAULT_LEARNING_QUESTION,
        timeout_seconds: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
        cancellation_token=None,
    ) -> LearningApprovalResult:
        """Speak the learning-approval question and listen for up to
        `timeout_seconds` for an explicit "yes jarvis"/"no jarvis" (Phase 3),
        reusing this assistant's single microphone consumer instead of
        opening a second one (Phase 4). Must be called from a thread other
        than the audio thread -- e.g. the background worker that just
        finished driving a verified Claude teacher fix, never the
        `jarvis-audio` thread itself."""
        from .text_to_speech import speak

        def speak_fn(text: str) -> None:
            self._set_state(AssistantState.SPEAKING, "Learning approval question")
            try:
                speak(text)
            except Exception:
                self.log.exception("Failed to speak learning-approval question")

        def listen_fn(remaining: float):
            return self._capture_utterance_via_audio_thread(remaining)

        return _run_learning_approval(
            speak_fn=speak_fn, listen_fn=listen_fn, clock=self.clock,
            timeout_seconds=timeout_seconds, question=question, cancellation_token=cancellation_token,
        )

    def _elevenlabs_stt_enabled(self) -> bool:
        from .elevenlabs_realtime_stt import is_configured
        return is_configured()

    def _start_realtime_stt(self, perf):
        """Open an ElevenLabs realtime STT session for the utterance that
        just started (Part A/M): only ever called right after a wake event,
        never while idle, and always closed by the caller once the
        interaction ends. Returns None (never raises) if ElevenLabs STT is
        not configured/enabled or fails to start -- the caller's existing
        Whisper-based capture path is entirely unaffected either way."""
        if not self._elevenlabs_stt_enabled():
            return None
        try:
            from .realtime_capture import RealtimeSTTController
            controller = RealtimeSTTController(
                perf=perf, on_speculative_action=self._on_speculative_action,
                sample_rate=self.wake_engine.sample_rate,
            )
            controller.start()
            return controller
        except Exception:
            self.log.exception("Failed to start ElevenLabs realtime STT session")
            return None

    def _on_speculative_action(self, action) -> None:
        """Invoked (from the ElevenLabs receive thread) the moment a
        partial transcript resolves into a stable, SAFE, deterministic
        action (Part B). Acknowledgement speech and the actual action
        execution are dispatched on two independent threads so neither
        blocks the other or the STT receive thread itself (Part E)."""
        self.log.info("Speculative partial action: tool=%s", action.route.get("tool"))
        ack_text = self._speculative_ack_text(action.route)
        if ack_text and not self.muted:
            self._start_speech_task(ack_text, "en", None)
        threading.Thread(
            target=self._run_speculative_action, args=(action,),
            name="jarvis-speculative-exec", daemon=True,
        ).start()

    def _run_speculative_action(self, action) -> None:
        try:
            from brain.agent import run_agent
            action.result = run_agent(action.partial_text, route=action.route)
        except Exception:
            self.log.exception("Speculative action execution failed")

    @staticmethod
    def _speculative_ack_text(route: dict) -> str:
        tool = route.get("tool")
        if tool in {"open_application", "open_website"}:
            return "Opening it now, sir."
        if tool in {"volume_up", "volume_down", "mute_volume"}:
            return "Right away, sir."
        return "On it, sir."

    @staticmethod
    def _command_needs_planning(command: str, route: dict) -> bool:
        """Part J: reuses the exact predicate `brain.agent.run_agent`
        itself uses to decide whether task planning/cloud calls are
        needed -- no new heuristic invented -- plus the router's own
        explicit "this always needs an LLM" route types."""
        try:
            from brain.task_planner import should_use_task_planner
            return route.get("type") in {"plan", "tools", "ai"} or should_use_task_planner(command)
        except Exception:
            return route.get("type") in {"plan", "tools", "ai"}

    def _process_capture(self, frames, *, elevenlabs_transcript: str | None = None, ledger=None, perf=None) -> None:
        import numpy as np

        from .voice_perf import VoiceInteractionTimer
        if perf is None:
            perf = VoiceInteractionTimer(clock=self.clock)
        transcript = (elevenlabs_transcript or "").strip() or None
        path = None
        if transcript is None:
            import soundfile as sf
            audio = np.concatenate(frames).astype("float32") / 32768.0
            temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            path = Path(temp.name)
            temp.close()
            sf.write(path, audio, self.wake_engine.sample_rate)
        try:
            self._set_state(AssistantState.PROCESSING, "Transcribing")
            if transcript is not None:
                self.log.info("Using ElevenLabs realtime committed transcript (Whisper skipped)")
            else:
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
            from brain.agent import run_agent, agent_runtime
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
            # Resolving references here (before run_agent bumps the turn for
            # this command) still only ever sees PRIOR turns' entities --
            # correct, since nothing from this command has been recorded yet.
            # `agent_runtime` is already imported into this scope above.
            route = route_command(command, context=agent_runtime.context)
            perf.mark("committed_transcript")
            if ledger is not None and ledger.has_fired_anything():
                from brain.speculative_execution import reconcile_final_route, reconcile_local_plan_actions
                if route.get("type") == "local_plan":
                    remaining_actions = reconcile_local_plan_actions(ledger, route.get("actions") or [])
                    if not remaining_actions:
                        self.log.info("Final command fully satisfied by an earlier speculative partial action")
                        recorder.record(EventType.TASK_COMPLETED, {"success": True, "verified": True, "reason": "already_completed_by_speculative_partial_action"}, interaction_id)
                        recorder.finalize(interaction_id, success=True, verified=True, response="Already completed from an earlier action.")
                        self._set_state(AssistantState.IDLE, "Wake word ready" if self.wake_enabled else "Wake word disabled")
                        perf.mark("interaction_finished");perf.log_summary()
                        return
                    route = {**route, "actions": remaining_actions}
                else:
                    reconciled_route, matched = reconcile_final_route(ledger, route)
                    if reconciled_route is None and matched is not None:
                        self.log.info("Final command fully satisfied by an earlier speculative partial action")
                        recorder.record(EventType.TASK_COMPLETED, {"success": True, "verified": True, "reason": "already_completed_by_speculative_partial_action"}, interaction_id)
                        recorder.finalize(interaction_id, success=True, verified=True, response="Already completed from an earlier action.")
                        self._set_state(AssistantState.IDLE, "Wake word ready" if self.wake_enabled else "Wake word disabled")
                        perf.mark("interaction_finished");perf.log_summary()
                        return
                    route = reconciled_route
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
            if route.get("type") == "contextual_question" and not self._is_agent_route(route):
                # No agent provider available -- brain/agent.py's
                # `_run_agent_impl` falls back to the same cancellable
                # web-answer path an ordinary "question" gets (with the
                # resolved context folded into the query), so dispatch it
                # the same way here too instead of letting it fall through
                # to the synchronous default path below and block the
                # microphone.
                self._start_question_task(command, route, interaction_id, transcript, question_pipeline_started)
                return
            if route.get("type") == "start_learning":
                self._start_learning_task(interaction_id)
                return
            if route.get("type") == "stop_learning":
                self._stop_learning_task(interaction_id)
                return
            if route.get("type") == "learning_status":
                self._learning_status_task(interaction_id)
                return
            if route.get("type") == "coding_task":
                self._start_coding_task(route.get("task") or command, interaction_id)
                return
            if (re.match(r"^(?:send|message|tell)\s+",command,re.I) and route.get("type")!="revise_whatsapp_recipient") or (route.get("type")=="tool" and route.get("tool")=="analyze_screen"):
                self._start_cancellable_action_task(command,route,interaction_id,transcript)
                return
            if self._is_agent_route(route):
                # A multi-step agent run can take minutes. Running it on this
                # capture thread would block the microphone for its whole
                # duration; the cancellable-action path runs it off-thread and
                # registers a token, so the user can still speak -- and can say
                # "cancel" -- while it works.
                self.log.info("Dispatching to the agent runtime off the capture thread: route=%s", route.get("type"))
                perf.mark("agent_dispatched")
                if not self.muted:
                    from .response_formatter import generic_acknowledgement
                    from .voice_language import resolve_utterance_language
                    if resolve_utterance_language(transcript) == "he":
                        self._start_speech_task(generic_acknowledgement(), "en", None)
                    else:
                        self._start_speech_task("I'll work on that, sir.", "en", None)
                self._start_cancellable_action_task(command,route,interaction_id,transcript,narrate=True)
                return
            needs_planning = self._command_needs_planning(command, route)
            # Bilingual voice UX: VOICE_LANGUAGE (auto/en/he) controls STT
            # INPUT only -- the ACTUAL per-utterance input language (what
            # decides ack wording) is resolved here, per utterance, never
            # assumed from the configured mode alone (auto mode especially
            # must not treat every utterance as one fixed language).
            from .voice_language import resolve_utterance_language
            input_language = resolve_utterance_language(transcript)
            hebrew_mode = input_language == "he"
            if not self.muted:
                perf.mark("acknowledgement_tts_request")
                if hebrew_mode:
                    # TTS speaks English only, always. A Hebrew command's
                    # action (which DOES receive/preserve the exact
                    # recognized Hebrew entities -- see
                    # brain/music_intent.py) must never have its Hebrew
                    # payload spoken back, translated, or transliterated.
                    from .response_formatter import generic_acknowledgement
                    self._start_speech_task(generic_acknowledgement(), "en", None)
                else:
                    # Deterministic, non-LLM contextual ack ("Opening
                    # YouTube, sir." / "Playing Starboy, sir.") fired
                    # immediately, BEFORE run_agent below even starts --
                    # action and acknowledgement genuinely overlap.
                    from .response_formatter import compose_contextual_ack
                    self._start_speech_task(compose_contextual_ack(route), "en", None)
            self._perf("execution_started")
            if needs_planning:
                perf.mark("planner_started")
            execution_outcome = {}
            response = run_agent(command, route=route, interaction_id=interaction_id, original_user_text=transcript, execution_outcome=execution_outcome, speculative_ledger=ledger)
            if needs_planning:
                perf.mark("planner_finished")
            perf.mark("final_action_finished")
            self._perf("execution_completed")
            if self._stop.is_set():
                return
            response_text = response.get("message") if isinstance(response, dict) else str(response)
            self.log.info("Final action result: %s", _redact_for_log(response_text))
            # Success: the immediate ack above already covered it -- no
            # additional speech, for EITHER language (never speak twice for
            # the same outcome). Failure: exactly ONE short English
            # follow-up. Hebrew always uses the generic, entity-free
            # failure message (the tool's own failure text could contain
            # the recognized Hebrew entity); English reuses the specific,
            # user-friendly tool failure message where one exists.
            if execution_outcome.get("success"):
                lang, spoken = "en", None
            elif hebrew_mode:
                from .response_formatter import generic_failure_message
                lang, spoken = "en", generic_failure_message()
            else:
                lang = "en"
                spoken = format_spoken_response(command, route, response_text, lang=lang, execution=execution_outcome)
            self._perf("formatter_completed")
            self.log.info("Formatter result / final spoken response: %s",_redact_for_log(spoken))
            if not self.muted and spoken:
                perf.mark("final_tts_started")
                self._start_speech_task(spoken,lang,interaction_id)
                self._perf("tts_dispatched")
            perf.mark("interaction_finished")
            perf.log_summary()
        finally:
            if path is not None and not self.debug_audio:
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
                    from .speech_coordinator import get_speech_coordinator
                    from .speech_journal import begin_speech,finish_speech
                    speech_journal=begin_speech(lang,len(spoken),interaction_id)
                    tts_start_ms=(time.perf_counter()-question_pipeline_started)*1000
                    self.log.info("Question performance: intent_classification_ms=%.1f tts_start_ms=%.1f",route.get("intent_classification_ms",0.0),tts_start_ms)
                    try:speech_result=get_speech_coordinator().speak(spoken,lang=lang)
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

    def _start_speech_task(self,spoken,lang,interaction_id=None,priority=None) -> None:
        from .speech_coordinator import PRIORITY_STATUS
        if priority is None:priority=PRIORITY_STATUS
        def work():
            try:
                from brain.agent import agent_runtime
                agent_runtime.context.interrupted_response=spoken
                self._set_state(AssistantState.SPEAKING)
                from .speech_coordinator import get_speech_coordinator
                from .speech_journal import begin_speech,finish_speech
                speech_journal=begin_speech(lang,len(spoken),interaction_id)
                try:speech_result=get_speech_coordinator().speak(spoken,lang=lang,priority=priority)
                except Exception as exc:finish_speech(speech_journal,error=exc);raise
                finish_speech(speech_journal,speech_result)
                if not isinstance(speech_result,dict) or speech_result.get("success"):
                    agent_runtime.context.last_spoken_response=spoken
            except Exception:self.log.exception("Speech task failed")
            finally:
                if self.state is AssistantState.SPEAKING:self._set_state(AssistantState.IDLE,"Wake word ready" if self.wake_enabled else "Wake word disabled")
        threading.Thread(target=work,name="jarvis-speech",daemon=True).start()

    @staticmethod
    def _is_agent_route(route) -> bool:
        """Will `run_agent` hand this request to the agent runtime?

        Mirrors `brain/agent.py::_run_agent_impl`'s escalation condition
        rather than guessing: an explicit `agent_task`, or one of the
        "local routing could not resolve this" routes at a moment when a
        provider is genuinely configured. With no API key this is always
        False, so the voice path is byte-for-byte unchanged.

        `contextual_question` (brain/conversational_context.py's resolved
        explanatory follow-up) is included alongside `plan`/`ai`: like
        them, `_run_agent_impl` conditionally escalates it to
        `_run_agent_with_loop` only when a provider is available. Without
        this, a follow-up like "What does that mean?" would run the whole
        agent turn SYNCHRONOUSLY on this capture thread instead of through
        the off-thread, cancellable, narrated dispatch every other agent
        turn gets -- blocking the microphone for as long as the model
        takes to answer.
        """
        route_type = (route or {}).get("type")
        if route_type == "agent_task":
            return True
        if route_type not in {"plan", "ai", "contextual_question"}:
            return False
        from brain.agent import _agent_escalation_available

        return _agent_escalation_available()

    def _start_cancellable_action_task(self,command,route,interaction_id,transcript,narrate=False):
        """Run a long action off the capture thread.

        `narrate=True` (agent routes) additionally speaks REAL progress while
        the agent works and streams the final answer sentence by sentence, so
        a minute-long task is not a minute of silence. Both are dispatched on
        their own speech threads and never block the agent -- see
        `voice/agent_narration.py`."""
        def work():
            from brain.task_supervisor import CancellationToken,register_interactive_task,unregister_interactive_task
            token=CancellationToken();task_id=register_interactive_task(token)
            narrator=None;stream=None
            try:
                from brain.agent import run_agent
                execution_outcome = {}
                if narrate and not self.muted:
                    from .agent_narration import AgentNarrator
                    from .speech_coordinator import PRIORITY_STATUS,PRIORITY_FINAL
                    narrator=AgentNarrator(
                        speak=lambda text: self._start_speech_task(text,"en",None,priority=PRIORITY_STATUS),
                        speak_final=lambda text: self._start_speech_task(text,"en",None,priority=PRIORITY_FINAL),
                    )
                    stream=narrator.answer_stream()
                    # Fills a long gap while the model reasons between tools.
                    # Runs on its own thread; the agent never waits for it.
                    narrator.start_heartbeat()
                response=run_agent(
                    command,route=route,interaction_id=interaction_id,original_user_text=transcript,
                    cancellation_token=token,execution_outcome=execution_outcome,
                    progress=None if narrator is None else narrator.on_event,
                    on_answer_text=None if stream is None else stream.feed,
                )
                if token.cancelled or self._stop.is_set():
                    if stream is not None:stream.discard()
                    return
                text=str(response);lang="en"
                from .response_formatter import format_spoken_response
                spoken=format_spoken_response(command,route,text,lang=lang,execution=execution_outcome)
                self.log.info("Formatter result / final spoken response: %s",_redact_for_log(spoken))
                if stream is not None and stream.has_spoken:
                    # The answer was already streamed sentence by sentence as
                    # the model produced it. Speaking `spoken` now would say
                    # the whole thing a second time; flush whatever is still
                    # buffered instead so nothing is lost.
                    stream.flush()
                    self.log.info("Final answer was streamed in %d chunk(s); not repeated",len(stream.spoken_chunks))
                    return
                if stream is not None:stream.discard()
                if spoken:
                    from .speech_coordinator import PRIORITY_FINAL
                    self._start_speech_task(spoken,lang,interaction_id,priority=PRIORITY_FINAL)
            except Exception:self.log.exception("Cancellable action task failed")
            finally:
                if narrator is not None:narrator.stop()
                unregister_interactive_task(task_id)
        threading.Thread(target=work,name="jarvis-interactive-action",daemon=True).start()

    def _start_learning_task(self, interaction_id=None) -> None:
        """"Hey Jarvis, start learning" (Phase 11): a deterministic local
        command, matched in brain/router.py before any planner/intent-model
        involvement. Reuses the SAME interactive-task cancellation registry
        (Phase 19/20) other long-running voice-triggered actions already use
        -- the existing "cancel"/"stop" command cancels this too, in
        addition to the dedicated "stop learning" phrase."""
        from brain.task_supervisor import CancellationToken, register_interactive_task, unregister_interactive_task
        from brain.learning_store import get_learning_job_store

        with self._learning_lock:
            if self._learning_token is not None and not self._learning_token.cancelled:
                self._start_speech_task("I'm already learning, sir.", "en", interaction_id)
                return
            token = CancellationToken()
            self._learning_token = token
        task_id = register_interactive_task(token)

        try:
            job_count = len(get_learning_job_store().query_trainable())
        except Exception:
            self.log.exception("Could not query the learning job queue")
            job_count = 0

        if job_count:
            plural = "s" if job_count != 1 else ""
            announce = f"I have {job_count} approved learning task{plural}. I'll start learning now, sir."
        else:
            announce = "I don't have any approved learning tasks right now, sir."
        self._start_speech_task(announce, "en", interaction_id)

        def work() -> None:
            try:
                from pathlib import Path

                from brain.improvement_coding_agent import ClaudeCodeAdapter
                from brain.learning_orchestrator import start_learning
                from brain.learning_training import TrainingPolicy
                from training.code_model.production import build_production_backend, build_production_benchmark, build_training_config

                repository_root = str(Path(__file__).resolve().parent.parent)
                # Real backend/benchmark (Phase 19): no FakeTrainingBackend
                # or FakeBenchmark in this production path. Which base
                # model/config is used is controlled by
                # JARVIS_CODE_MODEL_CONFIG (default: "qlora_7b", this
                # project's recommended real config -- see
                # training/code_model/configs/). If this machine can't
                # actually run it, backend.is_available() honestly reports
                # that (Phase 6/20) and start_learning stops before
                # claiming training began; it never silently substitutes a
                # fake.
                backend = build_production_backend()
                benchmark = build_production_benchmark(backend.code_model_config)
                summary = start_learning(
                    coding_agent=ClaudeCodeAdapter(),
                    repository_root=repository_root,
                    backend=backend,
                    benchmark=benchmark,
                    training_config=build_training_config(backend.code_model_config),
                    policy=TrainingPolicy(mode="manual_only"),
                    explicit_command=True,
                    cancellation_token=token,
                    progress_callback=self._on_learning_progress,
                )
                spoken = None
                if summary.status == "CANCELLED":
                    spoken = "Learning stopped, sir."
                elif summary.status == "COMPLETED" and summary.promoted:
                    spoken = "Learning is complete. The new coding model performed better and is now active, sir."
                elif summary.status == "COMPLETED" and summary.promoted is False:
                    spoken = "Learning is complete, but the new model did not outperform the current one, so I kept the existing model, sir."
                elif summary.status == "FAILED" and summary.error == "pre-training checks failed":
                    spoken = "I've prepared the training data, but training requires external compute that isn't configured yet, sir."
                elif summary.job_count == 0:
                    spoken = None  # already told the user before starting; nothing new to report
                else:
                    spoken = "I couldn't complete the learning run, sir."
                if spoken and not self._stop.is_set():
                    self._start_speech_task(spoken, "en", interaction_id)
            except Exception:
                self.log.exception("start_learning task failed")
                if not self._stop.is_set():
                    self._start_speech_task("I couldn't complete the learning run, sir.", "en", interaction_id)
            finally:
                unregister_interactive_task(task_id)
                with self._learning_lock:
                    if self._learning_token is token:
                        self._learning_token = None

        threading.Thread(target=work, name="jarvis-start-learning", daemon=True).start()

    def _stop_learning_task(self, interaction_id=None) -> None:
        """"Hey Jarvis, stop learning" (Phase 20)."""
        with self._learning_lock:
            token = self._learning_token
        if token is not None and not token.cancelled:
            token.cancel()
            self._start_speech_task("Stopping learning, sir.", "en", interaction_id)
        else:
            self._start_speech_task("I'm not learning right now, sir.", "en", interaction_id)

    def _on_learning_progress(self, status: str, detail: str) -> None:
        self._learning_last_status = (status, detail)
        self.log.info("[learning] %s %s", status, detail)

    def _learning_status_task(self, interaction_id=None) -> None:
        """"Hey Jarvis, learning status" (Phase 21): reports the current
        stage of an in-progress run (if any, from the last
        `progress_callback` update `start_learning` sent), plus queue and
        latest-model summary from the persisted stores -- never requires an
        active run to answer something useful."""
        def work() -> None:
            try:
                from brain.learning_store import get_learning_job_store
                from brain.learning_training import get_model_registry

                with self._learning_lock:
                    active = self._learning_token is not None and not self._learning_token.cancelled
                    last_status = self._learning_last_status

                parts = []
                if active and last_status:
                    parts.append(f"Learning is currently in the {last_status[0].replace('_', ' ').lower()} stage.")
                elif active:
                    parts.append("Learning is currently running.")
                else:
                    parts.append("No learning run is active right now.")

                try:
                    trainable = len(get_learning_job_store().query_trainable())
                    parts.append(f"{trainable} approved learning task{'s' if trainable != 1 else ''} waiting.")
                except Exception:
                    self.log.exception("Could not query the learning job queue for status")

                try:
                    history = get_model_registry().history(limit=1)
                    if history:
                        latest = history[0]
                        parts.append(f"Most recent model: {latest.model_version}, status {latest.status.lower()}.")
                except Exception:
                    self.log.exception("Could not query the model registry for status")

                self._start_speech_task(" ".join(parts), "en", interaction_id)
            except Exception:
                self.log.exception("learning status task failed")
                self._start_speech_task("I couldn't check the learning status, sir.", "en", interaction_id)

        threading.Thread(target=work, name="jarvis-learning-status", daemon=True).start()

    def _start_coding_task(self, task, interaction_id=None) -> None:
        """A genuine coding/self-improvement request (Part A of the
        student-first production runtime): deterministically matched in
        `brain/router.py` via `brain.coding_task_intent.is_coding_task`
        (ordinary commands never reach here). Runs
        `brain.improvement_student_teacher.run_coding_task` -- ACTIVE local
        student first, Claude teacher fallback, existing voice-approved
        learning trigger on verified teacher success -- on a background
        thread, never blocking the audio/wake-word loop.

        Shares `self._learning_token`/`self._learning_lock` with
        "start learning": both are GPU-bound background AI work on this
        machine, so they're deliberately mutually exclusive (and "stop
        learning" already cancels either one)."""
        from brain.task_supervisor import CancellationToken, register_interactive_task, unregister_interactive_task

        with self._learning_lock:
            if self._learning_token is not None and not self._learning_token.cancelled:
                self._start_speech_task("I'm already working on something, sir. Say stop learning if you'd like me to stop first.", "en", interaction_id)
                return
            token = CancellationToken()
            self._learning_token = token
        task_id = register_interactive_task(token)
        self._start_speech_task("Let me look into that, sir.", "en", interaction_id)

        def work() -> None:
            try:
                from pathlib import Path

                from brain.improvement_student_teacher import run_coding_task

                repository_root = str(Path(__file__).resolve().parent.parent)
                result = run_coding_task(
                    task, repository_root=repository_root,
                    request_approval=self.request_learning_approval,
                    cancellation_token=token,
                )
                self.log.info("[coding-task] %s", result.to_dict())

                spoken = self._coding_task_result_speech(result)
                if spoken and not self._stop.is_set():
                    self._start_speech_task(spoken, "en", interaction_id)
            except Exception:
                self.log.exception("coding task failed")
                if not self._stop.is_set():
                    self._start_speech_task("I ran into a problem working on that, sir.", "en", interaction_id)
            finally:
                unregister_interactive_task(task_id)
                with self._learning_lock:
                    if self._learning_token is token:
                        self._learning_token = None

        threading.Thread(target=work, name="jarvis-coding-task", daemon=True).start()

    @staticmethod
    def _coding_task_result_speech(result) -> str | None:
        """The exact confirmation phrases this pipeline's voice UX was
        designed around (see CLAUDE.md's voice-approved-learning section):
        "Do you want me to learn how to do that, sir?" already got spoken
        (and answered) inside `request_learning_approval`, called from
        within `run_coding_task` itself -- this is only the follow-up
        wrap-up line."""
        if result.solved_by == "student":
            return "I fixed it myself, sir."
        if result.solved_by == "teacher":
            offer = result.learning_offer
            if offer is not None and offer.offered:
                if offer.approval_outcome == "APPROVED":
                    return "I've added that to my learning queue, sir."
                if offer.approval_outcome == "DECLINED":
                    return "Understood, sir."
                return None  # TIMED_OUT/CANCELLED -> silently return to normal state
            return "It's fixed and verified, sir."
        return "I wasn't able to fix that, sir."
