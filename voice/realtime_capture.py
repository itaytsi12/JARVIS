"""Bridges JARVIS's single, existing microphone-owning audio thread
(`voice/background_assistant.py`) to a background ElevenLabs realtime STT
session -- WITHOUT ever becoming a second microphone consumer. This module
never opens an audio device itself; the caller feeds it PCM frames it is
already reading from the one real `NativeMicrophoneStream`.

Connecting to ElevenLabs is done on a small background thread so a slow or
unreachable network never blocks the audio thread's frame-read loop (Part M:
the realtime session only ever exists for the duration of one wake/listen
interaction). Frames captured before the connection finishes are buffered
locally and flushed once connected, rather than dropped -- a stalled network
degrades to "ElevenLabs sees the whole utterance a little late" rather than
losing audio outright; Whisper's fallback transcription always has the
complete, un-degraded frame buffer regardless.

Every partial transcript is fed through `brain.speculative_execution`'s
`PartialActionLedger`; when a candidate action becomes stable, this class
invokes the caller-supplied `on_speculative_action` callback so the actual
speech/execution side effects live in `voice/background_assistant.py`
(kept here as a callback rather than a hard dependency so this module is
unit-testable without a running `AlwaysOnAssistant`).
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from brain.speculative_execution import PartialActionLedger
from voice.elevenlabs_realtime_stt import ElevenLabsRealtimeSTT, ElevenLabsSTTError, TranscriptEvent
from voice.voice_perf import VoiceInteractionTimer

log = logging.getLogger("jarvis.realtime_capture")


class RealtimeSTTController:
    def __init__(
        self,
        *,
        perf: VoiceInteractionTimer,
        on_speculative_action: Optional[Callable[["object"], None]] = None,
        sample_rate: int = 16000,
        stt_factory: Callable[..., ElevenLabsRealtimeSTT] = ElevenLabsRealtimeSTT,
        min_stable_partials: int = 2,
    ):
        self._perf = perf
        self._on_speculative_action = on_speculative_action
        self.sample_rate = sample_rate
        self._stt_factory = stt_factory
        self.ledger = PartialActionLedger(min_stable=min_stable_partials)
        self._session: Optional[ElevenLabsRealtimeSTT] = None
        self._connect_error: Optional[Exception] = None
        self._connected = threading.Event()
        self._pending_frames: list[bytes] = []
        self._pending_lock = threading.Lock()
        self._closed = False
        self._send_failed = False

    def start(self) -> None:
        self._perf.mark("cloud_stt_connect_start")

        def connect() -> None:
            try:
                session = self._stt_factory(sample_rate=self.sample_rate, on_event=self._handle_event)
                session.connect()
                self._session = session
                self._perf.mark("cloud_stt_connected")
                self._flush_pending()
            except Exception as exc:
                self._connect_error = exc
                log.info(
                    "[STT] active_provider=whisper_fallback provider_fallback_reason=%s: %s",
                    type(exc).__name__, str(exc)[:300],
                )
            finally:
                self._connected.set()

        threading.Thread(target=connect, name="jarvis-elevenlabs-connect", daemon=True).start()

    @property
    def available(self) -> bool:
        return self._session is not None and self._connect_error is None and not self._send_failed

    def feed(self, pcm_bytes: bytes) -> None:
        if self._closed or self._send_failed:
            return
        if self._session is None:
            with self._pending_lock:
                if self._session is None:
                    if self._connect_error is None:
                        self._pending_frames.append(pcm_bytes)
                    return
        self._send(pcm_bytes)

    def _flush_pending(self) -> None:
        with self._pending_lock:
            pending, self._pending_frames = self._pending_frames, []
        for chunk in pending:
            self._send(chunk)

    def _send(self, pcm_bytes: bytes) -> None:
        try:
            self._session.send_audio(pcm_bytes)
            self._perf.mark("first_audio_chunk_sent")
        except Exception:
            self._send_failed = True

    def _handle_event(self, event: TranscriptEvent) -> None:
        if event.kind == "partial":
            self._perf.mark("first_partial_transcript")
            if not event.text:
                return
            action = self.ledger.observe_partial(event.text)
            if action is not None:
                self._perf.mark("first_stable_intent")
                if self._on_speculative_action is not None:
                    try:
                        self._on_speculative_action(action)
                    except Exception:
                        log.exception("Speculative action dispatch failed")
        elif event.kind == "error":
            log.info("ElevenLabs realtime STT reported an error mid-session: %r", event.raw.get("message_type"))

    def commit_and_close(self, timeout: float = 5.0) -> str | None:
        """Block for up to `timeout` seconds for the connection to be
        established (if it isn't already) and for the final committed
        transcript. Always returns (never raises); None means "no usable
        realtime transcript this turn -- fall back to Whisper"."""
        if not self._connected.wait(max(0.0, timeout)):
            self.close()
            return None
        if self._session is None:
            return None
        try:
            transcript = self._session.commit(timeout=timeout)
        except Exception:
            transcript = None
        self._perf.mark("committed_transcript")
        self.close()
        return transcript

    def close(self) -> None:
        self._closed = True
        if self._session is not None:
            self._session.close()
