"""Incremental local Whisper transcription fed by the existing mic owner.

This controller never opens an audio device and never persists partial text.
It periodically transcribes an in-memory snapshot while the user is speaking,
feeding stable deterministic routes into the same speculative-action ledger as
the cloud realtime provider. The final full snapshot remains authoritative.
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from brain.speculative_execution import PartialActionLedger

log = logging.getLogger("jarvis.local_realtime_stt")


class LocalRealtimeSTTController:
    def __init__(self, *, perf, on_speculative_action: Optional[Callable] = None,
                 sample_rate: int = 16000, min_stable_partials: int = 2,
                 interval_seconds: float | None = None):
        self._perf = perf
        self._on_speculative_action = on_speculative_action
        self.sample_rate = sample_rate
        self.interval_seconds = interval_seconds or float(os.getenv("WHISPER_PARTIAL_INTERVAL_SECONDS", "0.8") or "0.8")
        self.ledger = PartialActionLedger(min_stable=min_stable_partials)
        self._audio = bytearray()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._worker: threading.Thread | None = None
        self._submitted_bytes = 0

    @property
    def has_stable_partial(self) -> bool:
        return self.ledger.has_fired_anything()

    def start(self) -> None:
        self._worker = threading.Thread(target=self._run, name="jarvis-local-stt", daemon=True)
        self._worker.start()

    def feed(self, pcm_bytes: bytes) -> None:
        if self._closed.is_set():
            return
        with self._lock:
            self._audio.extend(pcm_bytes)
        # Trigger on audio duration, not wall time: the first 80 ms frame is
        # too short to produce a useful partial and only wastes inference.
        interval_bytes = int(self.sample_rate * 2 * self.interval_seconds)
        if len(self._audio) - self._submitted_bytes >= interval_bytes:
            self._submitted_bytes = len(self._audio)
            self._wake.set()

    def _snapshot(self) -> bytes:
        with self._lock:
            return bytes(self._audio)

    def _transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        import numpy as np
        import soundfile as sf
        descriptor, raw_path = tempfile.mkstemp(suffix=".wav")
        os.close(descriptor)
        path = Path(raw_path)
        try:
            audio = np.frombuffer(pcm, dtype=np.int16).astype("float32") / 32768.0
            sf.write(path, audio, self.sample_rate)
            from .speech_to_text import transcribe_audio
            return transcribe_audio(str(path)).strip()
        finally:
            path.unlink(missing_ok=True)

    def _observe(self, text: str) -> None:
        if not text:
            return
        self._perf.mark("first_partial_transcript")
        action = self.ledger.observe_partial(text)
        if action is not None:
            self._perf.mark("first_stable_intent")
            if self._on_speculative_action is not None:
                self._on_speculative_action(action)

    def _run(self) -> None:
        while not self._closed.is_set():
            self._wake.wait(0.1)
            self._wake.clear()
            if self._closed.is_set():
                break
            try:
                self._observe(self._transcribe(self._snapshot()))
            except Exception:
                log.exception("Incremental local transcription failed; final fallback remains available")

    def commit_and_close(self, timeout: float = 5.0) -> str | None:
        self._closed.set()
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=max(0.0, timeout))
        try:
            transcript = self._transcribe(self._snapshot())
        except Exception:
            log.exception("Final local streaming transcription failed")
            transcript = None
        self._perf.mark("committed_transcript")
        return transcript or None

    def close(self) -> None:
        self._closed.set()
        self._wake.set()
