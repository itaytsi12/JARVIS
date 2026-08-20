"""ElevenLabs Scribe realtime speech-to-text.

A synchronous, threaded websocket client meant to be FED audio frames by
JARVIS's single existing microphone-owning thread
(`voice/background_assistant.py`) -- it never opens its own audio device and
must never become a second microphone consumer. One instance == one captured
utterance: callers construct a fresh session only after local wake-word
detection and close it once the interaction ends (see Part M's connection
lifecycle -- no idle cloud STT streaming).

The wire protocol (endpoint, query params, message shapes) follows
ElevenLabs' realtime speech-to-text API: connect to
`wss://api.elevenlabs.io/v1/speech-to-text/realtime`, authenticate via the
`xi-api-key` header, send `input_audio_chunk` messages with base64 PCM audio,
and receive `partial_transcript` / `committed_transcript` / `final_transcript`
/ error messages back.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger("jarvis.elevenlabs_stt")

try:
    import websocket  # websocket-client
    _WEBSOCKET_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    websocket = None
    _WEBSOCKET_AVAILABLE = False

ENDPOINT = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

_ERROR_MESSAGE_TYPES = {
    "auth_error", "quota_exceeded", "rate_limited", "resource_exhausted",
    "session_time_limit_exceeded", "insufficient_audio_activity",
}


class ElevenLabsSTTError(RuntimeError):
    """Connect/auth/quota/protocol failure. Callers (background_assistant)
    must catch this and fall back to the existing local Whisper path --
    never let it crash the always-on assistant."""


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _safe_error_reason(exc: Exception) -> str:
    """A loggable connect-failure reason with the API key stripped, in case
    it ever ended up embedded in an underlying library's error string."""
    text = str(exc)
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if api_key:
        text = text.replace(api_key, "<redacted>")
    return text[:300]


def is_configured() -> bool:
    """Whether ElevenLabs realtime STT is both configured AND its client
    library is installed -- checked cheaply, no network call, no credits
    spent (Part Q's startup validation contract)."""
    if not _env_flag("ELEVENLABS_STT_ENABLED", "true"):
        return False
    if os.getenv("STT_PROVIDER", "elevenlabs").strip().lower() != "elevenlabs":
        return False
    return bool(os.getenv("ELEVENLABS_API_KEY")) and _WEBSOCKET_AVAILABLE


@dataclass
class TranscriptEvent:
    kind: str  # partial | committed | final | session_started | error | unknown
    text: str = ""
    raw: dict = field(default_factory=dict)


class _DefaultWSApp:
    """Thin adapter around `websocket.WebSocketApp` matching the small
    surface `ElevenLabsRealtimeSTT` needs -- swapped out in tests via
    `ws_app_factory` so unit tests never open a real socket."""

    def __init__(self, url, header, on_open, on_message, on_error, on_close):
        self._app = websocket.WebSocketApp(
            url, header=header, on_open=on_open, on_message=on_message,
            on_error=on_error, on_close=on_close,
        )

    def run_forever(self, **kwargs):
        self._app.run_forever(**kwargs)

    def send(self, payload: str) -> None:
        self._app.send(payload)

    def close(self) -> None:
        self._app.close()


class ElevenLabsRealtimeSTT:
    """One realtime STT session. Not reusable across interactions."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        on_event: Optional[Callable[[TranscriptEvent], None]] = None,
        connect_timeout: float | None = None,
        model: str | None = None,
        silence_ms: float | None = None,
        api_key: str | None = None,
        ws_app_factory: Callable[..., object] | None = None,
    ):
        api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ElevenLabsSTTError("ELEVENLABS_API_KEY is not configured")
        if ws_app_factory is None and not _WEBSOCKET_AVAILABLE:
            raise ElevenLabsSTTError("websocket-client is not installed")
        self._api_key = api_key
        self.sample_rate = int(sample_rate)
        self.model = model or os.getenv("ELEVENLABS_STT_MODEL", "scribe_v2_realtime")
        self.connect_timeout = (
            connect_timeout if connect_timeout is not None
            else _env_float("ELEVENLABS_STT_CONNECT_TIMEOUT", 5.0)
        )
        self._silence_secs = (
            (silence_ms if silence_ms is not None else _env_float("ELEVENLABS_STT_SILENCE_MS", 700.0))
            / 1000.0
        )
        self._on_event = on_event
        self._ws_app_factory = ws_app_factory or _DefaultWSApp
        self._ws = None
        self._recv_thread: threading.Thread | None = None
        self._connected = threading.Event()
        self._closed = threading.Event()
        self._error: Optional[Exception] = None
        self._committed_event = threading.Event()
        self._committed_text = ""
        self._lock = threading.Lock()
        self.audio_chunks_sent = 0
        self.first_partial_at: float | None = None

    # -- lifecycle -----------------------------------------------------
    def connect(self) -> None:
        # `language_code` and `include_language_detection` are both real,
        # documented query parameters of ElevenLabs' realtime Scribe
        # endpoint (confirmed against the live API reference -- no guessed
        # parameter names).
        #
        # Which of the two is sent comes from `stt_language_code`, the ONE
        # shared rule `voice/speech_to_text.py`'s Whisper fallback also
        # uses: force `language_code` only when the configuration expects
        # exactly one language, and otherwise ask Scribe to detect the
        # language per utterance. Deciding this here, from the mode name,
        # is what previously let the primary and the fallback provider
        # disagree about what counts as expected input.
        from .voice_language import get_voice_language, stt_language_code
        voice_language = get_voice_language()
        forced_language = stt_language_code(voice_language)
        params = (
            f"model_id={self.model}&audio_format=pcm_{self.sample_rate}"
            f"&commit_strategy=vad&vad_silence_threshold_secs={self._silence_secs}"
        )
        if forced_language is None:
            params += "&include_language_detection=true"
        else:
            params += f"&language_code={forced_language}"
        url = f"{ENDPOINT}?{params}"
        log.info(
            "[STT] configured_provider=elevenlabs voice_language_mode=%s forced_language=%s model=%s",
            voice_language, forced_language or "auto-detect", self.model,
        )
        ready = threading.Event()
        error_or_close = threading.Event()

        def on_open(_ws):
            self._connected.set()
            ready.set()

        def on_message(_ws, message):
            self._handle_message(message)
            if self._error is not None:
                error_or_close.set()

        def on_error(_ws, error):
            self._error = error if isinstance(error, Exception) else RuntimeError(str(error))
            error_or_close.set()
            ready.set()

        def on_close(_ws, _status_code, _msg):
            self._closed.set()
            error_or_close.set()
            ready.set()

        self._ws = self._ws_app_factory(
            url, header=[f"xi-api-key: {self._api_key}"],
            on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close,
        )
        self._recv_thread = threading.Thread(
            target=self._ws.run_forever, name="jarvis-elevenlabs-stt", daemon=True,
            kwargs={"ping_interval": 20, "ping_timeout": 10},
        )
        self._recv_thread.start()
        if not ready.wait(self.connect_timeout):
            self.close()
            log.info("[STT] active_provider=elevenlabs_realtime provider_fallback_reason=connect_timeout")
            raise ElevenLabsSTTError("timed out connecting to ElevenLabs realtime STT")
        # The raw WebSocket handshake succeeding (on_open) is NOT itself
        # proof of a successfully AUTHENTICATED realtime session: confirmed
        # live that ElevenLabs can accept the handshake, then send a
        # business-logic `auth_error` message and close the connection --
        # both arriving shortly AFTER on_open, not instead of it. Without
        # this brief grace window, `connect()` could return "success" for a
        # session that is already failing, and the caller would silently
        # get no transcript 5s later with no logged reason at all (this
        # reproduced the exact live symptom: silent fallback to Whisper).
        if not self._error and not self._closed.is_set():
            grace_secs = _env_float("ELEVENLABS_STT_AUTH_GRACE_MS", 300.0) / 1000.0
            error_or_close.wait(grace_secs)
        if self._error is not None:
            self.close()
            log.info("[STT] active_provider=elevenlabs_realtime provider_fallback_reason=%s", type(self._error).__name__ + ": " + _safe_error_reason(self._error))
            raise ElevenLabsSTTError(f"ElevenLabs realtime STT connection failed: {self._error}")
        if self._closed.is_set():
            log.info("[STT] active_provider=elevenlabs_realtime provider_fallback_reason=connection_closed_immediately")
            raise ElevenLabsSTTError("ElevenLabs realtime STT connection closed immediately")
        log.info("[STT] active_provider=elevenlabs_realtime provider_fallback_reason=none (connected)")

    @staticmethod
    def _log_committed(data: dict, text: str) -> None:
        """Safe diagnostic logging (Part 13): the transcript itself IS the
        thing being diagnosed here (item 2/13's explicit ask -- to tell
        which layer a live Hebrew failure broke at), never a secret. No
        API key/token is ever included."""
        detected_language = data.get("language_code") or data.get("language") or data.get("detected_language") or "unknown"
        log.info("[STT] committed_language=%s committed_text=%r", detected_language, text)

    def _handle_message(self, raw_message) -> None:
        try:
            data = json.loads(raw_message)
        except Exception:
            return
        message_type = data.get("message_type") or data.get("type") or ""
        text = data.get("text") or data.get("transcript") or ""
        if message_type == "partial_transcript":
            if self.first_partial_at is None:
                self.first_partial_at = time.monotonic()
            event = TranscriptEvent("partial", text, data)
        elif message_type in {"committed_transcript", "committed_transcript_with_timestamps"}:
            with self._lock:
                self._committed_text = text
            event = TranscriptEvent("committed", text, data)
            self._committed_event.set()
            self._log_committed(data, text)
        elif message_type in {"final_transcript", "final_transcript_with_timestamps"}:
            with self._lock:
                self._committed_text = text
            event = TranscriptEvent("final", text, data)
            self._committed_event.set()
            self._log_committed(data, text)
        elif message_type == "session_started":
            event = TranscriptEvent("session_started", "", data)
        elif message_type in _ERROR_MESSAGE_TYPES:
            self._error = ElevenLabsSTTError(f"{message_type}: {data.get('message', '')}")
            event = TranscriptEvent("error", "", data)
            # An error mid-session still must not hang a caller blocked in
            # commit()/close(); surface it as if the turn had committed.
            self._committed_event.set()
        else:
            event = TranscriptEvent(message_type or "unknown", text, data)
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                log.exception("ElevenLabs STT event callback failed")

    # -- audio -----------------------------------------------------------
    def send_audio(self, pcm_bytes: bytes, *, commit: bool = False) -> None:
        if self._ws is None or self._closed.is_set():
            return
        payload = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm_bytes).decode("ascii"),
            "commit": commit,
            "sample_rate": self.sample_rate,
        }
        try:
            self._ws.send(json.dumps(payload))
            self.audio_chunks_sent += 1
        except Exception as exc:
            self._error = exc
            raise ElevenLabsSTTError(f"failed to send audio to ElevenLabs: {exc}") from exc

    def commit(self, timeout: float = 5.0) -> str | None:
        """Signal end-of-turn and block for the committed/final transcript.
        Returns None (never raises) if nothing usable arrived in time or an
        error occurred -- callers fall back to Whisper on None."""
        if self._closed.is_set() and not self._committed_event.is_set():
            return None
        try:
            self.send_audio(b"", commit=True)
        except ElevenLabsSTTError:
            return None
        if not self._committed_event.wait(timeout):
            return None
        with self._lock:
            text = self._committed_text
        return text or None

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        self._closed.set()

    @property
    def connected(self) -> bool:
        return self._connected.is_set() and not self._closed.is_set()

    @property
    def error(self) -> Optional[Exception]:
        return self._error
