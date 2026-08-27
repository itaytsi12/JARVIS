"""ElevenLabs low-latency streaming text-to-speech.

Follows the same provider contract as `voice/tts/openai_tts.py` and
`voice/tts/chatterbox_tts.py` (`is_available()`, `speak(text, lang=None)`,
`stop()`) so `voice/text_to_speech.py` can slot it into the existing
provider-order/fallback machinery unchanged.

Unlike the other two providers this one never writes a temporary audio
file: it requests raw PCM audio over HTTP streaming
(`POST /v1/text-to-speech/{voice_id}/stream?output_format=pcm_<rate>`) and
writes each chunk straight into a live `sounddevice.OutputStream` as it
arrives, so playback can start before the full response exists. The voice
ID and model are always read from `ELEVENLABS_VOICE_ID` /
`ELEVENLABS_TTS_MODEL` at call time -- never hardcoded.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Optional

from voice.provider_health import get_provider_health

_API_BASE = "https://api.elevenlabs.io/v1/text-to-speech"
_HEALTH_NAME = "elevenlabs_tts"

_PLAYBACK_LOCK = threading.Lock()
_STOP_EVENT = threading.Event()
_CURRENT_RESPONSE = None
_CURRENT_RESPONSE_LOCK = threading.Lock()


def _sample_rate() -> int:
    try:
        return int(os.getenv("ELEVENLABS_TTS_SAMPLE_RATE", "24000"))
    except ValueError:
        return 24000


def is_available() -> bool:
    if os.getenv("ELEVENLABS_TTS_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    # A known non-transient failure (quota/insufficient funds) this
    # session -- see voice/provider_health.py -- skips straight past this
    # provider instead of retrying a doomed request on every command.
    if not get_provider_health(_HEALTH_NAME).available:
        return False
    return bool(os.getenv("ELEVENLABS_API_KEY")) and bool(os.getenv("ELEVENLABS_VOICE_ID"))


def _headers() -> dict:
    return {"xi-api-key": os.getenv("ELEVENLABS_API_KEY", ""), "Content-Type": "application/json"}


def _voice_id() -> str:
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    if not voice_id:
        raise RuntimeError("ELEVENLABS_VOICE_ID is not configured")
    return voice_id


def stream_pcm_chunks(text: str, *, timeout: float = 20.0, chunk_bytes: int = 4096):
    """Yield raw PCM S16LE mono chunks at `_sample_rate()` as they arrive
    over the network. The HTTP response is stored so `stop()` can close it
    mid-stream for fast barge-in."""
    if not text:
        return
    import requests

    voice_id = _voice_id()
    model_id = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")
    url = f"{_API_BASE}/{voice_id}/stream"
    params = {
        "optimize_streaming_latency": os.getenv("ELEVENLABS_TTS_LATENCY_LEVEL", "3"),
        "output_format": f"pcm_{_sample_rate()}",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": float(os.getenv("ELEVENLABS_TTS_STABILITY", "0.5")),
            "similarity_boost": float(os.getenv("ELEVENLABS_TTS_SIMILARITY", "0.75")),
        },
    }
    response = requests.post(url, headers=_headers(), params=params, json=payload, stream=True, timeout=timeout)
    global _CURRENT_RESPONSE
    with _CURRENT_RESPONSE_LOCK:
        _CURRENT_RESPONSE = response
    try:
        if response.status_code != 200:
            detail = response.text[:300]
            raise RuntimeError(f"ElevenLabs TTS request failed: HTTP {response.status_code}: {detail}")
        for chunk in response.iter_content(chunk_size=chunk_bytes):
            if _STOP_EVENT.is_set():
                break
            if chunk:
                yield chunk
    finally:
        response.close()
        with _CURRENT_RESPONSE_LOCK:
            if _CURRENT_RESPONSE is response:
                _CURRENT_RESPONSE = None


def speak(text: str, lang: Optional[str] = None, on_first_audio_chunk: Optional[Callable[[], None]] = None) -> None:
    """Synthesize and play `text` with the configured JARVIS voice,
    streaming playback as chunks arrive. Raises on failure so
    `voice/text_to_speech.py` can fall back to the next provider."""
    if not text:
        return
    try:
        _speak_impl(text, on_first_audio_chunk)
    except Exception as exc:
        # Only a known non-transient error (quota/funds -- see
        # voice/provider_health.py) marks the provider unavailable; an
        # ordinary transient failure is left alone so it can be retried
        # next time exactly as before this health tracking existed.
        get_provider_health(_HEALTH_NAME).note_result(exc)
        raise


def _speak_impl(text: str, on_first_audio_chunk: Optional[Callable[[], None]] = None) -> None:
    import numpy as np
    import sounddevice as sd

    _STOP_EVENT.clear()
    first_chunk_played = False
    with _PLAYBACK_LOCK:
        stream = sd.OutputStream(samplerate=_sample_rate(), channels=1, dtype="int16")
        stream.start()
        try:
            leftover = b""
            for chunk in stream_pcm_chunks(text):
                if _STOP_EVENT.is_set():
                    break
                data = leftover + chunk
                usable_len = len(data) - (len(data) % 2)
                leftover = data[usable_len:]
                if not usable_len:
                    continue
                samples = np.frombuffer(data[:usable_len], dtype="int16")
                stream.write(samples)
                if not first_chunk_played:
                    first_chunk_played = True
                    if on_first_audio_chunk is not None:
                        try:
                            on_first_audio_chunk()
                        except Exception:
                            pass
        finally:
            stream.stop()
            stream.close()
    if not first_chunk_played and not _STOP_EVENT.is_set():
        raise RuntimeError("ElevenLabs TTS produced no audio")


def stop() -> None:
    """Stop current playback quickly (barge-in): close the in-flight HTTP
    stream so no further chunks are read, signal the write loop to stop,
    and halt any audio already queued on the output device."""
    _STOP_EVENT.set()
    with _CURRENT_RESPONSE_LOCK:
        response = _CURRENT_RESPONSE
    if response is not None:
        try:
            response.close()
        except Exception:
            pass
    try:
        import sounddevice as sd
        sd.stop()
    except Exception:
        pass
