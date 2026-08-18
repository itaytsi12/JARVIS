"""Startup provider validation (Part Q).

Logs which STT/TTS providers are configured and available WITHOUT spending
any API credits -- this only inspects environment configuration and local
package availability, never makes a real transcription/synthesis request
just to prove startup health.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("jarvis.startup")


def _tts_fallback_available() -> bool:
    from voice import text_to_speech as tts
    return bool(
        tts._openai_is_available()
        or tts._pyttsx3_available
        or (tts._chatterbox_available and tts._chatterbox_provider is not None)
    )


def log_provider_status() -> None:
    from voice.elevenlabs_realtime_stt import is_configured as elevenlabs_stt_configured
    from voice.speech_to_text import is_available as whisper_available
    from voice import text_to_speech as tts

    stt_line = "ElevenLabs realtime" if elevenlabs_stt_configured() else "Whisper (local)"
    tts_line = tts._provider_label(tts._active_provider())
    voice_id_configured = bool(os.getenv("ELEVENLABS_VOICE_ID"))

    lines = (
        f"STT provider: {stt_line}",
        f"TTS provider: {tts_line}",
        f"JARVIS voice: {'configured' if voice_id_configured else 'not configured'}",
        f"Whisper fallback: {'available' if whisper_available() else 'unavailable'}",
        f"TTS fallback: {'available' if _tts_fallback_available() else 'unavailable'}",
    )
    for line in lines:
        print(line)
        log.info(line)
