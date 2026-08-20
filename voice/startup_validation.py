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

    from voice.voice_language import expected_input_languages, get_voice_language, stt_language_code

    stt_line = "ElevenLabs realtime" if elevenlabs_stt_configured() else "Whisper (local)"
    tts_line = tts._provider_label(tts._active_provider())
    voice_id_configured = bool(os.getenv("ELEVENLABS_VOICE_ID"))
    # Report the RESOLVED language policy, not just the configured mode:
    # "he" recognizes English too, and only a single-language mode forces a
    # code on a provider (see voice/voice_language.py). A live failure was
    # invisible precisely because the mode name alone said nothing about
    # what the fallback would actually do.
    mode = get_voice_language()
    forced = stt_language_code(mode)

    lines = (
        f"STT provider: {stt_line}",
        f"TTS provider: {tts_line}",
        f"Voice input language: mode={mode} recognizes={'/'.join(expected_input_languages(mode))} "
        f"forced_language={forced or 'auto-detect (both providers)'}",
        f"JARVIS voice: {'configured' if voice_id_configured else 'not configured'}",
        f"Whisper fallback: {'available' if whisper_available() else 'unavailable'}",
        f"TTS fallback: {'available' if _tts_fallback_available() else 'unavailable'}",
    )
    for line in lines:
        print(line)
        log.info(line)
