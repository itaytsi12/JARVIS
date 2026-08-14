from __future__ import annotations

import os
import tempfile
from threading import Thread
from typing import Any, Callable, Optional, Tuple

from brain.agent import run_agent
from brain.router import route_command

from .listener import listen_push_to_talk, is_available as listener_available
from .speech_to_text import transcribe_audio, is_available as stt_available
from .text_to_speech import speak
from .response_formatter import format_spoken_response
from .text_normalizer import normalize_transcript
from .language_utils import detect_dominant_language


def _run_with_interruptible_thread(func: Callable[..., Any], *args, **kwargs) -> Tuple[Optional[Any], Optional[Exception]]:
    """Run `func` in a background thread while keeping the main thread
    responsive to KeyboardInterrupt. Returns (result, exception).

    Note: the background thread cannot be forcefully killed; if interrupted,
    we return early and let the thread finish in background.
    """
    result_container = {}

    def target():
        try:
            result_container['result'] = func(*args, **kwargs)
        except Exception as e:
            result_container['error'] = e

    t = Thread(target=target, daemon=True)
    t.start()

    try:
        while t.is_alive():
            t.join(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user (cancelling current voice operation)...")
        return None, KeyboardInterrupt()

    return result_container.get('result'), result_container.get('error')


def one_round_push_to_talk():
    """Record, transcribe, run agent, and speak, but keep responsiveness."""
    if not listener_available():
        print("Recording dependencies missing (sounddevice/soundfile).")
        return

    if not stt_available():
        print("STT model (faster-whisper) not available.")
        return

    wav = listen_push_to_talk()

    if not wav:
        print("No audio recorded.")
        return

    # Transcribe in background so user can interrupt with Ctrl+C
    print("Transcribing... (Ctrl+C to cancel)")
    text, err = _run_with_interruptible_thread(transcribe_audio, wav)

    # Cleanup temp file
    try:
        os.unlink(wav)
    except Exception:
        pass

    if isinstance(err, KeyboardInterrupt):
        return

    if err:
        print(f"Transcription failed: {err}")
        return

    if not text:
        print("(no speech detected)")
        return

    print(f"You said: {text}")

    # Normalize transcript: remove wake prefixes and common aliases
    cleaned, wake_removed = normalize_transcript(text)
    if not cleaned:
        print("No command after wake-word removal.")
        return

    print(f"Interpreting as: {cleaned}")

    # Run agent (synchronous, small/fast) using cleaned text
    try:
        response = run_agent(cleaned)
    except KeyboardInterrupt:
        print("Agent interrupted by user.")
        return
    except Exception as e:
        print(f"Agent error: {e}")
        return

    if isinstance(response, dict):
        resp_text = response.get("message") or str(response)
    else:
        resp_text = str(response)

    print(f"Jarvis: {resp_text}")

    # Generate a deterministic, short spoken response using the route
    try:
        route = route_command(cleaned)
    except Exception:
        route = None

    # Detect dominant language for spoken replies
    lang = detect_dominant_language(text)

    spoken = format_spoken_response(cleaned, route, resp_text, lang=lang)

    # Speak in background so Ctrl+C can stop further loops quickly
    print("Speaking... (Ctrl+C to interrupt)")
    _, speak_err = _run_with_interruptible_thread(speak, spoken)
    if isinstance(speak_err, KeyboardInterrupt):
        return
    if speak_err:
        print(f"TTS/playback failed: {speak_err}")


def run_voice_loop():
    print("Voice mode: push-to-talk. Press Ctrl+C to stop.")
    try:
        while True:
            one_round_push_to_talk()
    except KeyboardInterrupt:
        print("Voice mode stopped by user.")
