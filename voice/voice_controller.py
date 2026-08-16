from __future__ import annotations

import os
import tempfile
from threading import Thread
from typing import Any, Callable, Optional, Tuple

from brain.agent import run_agent
from brain.router import route_command

from .listener import listen_push_to_talk, is_available as listener_available
from .speech_to_text import transcribe_audio, is_available as stt_available
from .text_to_speech import speak, start_background
from .response_formatter import format_spoken_response
from .text_normalizer import normalize_transcript
from .language_utils import detect_dominant_language
from training_data import EventType,get_recorder
from training_data.sanitizer import sanitize_user_request


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
    recorder=get_recorder();interaction_id=recorder.begin(text,metadata={"input_mode":"voice_push_to_talk","audio_captured":False})
    recorder.record(EventType.STT_TRANSCRIPT,{"raw_stt_transcript":sanitize_user_request(text),"audio_path":None},interaction_id)
    safe_command=sanitize_user_request(cleaned)
    recorder.record(EventType.NORMALIZED_REQUEST,{"normalized_text":safe_command,"final_interpreted_command":safe_command},interaction_id)

    try:
        route = route_command(cleaned)
    except Exception:
        route = None

    # Interactive web questions are cancellable. Existing desktop actions keep
    # their original synchronous execution path and scheduler.
    if route and route.get("type") == "question":
        response, agent_error = _run_with_interruptible_thread(run_agent, cleaned, route=route,interaction_id=interaction_id,original_user_text=text)
        if isinstance(agent_error, KeyboardInterrupt):
            from brain.task_supervisor import cancel_read_only_tasks
            cancel_read_only_tasks()
            return
        if agent_error:
            print(f"Agent error: {agent_error}")
            return
    else:
        try:
            response = run_agent(cleaned, route=route,interaction_id=interaction_id,original_user_text=text)
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

    # Detect dominant language for spoken replies
    lang = detect_dominant_language(text)

    spoken = format_spoken_response(cleaned, route, resp_text, lang=lang)

    # Speak in background so Ctrl+C can stop further loops quickly
    print("Speaking... (Ctrl+C to interrupt)")
    from brain.agent import agent_runtime
    from .speech_journal import begin_speech,finish_speech
    agent_runtime.context.interrupted_response=spoken
    speech_journal=begin_speech(lang,len(spoken),interaction_id)
    speech_result, speak_err = _run_with_interruptible_thread(speak, spoken, lang=lang)
    if isinstance(speak_err, KeyboardInterrupt):
        finish_speech(speech_journal,error="interrupted")
        return
    if speak_err:
        finish_speech(speech_journal,error=speak_err)
        print(f"TTS/playback failed: {speak_err}")
        return
    finish_speech(speech_journal,speech_result)
    if isinstance(speech_result,dict) and not speech_result.get("success"):
        print(f"TTS/playback failed: {speech_result.get('error','unknown error')}")
        return
    agent_runtime.context.last_spoken_response=spoken


def _start_stt_warmup() -> None:
    """Load Whisper off the startup path while the user waits to record."""
    if not stt_available():
        return
    def warm():
        try:
            from .speech_to_text import warm_up
            warm_up()
        except Exception as exc:
            print(f"STT warm-up failed: {exc}")
    Thread(target=warm,name="jarvis-stt-warmup",daemon=True).start()


def run_voice_loop():
    start_background()
    _start_stt_warmup()
    print("Voice mode: push-to-talk. Press Ctrl+C to stop.")
    try:
        while True:
            one_round_push_to_talk()
    except KeyboardInterrupt:
        print("Voice mode stopped by user.")
