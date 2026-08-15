"""Low-latency natural speech through OpenAI's speech endpoint."""
from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path
from urllib.request import Request, urlopen


_API_URL = "https://api.openai.com/v1/audio/speech"
_MODEL = os.getenv("JARVIS_OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
_VOICE = os.getenv("JARVIS_OPENAI_TTS_VOICE", "cedar")
_INSTRUCTIONS = os.getenv(
    "JARVIS_OPENAI_TTS_INSTRUCTIONS",
    (
        "Speak like a polished AI assistant: calm, confident, warm, concise, "
        "and conversational. Use natural pacing and subtle expressiveness. "
        "Never sound robotic, theatrical, or overly excited. Clearly emphasize "
        "the action that was completed. Speak Hebrew naturally when the text is Hebrew."
    ),
)


def is_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def synthesize_wav(text: str, timeout: float = 20.0) -> bytes:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    payload = json.dumps(
        {
            "model": _MODEL,
            "voice": _VOICE,
            "input": text,
            "instructions": _INSTRUCTIONS,
            "response_format": "wav",
        }
    ).encode("utf-8")
    request = Request(
        _API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def play_wav(path: str | Path) -> None:
    """Decode a WAV container and play its samples directly."""
    import soundfile as sf
    import sounddevice as sd

    audio, sample_rate = sf.read(path, dtype="float32")
    print("shape:", audio.shape)
    print("sample_rate:", sample_rate)
    print("min:", float(audio.min()))
    print("max:", float(audio.max()))
    sd.play(audio, sample_rate)
    sd.wait()


def speak(text: str) -> None:
    if not text:
        return

    try:
        audio = synthesize_wav(text)
    except Exception:
        print("OpenAI TTS generation exception:")
        traceback.print_exc()
        raise

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(audio)
            output_path = Path(temp.name)
    except Exception:
        print("OpenAI TTS file writing exception:")
        traceback.print_exc()
        raise

    try:
        play_wav(output_path)
    except Exception:
        print("OpenAI TTS decoding/playback exception:")
        traceback.print_exc()
        raise
    finally:
        output_path.unlink(missing_ok=True)
