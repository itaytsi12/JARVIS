"""OPTIONAL manual smoke test for the real ElevenLabs STT + TTS integration.

This makes REAL, PAID ElevenLabs API calls -- it is never run automatically
(not part of `pytest`/CI) and requires an explicit `--run` flag so it is
never triggered by accident. Requires ELEVENLABS_API_KEY and
ELEVENLABS_VOICE_ID to be configured (see .env.example).

Usage:
    python scripts/test_elevenlabs_voice.py --run
    python scripts/test_elevenlabs_voice.py --run --record-seconds 4
    python scripts/test_elevenlabs_voice.py --run --stt-only
    python scripts/test_elevenlabs_voice.py --run --tts-only --phrase "At your service, sir."
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")


def _record_short_phrase(seconds: float, sample_rate: int = 16000) -> bytes:
    import numpy as np
    import sounddevice as sd

    print(f"Recording for {seconds:.1f}s -- speak a short phrase now...")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    print("Recording finished.")
    return audio.reshape(-1).astype(np.int16).tobytes()


def _run_stt_smoke_test(record_seconds: float) -> None:
    from voice.elevenlabs_realtime_stt import ElevenLabsRealtimeSTT, ElevenLabsSTTError

    print("\n--- ElevenLabs Scribe realtime STT smoke test ---")
    events = []
    session = ElevenLabsRealtimeSTT(on_event=events.append)

    connect_started = time.perf_counter()
    try:
        session.connect()
    except ElevenLabsSTTError as exc:
        print(f"FAILED to connect: {exc}")
        return
    connect_ms = (time.perf_counter() - connect_started) * 1000
    print(f"Connected in {connect_ms:.0f} ms")

    pcm = _record_short_phrase(record_seconds, sample_rate=session.sample_rate)

    chunk_bytes = session.sample_rate * 2 // 5  # ~100ms chunks of 16-bit mono PCM
    send_started = time.perf_counter()
    for offset in range(0, len(pcm), chunk_bytes):
        session.send_audio(pcm[offset:offset + chunk_bytes])
    send_ms = (time.perf_counter() - send_started) * 1000
    print(f"Sent {len(pcm)} bytes of audio in {send_ms:.0f} ms")

    commit_started = time.perf_counter()
    transcript = session.commit(timeout=10)
    commit_ms = (time.perf_counter() - commit_started) * 1000
    session.close()

    partials = [e for e in events if e.kind == "partial"]
    print(f"Partial transcripts received: {len(partials)}")
    if partials:
        print(f"  first partial: {partials[0].text!r}")
    print(f"Committed transcript ({commit_ms:.0f} ms to arrive): {transcript!r}")


def _run_tts_smoke_test(phrase: str) -> None:
    from voice.tts import elevenlabs_tts

    print("\n--- ElevenLabs streaming TTS smoke test ---")
    if not elevenlabs_tts.is_available():
        print("FAILED: ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID not configured.")
        return

    print(f"Synthesizing and playing: {phrase!r}")
    first_chunk_ms = {}
    started = time.perf_counter()

    def on_first_chunk() -> None:
        first_chunk_ms["value"] = (time.perf_counter() - started) * 1000

    try:
        elevenlabs_tts.speak(phrase, on_first_audio_chunk=on_first_chunk)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return
    total_ms = (time.perf_counter() - started) * 1000
    print(f"First audio chunk played after {first_chunk_ms.get('value', total_ms):.0f} ms")
    print(f"Total synthesis+playback time: {total_ms:.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", required=True, help="Required: confirms you intend to make real, paid ElevenLabs API calls.")
    parser.add_argument("--record-seconds", type=float, default=3.0, help="How long to record for the STT test (default: 3s).")
    parser.add_argument("--phrase", type=str, default="At your service, sir.", help="Phrase to synthesize for the TTS test.")
    parser.add_argument("--stt-only", action="store_true", help="Only run the STT half.")
    parser.add_argument("--tts-only", action="store_true", help="Only run the TTS half.")
    args = parser.parse_args()

    _load_env()

    if not os.getenv("ELEVENLABS_API_KEY"):
        print("ELEVENLABS_API_KEY is not set (see .env.example). Nothing to test.")
        raise SystemExit(1)

    print("This makes REAL, PAID requests to the ElevenLabs API.")

    if not args.tts_only:
        _run_stt_smoke_test(args.record_seconds)
    if not args.stt_only:
        _run_tts_smoke_test(args.phrase)


if __name__ == "__main__":
    main()
