"""Chatterbox Multilingual TTS HTTP service.

Run this inside the .venv-chatterbox (Python 3.11) so it uses the venv
installation of `chatterbox` and `torchaudio`.

Endpoints:
 - GET /health -> {"status": "ready"} once model loaded
 - POST /synthesize -> JSON {"text": "...", "language_id": "en"}

The service loads the model once at startup using the exact code that
worked in `test_chatterbox.py` and uses `model.generate(...)` and
`torchaudio.save(...)` to write audio files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

_SERVICE_MODEL = None

PORT_DEFAULT = 5001


class Handler(BaseHTTPRequestHandler):
    server_version = "ChatterboxTTS/0.1"

    def _set_json(self, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            status = "ready" if _SERVICE_MODEL is not None else "loading"
            code = 200 if _SERVICE_MODEL is not None else 503
            self._set_json(code)
            self.wfile.write(json.dumps({"status": status}).encode("utf-8"))
            return
        self._set_json(404)
        self.wfile.write(json.dumps({"error": "not found"}).encode("utf-8"))

    def do_POST(self) -> None:
        if self.path != "/synthesize":
            self._set_json(404)
            self.wfile.write(json.dumps({"error": "not found"}).encode("utf-8"))
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "invalid json"}).encode("utf-8"))
            return

        text = data.get("text")
        # accept either 'language_id' or older 'lang' key
        language_id = data.get("language_id") or data.get("lang") or "en"

        if not text:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "empty text"}).encode("utf-8"))
            return

        try:
            synth_and_play(text, language_id)
        except Exception as e:
            # return error and include exception text
            tb = traceback.format_exc()
            self._set_json(500)
            self.wfile.write(json.dumps({"status": "error", "error": str(e), "trace": tb}).encode("utf-8"))
            return

        self._set_json(200)
        self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))


def synth_and_play(text: str, language_id: Optional[str] = None) -> None:
    """Synthesize `text` with the preloaded Chatterbox model and play it.

    This uses the exact pattern from test_chatterbox.py: `model.generate(...)`
    and `torchaudio.save(...)`. The model must already be loaded at service
    startup and stored in the `_SERVICE_MODEL` global.
    """
    global _SERVICE_MODEL
    if _SERVICE_MODEL is None:
        raise RuntimeError("Chatterbox model not loaded; service not ready")

    lang = language_id or "en"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out_path = tmp.name
    tmp.close()

    try:
        print(f"[CHATTERBOX] Generating language={lang}")
        t0 = time.perf_counter()
        try:
            wav = _SERVICE_MODEL.generate(text, language_id=lang)
        except Exception:
            # surface exact exception
            traceback.print_exc()
            raise
        dt = time.perf_counter() - t0
        print(f"[CHATTERBOX] Generated in {dt:.2f}s")

        # Save using torchaudio exactly like the working test
        try:
            import torchaudio as ta

            ta.save(out_path, wav, _SERVICE_MODEL.sr)
        except Exception:
            traceback.print_exc()
            raise

        # Play the wav (prefer sounddevice/soundfile, else OS player)
        try:
            import soundfile as sf
            import sounddevice as sd

            data, sr = sf.read(out_path)
            sd.play(data, sr)
            sd.wait()
        except Exception:
            if os.name == "nt":
                import subprocess

                subprocess.Popen(["powershell", "-Command", "Start-Process", out_path])
            else:
                import subprocess

                subprocess.Popen(["xdg-open", out_path])

    finally:
        # cleanup
        try:
            os.unlink(out_path)
        except Exception:
            pass


def run_server(port: int) -> None:
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Chatterbox service listening on http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def main() -> None:
    global _SERVICE_MODEL

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=PORT_DEFAULT)
    args = parser.parse_args()

    # Load model exactly like test_chatterbox.py
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    print(f"Using device: {device}")
    print("Loading Chatterbox Multilingual...")
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        _SERVICE_MODEL = ChatterboxMultilingualTTS.from_pretrained(device=device)
        print("Model loaded.")
    except Exception:
        # Print full traceback and exit; caller (client) should fallback to pyttsx3
        traceback.print_exc()
        print("Failed to load Chatterbox model at startup; exiting")
        sys.exit(1)

    run_server(args.port)


if __name__ == "__main__":
    main()
