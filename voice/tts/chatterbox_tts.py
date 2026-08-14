"""Chatterbox Multilingual TTS provider wrapper.

This module is defensive: it attempts to import the `chatterbox` package and
use it to synthesize audio. If the package is not installed or the runtime
is incompatible, it raises ImportError so callers can fall back to other TTS
providers.

The wrapper loads the model once and reuses it. It writes synthesized audio to
a temporary WAV file and plays it back locally, then deletes the temp file.
"""
from __future__ import annotations

import json
import os
import time
import subprocess
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# Configuration: service port and venv location
_SERVICE_PORT = 5001
_SERVICE_URL = f'http://127.0.0.1:{_SERVICE_PORT}'
_VENV_DIR = Path(__file__).resolve().parents[2] / '.venv-chatterbox'
_VENV_PY = _VENV_DIR / 'Scripts' / 'python.exe'
_SERVICE_PROC = None
_LOG_PATH = Path(__file__).resolve().parents[1] / 'chatterbox_service.log'


def _is_service_running(timeout: float = 0.5) -> bool:
    try:
        req = Request(f'{_SERVICE_URL}/health')
        with urlopen(req, timeout=timeout) as r:
            data = json.load(r)
            return data.get('status') == 'ready'
    except Exception:
        return False


def _start_service(timeout: float = 60.0) -> bool:
    """Start the chatterbox service using the venv python, wait until healthy.
    Returns True if service is ready within timeout.
    """
    global _SERVICE_PROC
    if _is_service_running():
        return True

    python_exe = str(_VENV_PY)
    service_script = str(Path(__file__).resolve().parents[1] / 'chatterbox_service.py')
    if not Path(python_exe).exists():
        raise FileNotFoundError(f'Venv python not found: {python_exe}')

    # Start subprocess
    # Ensure log dir exists
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logfile = open(_LOG_PATH, 'ab')
    except Exception:
        logfile = subprocess.DEVNULL

    _SERVICE_PROC = subprocess.Popen([python_exe, service_script, '--port', str(_SERVICE_PORT)],
                                     stdout=logfile, stderr=logfile)

    # Poll for readiness
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_service_running(timeout=0.5):
            return True
        time.sleep(0.5)

    return False


def _read_service_log(lines: int = 40) -> str:
    try:
        with open(_LOG_PATH, 'rb') as f:
            data = f.read()
        text = data.decode('utf-8', errors='replace')
        return '\n'.join(text.splitlines()[-lines:])
    except Exception:
        return '<no log available>'

    return False


def _call_service_synthesize(text: str, lang: Optional[str] = None, timeout: float = 15.0) -> dict:
    payload = json.dumps({'text': text, 'lang': lang or ''}).encode('utf-8')
    req = Request(f'{_SERVICE_URL}/synthesize', data=payload, headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except HTTPError as e:
        try:
            return json.load(e)
        except Exception:
            raise
    except URLError as e:
        raise


def speak(text: str, lang: Optional[str] = None) -> None:
    """Send text to the Chatterbox service (running in the 3.11 venv).

    If the service is not running, attempt to start it. On failure, raise
    ImportError so callers can fallback to other TTS providers.
    """
    if not text:
        return

    # Ensure service available
    if not _is_service_running():
        try:
            ok = _start_service()
        except Exception as e:
            logtail = _read_service_log()
            raise ImportError('Failed to start Chatterbox service: %s\nLog tail:\n%s' % (e, logtail))
        if not ok:
            logtail = _read_service_log()
            raise ImportError('Chatterbox service did not become ready within timeout. Log tail:\n%s' % logtail)

    # Call synth endpoint (this will play audio on service side)
    resp = _call_service_synthesize(text, lang=lang)
    if resp.get('status') != 'ok':
        raise RuntimeError('Chatterbox service failed: %s' % resp.get('error'))
