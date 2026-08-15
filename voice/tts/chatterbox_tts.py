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
import threading
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from tools.windows_process import hidden_process_kwargs

# Configuration: service port and venv location
_SERVICE_PORT = int(os.environ.get('JARVIS_CHATTERBOX_PORT', '5002'))
_SERVICE_URL = f'http://127.0.0.1:{_SERVICE_PORT}'
_VENV_DIR = Path(__file__).resolve().parents[2] / '.venv-chatterbox'
_VENV_PY = _VENV_DIR / 'Scripts' / 'python.exe'
_VENV_PYW = _VENV_DIR / 'Scripts' / 'pythonw.exe'
_SERVICE_PROC = None
_SERVICE_LOG_HANDLE = None
_SERVICE_START_LOCK = threading.Lock()
_LOG_PATH = Path(__file__).resolve().parents[1] / 'chatterbox_service.log'


def _get_health_status(timeout: float = 0.5) -> Optional[str]:
    """Return 'ready', 'loading', or None if service not reachable."""
    try:
        req = Request(f'{_SERVICE_URL}/health')
        with urlopen(req, timeout=timeout) as r:
            data = json.load(r)
            return data.get('status')
    except HTTPError as e:
        # If service returns HTTP error, try to read body for status
        try:
            payload = e.read().decode('utf-8')
            data = json.loads(payload)
            return data.get('status')
        except Exception:
            return None
    except Exception:
        return None


def _get_health(timeout: float = 0.5) -> Optional[dict]:
    try:
        req = Request(f'{_SERVICE_URL}/health')
        with urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            return json.loads(exc.read().decode('utf-8'))
        except Exception:
            return None
    except Exception:
        return None


def is_low_latency_ready() -> bool:
    """Return whether Chatterbox can answer promptly enough for voice mode."""
    health = _get_health(timeout=0.2)
    if not health or health.get('status') != 'ready':
        return False
    if os.environ.get('JARVIS_CHATTERBOX_ALLOW_SLOW') == '1':
        return True
    return health.get('device') not in (None, 'cpu')


def _start_service(timeout: float = 120.0) -> bool:
    with _SERVICE_START_LOCK:
        return _start_service_locked(timeout)


def _start_service_locked(timeout: float) -> bool:
    """Ensure the chatterbox service is running and becomes READY within timeout.

    Behavior:
    - If /health reports 'ready' -> return True immediately.
    - If /health reports 'loading' -> poll until 'ready' or timeout (do NOT start a new process).
    - If /health unreachable -> start the service process (unless already started) and poll.
    - Poll frequency: 0.25s.
    Returns True if service reports 'ready' within timeout, else False.
    """
    global _SERVICE_PROC, _SERVICE_LOG_HANDLE

    deadline = time.time() + timeout
    poll_interval = 0.25

    # Check current health
    status = _get_health_status(timeout=0.5)
    if status == 'ready':
        return True
    if status == 'error':
        return False

    # If service exists but is loading, wait until ready
    if status == 'loading':
        while time.time() < deadline:
            status = _get_health_status(timeout=0.5)
            if status == 'ready':
                return True
            if status == 'error':
                return False
            time.sleep(poll_interval)
        return False

    # status is None -> no service responding on port, start it
    python_exe = str(_VENV_PYW if os.name == 'nt' and _VENV_PYW.exists() else _VENV_PY)
    service_script = str(Path(__file__).resolve().parents[1] / 'chatterbox_service.py')
    if not Path(python_exe).exists():
        raise FileNotFoundError(f'Venv python not found: {python_exe}')

    # If we already started a subprocess earlier, ensure it's still running
    if _SERVICE_PROC is not None:
        if _SERVICE_PROC.poll() is None:
            # process is running; just poll health
            while time.time() < deadline:
                status = _get_health_status(timeout=0.5)
                if status == 'ready':
                    return True
                if status == 'error':
                    return False
                time.sleep(poll_interval)
            return False

    # Start subprocess and redirect output to log file
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logfile = open(_LOG_PATH, 'ab')
        _SERVICE_LOG_HANDLE = logfile
    except Exception:
        logfile = subprocess.DEVNULL

    _SERVICE_PROC = subprocess.Popen(
        [python_exe, '-u', service_script, '--port', str(_SERVICE_PORT)],
        cwd=str(Path(__file__).resolve().parents[2]),
        stdin=subprocess.DEVNULL,
        stdout=logfile,
        stderr=logfile,
        close_fds=True,
        **hidden_process_kwargs(),
    )

    # Poll until ready or timeout
    while time.time() < deadline:
        status = _get_health_status(timeout=0.5)
        if status == 'ready':
            return True
        if status == 'error':
            return False
        # If the process exited prematurely, stop waiting
        if _SERVICE_PROC.poll() is not None:
            break
        time.sleep(poll_interval)

    return False


def start_service_background() -> None:
    """Start warming Chatterbox without delaying JARVIS voice-mode startup."""
    if _get_health_status(timeout=0.2) in ('loading', 'ready'):
        return

    def warmup() -> None:
        try:
            _start_service()
        except Exception as exc:
            print(f'Chatterbox warm-up failed: {exc}')

    threading.Thread(target=warmup, name='chatterbox-warmup', daemon=True).start()


def _read_service_log(lines: int = 40) -> str:
    try:
        with open(_LOG_PATH, 'rb') as f:
            data = f.read()
        text = data.decode('utf-8', errors='replace')
        return '\n'.join(text.splitlines()[-lines:])
    except Exception:
        return '<no log available>'

    return False


def _call_service_synthesize(text: str, lang: Optional[str] = None, timeout: float = 180.0) -> dict:
    payload = json.dumps({'text': text, 'language_id': lang or 'en'}).encode('utf-8')
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
    status = _get_health_status(timeout=0.5)
    if status != 'ready':
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
