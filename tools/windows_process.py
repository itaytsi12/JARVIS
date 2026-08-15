"""Windows subprocess options for background helpers, never user-facing apps."""
from __future__ import annotations

import os
import subprocess


def hidden_process_kwargs() -> dict:
    if os.name != "nt" or os.getenv("DEBUG_BACKGROUND_CONSOLE", "false").lower() in {"1", "true", "yes", "on"}:
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": subprocess.CREATE_NO_WINDOW,
        "startupinfo": startupinfo,
    }
