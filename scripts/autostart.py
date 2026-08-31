"""Shared per-user Task Scheduler configuration for JARVIS.

Installs ONE per-user scheduled task that runs `main.py --start` at logon:
the full desktop startup in `startup/launcher.py` (window, JARVIS's own
Chrome, the backend, voice, and the tray). It is a LeastPrivilege,
InteractiveToken task registered for the CURRENT user only, so it needs no
administrator rights, and every path in it -- the interpreter, the project
directory, the user name -- is resolved at install time rather than
hard-coded.

`pythonw.exe` is used deliberately: no console window is left open at
logon. That also means nothing printed is visible, which is why
`startup/launcher.py` installs the rotating log file
(`logs/jarvis_background.log`) before anything else runs.

`MultipleInstancesPolicy=IgnoreNew` stops Task Scheduler from starting a
second copy; `voice/single_instance.py`'s named mutex independently stops
one started any other way.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

from tools.windows_process import hidden_process_kwargs


TASK_NAME = "JARVIS Background Assistant"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def background_python() -> Path:
    project_pythonw = PROJECT_ROOT / ".venv-agent" / "Scripts" / "pythonw.exe"
    if project_pythonw.is_file():
        return project_pythonw
    current = Path(sys.executable)
    pythonw = current.with_name("pythonw.exe")
    return pythonw if pythonw.is_file() else current


def task_xml(delay: str = "PT10S") -> str:
    """The task definition. `delay` is a real Task Scheduler logon delay
    (ISO 8601), NOT a sleep inside Python -- the process starts late, it
    does not start early and then block."""
    command = escape(str(background_python()))
    # `--start` is the full desktop sequence (`startup/launcher.py`): the
    # window, JARVIS's own Chrome, the backend, voice AND the tray. The
    # older `--tray` action is still supported by `main.py`; it is simply
    # a strict subset of this, so the logon task uses the whole thing.
    arguments = escape(f'"{PROJECT_ROOT / "main.py"}" --start')
    working_directory = escape(str(PROJECT_ROOT))
    username = os.environ.get("USERNAME", "")
    domain = os.environ.get("USERDOMAIN", "")
    user = escape(f"{domain}\\{username}" if domain and username else username)
    return f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Start JARVIS (interface, Chrome, backend and voice) at user logon.</Description></RegistrationInfo>
  <Triggers><LogonTrigger><Enabled>true</Enabled><Delay>{delay}</Delay><UserId>{user}</UserId></LogonTrigger></Triggers>
  <Principals><Principal id="Author"><UserId>{user}</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><StartWhenAvailable>true</StartWhenAvailable><ExecutionTimeLimit>PT0S</ExecutionTimeLimit><Enabled>true</Enabled></Settings>
  <Actions Context="Author"><Exec><Command>{command}</Command><Arguments>{arguments}</Arguments><WorkingDirectory>{working_directory}</WorkingDirectory></Exec></Actions>
</Task>'''


def validate_installation() -> None:
    pythonw = background_python()
    python = pythonw.with_name("python.exe")
    if not pythonw.is_file() or not python.is_file():
        raise RuntimeError("Project-local .venv-agent is missing python.exe/pythonw.exe")
    model_dir = PROJECT_ROOT / "models" / "wake_word"
    required_models = ("melspectrogram.onnx", "embedding_model.onnx", "hey_jarvis_v0.1.onnx")
    missing = [name for name in required_models if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing wake-word models: {', '.join(missing)}")
    check = subprocess.run(
        [str(python), "-c", "import openwakeword, pystray, sounddevice, soundfile, faster_whisper"],
        capture_output=True,
        text=True,
        **hidden_process_kwargs(),
    )
    if check.returncode:
        raise RuntimeError(check.stderr.strip() or "Background Python dependencies are incomplete")


def install_autostart() -> None:
    validate_installation()
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as stream:
        path = Path(stream.name)
    try:
        path.write_text(task_xml(), encoding="utf-16")
        subprocess.run(["schtasks.exe", "/Create", "/TN", TASK_NAME, "/XML", str(path), "/F"], check=True, capture_output=True, text=True, **hidden_process_kwargs())
    finally:
        path.unlink(missing_ok=True)


def remove_autostart() -> None:
    result = subprocess.run(["schtasks.exe", "/Delete", "/TN", TASK_NAME, "/F"], capture_output=True, text=True, **hidden_process_kwargs())
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or "Could not remove JARVIS autostart task")


def is_autostart_enabled() -> bool:
    result = subprocess.run(["schtasks.exe", "/Query", "/TN", TASK_NAME], capture_output=True, text=True, **hidden_process_kwargs())
    return result.returncode == 0
