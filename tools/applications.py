import subprocess
import time
import ctypes
import os
import shutil
import logging
from pathlib import Path

from tools.window import find_application_window
from tools.windows_process import hidden_process_kwargs


log = logging.getLogger("jarvis.applications")


VS_CODE_ALIASES = {"vscode", "vs code", "visual studio code", "code"}


def _resolve_vscode_command() -> list[str] | None:
    path_command = shutil.which("code")
    if path_command:
        path = Path(path_command)
        installed_exe = path.parent.parent / "Code.exe"
        if installed_exe.is_file():
            return [str(installed_exe)]
        if path.suffix.lower() == ".exe":
            return [str(path)]
        return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", "call", str(path)]

    candidates = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.extend([
            Path(local_app_data) / "Programs" / "Microsoft VS Code" / "Code.exe",
            Path(local_app_data) / "Programs" / "Microsoft VS Code Insiders" / "Code - Insiders.exe",
        ])
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(variable)
        if base:
            candidates.extend([
                Path(base) / "Microsoft VS Code" / "Code.exe",
                Path(base) / "Microsoft VS Code Insiders" / "Code - Insiders.exe",
            ])
    for candidate in candidates:
        if candidate.is_file():
            return [str(candidate)]
    return None


def open_application(app_name: str) -> dict:
    app_name = app_name.lower().strip()

    if app_name in VS_CODE_ALIASES:
        app_name = "vscode"

    apps = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "paint": "mspaint.exe",
        "cmd": "cmd.exe",
        "powershell": "powershell.exe",
        "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "explorer": "explorer.exe",
    }

    command = _resolve_vscode_command() if app_name == "vscode" else apps.get(app_name)

    if command:
        try:
            log.info("Application requested: %s; executable resolved: %s", app_name, command[0] if isinstance(command, list) else command)
            helper_kwargs = {}
            if app_name == "vscode" and isinstance(command, list) and Path(command[0]).name.lower() in {"cmd.exe", "cmd"}:
                helper_kwargs = hidden_process_kwargs()
            proc = subprocess.Popen(command, **helper_kwargs)
            pid = None

            try:
                pid = proc.pid
            except Exception:
                pid = None

            

            # Return both a human message and the pid so callers can identify the launched process.
            result = {
                "success": True,
                "message": f"Opened {app_name} successfully.",
                "pid": pid,
            }
            log.info("Application subprocess started: app=%s pid=%s", app_name, pid)
            return result
        except Exception as e:
            log.exception("Application launch failed: %s", app_name)
            return {"success": False, "message": f"Failed to open {app_name}.", "error": str(e)}

    return {"success": False, "message": f"I don't know how to open '{app_name}' yet.", "error": "unknown_application"}


def close_application(app_name: str) -> dict:
    app_name = app_name.lower().strip()

    try:
        hwnd = find_application_window(app_name)
        if not hwnd:
            return {"success": False, "message": f"Could not find {app_name}.", "error": "window_not_found"}
        ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            if not ctypes.windll.user32.IsWindow(hwnd):
                return {"success": True, "message": f"Closed {app_name} successfully."}
            time.sleep(0.05)
        return {"success": False, "message": f"{app_name} did not close.", "error": "window_still_open"}

    except Exception as e:
        return {"success": False, "message": f"Failed to close {app_name}.", "error": str(e)}
