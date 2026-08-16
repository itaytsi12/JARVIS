import subprocess
import time
import ctypes
import os
import shutil
import logging
import json
import threading
from pathlib import Path

from tools.window import find_application_window,find_top_window_for_pid
from tools.windows_process import hidden_process_kwargs


log = logging.getLogger("jarvis.applications")


VS_CODE_ALIASES = {"vscode", "vs code", "visual studio code", "code"}
_START_APPS_CACHE=None
_START_APPS_CACHE_AT=0.0
_START_APPS_LOCK=threading.Lock()


def _start_apps_catalog() -> list[dict]:
    global _START_APPS_CACHE,_START_APPS_CACHE_AT
    ttl=max(0.0,float(os.getenv("JARVIS_START_APPS_CACHE_TTL","300")));now=time.monotonic()
    with _START_APPS_LOCK:
        if _START_APPS_CACHE is not None and now-_START_APPS_CACHE_AT<ttl:return list(_START_APPS_CACHE)
        script="Get-StartApps | Select-Object Name,AppID | ConvertTo-Json -Compress"
        result=subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-Command",script],text=True,capture_output=True,timeout=4,**hidden_process_kwargs())
        if result.returncode or not result.stdout.strip():items=[]
        else:
            payload=json.loads(result.stdout);items=payload if isinstance(payload,list) else [payload]
            items=[item for item in items if isinstance(item,dict) and item.get("Name") and item.get("AppID")]
        _START_APPS_CACHE=list(items);_START_APPS_CACHE_AT=now;return list(items)


def _resolve_start_app_command(app_name: str) -> list[str] | None:
    """Resolve one exact Windows Start Apps display name without guessing."""
    try:
        items=_start_apps_catalog()
        normalized=" ".join(app_name.casefold().split())
        matches=[item for item in items if isinstance(item,dict) and " ".join(str(item.get("Name","")).casefold().split())==normalized and item.get("AppID")]
        return ["explorer.exe",f"shell:AppsFolder\\{matches[0]['AppID']}"] if len(matches)==1 else None
    except (OSError,subprocess.SubprocessError,ValueError,json.JSONDecodeError):return None


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


def _resolve_whatsapp_command() -> list[str] | None:
    configured=os.getenv("JARVIS_WHATSAPP_EXE")
    candidates=[Path(configured)] if configured else []
    found=shutil.which("whatsapp")
    if found:candidates.append(Path(found))
    local=os.getenv("LOCALAPPDATA")
    if local:candidates.extend([Path(local)/"WhatsApp"/"WhatsApp.exe",Path(local)/"Programs"/"WhatsApp"/"WhatsApp.exe"])
    return [str(next((path for path in candidates if path.is_file()),""))] if any(path.is_file() for path in candidates) else None


def _wait_for_visible_window(app_name,pid,timeout=6.0):
    deadline=time.perf_counter()+timeout
    while time.perf_counter()<deadline:
        hwnd=find_top_window_for_pid(pid,timeout=.1) if pid else None
        hwnd=hwnd or find_application_window(app_name)
        if hwnd and ctypes.windll.user32.IsWindowVisible(hwnd) and not ctypes.windll.user32.IsHungAppWindow(hwnd):return hwnd
        time.sleep(.05)
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
        "task manager": "taskmgr.exe",
        "control panel": "control.exe",
    }

    command = _resolve_vscode_command() if app_name == "vscode" else _resolve_whatsapp_command() if app_name=="whatsapp" else apps.get(app_name)
    if not command:
        executable=shutil.which(app_name) or shutil.which(f"{app_name}.exe")
        command=executable or _resolve_start_app_command(app_name)

    if command:
        try:
            log.info("Application requested: %s; executable resolved: %s", app_name, command[0] if isinstance(command, list) else command)
            helper_kwargs = {}
            if isinstance(command, list) and Path(command[0]).name.lower() in {"cmd.exe", "cmd", "powershell.exe", "powershell", "pwsh.exe", "pwsh"}:
                helper_kwargs = hidden_process_kwargs()
            proc = subprocess.Popen(command, **helper_kwargs)
            pid = None

            try:
                pid = proc.pid
            except Exception:
                pid = None

            shell_activation=isinstance(command,list) and len(command)>1 and str(command[0]).lower().endswith("explorer.exe") and str(command[1]).lower().startswith("shell:appsfolder\\")
            if shell_activation:pid=None

            

            hwnd=_wait_for_visible_window(app_name,pid)
            if not hwnd:
                return {"success":False,"verified":False,"message":f"Started {app_name}, but no responsive window appeared.","error":"application_window_unverified","pid":pid}

            # Return both a human message and the process/window identity so
            # later steps can focus the verified application instance.
            result = {
                "success": True,
                "verified": True,
                "message": f"Opened {app_name} successfully.",
                "pid": pid,
                "hwnd": hwnd,
            }
            log.info("Application subprocess started: app=%s pid=%s", app_name, pid)
            return result
        except Exception as e:
            log.exception("Application launch failed: %s", app_name)
            return {"success": False, "message": f"Failed to open {app_name}.", "error": str(e)}

    return {"success": False, "message": f"I don't know how to open '{app_name}' yet.", "error": "unknown_application"}


def close_application(app_name: str, hwnd: int | None = None) -> dict:
    app_name = app_name.lower().strip()

    try:
        expected_hwnd=hwnd is not None
        hwnd = hwnd or find_application_window(app_name)
        if not hwnd:
            return {"success": False, "message": f"Could not find {app_name}.", "error": "window_not_found"}
        if expected_hwnd and (not ctypes.windll.user32.IsWindow(hwnd) or not ctypes.windll.user32.IsWindowVisible(hwnd)):
            return {"success":False,"verified":False,"message":f"The expected {app_name} window is no longer available.","error":"expected_window_unavailable"}
        ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            if not ctypes.windll.user32.IsWindow(hwnd):
                return {"success": True, "verified":True,"message": f"Closed {app_name} successfully."}
            time.sleep(0.05)
        return {"success": False, "message": f"{app_name} did not close.", "error": "window_still_open"}

    except Exception as e:
        return {"success": False, "message": f"Failed to close {app_name}.", "error": str(e)}
