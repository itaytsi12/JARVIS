"""Windows system-level utilities."""
import subprocess
import ctypes
from ctypes import wintypes
import os
from tools.applications import _wait_for_visible_window

user32 = ctypes.windll.user32


def lock_computer() -> dict:
    try:
        ok=bool(user32.LockWorkStation())
        return {"success":ok,"message":"Locked workstation." if ok else "Windows refused to lock the workstation.","error":None if ok else "lock_failed"}
    except Exception as e:
        return {"success":False,"message":"Failed to lock workstation.","error":str(e)}


def open_task_manager() -> dict:
    try:
        process=subprocess.Popen(["taskmgr"])
        hwnd=_wait_for_visible_window("task manager",process.pid,timeout=6.0)
        if not hwnd:return {"success":False,"verified":False,"pid":process.pid,"message":"Task Manager started, but no responsive window appeared.","error":"application_window_unverified"}
        return {"success":True,"verified":True,"pid":process.pid,"hwnd":hwnd,"message":"Opened Task Manager."}
    except Exception as e:
        return {"success":False,"message":"Failed to open Task Manager.","error":str(e)}


def open_file_explorer(path: str = None) -> dict:
    try:
        if not path:
            path = os.path.expanduser("~")

        os.startfile(path)
        return {"success":True,"message":f"Opened explorer at {path}.","path":str(path)}
    except Exception as e:
        return {"success":False,"message":"Failed to open explorer.","error":str(e)}


def show_desktop() -> dict:
    try:
        # Simulate Win+D
        user32.keybd_event(0x5B, 0, 0, 0)  # Win down
        user32.keybd_event(0x44, 0, 0, 0)  # D down
        user32.keybd_event(0x44, 0, 2, 0)  # D up
        user32.keybd_event(0x5B, 0, 2, 0)  # Win up
        return {"success":True,"message":"Show desktop toggled."}
    except Exception as e:
        return {"success":False,"message":"Failed to show desktop.","error":str(e)}


def minimize_foreground_window(hwnd: int | None = None) -> dict:
    try:
        expected=hwnd is not None;hwnd = hwnd or user32.GetForegroundWindow()
        if not hwnd:return {"success":False,"message":"No foreground window was available.","error":"window_not_found"}
        if expected and (not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd)):return {"success":False,"verified":False,"error":"expected_window_unavailable"}
        user32.ShowWindow(hwnd, 6)
        verified=bool(user32.IsIconic(hwnd));return {"success":verified,"verified":verified,"hwnd":hwnd,"message":"Minimized foreground window." if verified else "Could not verify window minimization.","error":None if verified else "verification_failed"}
    except Exception as e:
        return {"success":False,"message":"Failed to minimize window.","error":str(e)}


def maximize_foreground_window(hwnd: int | None = None) -> dict:
    try:
        expected=hwnd is not None;hwnd = hwnd or user32.GetForegroundWindow()
        if not hwnd:return {"success":False,"message":"No foreground window was available.","error":"window_not_found"}
        if expected and (not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd)):return {"success":False,"verified":False,"error":"expected_window_unavailable"}
        user32.ShowWindow(hwnd, 3)
        verified=bool(user32.IsZoomed(hwnd));return {"success":verified,"verified":verified,"hwnd":hwnd,"message":"Maximized foreground window." if verified else "Could not verify window maximization.","error":None if verified else "verification_failed"}
    except Exception as e:
        return {"success":False,"message":"Failed to maximize window.","error":str(e)}


def restore_foreground_window(hwnd: int | None = None) -> dict:
    try:
        expected=hwnd is not None;hwnd = hwnd or user32.GetForegroundWindow()
        if not hwnd:return {"success":False,"message":"No foreground window was available.","error":"window_not_found"}
        if expected and not user32.IsWindow(hwnd):return {"success":False,"verified":False,"error":"expected_window_unavailable"}
        user32.ShowWindow(hwnd, 9)
        verified=bool(user32.IsWindowVisible(hwnd)) and not bool(user32.IsIconic(hwnd));return {"success":verified,"verified":verified,"hwnd":hwnd,"message":"Restored foreground window." if verified else "Could not verify window restoration.","error":None if verified else "verification_failed"}
    except Exception as e:
        return {"success":False,"message":"Failed to restore window.","error":str(e)}


def close_foreground_window(hwnd: int | None = None) -> dict:
    try:
        expected=hwnd is not None;hwnd = hwnd or user32.GetForegroundWindow()
        if not hwnd:return {"success":False,"message":"No foreground window was available.","error":"window_not_found"}
        if expected and (not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd)):return {"success":False,"verified":False,"error":"expected_window_unavailable"}
        posted=bool(user32.PostMessageW(hwnd,0x0010,0,0))
        return {"success":posted,"hwnd":hwnd,"message":"Close message sent to foreground window." if posted else "Windows refused the close request.","error":None if posted else "close_request_failed"}
    except Exception as e:
        return {"success":False,"message":"Failed to close window.","error":str(e)}

