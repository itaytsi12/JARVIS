"""Windows system-level utilities."""
import subprocess
import ctypes
from ctypes import wintypes
import os

user32 = ctypes.windll.user32


def lock_computer() -> str:
    try:
        user32.LockWorkStation()
        return "Locked workstation."
    except Exception as e:
        return f"Failed to lock workstation: {e}"


def open_task_manager() -> str:
    try:
        subprocess.Popen(["taskmgr"])
        return "Opened Task Manager."
    except Exception as e:
        return f"Failed to open Task Manager: {e}"


def open_file_explorer(path: str = None) -> str:
    try:
        if not path:
            path = os.path.expanduser("~")

        os.startfile(path)
        return f"Opened explorer at {path}."
    except Exception as e:
        return f"Failed to open explorer: {e}"


def show_desktop() -> str:
    try:
        # Simulate Win+D
        user32.keybd_event(0x5B, 0, 0, 0)  # Win down
        user32.keybd_event(0x44, 0, 0, 0)  # D down
        user32.keybd_event(0x44, 0, 2, 0)  # D up
        user32.keybd_event(0x5B, 0, 2, 0)  # Win up
        return "Show desktop toggled."
    except Exception as e:
        return f"Failed to show desktop: {e}"


def minimize_foreground_window() -> str:
    try:
        hwnd = user32.GetForegroundWindow()
        user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
        return "Minimized foreground window."
    except Exception as e:
        return f"Failed to minimize window: {e}"


def maximize_foreground_window() -> str:
    try:
        hwnd = user32.GetForegroundWindow()
        user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
        return "Maximized foreground window."
    except Exception as e:
        return f"Failed to maximize window: {e}"


def restore_foreground_window() -> str:
    try:
        hwnd = user32.GetForegroundWindow()
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        return "Restored foreground window."
    except Exception as e:
        return f"Failed to restore window: {e}"


def close_foreground_window() -> str:
    try:
        hwnd = user32.GetForegroundWindow()
        user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
        return "Close message sent to foreground window."
    except Exception as e:
        return f"Failed to close window: {e}"
"""System tool placeholder."""
