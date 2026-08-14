import ctypes
from ctypes import wintypes

import psutil


user32 = ctypes.windll.user32


def get_active_window_context() -> dict:
    hwnd = user32.GetForegroundWindow()

    if not hwnd:
        return {
            "title": None,
            "process": None,
            "pid": None,
        }

    # Get window title
    length = user32.GetWindowTextLengthW(hwnd)

    title_buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(
        hwnd,
        title_buffer,
        length + 1
    )

    title = title_buffer.value

    # Get process ID
    pid = wintypes.DWORD()

    user32.GetWindowThreadProcessId(
        hwnd,
        ctypes.byref(pid)
    )

    process_name = None

    try:
        process = psutil.Process(pid.value)
        process_name = process.name()

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    return {
        "title": title,
        "process": process_name,
        "pid": pid.value,
    }


def describe_active_window() -> str:
    context = get_active_window_context()

    return (
        f"Active window: {context['title']}\n"
        f"Process: {context['process']}\n"
        f"PID: {context['pid']}"
    )