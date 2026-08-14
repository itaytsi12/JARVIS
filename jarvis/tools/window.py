import ctypes
import time

import psutil


user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


SW_RESTORE = 9


APP_PROCESSES = {
    "notepad": ["notepad.exe"],
    "calculator": ["calculatorapp.exe", "calculator.exe"],
    "paint": ["mspaint.exe"],
    "cmd": ["cmd.exe"],
    "powershell": ["powershell.exe", "pwsh.exe"],
    "chrome": ["chrome.exe"],
    "spotify": ["spotify.exe"],
    "discord": ["discord.exe"],
    "vscode": ["code.exe"],
    "vs code": ["code.exe"],
}


def _get_window_pid(hwnd: int) -> int:
    pid = ctypes.c_ulong()

    user32.GetWindowThreadProcessId(
        hwnd,
        ctypes.byref(pid),
    )

    return pid.value


def _get_process_name(pid: int) -> str:
    try:
        return psutil.Process(pid).name().lower()
    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
    ):
        return ""


def find_application_window(
    app_name: str,
) -> int | None:
    app_name = app_name.lower().strip()

    process_names = APP_PROCESSES.get(
        app_name,
        [f"{app_name}.exe"],
    )

    found_hwnd = None

    EnumWindowsProc = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )

    def callback(hwnd, _):
        nonlocal found_hwnd

        if not user32.IsWindowVisible(hwnd):
            return True

        pid = _get_window_pid(hwnd)

        process_name = _get_process_name(pid)

        if process_name in process_names:
            found_hwnd = hwnd
            return False

        return True

    callback_function = EnumWindowsProc(
        callback
    )

    user32.EnumWindows(
        callback_function,
        0,
    )

    return found_hwnd


def focus_window(hwnd: int) -> bool:
    if not hwnd:
        return False

    if not user32.IsWindow(hwnd):
        return False

    try:
        # Simpler foreground attempt without AttachThreadInput to avoid blocking and complexity.
        user32.ShowWindow(
            hwnd,
            SW_RESTORE,
        )

        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        try:
            user32.SetActiveWindow(hwnd)
        except Exception:
            pass
        try:
            user32.SetFocus(hwnd)
        except Exception:
            pass

        return (
            user32.GetForegroundWindow()
            == hwnd
        )

    except Exception:
        return False


def focus_application(
    app_name: str,
    timeout: float = 3.0,
) -> bool:
    deadline = time.perf_counter() + timeout

    while time.perf_counter() < deadline:
        hwnd = find_application_window(
            app_name
        )

        if hwnd:
            if focus_window(hwnd):
                return True

        time.sleep(0.03)

    return False


def find_top_window_for_pid(pid: int, timeout: float = 1.0) -> int | None:
    """Find the top-level visible window HWND for a given process PID.

    Polls EnumWindows quickly until a matching HWND is found or timeout elapses.
    """
    if not pid:
        return None

    deadline = time.perf_counter() + timeout

    EnumWindowsProc = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )

    def enum_once():
        found = None

        def callback(hwnd, _):
            nonlocal found

            # Accept windows that either are visible or have a non-empty title.
            try:
                visible = bool(user32.IsWindowVisible(hwnd))
            except Exception:
                visible = False

            title_length = user32.GetWindowTextLengthW(hwnd)

            if not (visible or title_length > 0):
                return True

            win_pid = _get_window_pid(hwnd)

            if win_pid == pid:
                found = hwnd
                return False

            return True

        cb = EnumWindowsProc(callback)
        user32.EnumWindows(cb, 0)

        return found

    while time.perf_counter() < deadline:
        hwnd = enum_once()

        if hwnd:
            return hwnd

        # short pause, keep responsive
        time.sleep(0.02)

    return None


def bring_hwnd_to_foreground(hwnd: int) -> bool:
    """Best-effort bring hwnd to foreground without AttachThreadInput.

    Returns True if the foreground window matches hwnd after the attempt.
    """
    if not hwnd:
        return False
    try:
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)

        if user32.GetForegroundWindow() == hwnd:
            return True

        # If simple SetForegroundWindow failed (common when console has focus),
        # attempt a short AttachThreadInput sequence as a last-resort fallback.
        try:
            foreground = user32.GetForegroundWindow()

            foreground_thread = user32.GetWindowThreadProcessId(
                foreground,
                None,
            )

            target_thread = user32.GetWindowThreadProcessId(
                hwnd,
                None,
            )

            # Only attempt if we have valid distinct thread ids
            if foreground_thread and target_thread and foreground_thread != target_thread:
                try:
                    print(f"[DEBUG] window.bring_hwnd_to_foreground: attempting AttachThreadInput foreground={foreground_thread} target={target_thread}")
                    user32.AttachThreadInput(foreground_thread, target_thread, True)
                    user32.SetForegroundWindow(hwnd)
                    user32.BringWindowToTop(hwnd)
                finally:
                    try:
                        user32.AttachThreadInput(foreground_thread, target_thread, False)
                    except Exception:
                        pass

                return user32.GetForegroundWindow() == hwnd

        except Exception:
            pass

        return False
    except Exception:
        return False