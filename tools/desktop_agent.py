from __future__ import annotations

import time
import ctypes

from tools.window import bring_hwnd_to_foreground, find_application_window, find_top_window_for_pid


def wait_for_window(app_name: str, pid: int | None = None, timeout: float = 5.0) -> dict:
    deadline = time.perf_counter() + min(max(timeout, 0.1), 10.0)
    while time.perf_counter() < deadline:
        hwnd = find_top_window_for_pid(pid, timeout=0.1) if pid else None
        hwnd = hwnd or find_application_window(app_name)
        if (
            hwnd
            and ctypes.windll.user32.IsWindowVisible(hwnd)
            and not ctypes.windll.user32.IsHungAppWindow(hwnd)
        ):
            return {"success": True, "hwnd": hwnd, "message": f"{app_name} is ready."}
        time.sleep(0.05)
    return {"success": False, "error": f"Timed out waiting for {app_name}."}


def focus_target(app_name: str) -> dict:
    hwnd = find_application_window(app_name)
    if not hwnd or not ctypes.windll.user32.IsWindowVisible(hwnd):
        return {"success": False, "error": f"No window found for {app_name}."}
    focused = bring_hwnd_to_foreground(hwnd)
    return {"success": focused, "hwnd": hwnd, "message": f"Focused {app_name}." if focused else "Windows refused foreground focus."}


def get_controls(app_name: str, limit: int = 50) -> dict:
    """Read a concise UI Automation control list without relying on titles alone."""
    hwnd = find_application_window(app_name)
    if not hwnd:
        return {"success": False, "error": f"No window found for {app_name}."}
    try:
        from pywinauto import Desktop
    except ImportError:
        return {"success": False, "error": "pywinauto is not installed."}
    window = Desktop(backend="uia").window(handle=hwnd)
    controls = []
    for control in window.descendants()[:limit]:
        info = control.element_info
        if info.name or info.control_type:
            controls.append({"name": info.name, "type": info.control_type, "automation_id": info.automation_id})
    return {"success": True, "hwnd": hwnd, "controls": controls, "message": f"Found {len(controls)} controls."}


def click_control(app_name: str, name: str, control_type: str | None = None) -> dict:
    hwnd = find_application_window(app_name)
    if not hwnd:
        return {"success": False, "error": f"No window found for {app_name}."}
    try:
        from pywinauto import Desktop
    except ImportError:
        return {"success": False, "error": "pywinauto is not installed."}
    window = Desktop(backend="uia").window(handle=hwnd)
    criteria = {"title": name}
    if control_type:
        criteria["control_type"] = control_type
    control = window.child_window(**criteria)
    control.wait("visible enabled ready", timeout=5)
    control.click_input()
    return {"success": True, "hwnd": hwnd, "message": f"Clicked {name}."}


def type_into_control(app_name: str, text: str) -> dict:
    """Type through UI Automation, preserving SendInput as a later fallback."""
    hwnd = find_application_window(app_name)
    if not hwnd:
        return {"success": False, "error": f"No window found for {app_name}."}
    try:
        from pywinauto import Desktop
    except ImportError:
        return {"success": False, "error": "pywinauto is not installed."}
    try:
        window = Desktop(backend="uia").window(handle=hwnd)
        control = window.child_window(control_type="Edit")
        control.wait("exists visible enabled ready", timeout=2, retry_interval=0.1)
        control = control.wrapper_object()
        control.set_edit_text(text)
        value = ""
        try:
            value = control.get_value()
        except Exception:
            value = control.window_text()
        if text not in value:
            return {"success": False, "error": "UI Automation text verification failed."}
        return {"success": True, "hwnd": hwnd, "message": "Text entered and verified through UI Automation."}
    except Exception as exc:
        return {"success": False, "error": f"UI Automation typing failed: {exc}"}
