from __future__ import annotations

import time
import ctypes
from ctypes import wintypes

from tools.window import bring_hwnd_to_foreground, find_application_window, find_top_window_for_pid


def _select_notepad_editor(candidates:list[dict],top_area:int) -> dict | None:
    eligible=[item for item in candidates if item["visible"] and item["enabled"] and item["parent"]==item["top_hwnd"] and item["control_id"]==15 and item["area"]>=max(1,top_area)*.25]
    return eligible[0] if len(eligible)==1 else None


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
            return {"success": True, "verified": True, "hwnd": hwnd, "message": f"{app_name} is ready."}
        time.sleep(0.05)
    return {"success": False, "error": f"Timed out waiting for {app_name}."}


def focus_target(app_name: str, hwnd: int | None = None) -> dict:
    expected_hwnd=hwnd is not None
    hwnd = hwnd or find_application_window(app_name)
    if not hwnd or not ctypes.windll.user32.IsWindowVisible(hwnd):
        return {"success": False, "error": f"No window found for {app_name}."}
    if expected_hwnd and not ctypes.windll.user32.IsWindow(hwnd):
        return {"success":False,"verified":False,"error":"expected_window_unavailable"}
    focused = bring_hwnd_to_foreground(hwnd)
    return {"success": focused, "verified": focused, "hwnd": hwnd, "message": f"Focused {app_name}." if focused else "Windows refused foreground focus."}


def get_controls(app_name: str, limit: int = 50, hwnd: int | None = None) -> dict:
    """Read a concise UI Automation control list without relying on titles alone."""
    expected_hwnd=hwnd is not None
    hwnd = hwnd or find_application_window(app_name)
    if not hwnd:
        return {"success": False, "error": f"No window found for {app_name}."}
    if expected_hwnd and (not ctypes.windll.user32.IsWindow(hwnd) or not ctypes.windll.user32.IsWindowVisible(hwnd)):
        return {"success":False,"verified":False,"error":"expected_window_unavailable"}
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
    preview=[f"{item['name']} ({item['type']})" for item in controls if item["name"]][:8]
    message=f"Found {len(controls)} controls"+(f": {', '.join(preview)}" if preview else "")+"."
    return {"success": True, "verified": True, "hwnd": hwnd, "control_count":len(controls),"controls": controls, "message": message[:500]}


def click_control(app_name: str, name: str, control_type: str | None = None, hwnd: int | None = None) -> dict:
    expected_hwnd=hwnd is not None
    hwnd = hwnd or find_application_window(app_name)
    if not hwnd:
        return {"success": False, "error": f"No window found for {app_name}."}
    if expected_hwnd and (not ctypes.windll.user32.IsWindow(hwnd) or not ctypes.windll.user32.IsWindowVisible(hwnd)):
        return {"success":False,"verified":False,"error":"expected_window_unavailable"}
    try:
        from pywinauto import Desktop
    except ImportError:
        return {"success": False, "error": "pywinauto is not installed."}
    try:
        window = Desktop(backend="uia").window(handle=hwnd)
        candidates=window.descendants(control_type=control_type) if control_type else window.descendants()
        matches=[control for control in candidates if str(control.element_info.name).strip().casefold()==name.strip().casefold()]
        if len(matches)!=1:
            return {"success":False,"verified":False,"error":"ambiguous_ui_target","match_count":len(matches)}
        control=matches[0];control.wait("visible enabled ready", timeout=5)
        control.click_input()
        return {"success": True, "verified": False, "hwnd": hwnd, "message": f"Clicked {name}; the resulting application state was not independently verified."}
    except Exception as exc:
        return {"success":False,"verified":False,"error":f"UI Automation click failed: {exc}"}


def type_into_control(app_name: str, text: str, hwnd: int | None = None) -> dict:
    """Type through UI Automation, preserving SendInput as a later fallback."""
    expected_hwnd=hwnd is not None
    hwnd = hwnd or find_application_window(app_name)
    if not hwnd:
        return {"success": False, "error": f"No window found for {app_name}."}
    if expected_hwnd and (not ctypes.windll.user32.IsWindow(hwnd) or not ctypes.windll.user32.IsWindowVisible(hwnd)):
        return {"success":False,"verified":False,"error":"expected_window_unavailable"}
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
        if str(value).replace("\r\n","\n") != str(text).replace("\r\n","\n"):
            return {"success": False, "error": "UI Automation text verification failed."}
        return {"success": True, "verified": True, "hwnd": hwnd, "message": "Text entered and verified through UI Automation."}
    except Exception as exc:
        return {"success": False, "error": f"UI Automation typing failed: {exc}"}


def type_into_notepad_native(text:str,hwnd:int) -> dict:
    """Set and read back the exact Windows Notepad Edit child without UIA COM."""
    if not hwnd or not ctypes.windll.user32.IsWindow(hwnd) or not ctypes.windll.user32.IsWindowVisible(hwnd):
        return {"success":False,"verified":False,"error":"expected_window_unavailable"}
    children=[];buffer=ctypes.create_unicode_buffer(256)
    callback_type=ctypes.WINFUNCTYPE(ctypes.c_bool,ctypes.c_void_p,ctypes.c_void_p)
    def callback(child,_):
        ctypes.windll.user32.GetClassNameW(child,buffer,len(buffer))
        if buffer.value.casefold()=="edit":
            rect=wintypes.RECT();ctypes.windll.user32.GetWindowRect(child,ctypes.byref(rect));width=max(0,rect.right-rect.left);height=max(0,rect.bottom-rect.top)
            children.append({"hwnd":int(child),"top_hwnd":int(hwnd),"parent":int(ctypes.windll.user32.GetParent(child) or 0),"class_name":buffer.value,"control_id":int(ctypes.windll.user32.GetDlgCtrlID(child)),"visible":bool(ctypes.windll.user32.IsWindowVisible(child)),"enabled":bool(ctypes.windll.user32.IsWindowEnabled(child)),"rect":[rect.left,rect.top,rect.right,rect.bottom],"area":width*height})
        return True
    ctypes.windll.user32.EnumChildWindows(hwnd,callback_type(callback),0)
    top_rect=wintypes.RECT();ctypes.windll.user32.GetWindowRect(hwnd,ctypes.byref(top_rect));top_area=max(1,max(0,top_rect.right-top_rect.left)*max(0,top_rect.bottom-top_rect.top))
    selected=_select_notepad_editor(children,top_area)
    if selected is None:return {"success":False,"verified":False,"error":"notepad_edit_control_not_unique","match_count":len(children),"eligible_count":sum(1 for item in children if item["visible"] and item["enabled"] and item["parent"]==hwnd and item["control_id"]==15 and item["area"]>=top_area*.25),"editor_candidates":children}
    control_hwnd=selected["hwnd"]
    send=ctypes.windll.user32.SendMessageTimeoutW
    send.argtypes=[wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM,wintypes.UINT,wintypes.UINT,ctypes.POINTER(ctypes.c_size_t)]
    send.restype=wintypes.LPARAM
    value_pointer=ctypes.c_wchar_p(text);result=ctypes.c_size_t()
    ok=send(control_hwnd,0x000C,0,ctypes.cast(value_pointer,ctypes.c_void_p).value,0x0002,2000,ctypes.byref(result))
    if not ok:return {"success":False,"verified":False,"error":"notepad_text_set_timed_out"}
    length_result=ctypes.c_size_t();ok=send(control_hwnd,0x000E,0,0,0x0002,2000,ctypes.byref(length_result))
    if not ok:return {"success":False,"verified":False,"error":"notepad_text_length_timed_out"}
    value=ctypes.create_unicode_buffer(int(length_result.value)+1);read_result=ctypes.c_size_t()
    ok=send(control_hwnd,0x000D,len(value),ctypes.cast(value,ctypes.c_void_p).value,0x0002,2000,ctypes.byref(read_result))
    if not ok:return {"success":False,"verified":False,"error":"notepad_text_read_timed_out"}
    verified=value.value.replace("\r\n","\n")==text.replace("\r\n","\n")
    return {"success":verified,"verified":verified,"hwnd":hwnd,"control_hwnd":control_hwnd,"selected_editor":selected,"editor_candidates":children,"characters":len(text),"message":"Text entered and verified in Notepad." if verified else "Notepad text verification failed.","error":None if verified else "content_mismatch"}
