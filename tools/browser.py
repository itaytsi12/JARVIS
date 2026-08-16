from __future__ import annotations

import ctypes,os,re,shutil,subprocess,time
from pathlib import Path
from urllib.parse import urlparse

from tools.window import find_application_windows


def _resolve_chrome() -> str | None:
    configured=os.getenv("JARVIS_CHROME_EXE")
    candidates=[Path(configured)] if configured else []
    found=shutil.which("chrome") or shutil.which("chrome.exe")
    if found:candidates.append(Path(found))
    for variable in ("ProgramFiles","ProgramFiles(x86)","LOCALAPPDATA"):
        base=os.getenv(variable)
        if base:candidates.append(Path(base)/"Google"/"Chrome"/"Application"/"chrome.exe")
    return str(next((path for path in candidates if path.is_file()),"")) or None


def _window_text(hwnd: int) -> str:
    length=ctypes.windll.user32.GetWindowTextLengthW(hwnd);buffer=ctypes.create_unicode_buffer(length+1)
    ctypes.windll.user32.GetWindowTextW(hwnd,buffer,length+1);return buffer.value


def _address_values(hwnd: int) -> list[str]:
    try:
        from pywinauto import Desktop
        window=Desktop(backend="uia").window(handle=hwnd);values=[]
        for control in window.descendants(control_type="Edit")[:12]:
            try:value=control.get_value()
            except Exception:value=control.window_text()
            if value:values.append(str(value))
        return values
    except Exception:
        return []


def _verify_url_visible(url: str,timeout: float=4.0) -> dict:
    host=(urlparse(url).hostname or "").lower().removeprefix("www.");site=host.split(".")[0]
    deadline=time.perf_counter()+timeout
    while time.perf_counter()<deadline:
        for hwnd in find_application_windows("chrome"):
          if ctypes.windll.user32.IsWindowVisible(hwnd) and not ctypes.windll.user32.IsHungAppWindow(hwnd):
            title=_window_text(hwnd).lower();addresses=[value.lower() for value in _address_values(hwnd)]
            title_matches=host in title or len(site)>=3 and re.search(rf"\b{re.escape(site)}\b",title)
            if any(host in value for value in addresses) or title_matches:
                return {"success":True,"verified":True,"hwnd":hwnd,"evidence":"window_title_or_address"}
        time.sleep(.05)
    return {"success":False,"verified":False,"error":"website_navigation_unverified"}


def open_website(url: str) -> dict:
    url=url.strip()
    if not url.startswith(("http://","https://")):url="https://"+url
    executable=_resolve_chrome()
    if not executable:return {"success":False,"verified":False,"message":"Failed to open Google Chrome.","error":"chrome_not_found"}
    try:
        command=[executable,"--new-tab"]
        profile=os.getenv("JARVIS_CHROME_USER_DATA_DIR")
        if profile:command.append(f"--user-data-dir={profile}")
        if os.getenv("JARVIS_CHROME_DISABLE_GPU","false").lower() in {"1","true","yes","on"}:command.append("--disable-gpu")
        command.append(url)
        process=subprocess.Popen(command)
        verification=_verify_url_visible(url)
        if not verification["success"]:
            return {**verification,"message":f"Chrome started, but I could not verify that {urlparse(url).hostname or url} opened.","pid":process.pid,"process_exit_code":process.poll(),"executable":executable}
        return {**verification,"message":f"Opened {url} in Google Chrome.","pid":process.pid,"executable":executable}
    except Exception as exc:
        return {"success":False,"verified":False,"message":"Failed to open Google Chrome.","error":str(exc),"executable":executable}
