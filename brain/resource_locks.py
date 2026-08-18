"""Cancellable local and cross-process locks for shared JARVIS resources."""
from __future__ import annotations

import ctypes,hashlib,os,re,threading,time,weakref
from contextlib import contextmanager


DESKTOP_INPUT_TOOLS={"type_text","press_key","click_at","click_ui_element","focus_application","minimize_window","maximize_window","restore_window","close_window","send_whatsapp_message","save_current_document","save_active_document"}
BROWSER_TOOLS={"open_website","browser_open_url","browser_click_first_result","browser_click","browser_type","browser_select","browser_scroll","browser_go_back","browser_go_forward","browser_fullscreen"}
SPEAKER_TOOLS={"speak_response"}
# tools/browser_authenticated.py's AuthenticatedBrowserSession is a SEPARATE
# CDP attachment from the generic BROWSER_TOOLS one above
# (tools/browser_agent.py's ephemeral, unauthenticated browser) -- distinct
# lock so ordinary web browsing and authenticated-session use never
# serialize behind each other. Every tool that drives the shared
# authenticated session (Apple Music today; a future WhatsApp Web/Gmail/
# Calendar tool reuses the SAME session and MUST be added to this same set)
# shares one lock, so concurrent commands (e.g. a speculative "open music"
# racing the final "pause") never touch the same Playwright page object
# from two threads at once.
AUTHENTICATED_BROWSER_TOOLS={"open_music","music_pause","music_resume","music_stop","music_next","music_previous",
             "music_restart_track","music_shuffle_on","music_shuffle_off","music_repeat_on","music_repeat_off",
             "music_add_to_library","music_add_to_favorites","music_now_playing","music_artist_more",
             "music_queue_add","music_queue_next","music_play"}
_LOCKS={"action_plan":threading.RLock(),"desktop_input":threading.RLock(),"browser_session":threading.RLock(),"speaker":threading.RLock(),"authenticated_browser":threading.RLock()}
_DYNAMIC_LOCKS=weakref.WeakValueDictionary();_DYNAMIC_GUARD=threading.Lock()
WAIT_OBJECT_0=0;WAIT_ABANDONED=0x80;WAIT_TIMEOUT=0x102


def resource_for_tool(tool):
    if tool=="__action_plan__":return "action_plan"
    if tool in DESKTOP_INPUT_TOOLS:return "desktop_input"
    if tool in BROWSER_TOOLS:return "browser_session"
    if tool in SPEAKER_TOOLS:return "speaker"
    if tool in AUTHENTICATED_BROWSER_TOOLS:return "authenticated_browser"
    return None


def _acquire_named_mutex(resource,cancellation_token,deadline,kernel32=None):
    if os.name!="nt":return None
    kernel32=kernel32 or ctypes.windll.kernel32
    component=resource if re.fullmatch(r"[A-Za-z0-9_.-]+",resource) else hashlib.sha256(resource.encode()).hexdigest()
    handle=kernel32.CreateMutexW(None,False,f"Local\\JARVIS.Resource.{component}")
    if not handle:raise ctypes.WinError()
    try:
        while True:
            if cancellation_token is not None and cancellation_token.cancelled:raise RuntimeError("cancelled_while_waiting_for_resource")
            remaining=deadline-time.perf_counter()
            if remaining<=0:raise TimeoutError(f"resource_timeout:{resource}")
            result=kernel32.WaitForSingleObject(handle,max(1,min(50,int(remaining*1000))))
            if result in {WAIT_OBJECT_0,WAIT_ABANDONED}:return handle
            if result!=WAIT_TIMEOUT:raise OSError(f"WaitForSingleObject failed: {result}")
    except Exception:
        kernel32.CloseHandle(handle);raise


def _release_named_mutex(handle,kernel32=None):
    if handle is None:return
    kernel32=kernel32 or ctypes.windll.kernel32
    try:kernel32.ReleaseMutex(handle)
    finally:kernel32.CloseHandle(handle)


@contextmanager
def acquire_action_resource(tool,cancellation_token=None,timeout=30.0):
    resource=resource_for_tool(tool)
    if resource is None:
        yield {"resource":None,"wait_ms":0.0};return
    lock=_LOCKS[resource];started=time.perf_counter();acquired=False;deadline=started+timeout
    while not acquired:
        if cancellation_token is not None and cancellation_token.cancelled:raise RuntimeError("cancelled_while_waiting_for_resource")
        remaining=deadline-time.perf_counter()
        if remaining<=0:raise TimeoutError(f"resource_timeout:{resource}")
        acquired=lock.acquire(timeout=min(.05,remaining))
        if acquired and cancellation_token is not None and cancellation_token.cancelled:
            lock.release()
            acquired=False
            raise RuntimeError("cancelled_while_waiting_for_resource")
    named_handle=None
    try:
        named_handle=_acquire_named_mutex(resource,cancellation_token,deadline)
        yield {"resource":resource,"wait_ms":(time.perf_counter()-started)*1000,"cross_process":named_handle is not None}
    finally:
        _release_named_mutex(named_handle);lock.release()


@contextmanager
def acquire_named_resource(resource,cancellation_token=None,timeout=30.0):
    with _DYNAMIC_GUARD:
        lock=_DYNAMIC_LOCKS.get(resource)
        if lock is None:lock=threading.RLock();_DYNAMIC_LOCKS[resource]=lock
    started=time.perf_counter();deadline=started+timeout;acquired=False
    while not acquired:
        if cancellation_token is not None and cancellation_token.cancelled:raise RuntimeError("cancelled_while_waiting_for_resource")
        remaining=deadline-time.perf_counter()
        if remaining<=0:raise TimeoutError(f"resource_timeout:{resource}")
        acquired=lock.acquire(timeout=min(.05,remaining))
    named_handle=None
    try:
        named_handle=_acquire_named_mutex(resource,cancellation_token,deadline)
        yield {"resource":resource,"wait_ms":(time.perf_counter()-started)*1000,"cross_process":named_handle is not None}
    finally:_release_named_mutex(named_handle);lock.release()
