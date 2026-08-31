"""What this machine is doing right now: processes, resources, network,
and the exact system volume.

This is the read-mostly counterpart to `tools/system.py` (which performs
window and shell actions). Everything here except `set_volume` is purely
observational, so all of it is `read_only` in the catalog and safe to run
concurrently with anything else.

Three deliberate choices:

- **`psutil` is the source of truth for processes and resources.** It is
  already a dependency (`requirements-agent.txt`), it is cross-checked
  against nothing else, and every call here is wrapped so a process that
  exits mid-iteration -- extremely common -- is skipped rather than
  turning a listing into an exception.
- **`process_running` matches the way a person names an app**, not the
  way Windows does. "chrome", "Chrome", "chrome.exe" and "Google Chrome"
  all have to work, because the agent gets its argument from a spoken
  request. It reports every matching PID, so "is it running" and "how
  many" are one call.
- **Volume is read and set through the real Windows endpoint API**
  (`IAudioEndpointVolume` via `comtypes`, which `pywinauto` already
  brings in), not by pressing the volume key N times. Key presses cannot
  express "set it to 30%", they move in device-defined steps, and they
  cannot report the current level at all. If the COM interface is
  unavailable for any reason the tool says so plainly instead of
  pretending -- `tools/audio.py`'s relative media-key controls remain the
  fallback and are untouched.
"""
from __future__ import annotations

import logging
import os
import shutil
import socket
import time
from typing import Any

log = logging.getLogger("jarvis.tools")

#: Cap on how many processes a single listing returns. A desktop routinely
#: has 300+; handing all of them to a model is noise, so the biggest
#: consumers are returned and the total is always reported.
MAX_PROCESSES = 40


# ---------------------------------------------------------------------------
# processes
# ---------------------------------------------------------------------------
def _psutil():
    import psutil  # noqa: PLC0415 -- optional at import time

    return psutil


def _normalize(name: str) -> str:
    name = (name or "").strip().lower()
    return name[:-4] if name.endswith(".exe") else name


def _process_row(process, with_cpu: bool = False) -> dict[str, Any] | None:
    try:
        info = process.info
    except Exception:
        return None
    memory = info.get("memory_info")
    row = {
        "pid": info.get("pid"),
        "name": info.get("name") or "",
        "memory_mb": round((getattr(memory, "rss", 0) or 0) / (1024 * 1024), 1),
    }
    if with_cpu:
        row["cpu_percent"] = info.get("cpu_percent") or 0.0
    return row


def list_processes(name: str | None = None, limit: int = MAX_PROCESSES) -> dict[str, Any]:
    """Running processes, biggest memory consumers first.

    With `name`, only processes whose executable name contains it. Without
    one, the whole machine, truncated to `limit` with the real total
    reported so the truncation is never mistaken for the full picture.
    """
    try:
        psutil = _psutil()
    except Exception:
        return {"success": False, "message": "Process inspection is unavailable (psutil is not installed).", "error": "psutil_unavailable"}

    wanted = _normalize(name) if name else None
    rows: list[dict[str, Any]] = []
    total = 0
    for process in psutil.process_iter(["pid", "name", "memory_info"]):
        row = _process_row(process)
        if row is None:
            continue
        total += 1
        if wanted and wanted not in _normalize(row["name"]):
            continue
        rows.append(row)

    matched = len(rows)
    rows.sort(key=lambda item: item["memory_mb"], reverse=True)
    limit = max(1, min(int(limit or MAX_PROCESSES), 200))
    shown = rows[:limit]
    if wanted:
        message = (
            f"{matched} process{'' if matched == 1 else 'es'} matching {name!r}."
            if matched
            else f"Nothing matching {name!r} is running."
        )
    else:
        message = f"{total} processes running; showing the {len(shown)} largest by memory."
    return {
        "success": True,
        "verified": True,
        "message": message,
        "processes": shown,
        "matched": matched,
        "total": total,
        "truncated": matched > len(shown),
    }


def process_running(name: str) -> dict[str, Any]:
    """Is an application running? Answers the way a person asks it.

    Matches on the executable name with or without `.exe`, case
    insensitively, and also on each word of a multi-word name ("google
    chrome" -> "chrome") so a spoken app name resolves.
    """
    if not (name or "").strip():
        return {"success": False, "message": "I need an application name to check.", "error": "missing_name"}
    try:
        psutil = _psutil()
    except Exception:
        return {"success": False, "message": "Process inspection is unavailable (psutil is not installed).", "error": "psutil_unavailable"}

    wanted = _normalize(name)
    # "google chrome" should find chrome.exe. Longest word first so the
    # most specific token decides.
    tokens = sorted({wanted, *(part for part in wanted.split() if len(part) > 2)}, key=len, reverse=True)
    pids: list[int] = []
    matched_name = ""
    for process in psutil.process_iter(["pid", "name"]):
        try:
            executable = _normalize(process.info.get("name") or "")
        except Exception:
            continue
        if not executable:
            continue
        if any(token and (token in executable or executable in token) for token in tokens):
            pids.append(process.info["pid"])
            matched_name = matched_name or (process.info.get("name") or "")

    running = bool(pids)
    return {
        "success": True,
        "verified": True,
        "running": running,
        "message": f"{matched_name or name} is running ({len(pids)} process{'' if len(pids) == 1 else 'es'})."
        if running
        else f"{name} is not running.",
        "pids": pids[:50],
        "process_name": matched_name,
        "count": len(pids),
    }


# ---------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------
def system_status() -> dict[str, Any]:
    """CPU, memory, disk and battery in one call.

    One tool rather than four: the agent almost always wants "how is this
    machine doing", and four round trips to answer that would cost three
    extra model turns. Any individual probe that fails is simply absent
    from the payload rather than failing the whole call.
    """
    try:
        psutil = _psutil()
    except Exception:
        return {"success": False, "message": "System inspection is unavailable (psutil is not installed).", "error": "psutil_unavailable"}

    status: dict[str, Any] = {"success": True, "verified": True}
    parts: list[str] = []

    try:
        # A short real interval: `cpu_percent()` with no interval returns
        # the value since the last call, which is meaningless on the first.
        status["cpu_percent"] = psutil.cpu_percent(interval=0.15)
        status["cpu_cores"] = psutil.cpu_count(logical=True)
        parts.append(f"CPU {status['cpu_percent']:.0f}%")
    except Exception:
        log.debug("CPU probe failed", exc_info=True)

    try:
        memory = psutil.virtual_memory()
        status["memory_percent"] = memory.percent
        status["memory_used_gb"] = round(memory.used / (1024**3), 1)
        status["memory_total_gb"] = round(memory.total / (1024**3), 1)
        parts.append(f"memory {status['memory_percent']:.0f}% ({status['memory_used_gb']} of {status['memory_total_gb']} GB)")
    except Exception:
        log.debug("Memory probe failed", exc_info=True)

    try:
        usage = shutil.disk_usage(os.environ.get("SystemDrive", "C:") + "\\")
        status["disk_free_gb"] = round(usage.free / (1024**3), 1)
        status["disk_total_gb"] = round(usage.total / (1024**3), 1)
        parts.append(f"{status['disk_free_gb']} GB disk free")
    except Exception:
        log.debug("Disk probe failed", exc_info=True)

    try:
        battery = psutil.sensors_battery()
    except Exception:
        battery = None
    if battery is not None:
        status["battery_percent"] = round(battery.percent)
        status["power_plugged"] = bool(battery.power_plugged)
        # `secsleft` is a sentinel (unlimited/unknown) as often as it is a
        # real number; reporting a nonsense duration would be worse than
        # omitting it.
        seconds_left = getattr(battery, "secsleft", None)
        if isinstance(seconds_left, int) and seconds_left > 0:
            status["battery_minutes_left"] = seconds_left // 60
        parts.append(
            f"battery {status['battery_percent']}%" + (" (charging)" if status["power_plugged"] else "")
        )
    else:
        status["battery_percent"] = None
        status["power_plugged"] = None

    # NOT `.capitalize()`: it lowercases the rest of the string, turning
    # "CPU 12%, ... 8.0 GB" into "Cpu 12%, ... 8.0 gb". The first part
    # already starts with a capitalised token, so the join needs no
    # case surgery at all.
    status["message"] = ", ".join(parts) + "." if parts else "I could not read this machine's resource state."
    status["success"] = bool(parts)
    return status


def network_status(host: str = "1.1.1.1", port: int = 53, timeout: float = 2.0) -> dict[str, Any]:
    """Is this machine actually on the internet?

    A real TCP connect, not an interface check: an adapter can be "up"
    with a captive portal or a dead uplink behind it. DNS on 1.1.1.1 is
    used because it answers on a plain TCP socket, needs no DNS
    resolution of its own (avoiding a second failure mode), and returns in
    milliseconds.
    """
    started = time.perf_counter()
    online = False
    error: str | None = None
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout)):
            online = True
    except OSError as exc:
        error = type(exc).__name__

    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    hostname = ""
    try:
        hostname = socket.gethostname()
    except Exception:
        pass
    return {
        "success": True,
        "verified": True,
        "online": online,
        "message": f"This machine is online ({latency_ms} ms to {host})." if online else "This machine appears to be offline.",
        "latency_ms": latency_ms if online else None,
        "hostname": hostname,
        "error_kind": error,
    }


# ---------------------------------------------------------------------------
# volume
# ---------------------------------------------------------------------------
#: Standard Windows Core Audio identifiers. These are OS constants, not
#: guesses -- the same ones `pycaw` uses.
_CLSID_MMDEVICE_ENUMERATOR = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
_IID_IMMDEVICE_ENUMERATOR = "{A95664D2-9614-4F35-A746-DE8DB63617E6}"
_IID_IAUDIO_ENDPOINT_VOLUME = "{5CDF2C82-841E-4546-9722-0CF74078229A}"
_ERENDER = 0
_EMULTIMEDIA = 1
_CLSCTX_ALL = 23


def _endpoint_volume():
    """The default playback device's `IAudioEndpointVolume`, or None.

    The interface is declared here rather than pulled from `pycaw` because
    `comtypes` is already installed (via `pywinauto`) and `pycaw` is not;
    adding a dependency for two methods is not worth it. Everything below
    is standard Windows Core Audio.
    """
    import comtypes  # noqa: PLC0415
    from ctypes import HRESULT, POINTER, c_float, c_uint  # noqa: PLC0415
    from ctypes.wintypes import BOOL  # noqa: PLC0415

    class IAudioEndpointVolume(comtypes.IUnknown):
        _iid_ = comtypes.GUID(_IID_IAUDIO_ENDPOINT_VOLUME)
        # Only the five methods actually used are declared; the vtable
        # order is what matters and these are the first entries of it.
        _methods_ = (
            comtypes.STDMETHOD(HRESULT, "RegisterControlChangeNotify", []),
            comtypes.STDMETHOD(HRESULT, "UnregisterControlChangeNotify", []),
            comtypes.STDMETHOD(HRESULT, "GetChannelCount", [POINTER(c_uint)]),
            comtypes.STDMETHOD(HRESULT, "SetMasterVolumeLevel", [c_float, POINTER(comtypes.GUID)]),
            comtypes.STDMETHOD(HRESULT, "SetMasterVolumeLevelScalar", [c_float, POINTER(comtypes.GUID)]),
            comtypes.STDMETHOD(HRESULT, "GetMasterVolumeLevel", [POINTER(c_float)]),
            comtypes.STDMETHOD(HRESULT, "GetMasterVolumeLevelScalar", [POINTER(c_float)]),
            comtypes.STDMETHOD(HRESULT, "SetChannelVolumeLevel", []),
            comtypes.STDMETHOD(HRESULT, "SetChannelVolumeLevelScalar", []),
            comtypes.STDMETHOD(HRESULT, "GetChannelVolumeLevel", []),
            comtypes.STDMETHOD(HRESULT, "GetChannelVolumeLevelScalar", []),
            comtypes.STDMETHOD(HRESULT, "SetMute", [BOOL, POINTER(comtypes.GUID)]),
            comtypes.STDMETHOD(HRESULT, "GetMute", [POINTER(BOOL)]),
        )

    class IMMDevice(comtypes.IUnknown):
        _iid_ = comtypes.GUID("{D666063F-1587-4E43-81F1-B948E807363F}")
        _methods_ = (
            comtypes.STDMETHOD(HRESULT, "Activate", [POINTER(comtypes.GUID), c_uint, POINTER(comtypes.c_void_p), POINTER(POINTER(IAudioEndpointVolume))]),
        )

    class IMMDeviceEnumerator(comtypes.IUnknown):
        _iid_ = comtypes.GUID(_IID_IMMDEVICE_ENUMERATOR)
        _methods_ = (
            comtypes.STDMETHOD(HRESULT, "EnumAudioEndpoints", []),
            comtypes.STDMETHOD(HRESULT, "GetDefaultAudioEndpoint", [c_uint, c_uint, POINTER(POINTER(IMMDevice))]),
        )

    enumerator = comtypes.CoCreateInstance(
        comtypes.GUID(_CLSID_MMDEVICE_ENUMERATOR),
        IMMDeviceEnumerator,
        _CLSCTX_ALL,
    )
    device = POINTER(IMMDevice)()
    enumerator.GetDefaultAudioEndpoint(_ERENDER, _EMULTIMEDIA, device)
    interface = POINTER(IAudioEndpointVolume)()
    device.Activate(comtypes.GUID(_IID_IAUDIO_ENDPOINT_VOLUME), _CLSCTX_ALL, None, interface)
    return interface


def _read_volume(interface) -> tuple[int, bool]:
    from ctypes import c_float, byref  # noqa: PLC0415
    from ctypes.wintypes import BOOL  # noqa: PLC0415

    level = c_float()
    interface.GetMasterVolumeLevelScalar(byref(level))
    muted = BOOL()
    interface.GetMute(byref(muted))
    return round(level.value * 100), bool(muted.value)


def get_volume() -> dict[str, Any]:
    """The current system volume as a percentage, and whether it is muted."""
    try:
        interface = _endpoint_volume()
        percent, muted = _read_volume(interface)
    except Exception as exc:
        log.debug("Volume read failed", exc_info=True)
        return {
            "success": False,
            "message": "I could not read the system volume on this machine.",
            "error": f"volume_interface_unavailable:{type(exc).__name__}",
        }
    return {
        "success": True,
        "verified": True,
        "volume": percent,
        "muted": muted,
        "message": f"The volume is at {percent} percent{' and muted' if muted else ''}.",
    }


def set_volume(level: int) -> dict[str, Any]:
    """Set the system volume to an exact percentage (0-100).

    Verified by reading the level back: the COM call succeeding is not by
    itself proof the device accepted the value (some endpoints quantize),
    so `volume` in the result is what the device actually reports
    afterwards, and `verified` is whether it landed within one percent of
    what was asked for.
    """
    try:
        wanted = int(level)
    except (TypeError, ValueError):
        return {"success": False, "message": "The volume level has to be a number between 0 and 100.", "error": "invalid_level"}
    if not 0 <= wanted <= 100:
        return {"success": False, "message": "The volume level has to be between 0 and 100.", "error": "level_out_of_range"}

    from ctypes import c_float  # noqa: PLC0415

    try:
        interface = _endpoint_volume()
        interface.SetMasterVolumeLevelScalar(c_float(wanted / 100.0), None)
        actual, muted = _read_volume(interface)
    except Exception as exc:
        log.debug("Volume set failed", exc_info=True)
        return {
            "success": False,
            "message": "I could not set the system volume on this machine.",
            "error": f"volume_interface_unavailable:{type(exc).__name__}",
        }

    verified = abs(actual - wanted) <= 1
    return {
        "success": True,
        "verified": verified,
        "volume": actual,
        "muted": muted,
        "message": f"Volume set to {actual} percent." if verified else f"I asked for {wanted} percent; the device is reporting {actual}.",
    }


__all__ = [
    "list_processes",
    "process_running",
    "system_status",
    "network_status",
    "get_volume",
    "set_volume",
    "MAX_PROCESSES",
]
