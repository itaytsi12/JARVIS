import ctypes
import time


VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF


def _press_media_key(key_code: int, times: int = 1):
    t0 = time.perf_counter()
    for _ in range(times):
        ctypes.windll.user32.keybd_event(key_code, 0, 0, 0)
        ctypes.windll.user32.keybd_event(key_code, 0, 2, 0)


def volume_up(amount: int = 1) -> str:
    _press_media_key(VK_VOLUME_UP, amount)
    return "Volume increased."


def volume_down(amount: int = 1) -> str:
    _press_media_key(VK_VOLUME_DOWN, amount)
    return "Volume decreased."


def mute_volume() -> str:
    _press_media_key(VK_VOLUME_MUTE)
    return "Volume mute toggled."