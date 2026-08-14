import ctypes
import time
from ctypes import wintypes


user32 = ctypes.WinDLL("user32", use_last_error=True)


INPUT_KEYBOARD = 1

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004


ULONG_PTR = wintypes.WPARAM


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("ki", KEYBDINPUT),
        ("mi", MOUSEINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)

    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", INPUT_UNION),
    ]


user32.SendInput.argtypes = (
    wintypes.UINT,
    ctypes.POINTER(INPUT),
    ctypes.c_int,
)

user32.SendInput.restype = wintypes.UINT


def _send_unicode_char(char: str) -> None:
    code_point = ord(char)

    key_down = INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(
            wVk=0,
            wScan=code_point,
            dwFlags=KEYEVENTF_UNICODE,
            time=0,
            dwExtraInfo=0,
        ),
    )

    key_up = INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(
            wVk=0,
            wScan=code_point,
            dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
            time=0,
            dwExtraInfo=0,
        ),
    )

    inputs = (INPUT * 2)(
        key_down,
        key_up,
    )

    sent = user32.SendInput(
        2,
        inputs,
        ctypes.sizeof(INPUT),
    )

    if sent != 2:
        error = ctypes.get_last_error()

        raise RuntimeError(
            f"Windows SendInput failed. "
            f"Sent {sent}/2 events. "
            f"GetLastError={error}"
        )


def type_text(
    text: str,
    delay: float = 0.02,
) -> str:

    if not isinstance(text, str):
        raise ValueError(
            "Text must be a string."
        )

    if not text:
        return "Nothing to type."

    for char in text:
        _send_unicode_char(char)

        if delay > 0:
            time.sleep(delay)

    return "Text typed successfully."