import ctypes
import time


user32 = ctypes.windll.user32


VK_RETURN = 0x0D
VK_TAB = 0x09
VK_ESCAPE = 0x1B
VK_BACK = 0x08
VK_SPACE = 0x20
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_SHIFT = 0x10
VK_LWIN = 0x5B
VK_F5 = 0x74
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_UP = 0x26
VK_DOWN = 0x28


SPECIAL_KEYS = {
    "enter": VK_RETURN,
    "return": VK_RETURN,
    "tab": VK_TAB,
    "escape": VK_ESCAPE,
    "esc": VK_ESCAPE,
    "backspace": VK_BACK,
    "space": VK_SPACE,
    "f5": VK_F5,
    "left": VK_LEFT,
    "right": VK_RIGHT,
    "up": VK_UP,
    "down": VK_DOWN,
}


def press_key(key: str) -> str:
    """Press a key or key combination.

    Accepts single keys like 'enter', 'a', or combinations like
    'ctrl+s', 'ctrl+shift+t', 'alt+left'. Modifiers: ctrl, alt, shift, win.
    """
    key = key.lower().strip()

    # Normalize separators
    parts = [p.strip() for p in key.replace('+', ' ').split() if p.strip()]

    modifiers = []
    main_key = None

    for p in parts:
        if p in ("ctrl", "control"):
            modifiers.append(VK_CONTROL)
        elif p in ("alt",):
            modifiers.append(VK_MENU)
        elif p in ("shift",):
            modifiers.append(VK_SHIFT)
        elif p in ("win", "windows"):
            modifiers.append(0x5B)
        else:
            main_key = p

    if main_key is None:
        # If no main key specified, press modifiers alone (not typical)
        main_code = None
    else:
        main_code = SPECIAL_KEYS.get(main_key)

        if main_code is None:
            if len(main_key) == 1:
                main_code = ord(main_key.upper())
            else:
                raise ValueError(f"Unknown key: {main_key}")

    # Key down for modifiers
    for mod in modifiers:
        user32.keybd_event(mod, 0, 0, 0)

    # Main key down/up
    if main_code is not None:
        user32.keybd_event(main_code, 0, 0, 0)
        user32.keybd_event(main_code, 0, 2, 0)

    # Key up for modifiers (reverse order)
    for mod in reversed(modifiers):
        user32.keybd_event(mod, 0, 2, 0)

    return f"Pressed {key}."


def click_at(
    x: int,
    y: int,
) -> str:
    user32.SetCursorPos(
        x,
        y,
    )

    time.sleep(0.02)

    # Left button down
    user32.mouse_event(
        0x0002,
        0,
        0,
        0,
        0,
    )

    # Left button up
    user32.mouse_event(
        0x0004,
        0,
        0,
        0,
        0,
    )

    return f"Clicked at ({x}, {y})."


#: `mouse_event` wheel constants. One notch is 120 units (WHEEL_DELTA),
#: which is what every Windows application treats as "one scroll step".
_MOUSEEVENTF_WHEEL = 0x0800
_MOUSEEVENTF_HWHEEL = 0x01000
_WHEEL_DELTA = 120


def scroll_screen(direction: str = "down", clicks: int = 3, x: int | None = None, y: int | None = None) -> dict:
    """Scroll the window under the pointer.

    The desktop counterpart to `browser_scroll`, which only ever worked
    inside Playwright's own page. Windows delivers wheel input to whatever
    is under the CURSOR, not to the focused window, so an optional
    `x`/`y` moves the pointer first -- that is the only way to scroll a
    specific pane (a sidebar, a log panel) rather than whatever happened
    to be under the mouse.

    Scrolling changes nothing and is trivially undone, so this is SAFE and
    retry-safe. It reports where it scrolled rather than claiming to know
    that the content moved -- verifying that needs a screenshot, which is
    the caller's decision, not a cost paid on every scroll.
    """
    direction = (direction or "down").strip().lower()
    if direction not in {"up", "down", "left", "right"}:
        return {"success": False, "message": f"I can scroll up, down, left or right, not {direction!r}.", "error": "invalid_direction"}
    try:
        notches = max(1, min(int(clicks), 30))
    except (TypeError, ValueError):
        return {"success": False, "message": "The number of scroll clicks has to be a number.", "error": "invalid_clicks"}

    if x is not None and y is not None:
        user32.SetCursorPos(int(x), int(y))
        time.sleep(0.02)

    horizontal = direction in {"left", "right"}
    sign = -1 if direction in {"down", "left"} else 1
    event = _MOUSEEVENTF_HWHEEL if horizontal else _MOUSEEVENTF_WHEEL
    for _ in range(notches):
        # ctypes maps the negative delta through a signed int; a wheel
        # delta is a DWORD on the wire, hence the explicit two's complement.
        delta = sign * _WHEEL_DELTA
        user32.mouse_event(event, 0, 0, delta & 0xFFFFFFFF, 0)
        time.sleep(0.01)

    where = f" at ({x}, {y})" if x is not None and y is not None else ""
    return {
        "success": True,
        "verified": False,
        "message": f"Scrolled {direction} {notches} step{'' if notches == 1 else 's'}{where}.",
        "direction": direction,
        "clicks": notches,
    }
