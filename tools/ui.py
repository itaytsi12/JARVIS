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