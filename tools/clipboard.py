"""Read and write the Windows clipboard.

Why this exists
---------------
The clipboard is the cheapest reliable bridge between JARVIS and an
application it cannot script. "Copy that error and tell me what it means",
"put this in my clipboard so I can paste it" and, most usefully, the
read-back half of a verification -- select-all, copy, read -- all need it,
and none of them were possible before.

Implementation notes
--------------------
`win32clipboard` (pywin32) is the primary path and is already a dependency
of the environment the assistant runs in. It is used rather than a third
party wrapper because it is what is installed, and adding a package for
two calls would be gratuitous.

The clipboard is a genuinely shared, single-owner OS resource: another
process can hold it open, and `OpenClipboard` then fails with
`ERROR_ACCESS_DENIED`. That is transient and ordinary -- Chrome and Office
both do it -- so every operation retries briefly rather than reporting a
failure the first time another program blinks. It is never held open
across anything that could block.

Only text is handled. Reading an image or a file list returns a clear
"the clipboard does not contain text" rather than a confusing empty
string, and nothing here ever writes to disk.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger("jarvis.tools")

#: Windows caps a clipboard read for us at whatever is there; this is
#: JARVIS's own cap so a 40MB copied document cannot land in a model
#: prompt. The full length is always reported alongside the truncation.
MAX_TEXT = 100_000

#: `OpenClipboard` fails while another process owns the clipboard. That is
#: normal and brief, so retry rather than reporting a failure.
_OPEN_ATTEMPTS = 8
_OPEN_DELAY = 0.05


def _clipboard_module():
    import win32clipboard  # noqa: PLC0415 -- optional, Windows-only

    return win32clipboard


class _Clipboard:
    """`with _Clipboard() as win32clipboard:` -- opens with retries, and
    always closes, including when the body raises."""

    def __enter__(self):
        module = _clipboard_module()
        last: Exception | None = None
        for _ in range(_OPEN_ATTEMPTS):
            try:
                module.OpenClipboard()
                self._module = module
                return module
            except Exception as exc:  # another process owns it right now
                last = exc
                time.sleep(_OPEN_DELAY)
        raise RuntimeError("clipboard_busy") from last

    def __exit__(self, *_exc):
        try:
            self._module.CloseClipboard()
        except Exception:
            log.debug("Closing the clipboard failed", exc_info=True)
        return False


def read_clipboard() -> dict:
    """The clipboard's text content.

    `verified` is True whenever the read genuinely happened -- including
    when the clipboard is legitimately empty, which is a fact, not a
    failure.
    """
    try:
        with _Clipboard() as win32clipboard:
            formats = {
                "text": win32clipboard.CF_UNICODETEXT,
            }
            if not win32clipboard.IsClipboardFormatAvailable(formats["text"]):
                return {
                    "success": True,
                    "verified": True,
                    "message": "The clipboard does not contain text.",
                    "text": "",
                    "empty": True,
                    "length": 0,
                }
            raw = win32clipboard.GetClipboardData(formats["text"]) or ""
    except RuntimeError:
        return {
            "success": False,
            "message": "Another program is holding the clipboard; I could not read it.",
            "error": "clipboard_busy",
        }
    except Exception as exc:
        log.debug("Clipboard read failed", exc_info=True)
        return {"success": False, "message": "I could not read the clipboard.", "error": f"{type(exc).__name__}: {exc}"}

    text = str(raw)
    truncated = len(text) > MAX_TEXT
    return {
        "success": True,
        "verified": True,
        "message": f"Read {len(text)} characters from the clipboard.",
        "text": text[:MAX_TEXT],
        "length": len(text),
        "truncated": truncated,
        "empty": not text,
    }


def write_clipboard(text: str) -> dict:
    """Replace the clipboard's contents with `text`, then read it back.

    The read-back is the verification: `SetClipboardData` returning is not
    proof the clipboard actually holds the value -- a clipboard manager or
    a virtual-desktop tool can overwrite it immediately -- so `verified`
    reflects what is really there afterwards, and says so honestly when
    something else won the race.
    """
    text = "" if text is None else str(text)
    if len(text) > MAX_TEXT:
        return {
            "success": False,
            "message": f"That text is too long for the clipboard tool ({len(text)} characters, limit {MAX_TEXT}).",
            "error": "text_too_long",
        }
    try:
        with _Clipboard() as win32clipboard:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, text)
    except RuntimeError:
        return {
            "success": False,
            "message": "Another program is holding the clipboard; I could not write to it.",
            "error": "clipboard_busy",
        }
    except Exception as exc:
        log.debug("Clipboard write failed", exc_info=True)
        return {"success": False, "message": "I could not write to the clipboard.", "error": f"{type(exc).__name__}: {exc}"}

    readback = read_clipboard()
    verified = bool(readback.get("success")) and readback.get("text") == text[: MAX_TEXT]
    return {
        "success": True,
        "verified": verified,
        "message": "Copied to the clipboard." if verified else "I wrote to the clipboard but could not confirm the contents.",
        "length": len(text),
    }


__all__ = ["read_clipboard", "write_clipboard", "MAX_TEXT"]
