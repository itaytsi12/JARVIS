"""Windows System Media Transport Control (SMTC) key presses.

Same mechanism as `tools/audio.py`'s existing volume keys
(`keybd_event`/`VK_VOLUME_*`), extended with the play/pause/next/previous/
stop virtual key codes. Chrome registers itself as the active SMTC media
session for a tab using the Media Session API (Apple Music Web does this),
so these hardware-level key presses reach whichever tab Chrome currently
considers "now playing" WITHOUT any browser automation call -- this is the
fast path Part 6 of the music feature requires: no Playwright round-trip,
no LLM, sub-millisecond to issue.

This is a best-effort signal, not a verified action on its own: Windows
does not report back which application handled a media key, so callers
that need proof of a state change still confirm it via the Apple Music
page/DOM (see `tools/music/apple_music_provider.py`).
"""
from __future__ import annotations

import ctypes
import time

VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3


def _press(key_code: int) -> None:
    ctypes.windll.user32.keybd_event(key_code, 0, 0, 0)
    time.sleep(0.01)
    ctypes.windll.user32.keybd_event(key_code, 0, 2, 0)


def press_play_pause() -> None:
    _press(VK_MEDIA_PLAY_PAUSE)


def press_next() -> None:
    _press(VK_MEDIA_NEXT_TRACK)


def press_previous() -> None:
    _press(VK_MEDIA_PREV_TRACK)


def press_stop() -> None:
    _press(VK_MEDIA_STOP)
