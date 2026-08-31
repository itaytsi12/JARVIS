"""One controlled way to bring the whole desktop assistant up.

- `startup/launcher.py` -- the entry point (`main.py --start`, and what the
  Windows logon task runs). Single-instance guard, logging, Chrome, the
  backend/voice assistant, the tray, and the Qt window, in that order.
- `startup/chrome.py`   -- start JARVIS's OWN Chrome, and only if it is not
  already running.

Importing this package must stay free of Qt, audio and Playwright:
`startup/chrome.py` is imported by tests that have none of them, and the
launcher itself imports every heavy dependency lazily, inside the function
that needs it, so a missing one degrades to a reported, survivable failure
instead of an import-time crash.
"""
