"""Qt/QML application host for the JARVIS interface.

Owns the GUI thread and nothing else. The backend (voice loop, agent
runtime, tray) runs on its own threads and never touches Qt; it publishes
to `config/events.py` and `ui/ui_bridge.py` marshals those onto the GUI
thread. That separation is what keeps model calls and audio off the
render loop.

`run_ui` blocks until the window closes, so a caller that also needs the
backend must start the backend FIRST and then hand the main thread over
to this -- which is exactly what `startup/launcher.py` does.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

log = logging.getLogger("jarvis.ui")

QML_DIR = Path(__file__).resolve().parent / "qml"
MAIN_QML = QML_DIR / "main.qml"


def is_available() -> tuple[bool, str | None]:
    """Can the UI actually start in this interpreter?

    Returns `(available, reason)`. Never raises and never imports Qt into
    a process that only wanted to ask -- a missing PySide6 is a normal,
    reportable state (headless/CLI runs), not a crash.
    """
    try:
        import PySide6  # noqa: F401
        from PySide6 import QtQuick  # noqa: F401
    except Exception as exc:
        return False, f"pyside6_unavailable:{type(exc).__name__}"
    if not MAIN_QML.is_file():
        return False, f"missing_qml:{MAIN_QML}"
    return True, None


class JarvisUi:
    """One Qt application plus its bridge. Construct on the main thread."""

    def __init__(self, fullscreen: bool = False):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtQml import QQmlApplicationEngine
        # Importing QtQuick registers its types with the Python bindings.
        # Without it `rootObjects()[0]` comes back wrapped as a plain
        # QWindow rather than the QQuickWindow it really is, so callers
        # lose `grabWindow()` and the other Quick-only members.
        from PySide6 import QtQuick  # noqa: F401

        from ui.ui_bridge import UiBridge

        self._app = QGuiApplication.instance() or QGuiApplication([])
        self._app.setApplicationName("JARVIS")
        self._app.setOrganizationName("JARVIS")

        self.bridge = UiBridge()
        self._engine = QQmlApplicationEngine()
        # Exposed to QML as the single object name `bridge`.
        self._engine.rootContext().setContextProperty("bridge", self.bridge)
        self._engine.setInitialProperties({"startFullscreen": bool(fullscreen)})
        self._engine.load(QUrl.fromLocalFile(str(MAIN_QML)))
        if not self._engine.rootObjects():
            raise RuntimeError(f"QML failed to load: {MAIN_QML}")
        self.bridge.start()

    @property
    def app(self):
        return self._app

    def exec(self) -> int:
        try:
            return int(self._app.exec())
        finally:
            self.bridge.stop()
            # Tear the QML scene down BEFORE the bridge. QML objects hold
            # bindings onto `bridge`; if Python collects the bridge first
            # (attribute destruction order is not guaranteed) every one of
            # those bindings re-evaluates against a dangling context
            # property and floods the log with
            # "TypeError: Cannot read property ... of null" at exit.
            # Dropping the last reference to the engine here destroys the
            # scene deterministically, while the bridge is still alive.
            engine, self._engine = self._engine, None
            del engine

    def quit(self) -> None:
        self._app.quit()


def run_ui(fullscreen: bool = False, on_started: Callable[[JarvisUi], None] | None = None) -> int:
    """Create the window and run the Qt event loop until it closes.

    `on_started` is invoked with the live `JarvisUi` once the window
    exists but before the loop runs, so a caller can keep a reference to
    the bridge.
    """
    ui = JarvisUi(fullscreen=fullscreen)
    if on_started is not None:
        on_started(ui)
    log.info("JARVIS UI started (fullscreen=%s)", fullscreen)
    return ui.exec()


__all__ = ["JarvisUi", "run_ui", "is_available", "MAIN_QML", "QML_DIR"]
