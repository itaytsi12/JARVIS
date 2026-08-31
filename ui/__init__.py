"""The JARVIS graphical interface (PySide6 / Qt Quick).

- `ui/app.py`         -- Qt application host; owns the GUI thread.
- `ui/ui_bridge.py`   -- the single QObject QML binds to; subscribes to
                         `config/events.py` and marshals every update onto
                         the GUI thread.
- `ui/model_status.py`-- what model modules this install actually has, so
                         the UI never claims an unconfigured provider is
                         active.
- `ui/qml/`           -- `main.qml` plus `components/`.

Importing this package must NOT import Qt: `main.py --status`, the tests
and every headless path import `ui.model_status` freely. Qt is imported
only inside `ui/app.py` and `ui/ui_bridge.py`.
"""
