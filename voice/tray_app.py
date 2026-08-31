"""Windows notification-area host for the always-on assistant."""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import log_startup_status

from .background_assistant import AlwaysOnAssistant, AssistantState
from .single_instance import SingleInstance


log = logging.getLogger("jarvis.tray")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "jarvis_background.log"

# Every AssistantState must have an entry here. A state added to the enum
# without one used to raise KeyError from `make_icon` -- confirmed live for
# INTERRUPTED_LISTENING during barge-in, where the runtime entered a
# perfectly valid state and only the TRAY crashed. `UNKNOWN_STATE_COLOR`
# below makes that impossible to repeat: a colour is a cosmetic detail, and
# a missing one must never be able to take down the notification-area icon.
STATE_COLORS = {
    AssistantState.IDLE: "#2474d2",
    AssistantState.WAKE_DETECTED: "#18a558",
    AssistantState.LISTENING: "#18a558",
    # Barge-in: the user interrupted JARVIS mid-sentence and it is now
    # listening again. Same listening green, brightened, so it is
    # distinguishable from an ordinary wake at a glance.
    AssistantState.INTERRUPTED_LISTENING: "#00c853",
    AssistantState.WAITING_FOR_LEARNING_APPROVAL: "#00a3a3",
    AssistantState.PROCESSING: "#f39c12",
    AssistantState.EXECUTING: "#f39c12",
    AssistantState.SPEAKING: "#8e44ad",
    AssistantState.ERROR: "#d63031",
}

# Defensive fallback for a state this module has not been taught about --
# a future enum member, or a caller passing something unexpected.
UNKNOWN_STATE_COLOR = "#2474d2"


def state_color(state, disabled: bool = False) -> str:
    """The tray colour for `state`, never raising.

    Rendering the tray icon is not a place where correctness is worth a
    crash: an unmapped state degrades to the idle colour (and is logged
    once so it still gets noticed) instead of killing the icon thread.
    """
    if disabled:
        return "#777777"
    color = STATE_COLORS.get(state)
    if color is None:
        log.warning("No tray colour for state %r; using the default", getattr(state, "value", state))
        return UNKNOWN_STATE_COLOR
    return color


def configure_logging() -> None:
    """Install the rotating background log.

    The implementation moved to `config/logging_setup.py` so
    `startup/launcher.py` can install the SAME handler on the SAME file --
    a windowed (pythonw.exe) launch has no console, and its startup
    failures must land where this tray's "Open logs" item points. Calling
    this still does exactly what it always did.
    """
    from config import configure_file_logging

    configure_file_logging(LOG_FILE)


def make_icon(state: AssistantState, disabled: bool = False):
    from PIL import Image, ImageDraw, ImageFont

    color = state_color(state, disabled)
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((3, 3, 61, 61), fill=color, outline="white", width=3)

    font = None
    fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for font_path in (
        fonts_dir / "arialbd.ttf",
        fonts_dir / "segoeuib.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            font = ImageFont.truetype(str(font_path), 44)
            break
        except OSError:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=44)
        except TypeError:
            font = ImageFont.load_default()

    text_box = draw.textbbox((0, 0), "J", font=font, stroke_width=1)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    position = (
        (64 - text_width) / 2 - text_box[0],
        (64 - text_height) / 2 - text_box[1],
    )
    draw.text(position, "J", font=font, fill="white", stroke_width=1, stroke_fill="white")
    return image


def tray_title(state: AssistantState, detail: str | None = None) -> str:
    return f"JARVIS — {detail or state.value}"[:127]


class TrayApplication:
    """The notification-area host.

    `assistant` and `on_exit` exist so `startup/launcher.py` can run the
    tray ALONGSIDE the graphical UI: both surfaces then observe the SAME
    `AlwaysOnAssistant` (there is only ever one microphone owner), and
    choosing Exit in the tray also shuts the window down. Passing neither
    reproduces the original behaviour exactly, which is what `run_tray`
    and every existing test still do.
    """

    def __init__(self, assistant=None, on_exit=None):
        self.icon = None
        self._autostart_enabled = False
        self._on_exit = on_exit
        self.assistant = assistant if assistant is not None else AlwaysOnAssistant(state_callback=self._state_changed)
        if assistant is not None and getattr(assistant, "state_callback", None) is None:
            self.assistant.state_callback = self._state_changed

    def _state_changed(self, state, detail=None):
        if self.icon:
            self.icon.icon = make_icon(state, disabled=not self.assistant.wake_enabled)
            self.icon.title = tray_title(state, detail)
            self.icon.update_menu()

    def _status(self, _item=None):
        return f"Status: {self.assistant.state.value} — {self.assistant.status_detail}"

    def _listen(self, _icon, _item):
        self.assistant.request_listen()

    def _toggle_wake(self, _icon, _item):
        self.assistant.set_wake_enabled(not self.assistant.wake_enabled)
        self._state_changed(self.assistant.state, self.assistant.status_detail)

    def _toggle_mute(self, _icon, _item):
        self.assistant.set_muted(not self.assistant.muted)

    def _restart(self, _icon, _item):
        self.assistant.restart()

    def _open_logs(self, _icon, _item):
        LOG_DIR.mkdir(exist_ok=True)
        os.startfile(LOG_DIR)

    def _toggle_autostart(self, _icon, _item):
        from scripts.autostart import install_autostart, remove_autostart
        try:
            remove_autostart() if self._autostart_enabled else install_autostart()
            self._autostart_enabled = not self._autostart_enabled
            if self.icon:
                self.icon.update_menu()
        except Exception as exc:
            logging.getLogger(__name__).exception("Autostart update failed")
            self.assistant._set_state(AssistantState.ERROR, f"Autostart: {exc}")

    def _exit(self, icon, _item):
        self.assistant.stop()
        icon.stop()
        if self._on_exit is not None:
            try:
                self._on_exit()
            except Exception:
                log.exception("Exit hook failed")

    def run(self) -> int:
        import pystray
        from scripts.autostart import is_autostart_enabled
        self._autostart_enabled = is_autostart_enabled()
        menu = pystray.Menu(
            pystray.MenuItem(self._status, None, enabled=False),
            pystray.MenuItem("Listen now", self._listen),
            pystray.MenuItem("Enable wake word", self._toggle_wake, checked=lambda _item: self.assistant.wake_enabled),
            pystray.MenuItem("Mute spoken responses", self._toggle_mute, checked=lambda _item: self.assistant.muted),
            pystray.MenuItem("Restart JARVIS", self._restart),
            pystray.MenuItem("Open logs", self._open_logs),
            pystray.MenuItem("Start with Windows", self._toggle_autostart, checked=lambda _item: self._autostart_enabled),
            pystray.MenuItem("Exit", self._exit),
        )
        self.icon = pystray.Icon("JARVIS", make_icon(AssistantState.IDLE), "JARVIS — Starting", menu)
        self.assistant.start()
        try:
            self.icon.run()
        finally:
            self.assistant.stop()
        return 0


def run_tray() -> int:
    # `.env` is loaded from the project root by `config/settings.py`, which
    # `configure_logging` imports -- there is deliberately no second
    # `load_dotenv` here. The old one ran AFTER `main.py` had already
    # imported `config` and cached `get_config()`, so it populated
    # `os.environ` too late to affect anything and merely looked like the
    # configuration was being handled.
    configure_logging()
    # Requirements 6/7: say plainly whether the agent provider is usable,
    # and why not when it isn't. The SAME function the typed runtime calls
    # (`main.py`), so the two can never report different things.
    log_startup_status()
    logging.getLogger("jarvis.background").info(
        "Runtime entrypoint: pid=%s ppid=%s executable=%r argv=%r cwd=%r main=%r project_root=%r",
        os.getpid(),os.getppid(),sys.executable,sys.argv,os.getcwd(),str(Path(sys.argv[0]).resolve()),str(PROJECT_ROOT),
    )
    from .startup_validation import log_provider_status
    log_provider_status()
    instance = SingleInstance()
    if not instance.acquire():
        return 0
    try:
        return TrayApplication().run()
    finally:
        instance.release()
