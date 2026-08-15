"""Windows notification-area host for the always-on assistant."""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .background_assistant import AlwaysOnAssistant, AssistantState
from .single_instance import SingleInstance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "jarvis_background.log"

STATE_COLORS = {
    AssistantState.IDLE: "#2474d2",
    AssistantState.WAKE_DETECTED: "#18a558",
    AssistantState.LISTENING: "#18a558",
    AssistantState.PROCESSING: "#f39c12",
    AssistantState.EXECUTING: "#f39c12",
    AssistantState.SPEAKING: "#8e44ad",
    AssistantState.ERROR: "#d63031",
}


def configure_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(item, RotatingFileHandler) for item in root.handlers):
        root.addHandler(handler)


def make_icon(state: AssistantState, disabled: bool = False):
    from PIL import Image, ImageDraw, ImageFont

    color = "#777777" if disabled else STATE_COLORS[state]
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
    def __init__(self):
        self.icon = None
        self._autostart_enabled = False
        self.assistant = AlwaysOnAssistant(state_callback=self._state_changed)

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
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    configure_logging()
    instance = SingleInstance()
    if not instance.acquire():
        return 0
    try:
        return TrayApplication().run()
    finally:
        instance.release()
