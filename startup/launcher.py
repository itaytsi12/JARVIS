"""The one controlled JARVIS startup sequence.

`python main.py --start` -- and the Windows logon task, which runs exactly
that -- ends up here. It coordinates the pieces that already exist rather
than reimplementing any of them:

1. **Duplicate-instance protection** -- `voice/single_instance.py`'s
   Windows named mutex, under the SAME name the tray has always used, so
   `--start` and the pre-existing `--tray` cannot both be running. A
   second launch detects the first, says so, and exits 0 without opening a
   window, a backend, or a browser.
2. **Configuration and logging** -- `config/logging_setup.py`. The file
   handler is installed FIRST, which has a second, deliberate effect: a
   windowed (`pythonw.exe`) run has no console, and installing a handler
   before `configure_logging()` runs means it does not attach a
   `StreamHandler` to a `sys.stderr` that is `None`.
3. **The window** -- `ui/app.py`, on the MAIN thread, because Qt requires
   it. It is created first and everything below is started from
   `on_started`, so the core is on screen while the slow parts are still
   coming up rather than after them.
4. **JARVIS's own Chrome** -- `startup/chrome.py`, on a worker thread
   (launching and verifying it takes seconds). Never the user's personal
   profile, and never a second copy of JARVIS's own.
5. **The backend and voice** -- one `AlwaysOnAssistant`, the single owner
   of the one real microphone. `brain/agent.py`'s runtime is reached
   through it exactly as it always was.
6. **The tray** -- `voice/tray_app.py`'s `TrayApplication`, given THAT
   assistant rather than constructing a second one, on its own thread.

What this module deliberately does not do
-----------------------------------------
It does not replace `main.py --tray`, `main.py --voice` or the typed mode;
all three still work unchanged, and `--start --no-ui` reproduces the tray
behaviour exactly. It does not decide anything the configuration already
decides (`config/settings.py`'s `ui_enabled`, `ui_fullscreen`,
`auto_open_chrome`, `auto_start_voice`, `tray_enabled`) -- the CLI flags
below are per-run overrides of those settings, not a second source of
truth.

Every stage is individually survivable. Chrome failing, the tray failing,
or the voice stack failing is logged (and published to the UI) and the
rest of JARVIS still comes up; only the single-instance check can stop the
sequence, and that one exits 0 on purpose.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, replace
from typing import Any, Callable

log = logging.getLogger("jarvis.startup")

#: The mutex name. Identical to `voice/single_instance.py`'s default,
#: which is what `voice/tray_app.py::run_tray` uses -- if these differed,
#: `--start` and `--tray` would each think they were the only instance and
#: two processes would fight over one microphone.
MUTEX_NAME = r"Local\JARVIS.BackgroundAssistant"


@dataclass(frozen=True)
class StartupOptions:
    """What this run should bring up. Defaults come from `config`."""

    ui: bool = True
    fullscreen: bool = False
    chrome: bool = True
    voice: bool = True
    tray: bool = True

    @classmethod
    def from_config(cls) -> "StartupOptions":
        from config import get_config

        settings = get_config()
        return cls(
            ui=settings.ui_enabled,
            fullscreen=settings.ui_fullscreen,
            chrome=settings.auto_open_chrome,
            voice=settings.auto_start_voice,
            tray=settings.tray_enabled,
        )

    def with_overrides(self, **overrides: Any) -> "StartupOptions":
        """Apply only the overrides that were actually given (a CLI flag
        left unset is `None` and must not clobber the configured value)."""
        given = {key: bool(value) for key, value in overrides.items() if value is not None}
        return replace(self, **given) if given else self

    def describe(self) -> str:
        return (
            f"ui={self.ui} fullscreen={self.fullscreen} chrome={self.chrome} "
            f"voice={self.voice} tray={self.tray}"
        )


def _status(text: str) -> None:
    """Report one startup stage to the log and to the window.

    `config/events.py::publish` never raises and does nothing when nothing
    is subscribed, so this is equally safe in a headless run.
    """
    from config import events

    log.info("[startup] %s", text)
    events.publish(events.STATUS_TEXT, text=text)


class JarvisLauncher:
    """One startup sequence. Construct, `run()`, and it owns shutdown."""

    def __init__(self, options: StartupOptions):
        self.options = options
        self.ui = None
        self.assistant = None
        self.tray = None
        self.chrome_result: dict[str, Any] | None = None
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()
        #: Set when a headless run should keep the main thread alive.
        self._headless_idle = threading.Event()

    # -- stages ------------------------------------------------------------
    def start_chrome(self) -> None:
        """JARVIS's own Chrome, if it is not already up. Never raises."""
        from startup.chrome import ensure_jarvis_chrome

        _status("Checking JARVIS Chrome")
        self.chrome_result = ensure_jarvis_chrome(enabled=self.options.chrome)
        _status(f"Chrome: {self.chrome_result.get('action')}")

    def build_assistant(self):
        """Create the ONE always-on assistant. Returns None on failure.

        Imported here rather than at module scope so a machine without the
        audio stack can still run `--start --no-voice` and get a window.
        """
        try:
            from voice.background_assistant import AlwaysOnAssistant
            from voice.startup_validation import log_provider_status

            log_provider_status()
            self.assistant = AlwaysOnAssistant()
            return self.assistant
        except Exception:
            log.exception("Voice/backend startup failed; JARVIS is continuing without it")
            _status("Voice unavailable -- see the log")
            return None

    def start_backend(self) -> None:
        """Bring the assistant up, inside the tray when the tray is on.

        `TrayApplication.run()` starts and stops the assistant itself, so
        when the tray is running it -- and only it -- owns that lifecycle.
        Starting the assistant here as well would start it twice.
        """
        if not self.options.voice:
            _status("Voice disabled for this run")
            return
        assistant = self.build_assistant()
        if assistant is None:
            return

        if self.options.tray:
            try:
                self._run_tray(assistant)
                return
            except Exception:
                # The tray is a convenience; the assistant is not. Fall
                # through and start it directly rather than losing voice
                # because an icon could not be drawn.
                log.exception("Tray startup failed; starting the assistant without it")

        _status("Starting voice assistant")
        assistant.start()
        _status("JARVIS ready")

    def _run_tray(self, assistant) -> None:
        from voice.tray_app import TrayApplication

        self.tray = TrayApplication(assistant=assistant, on_exit=self.request_shutdown)
        _status("Starting tray and voice assistant")
        self._spawn("jarvis-tray", self.tray.run)

    def _spawn(self, name: str, target: Callable[[], Any]) -> threading.Thread:
        def work() -> None:
            try:
                target()
            except Exception:
                log.exception("Startup thread %r failed", name)

        thread = threading.Thread(target=work, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)
        return thread

    # -- shutdown ----------------------------------------------------------
    def request_shutdown(self) -> None:
        """Close everything. Safe to call from any thread -- the tray's
        Exit item calls it from the tray's own thread."""
        if self._stopping.is_set():
            return
        self._stopping.set()
        self._headless_idle.set()
        if self.ui is not None:
            # Qt must be touched on the GUI thread; the bridge already owns
            # the queued-connection marshalling for exactly this.
            try:
                self.ui.bridge.run_on_gui_thread(self.ui.quit)
            except Exception:
                log.exception("Could not close the JARVIS window")

    def shutdown(self) -> None:
        """Stop the assistant and the tray. Idempotent."""
        self._stopping.set()
        self._headless_idle.set()
        if self.tray is not None and getattr(self.tray, "icon", None) is not None:
            try:
                # Ends TrayApplication.run(), whose own `finally` stops the
                # assistant -- so the assistant is not stopped twice here.
                self.tray.icon.stop()
            except Exception:
                log.exception("Could not stop the tray icon")
        elif self.assistant is not None:
            try:
                self.assistant.stop()
            except Exception:
                log.exception("Could not stop the voice assistant")
        for thread in self._threads:
            thread.join(timeout=5)

    # -- entry point -------------------------------------------------------
    def run(self) -> int:
        options = self.options
        log.info("JARVIS startup: %s", options.describe())

        if options.ui:
            from ui.app import is_available, run_ui

            available, reason = is_available()
            if available:
                try:
                    return self._run_with_ui(run_ui)
                finally:
                    self.shutdown()
            # A missing PySide6 must not mean a missing JARVIS.
            log.error("The graphical interface is unavailable (%s); starting without it", reason)

        try:
            return self._run_headless()
        finally:
            self.shutdown()

    def _run_with_ui(self, run_ui) -> int:
        def on_started(ui) -> None:
            # The window exists but the Qt loop has not started yet.
            # Everything below is dispatched to worker threads, so the
            # loop starts immediately and the core is visible while Chrome
            # and the wake-word model are still loading.
            self.ui = ui
            _status("Starting JARVIS")
            self._spawn("jarvis-chrome-startup", self.start_chrome)
            self._spawn("jarvis-backend-startup", self.start_backend)

        return int(run_ui(fullscreen=self.options.fullscreen, on_started=on_started) or 0)

    def _run_headless(self) -> int:
        """No window. The tray (if enabled) owns the main thread, exactly
        as `voice/tray_app.py::run_tray` always has."""
        self._spawn("jarvis-chrome-startup", self.start_chrome)

        if not self.options.voice:
            log.error("Nothing to run: the interface is off and voice is disabled.")
            return 1

        assistant = self.build_assistant()
        if assistant is None:
            return 1

        if self.options.tray:
            from voice.tray_app import TrayApplication

            self.tray = TrayApplication(assistant=assistant, on_exit=self.request_shutdown)
            _status("Starting tray and voice assistant")
            return int(self.tray.run() or 0)

        _status("Starting voice assistant")
        assistant.start()
        _status("JARVIS ready")
        try:
            # Wait rather than sleep-poll: nothing here burns a core, and
            # Ctrl+C still interrupts it.
            self._headless_idle.wait()
        except KeyboardInterrupt:
            log.info("Interrupted; shutting down.")
        return 0


def start_jarvis(
    ui: bool | None = None,
    fullscreen: bool | None = None,
    chrome: bool | None = None,
    voice: bool | None = None,
    tray: bool | None = None,
) -> int:
    """Bring JARVIS up. The single entry point `main.py --start` calls.

    Each argument overrides the corresponding setting for this run only;
    `None` (the default) means "use the configured value".
    """
    from config import configure_file_logging, configure_logging, log_startup_status
    from voice.single_instance import SingleInstance

    # Order matters: the file handler goes in before `configure_logging`
    # so a windowed run (no console, `sys.stderr is None`) still records
    # everything below, and so nothing is written to a dead stream.
    log_path = configure_file_logging()
    configure_logging()

    instance = SingleInstance(MUTEX_NAME)
    if not instance.acquire():
        # Requirement 3: a second launch opens no window, starts no
        # backend and touches no browser. Exiting 0 is deliberate -- this
        # is the expected outcome of double-clicking twice, not a failure.
        message = "JARVIS is already running; this second launch is exiting without starting anything."
        log.warning(message)
        print(message)
        return 0

    try:
        log.info("JARVIS startup logging to %s", log_path)
        # Says whether the agent provider is usable and, when it is not,
        # why. The same call every other entry point makes.
        log_startup_status()
        options = StartupOptions.from_config().with_overrides(
            ui=ui, fullscreen=fullscreen, chrome=chrome, voice=voice, tray=tray
        )
        return JarvisLauncher(options).run()
    finally:
        instance.release()


__all__ = ["MUTEX_NAME", "StartupOptions", "JarvisLauncher", "start_jarvis"]
