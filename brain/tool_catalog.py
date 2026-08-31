"""The machine-readable tool catalog.

Every tool the agent runtime can use is declared here exactly once, with:

- a name and a description written for a model to read,
- a JSON Schema for its arguments,
- a category (so a skill can request a coherent subset),
- a risk level (reusing `security.safety`'s existing vocabulary),
- which exclusive resource it needs (desktop input, a browser session),
- whether it is read-only.

Execution goes through the pre-existing `brain.executor.Executor` /
`brain.agent_runtime.AgentRuntime`, which already own the resource
locking, retry and `ToolResult` conversion -- this module adds the schema
and the description layer, it does not add a second execution path.
Likewise the result type is the project's existing
`brain.models.ToolResult`, never a new incompatible one.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from brain.models import Action, ActionRisk, Plan, ToolResult
from providers.base import ToolSpec

log = logging.getLogger("jarvis.tools")

# Categories used to hand a skill a coherent subset of tools.
COMPUTER = "computer"
FILESYSTEM = "filesystem"
TERMINAL = "terminal"
CODE = "code"
BROWSER = "browser"
AUDIO = "audio"
VISION = "vision"
MEMORY = "memory"
INFO = "info"

# Exclusive resources, matching brain/resource_locks.py's vocabulary.
DESKTOP_INPUT = "desktop_input"
BROWSER_SESSION = "browser_session"

# Tools whose correctness depends on the shared desktop/browser session
# (which window was opened, which one has focus, which page is loaded).
# They go through `AgentRuntime`, which owns that state AND the
# process-wide plan lock. Everything else runs without that lock so
# independent background tasks are genuinely concurrent -- see
# `ToolCatalog._dispatch`.
SESSION_AWARE_CATEGORIES = frozenset({COMPUTER, BROWSER})
SESSION_AWARE_TOOLS = frozenset({
    # Launches a file in its associated desktop application, so it belongs
    # with the desktop tools even though it is filed under filesystem.
    "open_path",
})


def _schema(properties: dict[str, Any], required: Iterable[str] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


_STRING = {"type": "string"}
_INTEGER = {"type": "integer"}
_BOOLEAN = {"type": "boolean"}


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    category: str
    risk: ActionRisk = ActionRisk.SAFE
    exclusive_resource: str | None = None
    read_only: bool = False
    #: Is running this tool a second time with the same arguments
    #: equivalent to running it once? Every read-only tool is; so are the
    #: idempotent writers (`create_directory` lands on the same end state,
    #: `write_text_file` overwrites to the same content). `append_text_file`
    #: is NOT -- a retry duplicates the text -- and neither is anything
    #: that types, clicks or sends. `brain/recovery.py` and the agent loop
    #: consult this before ever retrying a failure, instead of each
    #: re-deriving it from the tool name.
    retry_safe: bool = False
    #: How long this tool may reasonably take, in seconds. Advisory: it is
    #: reported to the model so a slow tool is expected rather than assumed
    #: hung, and it is what a caller should use when choosing its own
    #: deadline. It is deliberately NOT a hard kill -- interrupting a
    #: half-finished desktop action is worse than waiting for it.
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        # A read-only tool that claimed to be retry-unsafe would be a
        # contradiction, and one nobody would notice; normalise instead of
        # trusting every call site to remember.
        if self.read_only and not self.retry_safe:
            object.__setattr__(self, "retry_safe", True)

    def to_spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, input_schema=self.parameters)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "risk": self.risk.value,
            "exclusive_resource": self.exclusive_resource,
            "read_only": self.read_only,
            "retry_safe": self.retry_safe,
            "timeout_seconds": self.timeout_seconds,
            "parameters": self.parameters,
        }


#: Deliberately NOT described here, though `brain/tool_router.py` can
#: dispatch them: `lock_computer` (locking the machine is something the
#: user asks for explicitly, never something an agent should decide is a
#: reasonable step toward a goal) and `send_whatsapp_message` (sending a
#: message is irreversible and outward-facing; it stays on the voice path,
#: which asks first). Both remain reachable through their own routes --
#: this list controls what the AGENT is offered, not what JARVIS can do.
DEFINITIONS: tuple[ToolDefinition, ...] = (
    # ---- computer / applications ------------------------------------
    ToolDefinition(
        "open_application",
        "Open a Windows application by name, for example 'notepad', 'spotify', 'chrome'.",
        _schema({"app_name": _STRING}, ["app_name"]),
        COMPUTER,
    ),
    ToolDefinition(
        "close_application",
        "Close a running Windows application by name. Unsaved work in that application may be lost.",
        _schema({"app_name": _STRING}, ["app_name"]),
        COMPUTER,
        risk=ActionRisk.CAUTION,
        exclusive_resource=DESKTOP_INPUT,
    ),
    ToolDefinition(
        "focus_application",
        "Bring an already-open application's window to the foreground.",
        _schema({"app_name": _STRING}, ["app_name"]),
        COMPUTER,
        exclusive_resource=DESKTOP_INPUT,
    ),
    ToolDefinition(
        "active_window",
        "Report which application and window currently has focus.",
        _schema({}),
        COMPUTER,
        read_only=True,
    ),
    ToolDefinition(
        "inspect_window",
        "List the accessible UI controls of an open application. Reads only; clicks nothing.",
        _schema({"app_name": _STRING, "limit": _INTEGER}, ["app_name"]),
        COMPUTER,
        read_only=True,
    ),
    ToolDefinition(
        "click_ui_element",
        "Click one named UI control inside an open application.",
        _schema({"app_name": _STRING, "name": _STRING, "control_type": _STRING}, ["app_name", "name"]),
        COMPUTER,
        risk=ActionRisk.CAUTION,
        exclusive_resource=DESKTOP_INPUT,
    ),
    ToolDefinition(
        "type_text",
        "Type text into the currently focused application window.",
        _schema({"text": _STRING, "delay": {"type": "number"}}, ["text"]),
        COMPUTER,
        exclusive_resource=DESKTOP_INPUT,
    ),
    ToolDefinition(
        "press_key",
        "Press a key or keyboard shortcut, for example 'enter' or 'ctrl+s'.",
        _schema({"key": _STRING}, ["key"]),
        COMPUTER,
        exclusive_resource=DESKTOP_INPUT,
    ),
    ToolDefinition(
        "click_at",
        "Click at absolute screen coordinates. Prefer click_ui_element when a named control exists.",
        _schema({"x": _INTEGER, "y": _INTEGER}, ["x", "y"]),
        COMPUTER,
        risk=ActionRisk.CAUTION,
        exclusive_resource=DESKTOP_INPUT,
    ),
    ToolDefinition(
        "minimize_window", "Minimize the foreground window.", _schema({"hwnd": _INTEGER}), COMPUTER, exclusive_resource=DESKTOP_INPUT
    ),
    ToolDefinition(
        "maximize_window", "Maximize the foreground window.", _schema({"hwnd": _INTEGER}), COMPUTER, exclusive_resource=DESKTOP_INPUT
    ),
    ToolDefinition(
        "show_desktop", "Minimize everything and show the desktop.", _schema({}), COMPUTER, exclusive_resource=DESKTOP_INPUT
    ),
    ToolDefinition(
        "open_task_manager", "Open Windows Task Manager.", _schema({}), COMPUTER
    ),
    # ---- audio -------------------------------------------------------
    # NOT retry-safe: each run moves the volume again, so a retry after an
    # ambiguous failure would change it twice. `set_volume` is the tool to
    # use when a specific level is wanted.
    ToolDefinition("volume_up", "Raise the system volume by one step.", _schema({"amount": _INTEGER}), AUDIO),
    ToolDefinition("volume_down", "Lower the system volume by one step.", _schema({"amount": _INTEGER}), AUDIO),
    ToolDefinition("mute_volume", "Toggle the system mute on or off.", _schema({}), AUDIO),
    # ---- vision ------------------------------------------------------
    ToolDefinition(
        "take_screenshot",
        "Capture the screen to an image file and return its path.",
        _schema({}),
        VISION,
        read_only=True,
    ),
    ToolDefinition(
        "analyze_screen",
        "Capture the screen and answer a question about what is visible. Use this only when the UI state cannot be determined any other way.",
        _schema({"question": _STRING}, ["question"]),
        VISION,
        read_only=True,
    ),
    # ---- browser -----------------------------------------------------
    ToolDefinition(
        "open_website",
        "Open a URL in the browser. The URL must include http:// or https://.",
        _schema({"url": _STRING}, ["url"]),
        BROWSER,
        exclusive_resource=BROWSER_SESSION,
    ),
    ToolDefinition(
        "browser_open_url",
        "Navigate the automated browser session to a URL and return the resulting page state.",
        _schema({"url": _STRING}, ["url"]),
        BROWSER,
        exclusive_resource=BROWSER_SESSION,
    ),
    ToolDefinition(
        "browser_click",
        "Click a link, button or element in the automated browser session, addressed by its visible text.",
        _schema({"target": _STRING, "kind": _STRING}, ["target"]),
        BROWSER,
        risk=ActionRisk.CAUTION,
        exclusive_resource=BROWSER_SESSION,
    ),
    ToolDefinition(
        "browser_type",
        "Type into a named field in the automated browser session.",
        _schema({"target": _STRING, "text": _STRING, "clear": _BOOLEAN}, ["target", "text"]),
        BROWSER,
        exclusive_resource=BROWSER_SESSION,
    ),
    ToolDefinition(
        "browser_click_first_result",
        "Click the first search result on the current results page.",
        _schema({}),
        BROWSER,
        exclusive_resource=BROWSER_SESSION,
    ),
    ToolDefinition(
        "browser_scroll",
        "Scroll the current browser page.",
        _schema({"direction": _STRING, "amount": _INTEGER}),
        BROWSER,
        exclusive_resource=BROWSER_SESSION,
    ),
    # ---- filesystem ---------------------------------------------------
    ToolDefinition(
        "list_files", "List the entries of a directory.", _schema({"path": _STRING}, ["path"]), FILESYSTEM, read_only=True
    ),
    ToolDefinition(
        "exists", "Check whether a path exists.", _schema({"path": _STRING}, ["path"]), FILESYSTEM, read_only=True
    ),
    ToolDefinition(
        "read_text_file", "Read a UTF-8 text file in full.", _schema({"path": _STRING}, ["path"]), FILESYSTEM, read_only=True
    ),
    ToolDefinition(
        "create_text_file",
        "Create a text file. Fails if the file already exists unless overwrite is true.",
        _schema({"path": _STRING, "contents": _STRING, "overwrite": _BOOLEAN}, ["path", "contents"]),
        FILESYSTEM,
        risk=ActionRisk.CAUTION,
    ),
    ToolDefinition(
        "write_text_file",
        "Write a text file, replacing its contents when overwrite is true.",
        _schema({"path": _STRING, "contents": _STRING, "overwrite": _BOOLEAN}, ["path", "contents"]),
        FILESYSTEM,
        risk=ActionRisk.CAUTION,
    ),
    ToolDefinition(
        "append_text_file",
        "Append text to an existing file.",
        _schema({"path": _STRING, "contents": _STRING}, ["path", "contents"]),
        FILESYSTEM,
        risk=ActionRisk.CAUTION,
    ),
    ToolDefinition(
        # Retry-safe: the end state is "this directory exists", which a
        # second run reaches identically.
        "create_directory",
        "Create a directory, including any missing parent directories.",
        _schema({"path": _STRING}, ["path"]),
        FILESYSTEM,
    ),
    ToolDefinition(
        "move_path",
        "Move or rename a file or directory. Refuses to overwrite an existing destination.",
        _schema({"source": _STRING, "destination": _STRING}, ["source", "destination"]),
        FILESYSTEM,
        risk=ActionRisk.CAUTION,
    ),
    ToolDefinition(
        "copy_path",
        "Copy a file. Refuses to overwrite an existing destination.",
        _schema({"source": _STRING, "destination": _STRING}, ["source", "destination"]),
        FILESYSTEM,
    ),
    ToolDefinition(
        "rename_path",
        "Rename a file or directory in place.",
        _schema({"path": _STRING, "new_name": _STRING}, ["path", "new_name"]),
        FILESYSTEM,
        risk=ActionRisk.CAUTION,
    ),
    ToolDefinition(
        "find_file",
        "Find files by name pattern under a directory.",
        _schema({"path": _STRING, "name": _STRING}, ["path", "name"]),
        FILESYSTEM,
        read_only=True,
    ),
    ToolDefinition(
        "search_text",
        "Search file contents under a directory for a literal string.",
        _schema({"path": _STRING, "query": _STRING}, ["path", "query"]),
        FILESYSTEM,
        read_only=True,
    ),
    ToolDefinition(
        "verify_file",
        "Verify a file exists and optionally that its contents match exactly.",
        _schema({"path": _STRING, "expected_content": _STRING}, ["path"]),
        FILESYSTEM,
        read_only=True,
    ),
    ToolDefinition(
        "open_path", "Open a file or folder with its default application.", _schema({"path": _STRING}, ["path"]), FILESYSTEM
    ),
    # ---- terminal -----------------------------------------------------
    ToolDefinition(
        "run_command",
        (
            "Run a command and capture stdout, stderr and the exit code. Use this to run a program "
            "or a test suite, for example 'python -m pytest -q'. Only well-known development "
            "executables are permitted; destructive commands are refused."
        ),
        _schema(
            {
                "command": _STRING,
                "working_directory": _STRING,
                "timeout": {"type": "number"},
            },
            ["command"],
        ),
        TERMINAL,
        risk=ActionRisk.CAUTION,
    ),
    # ---- code ---------------------------------------------------------
    ToolDefinition(
        "inspect_project",
        "Summarize a project directory: its markers, entry points, layout and file counts.",
        _schema({"path": _STRING, "max_files": _INTEGER}, ["path"]),
        CODE,
        read_only=True,
    ),
    ToolDefinition(
        "read_code",
        "Read a line-numbered slice of a source file. Prefer a slice over the whole file for large files.",
        _schema({"path": _STRING, "start_line": _INTEGER, "end_line": _INTEGER, "max_lines": _INTEGER}, ["path"]),
        CODE,
        read_only=True,
    ),
    ToolDefinition(
        "search_code",
        "Search a project's source files for a string and return file/line matches.",
        _schema({"path": _STRING, "query": _STRING, "max_results": _INTEGER}, ["path", "query"]),
        CODE,
        read_only=True,
    ),
    ToolDefinition(
        "edit_code",
        (
            "Replace an exact snippet in a source file. The old_text must appear exactly once so the "
            "edit cannot land in the wrong place; include enough surrounding lines to make it unique."
        ),
        _schema(
            {"path": _STRING, "old_text": _STRING, "new_text": _STRING, "expect_unique": _BOOLEAN},
            ["path", "old_text", "new_text"],
        ),
        CODE,
        risk=ActionRisk.CAUTION,
    ),
    ToolDefinition(
        "check_syntax",
        "Parse a Python file and report any syntax error with its line number.",
        _schema({"path": _STRING}, ["path"]),
        CODE,
        read_only=True,
    ),
    # ---- information ---------------------------------------------------
    ToolDefinition("get_time", "Report the current local time.", _schema({}), INFO, read_only=True),
    ToolDefinition("get_date", "Report today's date.", _schema({}), INFO, read_only=True),
    ToolDefinition(
        "calculator",
        "Evaluate an arithmetic expression, for example '527 * 93'.",
        _schema({"expression": _STRING}, ["expression"]),
        INFO,
        read_only=True,
    ),
    # ---- memory ---------------------------------------------------------
    # ---- machine state (read-only observation) -----------------------
    ToolDefinition(
        "system_status",
        "Report this machine's CPU load, memory use, free disk space and battery level in one call.",
        _schema({}),
        INFO,
        read_only=True,
        timeout_seconds=10,
    ),
    ToolDefinition(
        "network_status",
        "Check whether this machine really has internet access, by opening a connection rather than trusting the adapter state.",
        _schema({}),
        INFO,
        read_only=True,
        timeout_seconds=10,
    ),
    ToolDefinition(
        "list_processes",
        "List running processes, largest by memory first. Pass 'name' to list only processes matching it.",
        _schema({"name": _STRING, "limit": _INTEGER}),
        COMPUTER,
        read_only=True,
        timeout_seconds=15,
    ),
    ToolDefinition(
        "process_running",
        "Check whether an application is running, and how many processes it has. Accepts a spoken name like 'chrome' or 'Google Chrome'.",
        _schema({"name": _STRING}, ["name"]),
        COMPUTER,
        read_only=True,
        timeout_seconds=15,
    ),
    ToolDefinition(
        "get_volume",
        "Report the exact system volume percentage and whether audio is muted.",
        _schema({}),
        AUDIO,
        read_only=True,
        timeout_seconds=10,
    ),
    ToolDefinition(
        "set_volume",
        "Set the system volume to an exact percentage from 0 to 100. Use this rather than repeating volume_up/volume_down when a specific level is wanted.",
        _schema({"level": _INTEGER}, ["level"]),
        AUDIO,
        retry_safe=True,
        timeout_seconds=10,
    ),
    # ---- clipboard ---------------------------------------------------
    ToolDefinition(
        "read_clipboard",
        "Read the text currently on the Windows clipboard.",
        _schema({}),
        COMPUTER,
        read_only=True,
        timeout_seconds=10,
    ),
    ToolDefinition(
        "write_clipboard",
        "Replace the Windows clipboard with this text, so the user can paste it.",
        _schema({"text": _STRING}, ["text"]),
        COMPUTER,
        retry_safe=True,
        timeout_seconds=10,
    ),
    # ---- file discovery ----------------------------------------------
    ToolDefinition(
        "file_info",
        "Size, last-modified time and kind for one file or folder. Use this to answer 'when did I last change this' without reading the whole file.",
        _schema({"path": _STRING}, ["path"]),
        FILESYSTEM,
        read_only=True,
        timeout_seconds=10,
    ),
    ToolDefinition(
        "recent_files",
        "Files changed recently, newest first -- how to find 'the file I was working on yesterday'. "
        "'path' may be a folder or one of 'desktop', 'documents', 'downloads'; with no path it searches all three. "
        "'within_hours' defaults to 48; 'suffixes' limits it to e.g. ['.docx', '.py'].",
        _schema({"path": _STRING, "within_hours": {"type": "number"}, "limit": _INTEGER, "suffixes": {"type": "array", "items": _STRING}}),
        FILESYSTEM,
        read_only=True,
        timeout_seconds=45,
    ),
    # ---- desktop scrolling -------------------------------------------
    ToolDefinition(
        "scroll_screen",
        "Scroll the window under the mouse pointer. Give x and y to scroll a specific pane; otherwise it scrolls wherever the pointer already is.",
        _schema({"direction": _STRING, "clicks": _INTEGER, "x": _INTEGER, "y": _INTEGER}),
        COMPUTER,
        exclusive_resource=DESKTOP_INPUT,
        retry_safe=True,
        timeout_seconds=15,
    ),
    # ---- windows and shell (previously dispatchable but undescribed) --
    ToolDefinition(
        "restore_window",
        "Restore the foreground window from minimised or maximised back to its normal size.",
        _schema({"hwnd": _INTEGER}),
        COMPUTER,
        exclusive_resource=DESKTOP_INPUT,
        retry_safe=True,
    ),
    ToolDefinition(
        "close_window",
        "Close the foreground window. Asks the application to close, so unsaved work still prompts.",
        _schema({"hwnd": _INTEGER}),
        COMPUTER,
        risk=ActionRisk.CAUTION,
        exclusive_resource=DESKTOP_INPUT,
    ),
    ToolDefinition(
        "open_file_explorer",
        "Open File Explorer, at a specific folder when a path is given.",
        _schema({"path": _STRING}),
        FILESYSTEM,
        retry_safe=True,
    ),
    ToolDefinition(
        "open_folder",
        "Open one of the user's known folders by name -- desktop, documents, downloads, pictures.",
        _schema({"name": _STRING}, ["name"]),
        FILESYSTEM,
        retry_safe=True,
    ),
    ToolDefinition(
        "get_day",
        "Report today's day of the week.",
        _schema({}),
        INFO,
        read_only=True,
        timeout_seconds=5,
    ),
    # ---- music -------------------------------------------------------
    #
    # These were dispatchable in `brain/tool_router.py` but had no entry
    # here, which meant the agent runtime was never told they exist: a
    # request like "open Apple Music and make me a playlist" reached a
    # model holding no music tools at all. Describing them is the whole
    # fix -- the implementations were already there and already tested.
    ToolDefinition(
        "open_music",
        "Open Apple Music in JARVIS's browser session. Starts the browser itself if it is not already running.",
        _schema({}),
        BROWSER,
        exclusive_resource="authenticated_browser",
        retry_safe=True,
        timeout_seconds=60,
    ),
    ToolDefinition(
        "music_play",
        "Play music on Apple Music. 'intent' selects what to play: PLAY_SONG, PLAY_ARTIST, PLAY_ALBUM, PLAY_PLAYLIST, "
        "PLAY_QUERY, PLAY_MOOD, PLAY_LAST, PLAY_RECENT, RESUME_SESSION, PLAY_FAVORITES, PLAY_LIBRARY or PLAY_GENERIC. "
        "Supply whichever of song/artist/album/playlist/mood the intent needs.",
        _schema({
            "intent": _STRING, "song": _STRING, "artist": _STRING, "album": _STRING,
            "playlist": _STRING, "mood": _STRING, "scope": _STRING,
            "contextual": _BOOLEAN, "shuffle": _BOOLEAN,
        }, ["intent"]),
        BROWSER,
        exclusive_resource="authenticated_browser",
        timeout_seconds=90,
    ),
    ToolDefinition(
        "music_now_playing",
        "Report what Apple Music is playing right now. 'aspect' may be song, artist or album.",
        _schema({"aspect": _STRING}),
        BROWSER,
        exclusive_resource="authenticated_browser",
        read_only=True,
        timeout_seconds=45,
    ),
    ToolDefinition(
        "music_list_playlists",
        "List the playlists in the user's Apple Music library. Check this before acting on a playlist by name.",
        _schema({}),
        BROWSER,
        exclusive_resource="authenticated_browser",
        read_only=True,
        timeout_seconds=60,
    ),
    ToolDefinition(
        "music_create_playlist",
        "Create a new Apple Music playlist called 'name' containing 'songs'. At least one song is required -- "
        "Apple Music's web player cannot create an empty playlist. Verified against the library afterwards.",
        _schema({"name": _STRING, "songs": {"type": "array", "items": _STRING}, "artist": _STRING}, ["name", "songs"]),
        BROWSER,
        risk=ActionRisk.CAUTION,
        exclusive_resource="authenticated_browser",
        timeout_seconds=180,
    ),
    ToolDefinition(
        "music_add_to_playlist",
        "Add one song to an existing Apple Music playlist, then confirm it is really there. "
        "Reports the available playlist names if the one asked for does not exist.",
        _schema({"song": _STRING, "playlist": _STRING, "artist": _STRING}, ["song", "playlist"]),
        BROWSER,
        risk=ActionRisk.CAUTION,
        exclusive_resource="authenticated_browser",
        timeout_seconds=120,
    ),
    ToolDefinition("music_pause", "Pause whatever Apple Music is currently playing.", _schema({}), BROWSER, exclusive_resource="authenticated_browser", timeout_seconds=30),
    ToolDefinition("music_resume", "Resume Apple Music from where it was paused.", _schema({}), BROWSER, exclusive_resource="authenticated_browser", timeout_seconds=30),
    ToolDefinition("music_stop", "Stop Apple Music playback.", _schema({}), BROWSER, exclusive_resource="authenticated_browser", timeout_seconds=30),
    ToolDefinition("music_next", "Skip to the next track.", _schema({}), BROWSER, exclusive_resource="authenticated_browser", timeout_seconds=30),
    ToolDefinition("music_previous", "Go back to the previous track.", _schema({}), BROWSER, exclusive_resource="authenticated_browser", timeout_seconds=30),
    ToolDefinition("music_restart_track", "Start the current track again from the beginning.", _schema({}), BROWSER, exclusive_resource="authenticated_browser", timeout_seconds=30),
    ToolDefinition("music_shuffle_on", "Turn shuffle on for Apple Music playback.", _schema({}), BROWSER, exclusive_resource="authenticated_browser", retry_safe=True, timeout_seconds=30),
    ToolDefinition("music_shuffle_off", "Turn shuffle off for Apple Music playback.", _schema({}), BROWSER, exclusive_resource="authenticated_browser", retry_safe=True, timeout_seconds=30),
    ToolDefinition("music_repeat_on", "Turn repeat on for Apple Music playback.", _schema({}), BROWSER, exclusive_resource="authenticated_browser", retry_safe=True, timeout_seconds=30),
    ToolDefinition("music_repeat_off", "Turn repeat off for Apple Music playback.", _schema({}), BROWSER, exclusive_resource="authenticated_browser", retry_safe=True, timeout_seconds=30),
    ToolDefinition("music_add_to_library", "Add the current track to the user's library.", _schema({}), BROWSER, exclusive_resource="authenticated_browser", retry_safe=True, timeout_seconds=45),
    ToolDefinition("music_add_to_favorites", "Mark the current track as a favourite.", _schema({}), BROWSER, exclusive_resource="authenticated_browser", retry_safe=True, timeout_seconds=45),
    ToolDefinition("music_artist_more", "Open more from the current track's artist.", _schema({}), BROWSER, exclusive_resource="authenticated_browser", timeout_seconds=60),
    ToolDefinition(
        "music_queue_add",
        "Add a song to the end of the Apple Music queue without interrupting what is playing.",
        _schema({"song": _STRING, "contextual": _BOOLEAN}),
        BROWSER,
        exclusive_resource="authenticated_browser",
        timeout_seconds=90,
    ),
    ToolDefinition(
        "music_queue_next",
        "Queue a song to play next, without interrupting what is playing.",
        _schema({"song": _STRING, "contextual": _BOOLEAN}),
        BROWSER,
        exclusive_resource="authenticated_browser",
        timeout_seconds=90,
    ),
    ToolDefinition(
        "remember_fact",
        (
            "Store a durable fact about the user, their projects or their preferences, so it can be "
            "recalled in a future session. Use only for information that stays useful over time."
        ),
        _schema({"text": _STRING, "kind": _STRING, "importance": _INTEGER}, ["text"]),
        MEMORY,
    ),
    ToolDefinition(
        "recall_memory",
        "Search long-term memory for facts relevant to a query.",
        _schema({"query": _STRING, "limit": _INTEGER}, ["query"]),
        MEMORY,
        read_only=True,
    ),
)

BY_NAME: dict[str, ToolDefinition] = {definition.name: definition for definition in DEFINITIONS}

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_arguments(definition: ToolDefinition, arguments: dict[str, Any]) -> list[str]:
    """Check `arguments` against a tool's schema and return every problem.

    Structured validation instead of prose parsing: a model that invents
    an argument or a wrong type gets a precise, actionable error back
    rather than a confusing runtime exception.
    """
    if not isinstance(arguments, dict):
        return ["arguments_not_object"]
    schema = definition.parameters
    properties: dict[str, Any] = schema.get("properties", {})
    errors: list[str] = []
    for name in schema.get("required", []):
        if name not in arguments or arguments[name] is None:
            errors.append(f"missing_required:{name}")
    for name, value in arguments.items():
        if name not in properties:
            errors.append(f"unknown_argument:{name}")
            continue
        expected_name = properties[name].get("type")
        expected = _JSON_TYPES.get(expected_name)
        if expected is None or value is None:
            continue
        if expected_name in {"integer", "number"} and isinstance(value, bool):
            errors.append(f"invalid_type:{name}")
            continue
        if not isinstance(value, expected):
            errors.append(f"invalid_type:{name}")
    return errors


class ToolCatalog:
    """Lookup, description and typed execution for every agent tool."""

    def __init__(
        self,
        definitions: Iterable[ToolDefinition] = DEFINITIONS,
        runtime: Any | None = None,
        handlers: dict[str, Callable[[dict[str, Any]], ToolResult]] | None = None,
    ):
        self._definitions = tuple(definitions)
        self._by_name = {definition.name: definition for definition in self._definitions}
        self._runtime = runtime
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolResult]] = dict(handlers or {})
        self._executor = None

    # -- description ---------------------------------------------------
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return self._definitions

    def get(self, name: str) -> ToolDefinition | None:
        return self._by_name.get(name)

    def names(self) -> list[str]:
        return [definition.name for definition in self._definitions]

    def select(self, *, categories: Iterable[str] | None = None, names: Iterable[str] | None = None) -> list[ToolDefinition]:
        wanted_categories = set(categories) if categories else None
        wanted_names = set(names) if names else None
        return [
            definition
            for definition in self._definitions
            if (wanted_categories is None or definition.category in wanted_categories)
            and (wanted_names is None or definition.name in wanted_names)
        ]

    def specs(self, *, categories: Iterable[str] | None = None, names: Iterable[str] | None = None) -> list[ToolSpec]:
        return [definition.to_spec() for definition in self.select(categories=categories, names=names)]

    def register_handler(self, name: str, handler: Callable[[dict[str, Any]], ToolResult]) -> None:
        """Attach an in-process handler for a tool (used for the memory
        tools, which need the live memory objects rather than a
        module-level singleton)."""
        self._handlers[name] = handler

    # -- execution -----------------------------------------------------
    def execute(self, name: str, arguments: dict[str, Any] | None = None, cancellation_token: Any = None) -> ToolResult:
        """Run one tool and always return a `ToolResult` -- never raise.

        An unknown tool, invalid arguments and a tool that itself fails
        are all reported as an unsuccessful result carrying a specific
        error, because the agent loop must be able to read the failure
        and adapt rather than crash.
        """
        arguments = dict(arguments or {})
        definition = self._by_name.get(name)
        started = time.perf_counter()
        if definition is None:
            return ToolResult(
                False,
                name,
                f"There is no tool called {name!r}. Use one of the tools you were given.",
                {"verified": True, "available_tools": self.names()[:40]},
                "unknown_tool",
            )
        errors = validate_arguments(definition, arguments)
        if errors:
            return ToolResult(
                False,
                name,
                f"The arguments for {name} were not valid: {', '.join(errors)}.",
                {"verified": True, "validation_errors": errors, "schema": definition.parameters},
                "invalid_arguments",
            )
        if cancellation_token is not None and getattr(cancellation_token, "cancelled", False):
            return ToolResult(False, name, "Task cancelled.", {"verified": True}, "cancelled")

        try:
            result = self._dispatch(definition, arguments, cancellation_token)
        except Exception as exc:
            log.exception("Tool %s raised", name)
            result = ToolResult(False, name, f"The {name} tool failed.", {}, f"{type(exc).__name__}: {exc}")

        if not isinstance(result.data, dict):
            result.data = {}
        result.data.setdefault("verified", False)
        result.data["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        result.data["category"] = definition.category
        result.data["risk"] = definition.risk.value
        return result

    def _dispatch(self, definition: ToolDefinition, arguments: dict[str, Any], cancellation_token: Any) -> ToolResult:
        handler = self._handlers.get(definition.name)
        if handler is not None:
            return handler(arguments)
        action = Action(tool=definition.name, args=arguments, risk=definition.risk)

        if definition.name in SESSION_AWARE_TOOLS or definition.category in SESSION_AWARE_CATEGORIES:
            # Desktop and browser work: route through the session-aware
            # runtime so window/PID context, focus handling and the
            # process-wide "action_plan" lock all apply exactly as they do
            # for a voice command. Two agent tasks touching the desktop
            # therefore still serialize -- which is the point.
            if self._runtime is not None:
                results = self._runtime.execute(Plan(definition.name, [action]), cancellation_token=cancellation_token)
                return results[0] if results else ToolResult(False, definition.name, "The tool produced no result.", {}, "no_result")
            return self._get_executor().execute_action(action)

        # Everything else -- filesystem, terminal, code, info, memory --
        # touches no shared desktop state, so it must NOT take the
        # process-wide plan lock: doing so would serialize two independent
        # background tasks (say, running tests while researching) behind a
        # lock that exists to protect the keyboard. The per-tool resource
        # lock is still acquired inside `execute_action_unlocked_plan`.
        return self._get_executor().execute_action_unlocked_plan(action)

    def _get_executor(self):
        if self._executor is None:
            from brain.executor import Executor

            self._executor = Executor()
        return self._executor


_CATALOG: ToolCatalog | None = None


def get_tool_catalog() -> ToolCatalog:
    """The process-wide catalog used for description/lookup.

    Note this default instance has no runtime attached; the agent loop
    builds its own catalog bound to its `AgentRuntime` so session-aware
    tools behave correctly.
    """
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = ToolCatalog()
    return _CATALOG
