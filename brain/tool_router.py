from tools.applications import (
    open_application,
    close_application
)
from tools.system import (
    lock_computer,
    open_task_manager,
    open_file_explorer,
    show_desktop,
    minimize_foreground_window,
    maximize_foreground_window,
    restore_foreground_window,
    close_foreground_window,
)
from tools.files import (
    file_info,
    recent_files,
    create_directory,
    open_known_folder,
    open_path,
    list_files,
    exists,
    append_text_file,
    copy_path,
    create_text_file,
    find_file,
    move_path,
    read_text_file,
    rename_path,
    search_text,
    verify_file,
    write_text_file,
)
from tools.misc import (
    get_time,
    get_date,
    get_day,
)
from tools.calculator import calculate
from tools.audio import (
    volume_up,
    volume_down,
    mute_volume
)
from tools.ui import (
    press_key,
    click_at,
    scroll_screen,
)
from tools.clipboard import read_clipboard, write_clipboard
from tools.machine import (
    get_volume,
    list_processes,
    network_status,
    process_running,
    set_volume,
    system_status,
)
from tools.browser import open_website
from tools.whatsapp import send_whatsapp_message
from vision.screenshot import take_screenshot
from vision.screen_analyzer import analyze_screen
from tools.context import describe_active_window
from tools.keyboard import type_text
from tools.desktop_agent import click_control,get_controls
import os
from pathlib import Path
import time
from tools.window import (
    find_application_window,
    bring_hwnd_to_foreground,
)
from tools.music import apple_music_provider as music
from tools.terminal import run_command
from tools.code import check_syntax, edit_code, inspect_project, read_code, search_code


def execute_tool(
    tool_name: str,
    arguments: dict
):
    """The single tool dispatch point.

    Playwright-driving tools are run on their session's dedicated worker
    thread (`tools/playwright_runtime.py`) instead of on the caller's. The
    sync Playwright API leaves a RUNNING asyncio loop on whichever thread
    starts it, so a second sync-Playwright session on that thread fails with
    "It looks like you are using Playwright Sync API inside the asyncio loop"
    -- the live failure on "Open Music." / "Play Israeli playlist." -- and
    sync page objects are bound to their creating thread regardless. Routing
    the WHOLE tool call (not just the connect) keeps every page interaction
    it performs on that one thread.

    Non-browser tools are unaffected: `run_for_tool` calls them inline.
    """
    from tools.playwright_runtime import run_for_tool

    return run_for_tool(tool_name, _execute_tool_impl, tool_name, arguments)


def _execute_tool_impl(
    tool_name: str,
    arguments: dict
):
    t0 = time.perf_counter()
    if tool_name == "open_application":
        result = open_application(
            arguments["app_name"]
        )
        
        return result

    if tool_name == "close_application":
        result = close_application(
            arguments["app_name"]
        )
        
        return result

    if tool_name == "open_website":
        result = open_website(
            arguments["url"]
        )
        
        return result

    if tool_name == "calculator":
        result = calculate(
            arguments["expression"]
        )
        
        return result

    if tool_name == "volume_up":
        result = volume_up(
            arguments.get(
                "amount",
                1
            )
        )
        
        return result

    if tool_name == "volume_down":
        result = volume_down(
            arguments.get(
                "amount",
                1
            )
        )
        
        return result

    if tool_name == "mute_volume":
        result = mute_volume()
        
        return result

    if tool_name == "take_screenshot":
        result = take_screenshot()
        
        return result

    if tool_name == "open_folder":
        result = open_known_folder(arguments.get("name"))
        
        return result

    if tool_name == "list_files":
        result = list_files(arguments.get("path"))
        
        return result

    if tool_name == "exists":
        result = exists(arguments.get("path"))
        
        return result

    if tool_name == "open_path":
        return open_path(arguments["path"])

    if tool_name == "send_whatsapp_message":
        return send_whatsapp_message(arguments["recipient"],arguments["message"],arguments.get("literal",False))

    file_tools = {
        "create_text_file": lambda: create_text_file(arguments["path"], arguments["contents"], arguments.get("overwrite", False)),
        "read_text_file": lambda: read_text_file(arguments["path"]),
        "write_text_file": lambda: write_text_file(arguments["path"], arguments["contents"], arguments.get("overwrite", False)),
        "append_text_file": lambda: append_text_file(arguments["path"], arguments["contents"]),
        "verify_file": lambda: verify_file(arguments["path"], arguments.get("expected_content")),
        "rename_path": lambda: rename_path(arguments["path"], arguments["new_name"]),
        "copy_path": lambda: copy_path(arguments["source"], arguments["destination"]),
        "move_path": lambda: move_path(arguments["source"], arguments["destination"]),
        "find_file": lambda: find_file(arguments["path"], arguments["name"]),
        "search_text": lambda: search_text(arguments["path"], arguments["query"]),
        "create_directory": lambda: create_directory(arguments["path"]),
    }
    if tool_name in file_tools:
        return file_tools[tool_name]()

    if tool_name == "lock_computer":
        result = lock_computer()
        
        return result

    if tool_name == "open_task_manager":
        result = open_task_manager()
        
        return result

    if tool_name == "open_file_explorer":
        result = open_file_explorer(arguments.get("path"))
        
        return result

    if tool_name == "show_desktop":
        result = show_desktop()
        
        return result

    if tool_name == "minimize_window":
        # `hwnd` is optional and defaults to the foreground window. Passing
        # it through is what lets the agent act on a SPECIFIC window it
        # found with `inspect_window`, rather than only on whatever
        # happens to be in front by the time the action runs.
        result = minimize_foreground_window(arguments.get("hwnd"))
        
        return result

    if tool_name == "maximize_window":
        # `hwnd` is optional and defaults to the foreground window. Passing
        # it through is what lets the agent act on a SPECIFIC window it
        # found with `inspect_window`, rather than only on whatever
        # happens to be in front by the time the action runs.
        result = maximize_foreground_window(arguments.get("hwnd"))
        
        return result

    if tool_name == "restore_window":
        # `hwnd` is optional and defaults to the foreground window. Passing
        # it through is what lets the agent act on a SPECIFIC window it
        # found with `inspect_window`, rather than only on whatever
        # happens to be in front by the time the action runs.
        result = restore_foreground_window(arguments.get("hwnd"))
        
        return result

    if tool_name == "close_window":
        # `hwnd` is optional and defaults to the foreground window. Passing
        # it through is what lets the agent act on a SPECIFIC window it
        # found with `inspect_window`, rather than only on whatever
        # happens to be in front by the time the action runs.
        result = close_foreground_window(arguments.get("hwnd"))
        
        return result

    if tool_name == "get_time":
        result = get_time()
        
        return result

    if tool_name == "get_date":
        result = get_date()
        
        return result

    if tool_name == "get_day":
        result = get_day()
        
        return result

    if tool_name == "focus_application":
        app_name = arguments.get("app_name")
        hwnd = find_application_window(app_name)
        if hwnd:
            ok = bring_hwnd_to_foreground(hwnd)
            return {"success":bool(ok),"verified":bool(ok),"hwnd":hwnd,"message":f"Focused {app_name}." if ok else f"Windows refused to focus {app_name}.","error":None if ok else "focus_failed"}
        return {"success":False,"message":f"Could not find window for {app_name}.","error":"window_not_found"}

    if tool_name == "analyze_screen":
        screenshot = take_screenshot()

        if not screenshot.get("success"):
            return screenshot

        try:
            result = analyze_screen(screenshot["path"],arguments["question"])
            result["screenshot_retained"]=os.getenv("JARVIS_KEEP_ANALYSIS_SCREENSHOTS","false").lower() in {"1","true","yes","on"}
            return result
        finally:
            if os.getenv("JARVIS_KEEP_ANALYSIS_SCREENSHOTS","false").lower() not in {"1","true","yes","on"}:
                Path(screenshot["path"]).unlink(missing_ok=True)

    if tool_name == "active_window":
        result = describe_active_window()
        
        return result

    if tool_name == "inspect_window":
        return get_controls(arguments["app_name"],arguments.get("limit",50))

    if tool_name == "click_ui_element":
        return click_control(arguments["app_name"],arguments["name"],arguments.get("control_type"))

    if tool_name == "type_text":
        result = type_text(
            arguments["text"],
            arguments.get(
                "delay",
                0.02
            )
        )
        
        return result

    if tool_name == "press_key":
        result = press_key(
            arguments["key"]
        )
        
        return result

    if tool_name == "click_at":
        result = click_at(
            arguments["x"],
            arguments["y"],
        )

        return result

    # Observation and small utilities. Grouped in one table rather than as
    # a dozen `if` branches: every one takes its arguments straight from
    # the schema in `brain/tool_catalog.py`, so there is nothing
    # per-tool to say here.
    simple_tools = {
        "read_clipboard": lambda: read_clipboard(),
        "write_clipboard": lambda: write_clipboard(arguments["text"]),
        "list_processes": lambda: list_processes(arguments.get("name"), arguments.get("limit", 40)),
        "process_running": lambda: process_running(arguments["name"]),
        "system_status": lambda: system_status(),
        "network_status": lambda: network_status(),
        "get_volume": lambda: get_volume(),
        "set_volume": lambda: set_volume(arguments["level"]),
        "file_info": lambda: file_info(arguments["path"]),
        "recent_files": lambda: recent_files(
            arguments.get("path"),
            arguments.get("within_hours", 48),
            arguments.get("limit", 25),
            arguments.get("suffixes"),
        ),
        "scroll_screen": lambda: scroll_screen(
            arguments.get("direction", "down"),
            arguments.get("clicks", 3),
            arguments.get("x"),
            arguments.get("y"),
        ),
    }
    if tool_name in simple_tools:
        return simple_tools[tool_name]()

    music_no_arg_tools = {
        "open_music": music.open_music,
        "music_pause": music.music_pause,
        "music_resume": music.music_resume,
        "music_stop": music.music_stop,
        "music_next": music.music_next,
        "music_previous": music.music_previous,
        "music_restart_track": music.music_restart_track,
        "music_shuffle_on": music.music_shuffle_on,
        "music_shuffle_off": music.music_shuffle_off,
        "music_repeat_on": music.music_repeat_on,
        "music_repeat_off": music.music_repeat_off,
        "music_add_to_library": music.music_add_to_library,
        "music_add_to_favorites": music.music_add_to_favorites,
        "music_artist_more": music.music_artist_more,
    }
    if tool_name in music_no_arg_tools:
        return music_no_arg_tools[tool_name]()

    if tool_name == "music_now_playing":
        return music.music_now_playing(arguments.get("aspect", "song"))

    if tool_name == "music_queue_add":
        return music.music_queue_add(arguments.get("song"), arguments.get("contextual", False))

    if tool_name == "music_queue_next":
        return music.music_queue_next(arguments.get("song"), arguments.get("contextual", False))

    if tool_name == "music_list_playlists":
        return music.music_list_playlists()

    if tool_name == "music_create_playlist":
        return music.music_create_playlist(
            arguments["name"],
            arguments.get("songs"),
            arguments.get("artist"),
        )

    if tool_name == "music_add_to_playlist":
        return music.music_add_to_playlist(
            arguments["song"],
            arguments["playlist"],
            arguments.get("artist"),
        )

    if tool_name == "music_play":
        return music.music_play(
            arguments["intent"],
            song=arguments.get("song"),
            artist=arguments.get("artist"),
            album=arguments.get("album"),
            playlist=arguments.get("playlist"),
            mood=arguments.get("mood"),
            scope=arguments.get("scope"),
            contextual=arguments.get("contextual", False),
            shuffle=arguments.get("shuffle", False),
        )

    # Terminal + code tools (see tools/terminal.py and tools/code.py). These
    # are what the agent runtime's coding skill is built from; they are
    # dispatched here rather than in a second table so there stays exactly
    # one tool dispatch point in JARVIS.
    if tool_name == "run_command":
        return run_command(
            arguments["command"],
            arguments.get("working_directory"),
            arguments.get("timeout"),
            arguments.get("approved", False),
        )

    code_tools = {
        "inspect_project": lambda: inspect_project(arguments["path"], arguments.get("max_files", 200)),
        "read_code": lambda: read_code(arguments["path"], arguments.get("start_line", 1), arguments.get("end_line"), arguments.get("max_lines", 400)),
        "edit_code": lambda: edit_code(arguments["path"], arguments["old_text"], arguments["new_text"], arguments.get("expect_unique", True)),
        "check_syntax": lambda: check_syntax(arguments["path"]),
        "search_code": lambda: search_code(arguments["path"], arguments["query"], arguments.get("max_results", 60), arguments.get("suffixes")),
    }
    if tool_name in code_tools:
        return code_tools[tool_name]()

    # The Obsidian knowledge vault. Imported lazily so the vault package
    # is never on the import path of a request that does not touch it.
    if tool_name.startswith("vault_"):
        from vault import tools as vault_tools

        vault_dispatch = {
            "vault_search": lambda: vault_tools.vault_search(arguments["query"], arguments.get("note_type", ""), arguments.get("limit", 8)),
            "vault_read_note": lambda: vault_tools.vault_read_note(arguments["path"], arguments.get("section", "")),
            "vault_write_note": lambda: vault_tools.vault_write_note(
                arguments["path"],
                arguments["title"],
                arguments["note_type"],
                arguments["summary"],
                arguments["content"],
                arguments.get("tags", ""),
                arguments.get("quick_summary", ""),
            ),
            "vault_update_note": lambda: vault_tools.vault_update_note(
                arguments["path"], arguments["section"], arguments["content"], arguments.get("mode", "replace")
            ),
            "vault_record_lesson": lambda: vault_tools.vault_record_lesson(
                arguments["title"], arguments["summary"], arguments["lesson"], arguments.get("tags", "")
            ),
            "vault_record_working_method": lambda: vault_tools.vault_record_working_method(
                arguments["skill"], arguments["method"], arguments.get("failed_attempts", "")
            ),
            "vault_list_jobs": vault_tools.vault_list_jobs,
            "vault_status": vault_tools.vault_status,
        }
        if tool_name in vault_dispatch:
            return vault_dispatch[tool_name]()

    return {"success":False,"message":f"Unknown tool: {tool_name}","error":"unknown_tool"}
