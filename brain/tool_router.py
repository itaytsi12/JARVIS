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


def execute_tool(
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
        result = minimize_foreground_window()
        
        return result

    if tool_name == "maximize_window":
        result = maximize_foreground_window()
        
        return result

    if tool_name == "restore_window":
        result = restore_foreground_window()
        
        return result

    if tool_name == "close_window":
        result = close_foreground_window()
        
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

    return {"success":False,"message":f"Unknown tool: {tool_name}","error":"unknown_tool"}
