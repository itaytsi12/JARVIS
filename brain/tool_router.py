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
from vision.screenshot import take_screenshot
from vision.screen_analyzer import analyze_screen
from tools.context import describe_active_window
from tools.keyboard import type_text
import time
from tools.window import (
    find_application_window,
    bring_hwnd_to_foreground,
)


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
            return f"Focused {app_name}: {ok}"
        return f"Could not find window for {app_name}"

    if tool_name == "analyze_screen":
        screenshot_path = take_screenshot()

        result = analyze_screen(
            screenshot_path,
            arguments["question"]
        )
        
        return result

    if tool_name == "active_window":
        result = describe_active_window()
        
        return result

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

    return (
        f"Unknown tool: {tool_name}"
    )
