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


def execute_tool(
    tool_name: str,
    arguments: dict
):
    print(f"[DEBUG] ToolRouter: execute_tool called for '{tool_name}' with args: {arguments}")
    t0 = time.perf_counter()
    if tool_name == "open_application":
        result = open_application(
            arguments["app_name"]
        )
        print(f"[DEBUG] ToolRouter: open_application returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "close_application":
        result = close_application(
            arguments["app_name"]
        )
        print(f"[DEBUG] ToolRouter: close_application returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "open_website":
        result = open_website(
            arguments["url"]
        )
        print(f"[DEBUG] ToolRouter: open_website returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "calculator":
        result = calculate(
            arguments["expression"]
        )
        print(f"[DEBUG] ToolRouter: calculator returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "volume_up":
        result = volume_up(
            arguments.get(
                "amount",
                1
            )
        )
        print(f"[DEBUG] ToolRouter: volume_up returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "volume_down":
        result = volume_down(
            arguments.get(
                "amount",
                1
            )
        )
        print(f"[DEBUG] ToolRouter: volume_down returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "mute_volume":
        result = mute_volume()
        print(f"[DEBUG] ToolRouter: mute_volume returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "take_screenshot":
        result = take_screenshot()
        print(f"[DEBUG] ToolRouter: take_screenshot returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "open_folder":
        result = open_known_folder(arguments.get("name"))
        print(f"[DEBUG] ToolRouter: open_folder returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "list_files":
        result = list_files(arguments.get("path"))
        print(f"[DEBUG] ToolRouter: list_files returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "exists":
        result = exists(arguments.get("path"))
        print(f"[DEBUG] ToolRouter: exists returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "lock_computer":
        result = lock_computer()
        print(f"[DEBUG] ToolRouter: lock_computer returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "open_task_manager":
        result = open_task_manager()
        print(f"[DEBUG] ToolRouter: open_task_manager returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "open_file_explorer":
        result = open_file_explorer(arguments.get("path"))
        print(f"[DEBUG] ToolRouter: open_file_explorer returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "show_desktop":
        result = show_desktop()
        print(f"[DEBUG] ToolRouter: show_desktop returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "minimize_window":
        result = minimize_foreground_window()
        print(f"[DEBUG] ToolRouter: minimize_window returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "maximize_window":
        result = maximize_foreground_window()
        print(f"[DEBUG] ToolRouter: maximize_window returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "restore_window":
        result = restore_foreground_window()
        print(f"[DEBUG] ToolRouter: restore_window returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "close_window":
        result = close_foreground_window()
        print(f"[DEBUG] ToolRouter: close_window returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "get_time":
        result = get_time()
        print(f"[DEBUG] ToolRouter: get_time returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "get_date":
        result = get_date()
        print(f"[DEBUG] ToolRouter: get_date returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "get_day":
        result = get_day()
        print(f"[DEBUG] ToolRouter: get_day returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "analyze_screen":
        screenshot_path = take_screenshot()

        result = analyze_screen(
            screenshot_path,
            arguments["question"]
        )
        print(f"[DEBUG] ToolRouter: analyze_screen returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "active_window":
        result = describe_active_window()
        print(f"[DEBUG] ToolRouter: active_window returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "type_text":
        result = type_text(
            arguments["text"],
            arguments.get(
                "delay",
                0.02
            )
        )
        print(f"[DEBUG] ToolRouter: type_text returned in {time.perf_counter() - t0:.3f}s")
        return result

    if tool_name == "press_key":
        result = press_key(
            arguments["key"]
        )
        print(
            f"[DEBUG] ToolRouter: press_key returned in "
            f"{time.perf_counter() - t0:.3f}s"
        )
        return result

    if tool_name == "click_at":
        result = click_at(
            arguments["x"],
            arguments["y"],
        )
        print(
            f"[DEBUG] ToolRouter: click_at returned in "
            f"{time.perf_counter() - t0:.3f}s"
        )
        return result

    return (
        f"Unknown tool: {tool_name}"
    )