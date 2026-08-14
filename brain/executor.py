import time

from brain.models import Action, ToolResult
from brain.tool_router import execute_tool
from tools.window import (
    find_top_window_for_pid,
    bring_hwnd_to_foreground,
    find_application_window,
)


class Executor:
    def __init__(self):
        self.last_opened_app = None
        self.last_opened_pid: int | None = None
        self.last_opened_hwnd: int | None = None

    def execute_plan(
        self,
        actions: list[Action],
    ) -> list[ToolResult]:

        results = []
        # Reset plan state
        self.last_opened_app = None
        self.last_opened_pid = None
        self.last_opened_hwnd = None

        for action in actions:
            result = self.execute_action(action)
            results.append(result)

            if not result.success:
                break

        return results

    def execute_action(
        self,
        action: Action,
    ) -> ToolResult:

        try:
            # OPEN APPLICATION
            if action.tool == "open_application":
                start = time.perf_counter()

                raw_result = execute_tool(
                    action.tool,
                    action.args,
                )

                duration = time.perf_counter() - start

                self.last_opened_app = action.args.get(
                    "app_name"
                )

                # If the open_application returned a pid, remember it so future steps
                # (like type_text) can focus the correct window.
                try:
                    if isinstance(raw_result, dict) and raw_result.get("pid"):
                        self.last_opened_pid = int(raw_result.get("pid"))
                    else:
                        self.last_opened_pid = None
                except Exception:
                    self.last_opened_pid = None

                # Small temporary readiness delay (minimal to avoid blocking).
                time.sleep(0.05)

                # If the tool returned structured data (dict), preserve it in ToolResult.data
                if isinstance(raw_result, dict):
                    return ToolResult(
                        success=True,
                        tool=action.tool,
                        message=str(raw_result.get("message") or raw_result),
                        data=raw_result,
                    )

                return ToolResult(
                    success=True,
                    tool=action.tool,
                    message=str(raw_result),
                )

            # TYPE TEXT
            if action.tool == "type_text":
                start = time.perf_counter()
                # If we recently opened an app, attempt to find and focus its
                # top-level window by PID before typing so input goes to it.
                if self.last_opened_pid:
                    # Attempt a short PID-based lookup first; fallback to name-based lookup quickly.
                    # Try a short PID-based lookup first; fallback to name-based lookup
                    # quickly if not found. Keep timeout short to avoid blocking.
                    hwnd = find_top_window_for_pid(
                        self.last_opened_pid,
                        timeout=0.25,
                    )

                    if hwnd:
                        self.last_opened_hwnd = hwnd
                        focused = bring_hwnd_to_foreground(hwnd)
                    else:
                        # No PID window found; try by app name if available.
                        try:
                            hwnd = find_application_window(self.last_opened_app)
                            if hwnd:
                                self.last_opened_hwnd = hwnd
                                focused = bring_hwnd_to_foreground(hwnd)
                            else:
                                pass
                        except Exception as e:
                            pass

                raw_result = execute_tool(
                    action.tool,
                    action.args,
                )
                # type_text timing available via profiler when needed.

                if isinstance(raw_result, dict):
                    return ToolResult(
                        success=True,
                        tool=action.tool,
                        message=str(raw_result.get("message") or raw_result),
                        data=raw_result,
                    )

                return ToolResult(
                    success=True,
                    tool=action.tool,
                    message=str(raw_result),
                )

            # OTHER TOOLS
            start = time.perf_counter()
            raw_result = execute_tool(
                action.tool,
                action.args,
            )

            if isinstance(raw_result, dict):
                return ToolResult(
                    success=True,
                    tool=action.tool,
                    message=str(raw_result.get("message") or raw_result),
                    data=raw_result,
                )

            return ToolResult(
                success=True,
                tool=action.tool,
                message=str(raw_result),
            )

        except Exception as e:
            return ToolResult(
                success=False,
                tool=action.tool,
                message=f"Failed to execute {action.tool}.",
                error=str(e),
            )