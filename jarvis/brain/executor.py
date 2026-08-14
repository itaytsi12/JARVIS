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
        self.last_opened_app = None

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
                print("[DEBUG] Executor: starting open_application")
                start = time.perf_counter()

                raw_result = execute_tool(
                    action.tool,
                    action.args,
                )

                duration = time.perf_counter() - start
                print(f"[DEBUG] Executor: open_application returned: {duration:.3f}s")

                self.last_opened_app = action.args.get(
                    "app_name"
                )

                # If the open_application returned a pid, remember it so future steps
                # (like type_text) can focus the correct window.
                try:
                    if isinstance(raw_result, dict) and raw_result.get("pid"):
                        self.last_opened_pid = int(raw_result.get("pid"))
                        print(f"[DEBUG] Executor: recorded last_opened_pid={self.last_opened_pid}")
                    else:
                        self.last_opened_pid = None
                except Exception:
                    self.last_opened_pid = None

                # Small temporary readiness delay.
                # Reduced to minimal sleep to avoid blocking the next action.
                print("[DEBUG] Executor: starting post-open sleep 0.05s")
                t0 = time.perf_counter()
                time.sleep(0.05)
                print(f"[DEBUG] Executor: post-open sleep done: {time.perf_counter() - t0:.3f}s")

                return ToolResult(
                    success=True,
                    tool=action.tool,
                    message=str(raw_result),
                )

            # TYPE TEXT
            if action.tool == "type_text":
                print("[DEBUG] Executor: starting type_text")
                start = time.perf_counter()
                # If we recently opened an app, attempt to find and focus its
                # top-level window by PID before typing so input goes to it.
                if self.last_opened_pid:
                    print(f"[DEBUG] Executor: attempting to find window for pid {self.last_opened_pid}")
                    # Try a short PID-based lookup first; fallback to name-based lookup
                    # quickly if not found. Keep timeout short to avoid blocking.
                    hwnd = find_top_window_for_pid(
                        self.last_opened_pid,
                        timeout=0.25,
                    )

                    if hwnd:
                        print(f"[DEBUG] Executor: found hwnd {hwnd} for pid {self.last_opened_pid}")
                        self.last_opened_hwnd = hwnd
                        focused = bring_hwnd_to_foreground(hwnd)
                        print(f"[DEBUG] Executor: bring_hwnd_to_foreground returned {focused}")
                    else:
                        print(f"[DEBUG] Executor: no window found for pid {self.last_opened_pid}, trying by app name {self.last_opened_app}")
                        try:
                            hwnd = find_application_window(self.last_opened_app)
                            if hwnd:
                                print(f"[DEBUG] Executor: find_application_window found hwnd {hwnd} for {self.last_opened_app}")
                                self.last_opened_hwnd = hwnd
                                focused = bring_hwnd_to_foreground(hwnd)
                                print(f"[DEBUG] Executor: bring_hwnd_to_foreground returned {focused}")
                            else:
                                print(f"[DEBUG] Executor: find_application_window did not find a window for {self.last_opened_app}")
                        except Exception as e:
                            print(f"[DEBUG] Executor: find_application_window raised: {e}")

                raw_result = execute_tool(
                    action.tool,
                    action.args,
                )
                print(f"[DEBUG] Executor: type_text returned: {time.perf_counter() - start:.3f}s")

                return ToolResult(
                    success=True,
                    tool=action.tool,
                    message=str(raw_result),
                )

            # OTHER TOOLS
            print(f"[DEBUG] Executor: starting {action.tool}")
            start = time.perf_counter()
            raw_result = execute_tool(
                action.tool,
                action.args,
            )
            print(f"[DEBUG] Executor: {action.tool} returned: {time.perf_counter() - start:.3f}s")

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