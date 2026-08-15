from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from brain.executor import Executor
from brain.models import Action, ActionRisk, Plan, PlanStatus, ToolResult
from brain.session_context import SessionContext
from security.safety import may_auto_execute
from tools.browser_agent import BrowserAgent, HumanActionRequired
from tools.desktop_agent import focus_target, type_into_control, wait_for_window
from tools.files import create_text_file, verify_file, write_text_file
from tools.ui import press_key


class AgentRuntime:
    def __init__(self, context: SessionContext | None = None, browser: BrowserAgent | None = None, trace: bool = True):
        self.context = context or SessionContext()
        self.browser = browser or BrowserAgent()
        self.executor = Executor()
        self.trace = trace

    def _log(self, message: str) -> None:
        if self.trace:
            print(message)

    def execute(self, plan: Plan) -> list[ToolResult]:
        plan.status = PlanStatus.RUNNING
        self.context.current_plan = plan
        results = []
        self._log(f"[PLAN] Goal: {plan.safe_goal()}")
        self._log(f"[PLAN] {len(plan.actions)} steps")
        while plan.current_action_index < len(plan.actions):
            index = plan.current_action_index
            action = plan.actions[index]
            if any(dependency not in plan.completed_actions for dependency in action.depends_on):
                result = ToolResult(False, action.tool, "Dependency not completed.", error="dependency_failure")
            elif not may_auto_execute(action):
                result = ToolResult(False, action.tool, "High-impact action requires confirmation.", error="human_confirmation_required")
                plan.status = PlanStatus.PAUSED
            else:
                result = self._execute_with_retry(action)
            results.append(result)
            self.context.previous_action = action.tool
            self.context.previous_result = result.message
            if result.success:
                plan.completed_actions.append(index)
                plan.current_action_index += 1
                self._update_context(action, result)
                self._log("[OK]")
                continue
            if action.optional:
                plan.current_action_index += 1
                self._log(f"[SKIP] {result.error or result.message}")
                continue
            plan.failed_action = index
            plan.failure_information = result.error or result.message
            if plan.status is not PlanStatus.PAUSED:
                plan.status = PlanStatus.FAILED
            self._log(f"[STOP] {plan.failure_information}")
            break
        if plan.current_action_index == len(plan.actions):
            plan.status = PlanStatus.COMPLETED
        return results

    def _execute_with_retry(self, action: Action) -> ToolResult:
        attempts = 1 if action.risk is not ActionRisk.SAFE else 3
        last = None
        for attempt in range(attempts):
            self._log(f"[{self.context.current_plan.current_action_index + 1}/{len(self.context.current_plan.actions)}] {action.tool} {action.safe_args()}")
            try:
                last = self._execute_action(action)
            except HumanActionRequired as exc:
                self.context.current_plan.status = PlanStatus.PAUSED
                return ToolResult(False, action.tool, "Human action required.", error=str(exc))
            except Exception as exc:
                last = ToolResult(False, action.tool, f"Failed to execute {action.tool}.", error=f"{type(exc).__name__}: {exc}")
            if last.success:
                return last
            if attempt + 1 < attempts:
                self.context.current_plan.retry_count += 1
                time.sleep(0.05)
        return last

    def _execute_action(self, action: Action) -> ToolResult:
        tool, args = action.tool, action.args
        if tool == "wait_for_window":
            raw = wait_for_window(args["app_name"], self.context.last_pid, args.get("timeout", 5.0))
            return self._dict_result(tool, raw)
        if tool == "focus_application":
            return self._dict_result(tool, focus_target(args["app_name"]))
        if tool == "create_text_file":
            return self._dict_result(tool, create_text_file(args["path"], args["contents"]))
        if tool == "verify_file":
            return self._dict_result(tool, verify_file(args["path"], args.get("expected_content")))
        if tool == "write_text_file":
            return self._dict_result(tool, write_text_file(args["path"], args["contents"], args.get("overwrite", False)))
        if tool == "type_text" and self.context.active_app:
            uia_result = type_into_control(self.context.active_app, args["text"])
            if uia_result.get("success"):
                return self._dict_result(tool, uia_result)
        if tool == "save_active_document":
            press_key("ctrl+shift+s")
            time.sleep(0.15)
            from tools.keyboard import type_text
            type_text(args["path"], 0.005)
            press_key("enter")
            deadline = time.perf_counter() + 3
            while time.perf_counter() < deadline:
                if Path(args["path"]).exists():
                    return ToolResult(True, tool, f"Saved {args['path']}.")
                time.sleep(0.05)
            return ToolResult(False, tool, "Save dialog did not create the file.", error="verification_failed")
        if tool.startswith("browser_"):
            return self._browser_action(tool, args)
        result = self.executor.execute_action(action)
        return result

    def _browser_action(self, tool: str, args: dict) -> ToolResult:
        before = self.browser.get_current_url() if self.browser.page else None
        if tool == "browser_open_url": state = self.browser.open_url(args["url"])
        elif tool == "browser_click_first_result": state = self.browser.click_first_result()
        elif tool == "browser_click": state = self.browser.click_element(args["target"], args.get("kind"))
        elif tool == "browser_type":
            self.browser.type_into_field(args["target"], args["text"], args.get("clear", True)); state = self.browser.get_page_state()
        elif tool == "browser_select":
            self.browser.select_option(args["target"], args["option"]); state = self.browser.get_page_state()
        elif tool == "browser_scroll":
            self.browser.scroll(args.get("direction", "down"), args.get("amount", 700)); state = self.browser.get_page_state()
        elif tool == "browser_go_back": state = self.browser.go_back()
        elif tool == "browser_go_forward": state = self.browser.go_forward()
        elif tool == "browser_fullscreen":
            self.browser.press_key("f"); state = self.browser.get_page_state()
        else: return ToolResult(False, tool, error="unknown_browser_action")
        return ToolResult(True, tool, f"Browser: {state.title}", {"url": state.url, "title": state.title, "previous_url": before})

    @staticmethod
    def _dict_result(tool: str, raw: dict) -> ToolResult:
        return ToolResult(bool(raw.get("success")), tool, raw.get("message", ""), raw, raw.get("error"))

    def _update_context(self, action: Action, result: ToolResult) -> None:
        if action.tool == "open_application":
            self.context.last_opened_app = self.context.active_app = action.args["app_name"]
            self.context.last_pid = result.data.get("pid")
        elif action.tool in {"wait_for_window", "focus_application"}:
            self.context.last_hwnd = result.data.get("hwnd", self.context.last_hwnd)
            self.context.active_app = action.args.get("app_name", self.context.active_app)
        elif action.tool.startswith("browser_"):
            self.context.browser_active = True
            self.context.active_app = "browser"
            self.context.current_url = result.data.get("url")
            if action.tool == "browser_open_url":
                parsed = urlparse(action.args.get("url", ""))
                query = parse_qs(parsed.query)
                self.context.last_search_query = (query.get("q") or query.get("search_query") or [None])[0]
                self.context.last_search_provider = "youtube" if "youtube" in parsed.netloc else "google" if "google" in parsed.netloc else None
        elif action.tool in {"create_text_file", "write_text_file", "verify_file", "save_active_document"}:
            self.context.last_opened_file = action.args.get("path")
