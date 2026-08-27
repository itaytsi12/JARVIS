from __future__ import annotations

import logging
import os
import threading
import time
import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from brain.context_resolver import observe_tool_result
from brain.action_results import ReferenceError_, plan_has_references, resolve_arg_references, with_reference_dependencies
from brain.execution_graph import build_waves, partition_wave, plan_is_chain
from brain.executor import Executor
from brain.models import Action, ActionRisk, Plan, PlanStatus, ToolResult
from brain.recovery import plan_recovery
from brain.safe_tools import CONTEXT_INDEPENDENT_TOOLS
from brain.session_context import SessionContext
from brain.task_planner import _BROWSER_CAPABLE_APPS
from security.safety import may_auto_execute
from tools.browser_agent import BrowserAgent, HumanActionRequired
from tools.desktop_agent import click_control,focus_target,get_controls,type_into_control,type_into_notepad_native,wait_for_window
from tools.files import create_text_file, verify_file, write_text_file,open_path,read_text_file,rename_path,move_path
from tools.ui import press_key
from tools.whatsapp import send_whatsapp_message
from tools.applications import close_application
from tools.system import close_foreground_window,maximize_foreground_window,minimize_foreground_window,restore_foreground_window
from brain.resource_locks import acquire_action_resource
from training_data.sanitizer import sanitize_action_arguments

runtime_log = logging.getLogger("jarvis.runtime")
NON_RETRYABLE_TOOLS={
    "open_application","close_application","open_website","open_path","open_folder","open_file_explorer",
    "type_text","press_key","click_at","click_ui_element","browser_open_url","browser_click_first_result",
    "browser_click","browser_type","browser_select","browser_scroll","browser_go_back","browser_go_forward","browser_fullscreen",
    "create_text_file","write_text_file","append_text_file","rename_path","move_path","copy_path","save_current_document","save_active_document",
    "send_whatsapp_message","submit_form","upload_file","overwrite_file",
}
RETRYABLE_READ_ONLY_TOOLS={"wait_for_window","verify_file","exists","list_files","read_text_file","find_file","search_text","active_window","inspect_window","get_time","get_date","get_day"}


def _all_actions_independent(actions: list[Action]) -> bool:
    """Part H: a plan is safe to run concurrently only when EVERY action is
    both explicitly dependency-free (`depends_on` empty -- e.g. "open
    Chrome, then navigate, then click" always sets this) and its tool is in
    `CONTEXT_INDEPENDENT_TOOLS` (self-contained: reads no session state a
    sibling action writes). A single-action plan gains nothing from the
    parallel path, so it stays on the ordinary sequential one."""
    return len(actions) >= 2 and all(
        not action.depends_on and not action.optional and action.tool in CONTEXT_INDEPENDENT_TOOLS
        for action in actions
    )


class AgentRuntime:
    def __init__(self, context: SessionContext | None = None, browser: BrowserAgent | None = None, trace: bool | None = None, memory=None, session_id=None):
        # `trace` prints a per-step plan trace to stdout. It used to default
        # to True, which floods the terminal during ordinary agent work; it
        # now follows JARVIS_DEBUG instead. Callers that pass an explicit
        # value (every test does) are unaffected.
        if trace is None:
            from config import get_config

            trace = get_config().debug
        self.context = context or SessionContext()
        self.browser = browser or BrowserAgent()
        self.executor = Executor()
        self.trace = trace
        self.memory = memory
        self.session_id = session_id or (memory.start_session() if memory else None)
        self.context.memory = memory
        self.context.session_id = self.session_id
        self._context_lock = threading.Lock()

    def _log(self, message: str) -> None:
        if self.trace:
            print(message)

    def execute(self, plan: Plan, cancellation_token=None, action_observer=None) -> list[ToolResult]:
        with acquire_action_resource("__action_plan__",cancellation_token) as plan_resource:
            def observer(stage,index,action,result,context):
                if stage=="result" and result is not None:
                    if not isinstance(result.data,dict):result.data={}
                    result.data.update({"plan_resource":"action_plan","plan_resource_wait_ms":round(plan_resource.get("wait_ms",0.0),3),"plan_cross_process_lock":bool(plan_resource.get("cross_process"))})
                if action_observer is not None:action_observer(stage,index,action,result,context)
            results=self._execute_plan(plan,cancellation_token,observer)
            for result in results:
                if not isinstance(result.data,dict):result.data={}
                result.data.update({"plan_resource":"action_plan","plan_resource_wait_ms":round(plan_resource.get("wait_ms",0.0),3),"plan_cross_process_lock":bool(plan_resource.get("cross_process"))})
            return results

    def _execute_plan(self, plan: Plan, cancellation_token=None, action_observer=None) -> list[ToolResult]:
        plan.status = PlanStatus.RUNNING
        self.context.current_plan = plan
        results = []
        self._log(f"[PLAN] Goal: {plan.safe_goal()}")
        self._log(f"[PLAN] {len(plan.actions)} steps")
        # A pure chain (every action waiting on the one before it -- what
        # `brain/task_planner.py` emits) gains nothing from scheduling and
        # keeps the original sequential path below, byte for byte. Anything
        # with real structure goes through the dependency scheduler, which
        # runs each wave's independent actions together and everything else
        # in order. `_all_actions_independent` plans are simply the case
        # where that produces a single all-parallel wave.
        # A chain that passes results between its actions also goes through
        # the scheduler: it levels into single-action waves (so the order is
        # identical to the sequential path) but gains reference resolution,
        # which the sequential loop does not perform.
        if not plan_is_chain(plan.actions) or plan_has_references(plan.actions):
            self._log("[PLAN] Scheduling by dependency graph")
            return self._execute_plan_scheduled(plan, cancellation_token, action_observer)
        while plan.current_action_index < len(plan.actions):
            index = plan.current_action_index
            action = plan.actions[index]
            if action_observer is not None:action_observer("prepared",index,action,None,self.context)
            if cancellation_token is not None and cancellation_token.cancelled:
                result=ToolResult(False,action.tool,"Task cancelled.",error="cancelled");results.append(result)
                if action_observer is not None:action_observer("result",index,action,result,self.context)
                plan.status=PlanStatus.FAILED;break
            if any(dependency not in plan.completed_actions for dependency in action.depends_on):
                result = ToolResult(False, action.tool, "Dependency not completed.", error="dependency_failure")
            elif not may_auto_execute(action):
                result = ToolResult(False, action.tool, "High-impact action requires confirmation.", error="human_confirmation_required")
                plan.status = PlanStatus.PAUSED
            else:
                result = self._execute_with_retry(action,cancellation_token)
                if not result.success:
                    recovery = plan_recovery(action, result)
                    if recovery is not None:
                        result = self._attempt_recovery(plan, index, action, result, recovery, cancellation_token, None)
            results.append(result)
            self.context.previous_action = action.tool
            self.context.previous_result = result.message
            if result.success:
                plan.completed_actions.append(index)
                plan.current_action_index += 1
                self._update_context(action, result)
                if action_observer is not None:action_observer("result",index,action,result,self.context)
                self._log("[OK]")
                continue
            if action_observer is not None:action_observer("result",index,action,result,self.context)
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

    def _run_one_action(self, plan, index, action, cancellation_token, action_observer, plan_lock_held, results_by_index=None, allow_recovery=True):
        """Execute a single action and fold its result into shared state.

        Shared by the sequential-within-a-wave path and the concurrent one,
        so a parallel action and a sequential action are executed by exactly
        the same code -- only their scheduling differs. Every mutation of
        `self.context` happens under `self._context_lock` because several
        worker threads may finish at the same moment.
        """
        if action_observer is not None:
            action_observer("prepared", index, action, None, self.context)
        # Substitute any {"__from_result__": ...} argument with the field of
        # the earlier result it names. A bad reference fails THIS action --
        # attributed to the action that made it, never silently ignored.
        try:
            action = self._with_resolved_arguments(action, results_by_index)
        except ReferenceError_ as exc:
            result = ToolResult(False, action.tool, "Could not use the earlier result.", error=f"unresolved_reference: {exc}")
            with self._context_lock:
                self.context.previous_action = action.tool
                self.context.previous_result = result.message
            if action_observer is not None:
                action_observer("result", index, action, result, self.context)
            return result
        if cancellation_token is not None and cancellation_token.cancelled:
            result = ToolResult(False, action.tool, "Task cancelled.", error="cancelled")
        elif not may_auto_execute(action):
            result = ToolResult(False, action.tool, "High-impact action requires confirmation.", error="human_confirmation_required")
        else:
            try:
                result = self._execute_with_retry(action, cancellation_token, plan_lock_held=plan_lock_held)
            except Exception as exc:
                # An exception escaping one action must never abort its
                # siblings or lose the others' results. It is converted to a
                # failed ToolResult -- recorded, reported and visible -- not
                # swallowed.
                result = ToolResult(False, action.tool, f"Failed to execute {action.tool}.", error=f"{type(exc).__name__}: {exc}")
        # One bounded recovery attempt, and only for a genuine tool failure.
        # `allow_recovery=False` on the recovery actions themselves is what
        # makes an infinite retry loop structurally impossible: a failing
        # recovery can never generate more recovery.
        if allow_recovery and not result.success:
            recovery = plan_recovery(action, result)
            if recovery is not None:
                result = self._attempt_recovery(plan, index, action, result, recovery, cancellation_token, action_observer)
        with self._context_lock:
            self.context.previous_action = action.tool
            self.context.previous_result = result.message
            if result.success:
                self._update_context(action, result)
        if action_observer is not None:
            action_observer("result", index, action, result, self.context)
        return result

    def _attempt_recovery(self, plan, index, action, result, recovery, cancellation_token, action_observer):
        """Run a `Recovery` and return the result that should stand.

        A clarification replaces the raw tool error with a question for the
        user. Otherwise each proposed action runs once, with recovery
        disabled; if they all succeed the original action is reported as
        recovered, and if any fails the ORIGINAL failure is what stands --
        a recovery attempt never invents a success.
        """
        runtime_log.info("Recovering from %s on %s (%s)", result.error, action.tool, recovery.reason)
        if recovery.clarification:
            recovered = ToolResult(False, action.tool, recovery.clarification, error="needs_clarification")
            recovered.data = dict(result.data or {})
            recovered.data.update({"recovery_reason": recovery.reason, "original_error": result.error})
            return recovered
        for candidate in recovery.actions:
            attempt = self._run_one_action(plan, index, candidate, cancellation_token, None, False, None, allow_recovery=False)
            if not attempt.success:
                if not isinstance(result.data, dict):
                    result.data = {}
                result.data.update({"recovery_attempted": recovery.reason, "recovery_succeeded": False})
                return result
        recovered = ToolResult(True, action.tool, result.message or f"{action.tool} completed after recovery.")
        recovered.data = dict(result.data or {})
        recovered.data.update({"recovery_attempted": recovery.reason, "recovery_succeeded": True, "original_error": result.error})
        return recovered

    @staticmethod
    def _with_resolved_arguments(action, results_by_index):
        """A copy of `action` with its argument references substituted."""
        if not results_by_index:
            resolved = resolve_arg_references(action.args, {})
        else:
            resolved = resolve_arg_references(action.args, results_by_index)
        if resolved == action.args:
            return action
        return Action(
            tool=action.tool, args=resolved, depends_on=list(action.depends_on), optional=action.optional,
            verify=action.verify, stop_condition=action.stop_condition, risk=action.risk,
            sensitive_fields=set(action.sensitive_fields), max_attempts=action.max_attempts,
        )

    def _execute_plan_scheduled(self, plan: Plan, cancellation_token=None, action_observer=None) -> list[ToolResult]:
        """Execute a plan by dependency wave instead of strictly in order.

        `brain/execution_graph.py` levels the plan into waves and splits each
        wave into the actions that may genuinely run together and those that
        must not. Within a wave the parallel group runs on a bounded pool and
        the rest run one at a time; the next wave does not start until the
        current one has fully finished, which is what makes every
        `depends_on` edge hold.

        Failure semantics match the sequential path: an action whose
        dependency never completed is reported as `dependency_failure` and
        not executed, an `optional` failure is skipped, and any other failure
        stops the plan -- but only AFTER the wave it happened in has
        finished, so a sibling that was already running still returns its own
        result rather than being discarded.
        """
        results_by_index: dict[int, ToolResult] = {}
        needs_confirmation = False
        stop = False
        # `AgentRuntime.execute()` already holds the process-wide
        # "action_plan" lock for this whole plan on the calling thread, and
        # RLock reentrancy is per-thread -- a worker thread re-acquiring it
        # would deadlock against its own plan. See _execute_plan_parallel's
        # original note; the same reasoning applies to every worker here.
        # A reference to an earlier result is itself an ordering constraint,
        # so a plan that references without declaring `depends_on` still runs
        # in a correct order instead of reading a result that does not exist.
        plan.actions = with_reference_dependencies(plan.actions)
        for wave in build_waves(plan.actions):
            if stop:
                break
            runnable, blocked = [], []
            for index in wave:
                if any(dep not in plan.completed_actions for dep in plan.actions[index].depends_on):
                    blocked.append(index)
                else:
                    runnable.append(index)
            for index in blocked:
                action = plan.actions[index]
                if action_observer is not None:
                    action_observer("prepared", index, action, None, self.context)
                result = ToolResult(False, action.tool, "Dependency not completed.", error="dependency_failure")
                results_by_index[index] = result
                if action_observer is not None:
                    action_observer("result", index, action, result, self.context)
            parallel, sequential = partition_wave(plan.actions, runnable)
            if parallel:
                workers = max(1, min(len(parallel), self._max_parallel_actions()))
                self._log(f"[PLAN] Running {len(parallel)} independent steps concurrently")
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jarvis-parallel-action") as pool:
                    futures = {
                        index: pool.submit(self._run_one_action, plan, index, plan.actions[index], cancellation_token, action_observer, True, dict(results_by_index))
                        for index in parallel
                    }
                    for index, future in futures.items():
                        results_by_index[index] = future.result()
            for index in sequential:
                # Runs on THIS thread, which already holds the plan-level
                # resource lock reentrantly -- so `plan_lock_held` stays
                # False, exactly as on the original sequential path. Only a
                # worker thread needs the skip.
                results_by_index[index] = self._run_one_action(
                    plan, index, plan.actions[index], cancellation_token, action_observer, False, results_by_index
                )
            for index in sorted(results_by_index):
                if index in plan.completed_actions:
                    continue
                result = results_by_index[index]
                if result.success:
                    plan.completed_actions.append(index)
                    continue
                if result.error == "human_confirmation_required":
                    needs_confirmation = True
                    stop = True
                elif plan.actions[index].optional:
                    self._log(f"[SKIP] {result.error or result.message}")
                    continue
                elif result.error == "dependency_failure":
                    stop = True
                else:
                    stop = True
                if plan.failed_action is None:
                    plan.failed_action = index
                    plan.failure_information = result.error or result.message
        results = [results_by_index[index] for index in sorted(results_by_index)]
        plan.current_action_index = len(plan.actions) if len(plan.completed_actions) == len(plan.actions) else max(plan.completed_actions, default=-1) + 1
        if len(plan.completed_actions) == len(plan.actions):
            plan.status = PlanStatus.COMPLETED
        elif needs_confirmation:
            plan.status = PlanStatus.PAUSED
        elif all(result.success for result in results) and not stop:
            plan.status = PlanStatus.COMPLETED
        else:
            plan.status = PlanStatus.FAILED
        return results

    @staticmethod
    def _max_parallel_actions() -> int:
        from config import get_config

        return max(1, int(getattr(get_config(), "max_parallel_tools", 4) or 1))

    def _execute_plan_parallel(self, plan: Plan, cancellation_token=None, action_observer=None) -> list[ToolResult]:
        """Back-compatible entry point for an all-independent plan.

        This used to be a second, standalone execution engine for the one
        case where every action could run at once. That case is now just the
        shape `_execute_plan_scheduled` produces when the dependency graph
        levels into a single wave, so this delegates instead of maintaining a
        parallel implementation that could drift from the scheduled one.
        Kept as a named method because callers and tests refer to it.
        """
        return self._execute_plan_scheduled(plan, cancellation_token, action_observer)

    def _execute_with_retry(self, action: Action, cancellation_token=None, plan_lock_held: bool = False) -> ToolResult:
        attempts = action.max_attempts or (3 if action.tool in RETRYABLE_READ_ONLY_TOOLS and action.risk is ActionRisk.SAFE else 1)
        last = None;attempt_history=[]
        for attempt in range(attempts):
            self._log(f"[{self.context.current_plan.current_action_index + 1}/{len(self.context.current_plan.actions)}] {action.tool} {sanitize_action_arguments(action.tool,action.args)}")
            try:
                with acquire_action_resource(action.tool,cancellation_token) as resource_info:
                    # Preserve the exact historical call arity when neither
                    # new argument is in play, since test subclasses (e.g.
                    # tests/test_agent_runtime.py's FailingRuntime) override
                    # `_execute_action(self, action)` with that narrower
                    # signature.
                    if cancellation_token is not None and plan_lock_held:
                        last = self._execute_action(action,cancellation_token,plan_lock_held)
                    elif cancellation_token is not None:
                        last = self._execute_action(action,cancellation_token)
                    elif plan_lock_held:
                        last = self._execute_action(action,plan_lock_held=plan_lock_held)
                    else:
                        last = self._execute_action(action)
            except HumanActionRequired as exc:
                self.context.current_plan.status = PlanStatus.PAUSED
                return ToolResult(False, action.tool, "Human action required.", error=str(exc))
            except Exception as exc:
                last = ToolResult(False, action.tool, f"Failed to execute {action.tool}.", error=f"{type(exc).__name__}: {exc}")
                resource_info={"resource":None,"wait_ms":0.0}
            if not isinstance(last.data,dict):last.data={}
            attempt_history.append({"attempt":attempt+1,"success":bool(last.success),"error":last.error})
            last.data.update({"attempts":attempt+1,"retries":attempt,"attempt_history":list(attempt_history),"resource":resource_info.get("resource"),"resource_wait_ms":round(resource_info.get("wait_ms",0.0),3),"cross_process_lock":bool(resource_info.get("cross_process"))})
            if last.success:
                return last
            if attempt + 1 < attempts:
                self.context.current_plan.retry_count += 1
                time.sleep(0.05)
        return last

    def _execute_action(self, action: Action, cancellation_token=None, plan_lock_held: bool = False) -> ToolResult:
        tool, args = action.tool, action.args
        if tool == "type_text":
            if os.getenv("DEBUG_REFERENCE_RESOLUTION","false").lower() in {"1","true","yes","on"}:
                runtime_log.info("Exact text passed to type_text: %r; context_id=%s",args.get("text"),id(self.context))
            else:runtime_log.info("type_text payload length=%d; context_id=%s",len(str(args.get("text") or "")),id(self.context))
        if tool == "wait_for_window":
            raw = wait_for_window(args["app_name"], self.context.last_pid, args.get("timeout", 5.0))
            return self._dict_result(tool, raw)
        if tool == "focus_application":
            return self._dict_result(tool,focus_target(args["app_name"],self.context.application_windows.get(args["app_name"])))
        if tool == "close_application" and self.context.active_app==args["app_name"] and self.context.last_hwnd:
            return self._dict_result(tool,close_application(args["app_name"],self.context.last_hwnd))
        window_actions={"minimize_window":minimize_foreground_window,"maximize_window":maximize_foreground_window,"restore_window":restore_foreground_window,"close_window":close_foreground_window}
        if tool in window_actions and self.context.last_hwnd:
            return self._dict_result(tool,window_actions[tool](self.context.last_hwnd))
        if tool == "inspect_window":
            expected=self.context.last_hwnd if self.context.active_app==args["app_name"] else None
            return self._dict_result(tool,get_controls(args["app_name"],args.get("limit",50),expected))
        if tool == "click_ui_element":
            expected=self.context.last_hwnd if self.context.active_app==args["app_name"] else None
            if args["app_name"]=="notepad" and args["name"].strip().casefold()=="save" and expected:
                focused=focus_target("notepad",expected)
                if focused.get("success"):press_key("ctrl+s")
                else:
                    user32=ctypes.windll.user32
                    user32.PostMessageW(expected,0x0100,0x11,0);user32.PostMessageW(expected,0x0100,ord("S"),0)
                    user32.PostMessageW(expected,0x0101,ord("S"),0);user32.PostMessageW(expected,0x0101,0x11,0)
                deadline=time.perf_counter()+3
                while time.perf_counter()<deadline:
                    hwnd=ctypes.windll.user32.GetForegroundWindow();length=ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                    title=ctypes.create_unicode_buffer(length+1);ctypes.windll.user32.GetWindowTextW(hwnd,title,length+1)
                    if hwnd!=expected and "save" in title.value.casefold():return ToolResult(True,tool,"Opened Notepad's Save dialog.",{"verified":True,"hwnd":hwnd,"target_hwnd":expected})
                    time.sleep(.05)
                return ToolResult(False,tool,"Notepad's Save dialog did not appear.",{"verified":False,"target_hwnd":expected},"verification_failed")
            return self._dict_result(tool,click_control(args["app_name"],args["name"],args.get("control_type"),expected))
        if tool == "create_text_file":
            return self._dict_result(tool, create_text_file(args["path"], args["contents"]))
        if tool == "open_path":return self._dict_result(tool,open_path(args["path"]))
        if tool == "read_text_file":return self._dict_result(tool,read_text_file(args["path"]))
        if tool == "rename_path":return self._dict_result(tool,rename_path(args["path"],args["new_name"]))
        if tool == "move_path":return self._dict_result(tool,move_path(args["source"],args["destination"]))
        if tool == "verify_file":
            return self._dict_result(tool, verify_file(args["path"], args.get("expected_content")))
        if tool == "write_text_file":
            return self._dict_result(tool, write_text_file(args["path"], args["contents"], args.get("overwrite", False)))
        if tool == "type_text" and self.context.active_app and self.context.last_hwnd:
            if self.context.active_app=="notepad":
                return self._dict_result(tool,type_into_notepad_native(args["text"],self.context.last_hwnd))
            uia_result = type_into_control(self.context.active_app,args["text"],self.context.last_hwnd)
            if uia_result.get("success"):
                return self._dict_result(tool, uia_result)
            if self.context.last_hwnd:
                return self._dict_result(tool,uia_result)
        if tool == "type_text":
            return ToolResult(False,tool,"I don't have a verified target application to type into.",{"verified":False},"target_window_unverified")
        if tool == "press_key":
            if not self.context.active_app or not self.context.last_hwnd:
                return ToolResult(False,tool,"I don't have a verified target application for that key press.",{"verified":False},"target_window_unverified")
            focused=focus_target(self.context.active_app,self.context.last_hwnd)
            if not focused.get("success"):return self._dict_result(tool,focused)
            message=press_key(args["key"])
            return ToolResult(True,tool,message,{"verified":False,"hwnd":self.context.last_hwnd})
        if tool == "save_current_document":
            if self.context.active_app and self.context.last_hwnd:
                focused=focus_target(self.context.active_app,self.context.last_hwnd)
                if not focused.get("success"):return self._dict_result(tool,focused)
            press_key("ctrl+s")
            return ToolResult(True,tool,"Opened the current application's save flow.")
        if tool == "send_whatsapp_message":
            self.context.pending_messaging_recipient=args["recipient"];self.context.pending_messaging_message=args["message"];self.context.messaging_committed=False
            raw=send_whatsapp_message(args["recipient"],args["message"],args.get("literal",False),cancellation_token)
            if raw.get("committed"):self.context.messaging_committed=True;self.context.last_messaging_recipient=raw.get("recipient") or args["recipient"]
            return self._dict_result(tool,raw)
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
        result = self.executor.execute_action_unlocked_plan(action) if plan_lock_held else self.executor.execute_action(action)
        return result

    def _browser_action(self, tool: str, args: dict) -> ToolResult:
        """Browser actions are the agent runtime's own dispatch path (they
        never reach `brain/tool_router.py::execute_tool`), so they need the
        same Playwright-thread isolation applied here: sync Playwright leaves
        a running asyncio loop on its thread and binds its page objects to it
        (see `tools/playwright_runtime.py`)."""
        from tools.playwright_runtime import run_for_tool

        return run_for_tool(tool, self._browser_action_impl, tool, args)

    def _browser_action_impl(self, tool: str, args: dict) -> ToolResult:
        # This "before" snapshot is a diagnostic aid for the verification
        # comparison below, not the action itself -- a page that's already
        # dead (or dies while reading it) degrades to no prior URL rather
        # than failing the whole action before the real operation even gets
        # a chance to recover the session.
        try:
            is_live = self.browser.is_session_live() if hasattr(self.browser, "is_session_live") else bool(self.browser.page)
            before_state = self.browser.get_page_state() if is_live else None
        except Exception:
            before_state = None
        before = before_state.url if before_state else None
        verified=False
        if tool == "browser_open_url": state = self.browser.open_url(args["url"])
        elif tool == "browser_click_first_result": state = self.browser.click_first_result()
        elif tool == "browser_click": state = self.browser.click_element(args["target"], args.get("kind"))
        elif tool == "browser_type":
            verified=self.browser.type_into_field(args["target"],args["text"],args.get("clear",True));state=self.browser.get_page_state()
        elif tool == "browser_select":
            verified=self.browser.select_option(args["target"],args["option"]);state=self.browser.get_page_state()
        elif tool == "browser_scroll":
            self.browser.scroll(args.get("direction", "down"), args.get("amount", 700)); state = self.browser.get_page_state()
        elif tool == "browser_go_back": state = self.browser.go_back()
        elif tool == "browser_go_forward": state = self.browser.go_forward()
        elif tool == "browser_fullscreen":
            self.browser.press_key("f"); state = self.browser.get_page_state()
        else: return ToolResult(False, tool, error="unknown_browser_action")
        if tool=="browser_open_url":verified=bool(state.url) and (before is None or state.url!=before or state.url.startswith(args["url"]))
        elif tool in {"browser_click_first_result","browser_go_back","browser_go_forward"}:verified=bool(before and state.url and state.url!=before)
        elif tool == "browser_click":
            url_changed=bool(before and state.url and state.url!=before)
            state_changed=bool(before_state and (
                state.title != before_state.title
                or state.visible_text != before_state.visible_text
                or state.interactive_elements != before_state.interactive_elements
            ))
            verified=url_changed or state_changed
            if args.get("kind") == "link" and not url_changed:
                return ToolResult(False,tool,"Browser link click did not navigate.",{"url":state.url,"title":state.title,"previous_url":before,"verified":False},"verification_failed")
        if tool in {"browser_click_first_result","browser_go_back","browser_go_forward","browser_type","browser_select"} and not verified:return ToolResult(False,tool,"Browser action could not be verified.",{"url":state.url,"title":state.title,"previous_url":before,"verified":False},"verification_failed")
        message = f"Browser: {state.title}" if verified or tool != "browser_click" else "Browser click completed, but no resulting page change was independently verified."
        return ToolResult(True, tool, message, {"url": state.url, "title": state.title, "previous_url": before,"verified":verified})

    @staticmethod
    def _dict_result(tool: str, raw: dict) -> ToolResult:
        return ToolResult(bool(raw.get("success")), tool, raw.get("message", ""), raw, raw.get("error"))

    def _update_context(self, action: Action, result: ToolResult) -> None:
        # Generalized short-term context: records the command itself, any
        # error, and any structured list-like result (file listing, test
        # failures, ...) the same way regardless of which branch below
        # (if any) also updates the older, tool-specific fields.
        observe_tool_result(self.context, action.tool, action.args, result)
        if action.tool == "open_application":
            self.context.last_opened_app = self.context.active_app = action.args["app_name"]
            self.context.last_pid = result.data.get("pid")
            self.context.record_app_event(action.args["app_name"], "opened")
            if action.args["app_name"] in _BROWSER_CAPABLE_APPS:
                self.context.browser_active=True;self.context.browser_updated_at=time.monotonic();self.context.record_app_event("browser","opened")
            if result.data.get("hwnd"):self._remember_app_window(action.args["app_name"],result.data["hwnd"])
            if self.memory:
                self.memory.remember_entity("application", action.args["app_name"], self.session_id, pid=self.context.last_pid, hwnd=result.data.get("hwnd"), opened_by_jarvis=True)
        elif action.tool == "close_application":
            if self.context.active_app==action.args.get("app_name"):
                self.context.active_app=None;self.context.last_hwnd=None;self.context.last_pid=None
            self.context.application_windows.pop(action.args.get("app_name"),None)
            self.context.record_app_event(action.args.get("app_name"), "closed")
            if self.memory:
                resolution=self.memory.resolve(action.args["app_name"],self.session_id,"application")
                if resolution.status=="resolved":self.memory.update_entity(resolution.entity["id"],status="closed")
        elif action.tool in {"wait_for_window", "focus_application"}:
            self.context.last_hwnd = result.data.get("hwnd", self.context.last_hwnd)
            self.context.active_app = action.args.get("app_name", self.context.active_app)
            if self.context.active_app:self.context.record_app_event(self.context.active_app, "focused")
            if result.data.get("hwnd"):self._remember_app_window(self.context.active_app,result.data["hwnd"])
        elif action.tool.startswith("browser_"):
            self.context.browser_active = True
            self.context.active_app = "browser"
            self.context.current_url = result.data.get("url")
            self.context.browser_updated_at = time.monotonic()
            self.context.record_app_event("browser", "opened" if action.tool == "browser_open_url" else "focused")
            if action.tool == "browser_open_url":
                parsed = urlparse(action.args.get("url", ""))
                query = parse_qs(parsed.query)
                self.context.last_search_query = (query.get("q") or query.get("search_query") or [None])[0]
                self.context.last_search_provider = "youtube" if "youtube" in parsed.netloc else "google" if "google" in parsed.netloc else None
                if self.memory and self.context.last_search_query:
                    self.memory.remember_entity("search", self.context.last_search_query, self.session_id, provider=self.context.last_search_provider, url=result.data.get("url"))
        elif action.tool in {"create_text_file", "write_text_file", "verify_file", "save_active_document","open_path","read_text_file","rename_path","move_path"}:
            self.context.last_opened_file = result.data.get("path") or action.args.get("path") or action.args.get("destination")
            self.context.last_opened_file_at = time.monotonic()
            if self.memory and action.tool in {"create_text_file", "write_text_file"}:
                self.memory.remember_entity("file", Path(self.context.last_opened_file).name, self.session_id, path=self.context.last_opened_file)
        elif action.tool == "send_whatsapp_message":
            self.context.last_messaging_recipient=result.data.get("recipient") or action.args.get("recipient")
            self.context.messaging_committed=True

    def _remember_app_window(self,app_name,hwnd):
        windows=self.context.application_windows
        windows.pop(app_name,None);windows[app_name]=int(hwnd)
        while len(windows)>16:windows.pop(next(iter(windows)))
