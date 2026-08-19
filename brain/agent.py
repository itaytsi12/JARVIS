import logging
import hashlib
import os
import re
import time
import uuid
from datetime import datetime,timezone
from concurrent.futures import CancelledError,TimeoutError as FutureTimeoutError
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from openai import OpenAI

from brain.router import route_command
from brain.tool_router import execute_tool
from brain.planner import create_plan
from brain.executor import Executor
from brain.models import Action,ActionRisk,Plan,ToolResult
from brain.agent_runtime import AgentRuntime
from brain.task_planner import assess_plan_completeness,create_task_plan,should_use_task_planner,validate_goal_coverage
from brain.plan_validator import RUNTIME_TOOLS,validate_generated_actions,validate_plan_preflight
from memory import MemoryManager
from training_data import EventType, get_recorder
from training_data.sanitizer import privacy_safe_event,sanitize_text,sanitize_user_request
from brain.web_answer import get_web_answer_service
from brain.task_supervisor import ReadOnlyPriority,ReadOnlyTaskCancelled,active_interactive_task_summary,cancel_read_only_tasks,get_read_only_task_scheduler,wait_for_interactive_tasks


load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


executor = Executor()
memory_manager = MemoryManager()
agent_runtime = AgentRuntime(memory=memory_manager)
runtime_log = logging.getLogger("jarvis.runtime")
SESSION_AWARE_SINGLE_TOOLS={"press_key","close_application","focus_application","inspect_window","click_ui_element","minimize_window","maximize_window","restore_window","close_window"}
PRIVATE_RESPONSE_TOOLS={"read_text_file","analyze_screen"}


def _command_for_log(command):
    value=str(command)
    if os.getenv("DEBUG_REFERENCE_RESOLUTION","false").lower() not in {"1","true","yes","on"}:
        match=re.search(r"\b(?:send|message|tell|type|write)\b",value,re.I)
        if match:return f"{match.group(0).strip().lower()} request <content redacted; length={len(value)}>"
    return sanitize_text(value)


def ask_ai(message: str) -> str:
    response = client.responses.create(
        model="gpt-5-mini",
        input=message
    )

    return response.output_text


def run_agent(command: str, route: dict | None = None, interaction_id: str | None = None, original_user_text: str | None = None, cancellation_token=None, execution_outcome: dict | None = None, speculative_ledger=None) -> str:
    request_started = time.perf_counter()
    recorder = get_recorder()
    created_interaction = interaction_id is None
    iid = interaction_id or recorder.begin(original_user_text or command, agent_runtime.session_id)
    if created_interaction:
        safe_command=sanitize_user_request(command)
        recorder.record(EventType.NORMALIZED_REQUEST, {"normalized_text": safe_command, "final_interpreted_command": safe_command}, iid)
    recorder.record(EventType.TASK_STARTED, {"entrypoint": "run_agent", "input_mode": "voice" if original_user_text is not None else "text"}, iid)
    execution_meta = {}
    agent_runtime.context.last_user_message = original_user_text or command
    try:
        response = _run_agent_impl(command, route, recorder, iid, execution_meta, cancellation_token,original_user_text,speculative_ledger)
        explicit_success=execution_meta.get("success")
        textual_failure = explicit_success is None and ("error:" in response.lower() or "couldn't" in response.lower() or response.lower().startswith(("failed ","failure:")))
        failed = explicit_success is False or textual_failure
        verified = bool(execution_meta.get("verified",False)) and not failed
        recorder.record(
            EventType.TASK_FAILED if failed else EventType.TASK_COMPLETED,
            {
                "success": not failed,
                "verified": verified,
                "route_type": execution_meta.get("route_type"),
                "route_source": execution_meta.get("route_source"),
                "model_calls": execution_meta.get("model_calls",0),
            },
            iid,
        )
        recorded_response=f"<PRIVATE_RESPONSE_REDACTED; length={len(response)}>" if execution_meta.get("private_response") else response
        recorder.finalize(iid, success=not failed, verified=verified, response=recorded_response)
        agent_runtime.context.last_assistant_response = response
        outcome=_build_execution_outcome(execution_meta, overall_success=not failed)
        if execution_outcome is not None:
            execution_outcome.clear()
            execution_outcome.update(outcome)
        _log_request_performance(execution_meta, request_started, failed)
        _observe_for_improvement(command,route,outcome,original_user_text,iid,execution_meta,request_started)
        return response
    except Exception as exc:
        recorder.record(EventType.TASK_FAILED, {"error": f"{type(exc).__name__}: {exc}"}, iid)
        recorder.finalize(iid, success=False, response="Command failed")
        _log_request_performance(execution_meta, request_started, True)
        outcome={
            "executed": False, "success": False, "verified": False, "partial": False,
            "route_type": execution_meta.get("route_type"), "actions": [],
            "error": f"{type(exc).__name__}: {exc}",
            "exception_type": type(exc).__name__,
            "exception_message": sanitize_text(str(exc))[:500],
            "block_reason": execution_meta.get("block_reason"),
            "block_errors": list(execution_meta.get("block_errors") or []),
        }
        if execution_outcome is not None:
            execution_outcome.clear()
            execution_outcome.update(outcome)
        _observe_for_improvement(command,route,outcome,original_user_text,iid,execution_meta,request_started)
        raise


def _is_deterministic_music_route(route: dict | None) -> bool:
    """`brain/music_intent.py::route_music_command` (invoked from
    `brain/router.py::route_command` BEFORE any planner fallback) always
    returns a fully-resolved `{"type": "tool", "tool": "open_music" |
    "music_*", ...}` route -- there is nothing left to re-plan. Without
    this guard, `should_use_task_planner(command)` below is evaluated
    independent of what `route` already resolved to, so a real utterance
    could match one of its generic signals (e.g. a stray comma) and the
    planner would silently discard an already-correct music route and
    invent its own (e.g. opening music.youtube.com) -- confirmed live."""
    if not route or route.get("type") != "tool":
        return False
    tool = route.get("tool")
    return isinstance(tool, str) and (tool == "open_music" or tool.startswith("music_"))


def _active_agent_tasks():
    """Running agent tasks, or an empty list if the manager isn't up yet.

    Never raises: this only ever decorates a status answer, and a broken
    status lookup must not turn "what are you doing?" into an error.
    """
    try:
        from tasks.manager import get_task_manager
        from tasks.models import TaskStatus

        return [task for task in get_task_manager().active() if task.status is TaskStatus.RUNNING]
    except Exception:
        runtime_log.exception("Could not read agent task state for the status answer")
        return []


def _cancel_agent_tasks(reason: str = "user_cancelled") -> int:
    """Cancel every running agent task. Never raises, for the same reason."""
    try:
        from tasks.manager import get_task_manager

        return get_task_manager().cancel_all(reason)
    except Exception:
        runtime_log.exception("Could not cancel agent tasks")
        return 0


def _agent_escalation_available() -> bool:
    """Can a genuinely agentic request be handled right now?

    False whenever no model provider is configured -- which is the normal
    state until an API key is added -- so every deterministic local route
    below behaves exactly as it always has, with or without Claude.
    """
    try:
        from config import get_config
        from providers.registry import get_agent_provider

        if not get_config().agent_escalation_enabled:
            return False
        return get_agent_provider() is not None
    except Exception:
        runtime_log.exception("Agent availability check failed; staying on the local path")
        return False


def _run_agent_with_loop(goal: str, recorder, iid: str, execution_meta: dict, cancellation_token=None) -> str:
    """Hand a goal to the agent runtime (brain/agent_service.py).

    Records the same dataset events as every other route so an agentic
    interaction is not a blind spot in the training data.
    """
    from brain.agent_service import run_agent_task

    execution_meta["route_type"] = "agent_task"
    execution_meta["route_source"] = "agent_runtime"
    recorder.record(
        EventType.REASONING_REQUEST,
        {"operation": "agent_loop", "request": sanitize_user_request(goal)},
        iid,
    )
    outcome = run_agent_task(
        goal,
        runtime=agent_runtime,
        cancellation_token=cancellation_token,
        session_context=agent_runtime.context,
    )
    run = outcome.run
    execution_meta["success"] = outcome.success
    execution_meta["verified"] = outcome.verified
    execution_meta["model_calls"] = execution_meta.get("model_calls", 0) + run.model_calls
    execution_meta["cloud_ms"] = execution_meta.get("cloud_ms", 0.0) + run.duration_ms
    execution_meta["agent_episode_id"] = outcome.episode_id
    recorder.record(
        EventType.REASONING_RESPONSE,
        {
            "operation": "agent_loop",
            "model": run.model,
            "provider": run.provider,
            "model_calls": run.model_calls,
            "input_tokens": run.usage.input_tokens,
            "output_tokens": run.usage.output_tokens,
            "estimated_cost_usd": run.estimated_cost_usd,
            "stop_reason": run.stop_reason,
            "steps": len(run.steps),
            "success": outcome.success,
        },
        iid,
    )
    runtime_log.info("Agent runtime handled request: %s", outcome.describe())
    return outcome.answer


def _run_agent_impl(command: str, route: dict | None, recorder, iid: str, execution_meta: dict, cancellation_token=None,original_user_text: str | None=None,speculative_ledger=None) -> str:
    control_types={"cancel_read_only_task","task_status","resume_interrupted_response","correct_interrupted_response","revise_whatsapp_recipient","remember","agent_task"}
    if route is None and re.match(r"^(?:stop|cancel|never mind|continue|make it|shorten that|only (?:tell|give)|(?:don't send it|send it).+instead)",command,re.I):
        route=route_command(command)
    use_task_planner=(
        (not route or route.get("type") not in control_types)
        and not _is_deterministic_music_route(route)
        and should_use_task_planner(command)
    )
    runtime_log.info("Runtime decision: %s; command=%r; agent_runtime_id=%s context_id=%s", "task_planner" if use_task_planner else "deterministic_router", _command_for_log(command), id(agent_runtime), id(agent_runtime.context))
    if use_task_planner:
        execution_meta["route_type"] = "task_plan"
        execution_meta["route_source"] = "deterministic_planner"
        execution_meta["model_calls"] = 0
        if re.search(r"\b(it|that|earlier|previous|the browser|the file|the project|the task)\b", command, re.I):
            recorder.record(EventType.RESOLVED_REFERENCES, {"request": sanitize_user_request(command), "context": _bounded_context(agent_runtime.context)}, iid)
        plan = _timed(execution_meta,"planning_ms",lambda: create_task_plan(command, agent_runtime.context))
        if plan and plan.actions:
            completeness=assess_plan_completeness(command,plan)
            runtime_log.info("Local planner result: tools=%r clauses=%r represented=%r",[action.tool for action in plan.actions],completeness.get("clauses"),completeness.get("represented_clause_count"))
            runtime_log.info("Completeness decision: complete=%s reason=%s",completeness["complete"],completeness.get("reason"))
            if not completeness["complete"]:
                local_fidelity=validate_goal_coverage(command,plan.actions)
                recorder.record(EventType.PLAN_CREATED,{"route_type":"task_plan","route_source":"deterministic_planner","complete":False,"reason":"incomplete_local_plan","clauses":completeness["clauses"],"actions":[{"tool":a.tool,"arguments":_safe_action_arguments(a)} for a in plan.actions]},iid)
                if "app_identification_uncertain" in local_fidelity:
                    recorder.record(EventType.TASK_BLOCKED,{"reason":"app_identification_uncertain"},iid);execution_meta["success"]=False
                    execution_meta["block_reason"]="app_identification_uncertain";execution_meta["block_errors"]=["app_identification_uncertain"]
                    runtime_log.info("Cloud planner invoked: false; app identification is uncertain")
                    return "I couldn't confidently identify the music application. Please say its exact name."
                cloud_goal=original_user_text or command
                runtime_log.info("Cloud planner invoked: true; full_goal=%r",_command_for_log(cloud_goal))
                execution_meta["route_type"]="plan";execution_meta["route_source"]="cloud_planner";execution_meta["model_calls"]+=1
                recorder.record(EventType.REASONING_REQUEST,{"operation":"cloud_planner","model_calls":1,"request":sanitize_user_request(cloud_goal),"fallback_reason":"incomplete_local_plan"},iid)
                proposed=_timed(execution_meta,"cloud_ms",lambda: create_plan(cloud_goal));safe_proposal=privacy_safe_event("PLAN_CREATED",{"actions":proposed or []})["actions"]
                recorder.record(EventType.REASONING_RESPONSE,{"operation":"cloud_planner","actions":safe_proposal},iid)
                cloud_actions,validation_errors=validate_generated_actions(proposed)
                coverage_errors=validate_goal_coverage(command,cloud_actions) if cloud_actions else ["missing_complete_cloud_plan"]
                errors=validation_errors+coverage_errors
                runtime_log.info("Final cloud plan validation: valid=%s tools=%r errors=%r",not errors,[action.tool for action in cloud_actions],errors)
                if errors:
                    recorder.record(EventType.TASK_BLOCKED,{"reason":"cloud_plan_incomplete_or_invalid","errors":errors},iid);execution_meta["success"]=False
                    execution_meta["block_reason"]="cloud_plan_incomplete_or_invalid";execution_meta["block_errors"]=list(errors)
                    return "I identified the application request, but I don't have a complete validated UI plan to play that item, so I didn't open anything."
                cloud_plan=Plan(command,cloud_actions,context={"planning_trace":{"segments":completeness["clauses"]},"represented_clause_count":len(completeness["clauses"]),"fallback_from":"incomplete_local_plan"})
                recorder.record(EventType.PLAN_CREATED,{"route_type":"task_plan","route_source":"cloud_planner","goal":cloud_plan.safe_goal(),"actions":[{"tool":a.tool,"arguments":_safe_action_arguments(a)} for a in cloud_actions]},iid)
                results=_timed(execution_meta,"tool_execution_ms",lambda: _execute_recorded_plan(recorder,iid,cloud_plan,agent_runtime,cancellation_token,execution_meta,speculative_ledger))
                execution_meta["executed_actions"],execution_meta["executed_results"]=_paired_execution(cloud_actions,results)
                execution_meta["success"]=len(results)==len(cloud_actions) and all(result.success for result in results);execution_meta["verified"]=_results_verified(cloud_actions,results);execution_meta["model_calls"]+=_result_model_calls(results)
                return format_results(results)
            execution_meta["private_response"]=_has_private_response(plan.actions)
            reference_debug=os.getenv("DEBUG_REFERENCE_RESOLUTION","false").lower() in {"1","true","yes","on"}
            if reference_debug:runtime_log.info("SessionContext: last_user_message=%r last_assistant_response=%r last_spoken_response=%r", agent_runtime.context.last_user_message, agent_runtime.context.last_assistant_response, agent_runtime.context.last_spoken_response)
            else:runtime_log.info("SessionContext availability: last_user=%s last_assistant=%s last_spoken=%s",bool(agent_runtime.context.last_user_message),bool(agent_runtime.context.last_assistant_response),bool(agent_runtime.context.last_spoken_response))
            trace=plan.context.get("planning_trace",{})
            if reference_debug:
                runtime_log.info("Planner trace: segmented_clauses=%r type_payloads=%r",trace.get("segments"),trace.get("type_text"))
                runtime_log.info("Final structured plan: %r",[(action.tool,_safe_action_arguments(action)) for action in plan.actions])
            else:runtime_log.info("Planner structure: segments=%d tools=%r",len(trace.get("segments") or []),[action.tool for action in plan.actions])
            if plan.context.get("planning_trace"):
                recorder.record(EventType.RESOLVED_REFERENCES, {"planning_trace": _safe_reference_payload(plan.context["planning_trace"])}, iid)
            if plan.context.get("resolved_references"):
                recorder.record(EventType.RESOLVED_REFERENCES, _safe_reference_payload(plan.context["resolved_references"]), iid)
            recorder.record(EventType.PLAN_CREATED, {"route_type":"task_plan","route_source":"deterministic_planner","goal": plan.safe_goal(), "actions": [{"tool": a.tool, "arguments": _safe_action_arguments(a)} for a in plan.actions]}, iid)
            results = _timed(execution_meta,"tool_execution_ms",lambda: _execute_recorded_plan(recorder,iid,plan,agent_runtime,cancellation_token,execution_meta,speculative_ledger))
            execution_meta["executed_actions"],execution_meta["executed_results"]=_paired_execution(plan.actions,results)
            execution_meta["success"]=len(results)==len(plan.actions) and all(result.success for result in results)
            execution_meta["verified"]=_results_verified(plan.actions,results)
            execution_meta["model_calls"]+=_result_model_calls(results)
            return format_results(results)
        missing_completeness=assess_plan_completeness(command,plan)
        if len(missing_completeness.get("clauses") or [])>1:
            runtime_log.info("Local planner result: no complete plan; clauses=%r",missing_completeness["clauses"])
            fidelity=validate_goal_coverage(command,[])
            if "app_identification_uncertain" in fidelity:
                recorder.record(EventType.TASK_BLOCKED,{"reason":"app_identification_uncertain"},iid);execution_meta["success"]=False
                execution_meta["block_reason"]="app_identification_uncertain";execution_meta["block_errors"]=["app_identification_uncertain"]
                runtime_log.info("Completeness decision: incomplete; cloud planner invoked: false; app identification uncertain")
                return "I couldn't confidently identify the music application. Please say its exact name."
            cloud_goal=original_user_text or command;runtime_log.info("Completeness decision: incomplete; cloud planner invoked: true; full_goal=%r",_command_for_log(cloud_goal))
            execution_meta["route_type"]="plan";execution_meta["route_source"]="cloud_planner";execution_meta["model_calls"]+=1
            proposed=_timed(execution_meta,"cloud_ms",lambda: create_plan(cloud_goal));cloud_actions,validation_errors=validate_generated_actions(proposed)
            coverage_errors=validate_goal_coverage(command,cloud_actions) if cloud_actions else ["missing_complete_cloud_plan"];errors=validation_errors+coverage_errors
            runtime_log.info("Final cloud plan validation: valid=%s tools=%r errors=%r",not errors,[action.tool for action in cloud_actions],errors)
            if errors:
                recorder.record(EventType.TASK_BLOCKED,{"reason":"cloud_plan_incomplete_or_invalid","errors":errors},iid);execution_meta["success"]=False
                execution_meta["block_reason"]="cloud_plan_incomplete_or_invalid";execution_meta["block_errors"]=list(errors)
                return "I couldn't build a complete validated plan for every part of that request, so I didn't execute anything."
            cloud_plan=Plan(command,cloud_actions,context={"planning_trace":{"segments":missing_completeness["clauses"]},"represented_clause_count":len(missing_completeness["clauses"]),"fallback_from":"missing_local_plan"})
            recorder.record(EventType.PLAN_CREATED,{"route_type":"task_plan","route_source":"cloud_planner","goal":cloud_plan.safe_goal(),"actions":[{"tool":a.tool,"arguments":_safe_action_arguments(a)} for a in cloud_actions]},iid)
            results=_timed(execution_meta,"tool_execution_ms",lambda: _execute_recorded_plan(recorder,iid,cloud_plan,agent_runtime,cancellation_token,execution_meta,speculative_ledger))
            execution_meta["executed_actions"],execution_meta["executed_results"]=_paired_execution(cloud_actions,results)
            execution_meta["success"]=len(results)==len(cloud_actions) and all(result.success for result in results);execution_meta["verified"]=_results_verified(cloud_actions,results);execution_meta["model_calls"]+=_result_model_calls(results)
            return format_results(results)
        if plan and plan.context.get("clarification"):
            execution_meta["success"]=False;execution_meta["verified"]=False
            execution_meta["block_reason"]="clarification_required";execution_meta["block_errors"]=["clarification_required"]
            recorder.record(EventType.TASK_BLOCKED,{"reason":"clarification_required"},iid)
            return plan.context["clarification"]
        if _agent_escalation_available():
            # The deterministic planner genuinely could not represent this
            # request. That is exactly the "local understanding is
            # insufficient" condition the agent runtime exists for --
            # rather than telling the user no, hand it to the agent.
            runtime_log.info("Escalating to agent runtime: no complete local plan")
            return _run_agent_with_loop(original_user_text or command, recorder, iid, execution_meta, cancellation_token)
        execution_meta["block_reason"]="missing_local_plan";execution_meta["block_errors"]=["missing_local_plan"]
        return "I couldn't create a safe local plan for that task."

    if route is None:
        route = route_command(command)
    execution_meta["route_type"] = route["type"]
    route_source=route.get("route_source") or ("web_answer" if route["type"]=="question" else "cloud_planner" if route["type"]=="plan" else "cloud_intent_router" if route["type"] in {"tools","ai"} else "deterministic_router")
    execution_meta["route_source"]=route_source
    execution_meta["model_calls"]=int(route.get("model_calls",0))
    route_arguments=route.get("arguments")
    if route.get("tool") and isinstance(route_arguments,dict):
        route_arguments=_safe_action_arguments(Action(route["tool"],route_arguments))
    recorder.record(EventType.PLAN_CREATED, {"route_type": route["type"],"route_source":route_source,"model":route.get("model"),"model_calls":route.get("model_calls",0),"input_tokens":route.get("input_tokens",0),"output_tokens":route.get("output_tokens",0),"fallback_from":route.get("fallback_from",[]),"fallback_reason":route.get("fallback_reason"), "tool": route.get("tool"), "arguments": route_arguments}, iid)
    if route_source=="cloud_intent_router":
        recorder.record(EventType.REASONING_REQUEST,{"operation":"cloud_intent_router","model":route.get("model","gpt-5-mini"),"model_calls":route.get("model_calls",1),"input_tokens":route.get("input_tokens",0),"output_tokens":route.get("output_tokens",0),"fallback_reason":route.get("fallback_reason")},iid)

    # -------------------------
    # Escalation to the agent runtime
    # -------------------------
    # `plan`, `ai` and `coding_task` are precisely the routes that mean
    # "the deterministic local layer could not resolve this". They are the
    # right place -- and the only place -- to hand a request to the real
    # agent loop. Every deterministic route above is untouched, so simple
    # commands stay fast and never involve a model.
    #
    # `_agent_escalation_available()` is False whenever no provider is
    # configured, so with no API key this whole block is inert and the
    # pre-existing single-shot planner path below runs exactly as before.
    if route["type"] in {"plan", "ai", "coding_task"} and _agent_escalation_available():
        goal = route.get("task") or route.get("message") or original_user_text or command
        runtime_log.info("Escalating to agent runtime from route_type=%s", route["type"])
        return _run_agent_with_loop(goal, recorder, iid, execution_meta, cancellation_token)

    if route["type"] == "remember":
        # Explicit "remember that ..." -- a local, deterministic write to
        # long-term memory. No model call, and it works with no API key.
        from memory.agent_memory import get_agent_memory

        action=Action("remember_fact",{"text":route["text"]})
        prepared=_record_prepared_actions(recorder,iid,[action],agent_runtime.context)
        try:
            record=get_agent_memory().remember_explicitly(route["text"])
            result=ToolResult(True,action.tool,"I'll remember that, sir.",{"verified":True,"memory_id":record.memory_id,"kind":record.kind})
        except Exception as exc:
            result=ToolResult(False,action.tool,"I couldn't store that, sir.",{"verified":True},f"{type(exc).__name__}: {exc}")
        _record_action_results(recorder,iid,[action],[result],prepared,agent_runtime.context)
        execution_meta["success"]=result.success;execution_meta["verified"]=result.success
        execution_meta["executed_actions"]=[action];execution_meta["executed_results"]=[result]
        return result.message

    if route["type"] == "agent_task":
        # Explicitly requested agent run. Without a configured provider
        # this reports that honestly instead of silently doing nothing.
        return _run_agent_with_loop(route.get("goal") or command, recorder, iid, execution_meta, cancellation_token)

    if route["type"] == "cancel_read_only_task":
        action=Action("cancel_active_task",{})
        prepared=_record_prepared_actions(recorder,iid,[action],agent_runtime.context)
        # "Cancel" must reach BOTH task systems: the pre-existing read-only
        # scheduler and the agent task manager. Cancelling only one of them
        # would leave a long agent run going after the user said stop.
        cancelled = cancel_read_only_tasks() + _cancel_agent_tasks()
        if cancelled:
            agent_runtime.context.pending_messaging_recipient=None
            agent_runtime.context.pending_messaging_message=None
        result=ToolResult(bool(cancelled),"cancel_active_task","Cancelled." if cancelled else "Nothing cancellable.",{"verified":True})
        _record_action_results(recorder,iid,[action],[result],prepared,agent_runtime.context)
        execution_meta["success"]=True;execution_meta["verified"]=True
        execution_meta["executed_actions"]=[action];execution_meta["executed_results"]=[result]
        return "Cancelled." if cancelled else "There's nothing to cancel."

    if route["type"] == "task_status":
        action=Action("inspect_task_status",{})
        prepared=_record_prepared_actions(recorder,iid,[action],agent_runtime.context)
        summary=active_interactive_task_summary()
        # Agent tasks live in the newer tasks/ manager, which the older
        # read-only scheduler above knows nothing about. "What are you
        # doing?" must account for both, or a long agent run looks like
        # nothing is happening.
        agent_tasks=_active_agent_tasks()
        summary["agent_tasks"]=[task.to_dict(include_observations=False) for task in agent_tasks]
        summary["total"]+=len(agent_tasks)
        count=summary["total"]
        if agent_tasks:
            goals="; ".join(task.goal for task in agent_tasks[:2])
            response=f"I'm working on: {goals}." if count==len(agent_tasks) else f"{count} tasks are running, including: {goals}."
        else:
            response="No interactive tasks are running." if count==0 else f"{count} interactive task{' is' if count==1 else 's are'} running."
        result=ToolResult(True,action.tool,response,{**summary,"verified":True})
        _record_action_results(recorder,iid,[action],[result],prepared,agent_runtime.context)
        execution_meta["success"]=True;execution_meta["verified"]=True
        execution_meta["executed_actions"]=[action];execution_meta["executed_results"]=[result]
        return response

    if route["type"] == "resume_interrupted_response":
        buffered=agent_runtime.context.interrupted_response
        if not buffered:return "There's nothing to continue."
        action=Action("resume_buffered_response",{})
        prepared=_record_prepared_actions(recorder,iid,[action],agent_runtime.context)
        agent_runtime.context.speech_interrupted=False
        result=ToolResult(True,"resume_buffered_response","Resumed.",{"verified":True})
        _record_action_results(recorder,iid,[action],[result],prepared,agent_runtime.context)
        execution_meta["success"]=True;execution_meta["verified"]=True
        execution_meta["executed_actions"]=[action];execution_meta["executed_results"]=[result]
        return buffered

    if route["type"] == "correct_interrupted_response":
        buffered=agent_runtime.context.interrupted_response or agent_runtime.context.last_spoken_response
        if not buffered:return "There's no interrupted answer to change."
        instruction=route.get("instruction","").lower();count_match=re.search(r"(?:top\s+)?(\d+)",instruction)
        sentences=[part.strip() for part in re.split(r"(?<=[.!?])\s+",buffered) if part.strip()]
        if count_match:shortened=" ".join(sentences[:max(1,int(count_match.group(1)))])
        else:shortened=sentences[0] if sentences else buffered[:160].rsplit(" ",1)[0]
        action=Action("correct_buffered_response",{"instruction":instruction})
        prepared=_record_prepared_actions(recorder,iid,[action],agent_runtime.context)
        agent_runtime.context.interrupted_response=shortened
        agent_runtime.context.speech_interrupted=False
        recorder.correction(iid,"buffered_response",instruction,successful=True)
        result=ToolResult(True,"correct_buffered_response","Corrected.",{"verified":True,"output_length":len(shortened)})
        _record_action_results(recorder,iid,[action],[result],prepared,agent_runtime.context)
        execution_meta["success"]=True;execution_meta["verified"]=True
        execution_meta["executed_actions"]=[action];execution_meta["executed_results"]=[result]
        return shortened

    if route["type"] == "revise_whatsapp_recipient":
        context=agent_runtime.context;message=context.pending_messaging_message
        if not message:return "There's no pending WhatsApp message to change."
        original_recipient=context.pending_messaging_recipient
        cancel_read_only_tasks();wait_for_interactive_tasks()
        if context.messaging_committed:return "The previous WhatsApp message was already sent, so I didn't send another one."
        plan=Plan(command,[Action("send_whatsapp_message",{"recipient":route["recipient"],"message":message},risk=ActionRisk.CAUTION,max_attempts=1,sensitive_fields={"message"})])
        results=_timed(execution_meta,"tool_execution_ms",lambda: _execute_recorded_plan(recorder,iid,plan,agent_runtime,execution_meta=execution_meta))
        execution_meta["executed_actions"],execution_meta["executed_results"]=_paired_execution(plan.actions,results)
        successful=bool(results and results[0].success);recorder.correction(iid,{"recipient":original_recipient},{"recipient":route["recipient"]},successful=successful)
        execution_meta["success"]=successful;execution_meta["verified"]=_results_verified(plan.actions,results);execution_meta["model_calls"]+=_result_model_calls(results)
        return format_results(results)

    if route["type"] == "question":
        service = get_web_answer_service()
        execution_meta["model_calls"]=1
        web_action=Action("web_search",{"query":route["message"]});web_prepared=_record_prepared_actions(recorder,iid,[web_action],agent_runtime.context)
        execution_meta["executed_actions"]=[web_action]
        recorder.record(EventType.REASONING_REQUEST, {"provider":"openai","model":service.model,"operation":"read_only_web_answer","model_calls":1,"query":route["message"]}, iid)
        handle = get_read_only_task_scheduler().submit(
            lambda cancellation: service.answer(route["message"], cancellation),
            priority=ReadOnlyPriority.INTERACTIVE,
            interrupt_lower=True,
        )
        try:
            result = handle.result(timeout=float(service.timeout) + 2)
        except (CancelledError,ReadOnlyTaskCancelled):
            execution_meta["success"]=False
            cancelled_result=ToolResult(False,"web_search","Cancelled.",error="cancelled")
            execution_meta["executed_results"]=[cancelled_result]
            _record_action_results(recorder,iid,[web_action],[cancelled_result],web_prepared,agent_runtime.context)
            return "I stopped that question."
        except FutureTimeoutError:
            execution_meta["success"]=False
            handle.cancel()
            timeout_result=ToolResult(False,"web_search","Timed out.",error="scheduler_timeout")
            execution_meta["executed_results"]=[timeout_result]
            _record_action_results(recorder,iid,[web_action],[timeout_result],web_prepared,agent_runtime.context)
            return "I couldn't get a reliable answer from the web right now."
        web_result=ToolResult(result.success,"web_search","Web answer completed.",{"model":result.model,"model_calls":0 if result.cache_hit else 1,"input_tokens":result.input_tokens,"output_tokens":result.output_tokens,"web_request_ms":result.web_request_ms,"answer_processing_ms":result.answer_processing_ms,"source_count":len(result.sources or []),"cache_hit":result.cache_hit,"verified":False},result.error)
        execution_meta["executed_results"]=[web_result]
        _record_action_results(recorder,iid,[web_action],[web_result],web_prepared,agent_runtime.context)
        recorder.record(EventType.REASONING_RESPONSE,{"operation":"read_only_web_answer","model":result.model,"model_calls":0 if result.cache_hit else 1,"input_tokens":result.input_tokens,"output_tokens":result.output_tokens,"cache_hit":result.cache_hit,"success":result.success},iid)
        execution_meta["success"]=result.success;execution_meta["verified"]=False;execution_meta["model_calls"]=0 if result.cache_hit else 1
        return result.answer

    # -------------------------
    # Single local tool
    # -------------------------

    if route["type"] == "tool":
        action = Action(
            tool=route["tool"],
            args=route["arguments"]
        )
        execution_meta["private_response"]=_has_private_response([action])

        prepared=_record_prepared_actions(recorder,iid,[action],agent_runtime.context)
        if cancellation_token is not None and cancellation_token.cancelled:
            result=ToolResult(False,action.tool,"Task cancelled.",error="cancelled")
        else:
            if action.tool in SESSION_AWARE_SINGLE_TOOLS:
                result=agent_runtime.execute(Plan(command,[action]),cancellation_token=cancellation_token)[0] if cancellation_token is not None else agent_runtime.execute(Plan(command,[action]))[0]
            else:result = executor.execute_action(action)
            if cancellation_token is not None and cancellation_token.cancelled:
                result=ToolResult(False,action.tool,"Task cancelled; the late result was discarded.",error="cancelled")
        _record_action_results(recorder,iid,[action],[result],prepared,agent_runtime.context)
        execution_meta["success"]=result.success;execution_meta["verified"]=_results_verified([action],[result]);execution_meta["model_calls"]+=_result_model_calls([result])
        execution_meta["executed_actions"]=[action];execution_meta["executed_results"]=[result]

        if result.success:
            _remember_action(action, result)
            return result.message

        return (
            f"{result.message}\n"
            f"Error: {result.error}"
        )

    # -------------------------
    # Local multi-step plan
    # -------------------------

    if route["type"] == "local_plan":
        execution_meta["private_response"]=_has_private_response(route["actions"])
        results = _timed(execution_meta,"tool_execution_ms",lambda: _execute_recorded_plan(recorder,iid,Plan(command,route["actions"]),agent_runtime,cancellation_token,execution_meta,speculative_ledger))
        execution_meta["executed_actions"],execution_meta["executed_results"]=_paired_execution(route["actions"],results)
        execution_meta["success"]=len(results)==len(route["actions"]) and all(result.success for result in results);execution_meta["verified"]=_results_verified(route["actions"],results);execution_meta["model_calls"]+=_result_model_calls(results)

        return format_results(
            results
        )

    # -------------------------
    # Multi-step AI plan
    # -------------------------

    if route["type"] == "plan":
        execution_meta["model_calls"]+=1
        recorder.record(EventType.REASONING_REQUEST,{"operation":"cloud_planner","model_calls":1,"request":sanitize_user_request(route["message"])},iid)
        planned_actions = _timed(execution_meta,"cloud_ms",lambda: create_plan(
            route["message"]
        ))
        safe_proposal=privacy_safe_event("PLAN_CREATED",{"actions":planned_actions or []})["actions"]
        recorder.record(EventType.REASONING_RESPONSE,{"operation":"cloud_planner","actions":safe_proposal},iid)

        if not planned_actions:
            return "I couldn't create a plan for that."

        actions,validation_errors=validate_generated_actions(planned_actions)
        validation_errors+=validate_goal_coverage(route["message"],actions) if actions else ["missing_complete_cloud_plan"]
        if validation_errors:
            recorder.record(EventType.TASK_BLOCKED,{"reason":"generated_plan_validation_failed","errors":validation_errors},iid);execution_meta["success"]=False
            execution_meta["block_reason"]="generated_plan_validation_failed";execution_meta["block_errors"]=list(validation_errors)
            return "I rejected an unsafe or malformed generated plan."

        execution_meta["private_response"]=_has_private_response(actions)

        results = _timed(execution_meta,"tool_execution_ms",lambda: _execute_recorded_plan(recorder,iid,Plan(command,actions),agent_runtime,cancellation_token,execution_meta,speculative_ledger))
        execution_meta["executed_actions"],execution_meta["executed_results"]=_paired_execution(actions,results)
        execution_meta["success"]=len(results)==len(actions) and all(result.success for result in results);execution_meta["verified"]=_results_verified(actions,results);execution_meta["model_calls"]+=_result_model_calls(results)

        return format_results(
            results
        )

    # -------------------------
    # Intent router tools
    # -------------------------

    if route["type"] == "tools":
        actions,validation_errors=validate_generated_actions(route["actions"])
        validation_errors+=validate_goal_coverage(command,actions) if actions else ["missing_complete_cloud_plan"]
        if validation_errors:
            recorder.record(EventType.TASK_BLOCKED,{"reason":"generated_route_validation_failed","errors":validation_errors},iid);execution_meta["success"]=False
            execution_meta["block_reason"]="generated_route_validation_failed";execution_meta["block_errors"]=list(validation_errors)
            return "I rejected an unsafe or malformed generated action."

        execution_meta["private_response"]=_has_private_response(actions)

        results = _timed(execution_meta,"tool_execution_ms",lambda: _execute_recorded_plan(recorder,iid,Plan(command,actions),agent_runtime,cancellation_token,execution_meta,speculative_ledger))
        execution_meta["executed_actions"],execution_meta["executed_results"]=_paired_execution(actions,results)
        execution_meta["success"]=len(results)==len(actions) and all(result.success for result in results);execution_meta["verified"]=_results_verified(actions,results);execution_meta["model_calls"]+=_result_model_calls(results)

        return format_results(
            results
        )

    # -------------------------
    # Normal AI question
    # -------------------------

    if route["type"] == "ai":
        recorder.record(EventType.REASONING_REQUEST, {"provider": "openai", "model": "gpt-5-mini", "configuration": {}, "input": sanitize_user_request(route["message"])}, iid)
        answer = ask_ai(route["message"])
        execution_meta["model_calls"]+=1
        recorder.record(EventType.REASONING_RESPONSE, {"provider": "openai", "model": "gpt-5-mini", "response": answer}, iid)
        execution_meta["success"]=True;execution_meta["verified"]=False
        return answer

    return "I couldn't understand what to do."


def _observe_for_improvement(command,route,outcome,original_user_text,interaction_id,execution_meta,request_started) -> None:
    """Feed real execution evidence to the self-improvement observer.

    This must never affect the user-visible outcome of a request: any
    failure here is logged and swallowed, never raised. It runs after the
    response/execution_outcome are already finalized, using the exact same
    `outcome` dict handed to the response formatter -- never route/plan
    intent -- so a stale route can no longer contaminate improvement
    evidence than it can contaminate the spoken response.
    """
    try:
        from brain.improvement_observer import observe
        observe(
            command=command,
            route=route,
            execution_outcome=outcome,
            original_user_text=original_user_text,
            interaction_id=interaction_id,
            duration_ms=(time.perf_counter()-request_started)*1000,
            resource_wait_ms=_resource_wait_ms(execution_meta),
        )
    except Exception:
        runtime_log.exception("Improvement observation failed; continuing without it")


def _resource_wait_ms(execution_meta: dict) -> float:
    results=execution_meta.get("executed_results") or []
    return next((float(_result_data(r).get("plan_resource_wait_ms",0.0) or 0.0) for r in results),0.0)


def _log_request_performance(execution_meta: dict, request_started: float, failed: bool) -> None:
    """One consolidated per-request timing line -- not one line per stage,
    to avoid log spam. Fields follow the stage names requested for latency
    profiling; a stage not exercised by this request's route stays at 0.0
    rather than being fabricated."""
    total_ms=(time.perf_counter()-request_started)*1000
    resource_wait_ms=_resource_wait_ms(execution_meta)
    runtime_log.info(
        "Request performance: route_type=%s route_source=%s planning_ms=%.1f cloud_ms=%.1f tool_execution_ms=%.1f resource_wait_ms=%.1f model_calls=%s total_request_ms=%.1f success=%s",
        execution_meta.get("route_type"),execution_meta.get("route_source"),
        execution_meta.get("planning_ms",0.0),execution_meta.get("cloud_ms",0.0),
        execution_meta.get("tool_execution_ms",0.0),resource_wait_ms,
        execution_meta.get("model_calls",0),total_ms,not failed,
    )


def _bounded_context(context):
    return {"active_app":context.active_app,"last_opened_app":context.last_opened_app,"browser_active":context.browser_active,"has_last_assistant_response":bool(context.last_assistant_response),"has_last_spoken_response":bool(context.last_spoken_response),"has_pending_message":bool(context.pending_messaging_message)}


def _record_prepared_actions(recorder,iid,actions,context):
    prepared=[]
    before=_bounded_context(context)
    for index,action in enumerate(actions):
        started_at=datetime.now(timezone.utc).isoformat();item={"action_id":uuid.uuid4().hex,"index":index,"started_perf":time.perf_counter(),"started_at":started_at};prepared.append(item)
        recorder.record(EventType.TOOL_CALL,{"action_id":item["action_id"],"status":"prepared","name":action.tool,"arguments":_safe_action_arguments(action),"before_state":before,"prepared_at":started_at,"execution":{"started_at":started_at}},iid)
    return prepared


def _execute_recorded_plan(recorder,iid,plan,runtime,cancellation_token=None,execution_meta=None,speculative_ledger=None):
    if speculative_ledger is not None:
        # Part C: a leading step this plan is about to (re-)execute may
        # already have been completed early from a stable, safe partial
        # transcript (see brain/speculative_execution.py) -- never run the
        # same open/volume/mute action twice. Only ever trims from the
        # front and only actions with no unmet dependency wiring, so
        # execution order for genuinely dependent steps is untouched.
        from brain.speculative_execution import reconcile_local_plan_actions
        trimmed=reconcile_local_plan_actions(speculative_ledger,plan.actions)
        if plan.actions and not trimmed:
            # The whole plan (all of it) was already completed early --
            # report that real evidence instead of silently claiming
            # nothing happened. `plan.actions` is deliberately left
            # untouched so `_paired_execution` pairs this synthetic result
            # against the real action it corresponds to.
            recorder.record(EventType.TASK_COMPLETED,{"reason":"already_completed_by_speculative_partial_action"},iid)
            return [ToolResult(True,plan.actions[0].tool,"Already completed from an earlier action.",{"verified":True})]
        plan.actions=trimmed
    errors=validate_plan_preflight(plan,runtime.context)
    runtime_log.info("Complete plan before execution: %r",[(action.tool,_safe_action_arguments(action)) for action in plan.actions])
    for index,action in enumerate(plan.actions):runtime_log.info("Tool registry lookup: action=%d tool=%s registered=%s",index,action.tool,action.tool in RUNTIME_TOOLS)
    runtime_log.info("Plan validation result: valid=%s errors=%r context_id=%s",not errors,errors,id(runtime.context))
    if errors:
        recorder.record(EventType.TASK_BLOCKED,{"reason":"plan_preflight_failed","errors":errors},iid)
        if execution_meta is not None:execution_meta["block_reason"]="plan_preflight_failed";execution_meta["block_errors"]=list(errors)
        return [ToolResult(False,"plan_preflight","I couldn't safely validate the complete plan, so I didn't execute any part of it.",{"verified":True,"validation_errors":errors},"plan_preflight_failed")]
    prepared={};terminal=set()
    def observe(stage,index,action,result,context):
        if stage=="prepared":prepared[index]=_record_prepared_actions(recorder,iid,[action],context)[0]
        else:
            runtime_log.info("ToolResult: action=%d tool=%s success=%s error=%r metadata=%r",index,action.tool,result.success,result.error,_safe_result_metadata(action,result))
            _record_action_results(recorder,iid,[action],[result],[prepared[index]],context);terminal.add(index)
    results=[]
    try:
        results=runtime.execute(plan,cancellation_token=cancellation_token,action_observer=observe)
    except Exception as exc:
        for index,action in enumerate(plan.actions):
            item=[prepared[index]] if index in prepared else _record_prepared_actions(recorder,iid,[action],runtime.context)
            if index not in terminal:
                result=ToolResult(False,action.tool,"Action terminated by a runtime exception.",error=f"{type(exc).__name__}: {exc}") if index in prepared else None
                _record_action_results(recorder,iid,[action],[result] if result else [],item,runtime.context)
        raise
    for index,action in enumerate(plan.actions):
        if index in terminal:continue
        item=[prepared[index]] if index in prepared else _record_prepared_actions(recorder,iid,[action],runtime.context)
        if index in prepared and index<len(results):_record_action_results(recorder,iid,[action],[results[index]],item,runtime.context)
        else:_record_action_results(recorder,iid,[action],[],item,runtime.context)
    return results


def _record_action_results(recorder,iid,actions,results,prepared,context):
    after=_bounded_context(context)
    completed=len(results)
    for action,result,item in zip(actions,results,prepared):
        data=_result_data(result)
        recorder.record(EventType.TOOL_RESULT,{"action_id":item["action_id"],"status":"committed" if result.success else "failed","name":action.tool,"success":result.success,"error":getattr(result,"error",None),"message":_safe_result_message(action,result),"metadata":_safe_result_metadata(action,result),"execution":{"started_at":item["started_at"],"finished_at":datetime.now(timezone.utc).isoformat(),"latency_ms":round((time.perf_counter()-item["started_perf"])*1000,3)},"after_state":after,"automatically_verified":bool(data.get("verified"))},iid)
    for action,item in zip(actions[completed:],prepared[completed:]):
        recorder.record(EventType.TOOL_RESULT,{"action_id":item["action_id"],"status":"not_executed","name":action.tool,"success":False,"error":"prior_action_failed","message":"Action was not executed because the plan stopped earlier.","metadata":{"verified":False},"after_state":after,"automatically_verified":False},iid)


def _results_verified(actions,results):
    return bool(results) and len(results)==len(actions) and all(result.success and bool(_result_data(result).get("verified")) for result in results)


def _timed(execution_meta: dict, key: str, fn):
    """Time `fn()` and accumulate the elapsed milliseconds into
    execution_meta[key], forwarding fn's return value unchanged. A thin
    wrapper rather than inlining perf_counter() at each of the several call
    sites that need this -- fewer places for a copy-paste timing bug to hide."""
    started=time.perf_counter()
    result=fn()
    execution_meta[key]=execution_meta.get(key,0.0)+(time.perf_counter()-started)*1000
    return result


def _paired_execution(planned_actions, results):
    """Pair each real ToolResult with the Action that actually produced it.

    `_execute_recorded_plan` returns one result per action it actually
    attempted, in order, and stops appending once execution halts (a failed
    step, a cancellation, or a preflight block). It never returns more
    results than actions were attempted, so zipping the head of
    `planned_actions` against `results` is always the correct pairing.
    The single synthetic `plan_preflight` result is a special case: it means
    the plan was rejected before anything real ran, so it must not be
    reported as executed evidence at all.
    """
    if len(results) == 1 and getattr(results[0], "tool", None) == "plan_preflight":
        return [], []
    return list(planned_actions[:len(results)]), list(results)


def _build_execution_outcome(execution_meta, overall_success):
    """Summarize what actually ran, for callers that must describe evidence
    rather than a plan/route that may never have executed.

    `executed` is only True when at least one real ToolResult exists.
    `partial` marks a compound plan where some but not all actions
    succeeded. `success`/`verified` mirror the already-correct all-or-nothing
    flags in `execution_meta` -- this doesn't change what counts as success,
    it only exposes real per-action evidence to describe it with.
    """
    actions = execution_meta.get("executed_actions") or []
    results = execution_meta.get("executed_results") or []
    executed = bool(results)
    described = []
    for action, result in zip(actions, results):
        data = _result_data(result)
        error = getattr(result, "error", None)
        message = _safe_result_message(action, result)
        described.append({
            "tool": getattr(action, "tool", None),
            "arguments": action.safe_args(),
            "success": bool(getattr(result, "success", False)),
            "verified": bool(data.get("verified", False)),
            "error": sanitize_text(str(error))[:300] if error else None,
            "message": sanitize_text(str(message))[:300] if message else None,
        })
    any_success = any(item["success"] for item in described)
    full_success = bool(overall_success)
    return {
        "executed": executed,
        "route_type": execution_meta.get("route_type"),
        "route_source": execution_meta.get("route_source"),
        "success": full_success,
        "verified": bool(execution_meta.get("verified", False)) and full_success,
        "partial": executed and any_success and not full_success,
        "action_count": len(actions),
        "actions": described,
        "block_reason": execution_meta.get("block_reason"),
        "block_errors": list(execution_meta.get("block_errors") or []),
        "planning_ms": execution_meta.get("planning_ms", 0.0),
        "cloud_ms": execution_meta.get("cloud_ms", 0.0),
        "tool_execution_ms": execution_meta.get("tool_execution_ms", 0.0),
        "model_calls": execution_meta.get("model_calls", 0),
    }


def _result_model_calls(results):
    total=0
    for result in results:
        data=_result_data(result);total+=int(data.get("model_calls",0) or 0)
        if data.get("translated") and data.get("translation_model"):total+=1
    return total


def _has_private_response(actions):
    return any(action.tool in PRIVATE_RESPONSE_TOOLS for action in actions)


def _safe_result_metadata(action,result):
    data=_result_data(result)
    omitted={"contents","content","text","outgoing_message","password","items","matches","controls","answer"}
    safe={key:value for key,value in data.items() if key not in omitted}
    if action.tool=="send_whatsapp_message":
        safe={key:value for key,value in safe.items() if key in {"recipient_status","translated","translation_ms","translation_model","translation_input_tokens","translation_output_tokens","recipient_resolution_ms","app_ready_ms","chat_selection_ms","message_insert_ms","send_ms","total_ms","verified","committed"}}
        safe["message_length"]=len(str(action.args.get("message","")))
    if action.tool=="inspect_window":
        controls=data.get("controls") if isinstance(data.get("controls"),list) else []
        type_counts={}
        for control in controls:
            if not isinstance(control,dict):continue
            kind=str(control.get("control_type") or "unknown");type_counts[kind]=type_counts.get(kind,0)+1
        safe.update({"control_count":len(controls),"named_control_count":sum(bool(item.get("name")) for item in controls if isinstance(item,dict)),"control_type_counts":type_counts})
    return safe


def _safe_action_arguments(action):
    arguments=action.safe_args()
    private_field={"type_text":"text","browser_type":"text","send_whatsapp_message":"message"}.get(action.tool)
    if private_field and private_field in arguments:
        value=str(action.args.get(private_field, ""));arguments[private_field]="<REDACTED>";arguments[f"{private_field}_length"]=len(value)
    if "contents" in arguments and arguments["contents"]!="<REDACTED>":
        value=str(arguments.pop("contents"));arguments["content_metadata"]={"length":len(value),"sha256":hashlib.sha256(value.encode("utf-8")).hexdigest()}
    return arguments


def _safe_reference_payload(value,key=None):
    if key=="segments" and isinstance(value,list):return {"count":len(value)}
    if key in {"reference_resolution_result","resolved_value","type_payload_before_reference_resolution"} and isinstance(value,str):
        reference_like=bool(re.fullmatch(r"(?i)(?:exactly\s+)?(?:what you (?:just |last )?(?:said|told me)|your (?:last|previous) (?:answer|response)|the (?:previous (?:answer|response)|last thing(?: you said)?))",value.strip(" .?!")))
        if not reference_like:return {"available":True,"length":len(value),"sha256":hashlib.sha256(value.encode("utf-8")).hexdigest()}
    if isinstance(value,dict):return {item_key:_safe_reference_payload(item,item_key) for item_key,item in value.items()}
    if isinstance(value,list):return [_safe_reference_payload(item) for item in value]
    return value


def _safe_result_message(action,result):
    if action.tool=="read_text_file":return "Read text file successfully." if result.success else result.message
    if action.tool=="analyze_screen":return "Screen analysis completed." if result.success else result.message
    if action.tool=="inspect_window":return "Inspected window controls successfully." if result.success else result.message
    return result.message


def _result_data(result):
    data=getattr(result,"data",None)
    return data if isinstance(data,dict) else {}


def _remember_action(action: Action, result) -> None:
    try:
        data=result.data if isinstance(getattr(result,"data",None),dict) else {}
        context=agent_runtime.context
        if action.tool == "open_application":
            context.last_opened_app=context.active_app=action.args["app_name"];context.last_pid=data.get("pid");context.last_hwnd=data.get("hwnd")
            memory_manager.remember_entity("application", action.args["app_name"], agent_runtime.session_id, pid=data.get("pid"), hwnd=data.get("hwnd"), opened_by_jarvis=True)
        elif action.tool == "open_task_manager":
            context.last_opened_app=context.active_app="task manager";context.last_pid=data.get("pid");context.last_hwnd=data.get("hwnd")
            if data.get("hwnd"):context.application_windows["task manager"]=int(data["hwnd"])
            memory_manager.remember_entity("application","task manager",agent_runtime.session_id,pid=data.get("pid"),hwnd=data.get("hwnd"),opened_by_jarvis=True)
        elif action.tool == "open_website":
            url = action.args["url"]
            context.browser_active=True;context.current_url=url;context.active_app="browser";context.last_pid=data.get("pid");context.last_hwnd=data.get("hwnd")
            if data.get("hwnd"):context.application_windows["browser"]=int(data["hwnd"])
            memory_manager.remember_entity("website", urlparse(url).netloc or url, agent_runtime.session_id, url=url)
            query = (parse_qs(urlparse(url).query).get("search_query") or parse_qs(urlparse(url).query).get("q") or [None])[0]
            if query:
                provider = "youtube" if "youtube" in url else "google" if "google" in url else None
                memory_manager.remember_entity("search", query, agent_runtime.session_id, provider=provider, url=url)
        elif action.tool == "close_application":
            if context.active_app==action.args["app_name"]:context.active_app=None
            resolution = memory_manager.resolve(action.args["app_name"], agent_runtime.session_id, "application")
            if resolution.status == "resolved": memory_manager.update_entity(resolution.entity["id"], status="closed")
        elif action.tool in {"create_text_file","write_text_file","open_path","rename_path","move_path","copy_path"} and data.get("path"):
            context.last_opened_file=data["path"]
    except Exception:
        # Memory must never turn a successful user action into a failed action.
        pass


def format_results(results) -> str:
    messages = []

    for result in results:
        if result.success:
            messages.append(
                result.message
            )

        else:
            message = result.message

            if result.error:
                message += (
                    f"\nError: {result.error}"
                )

            messages.append(
                message
            )

    return "\n".join(
        messages
    )
