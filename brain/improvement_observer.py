"""Self-improvement observation: decides whether a completed request exposed
something worth improving, classifies it, deduplicates it against prior
occurrences of the same underlying problem, and persists a structured
ImprovementCandidate.

This module is the single integration point between the real execution
pipeline (`brain.agent.run_agent`) and the improvement-candidate store. It
is called exactly once per request, after `execution_outcome` has already
been finalized -- the same dict handed to the response formatter. That is
deliberate: `execution_outcome` is already the place this codebase records
what actually ran (see `brain/agent.py::_build_execution_outcome` and
`_paired_execution`), specifically to prevent a stale route/plan from being
mistaken for real execution. This module trusts that same evidence and adds
nothing else -- it does not re-derive "what happened" from `route` or from
any planned-but-possibly-unexecuted action list.

No cloud/LLM call is made anywhere in this module.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from brain.improvement_classifier import CLASSIFIER_SOURCE, classify
from brain.improvement_models import GapType, ImprovementCandidate, stable_hash
from brain.improvement_store import get_improvement_store
from brain.resource_locks import BROWSER_TOOLS, DESKTOP_INPUT_TOOLS
from training_data.sanitizer import sanitize_action_arguments, sanitize_text, sanitize_user_request

log = logging.getLogger("jarvis.improvement")

_CANCELLATION_CODES = {"cancelled", "cancelled_while_waiting_for_resource"}
_APPROVAL_CODES = {"human_confirmation_required"}
# A short, structured list of tool-reported error *codes* (not exception
# types -- see improvement_classifier for those) that are inherently
# external/transient rather than a JARVIS implementation defect.
_ENVIRONMENTAL_ERROR_CODES = {"chrome_not_found", "scheduler_timeout", "resource_timeout"}

_MESSAGING_TOOLS = {"send_whatsapp_message"}
_WEB_TOOLS = {"web_search"}
_FILESYSTEM_TOOLS = {
    "create_text_file", "write_text_file", "append_text_file", "read_text_file",
    "rename_path", "move_path", "copy_path", "verify_file", "open_path",
    "exists", "list_files", "find_file", "search_text",
}
_APP_LIFECYCLE_TOOLS = {"open_application", "close_application", "wait_for_window", "open_website", "open_folder", "open_file_explorer", "open_task_manager"}
_SYSTEM_TOOLS = {"volume_up", "volume_down", "mute_volume", "take_screenshot", "minimize_window", "maximize_window", "restore_window", "close_window"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _infer_subsystem(tool_names: list[str]) -> str | None:
    """Bucket real executed tool names into a small, fixed set of
    subsystems, reusing the same tool groupings brain/resource_locks.py
    already maintains for lock scoping -- one source of truth for "which
    tools belong to which part of the system", not a second one."""
    tools = {t for t in tool_names if t}
    if tools & BROWSER_TOOLS:
        return "browser"
    if tools & _MESSAGING_TOOLS:
        return "messaging"
    if tools & _WEB_TOOLS:
        return "web_answer"
    if tools & DESKTOP_INPUT_TOOLS:
        return "desktop_ui"
    if tools & _FILESYSTEM_TOOLS:
        return "filesystem"
    if tools & _APP_LIFECYCLE_TOOLS:
        return "app_lifecycle"
    if tools & _SYSTEM_TOOLS:
        return "system"
    if tools:
        return "general"
    return None


# The one tool in the codebase whose own runtime (agent_runtime.py
# ::_browser_action) intentionally allows success=True with verified=False:
# a click that happened but whose resulting page change couldn't be
# confirmed. Every *other* browser tool ties success to verified (success is
# forced False when unverified), so success+unverified never legitimately
# occurs for them -- and for everything outside browser tools (type_text,
# press_key, ...) unverified success is the normal, expected, harmless case
# by design and must never itself create noise.
_SOFT_SUCCESS_VERIFICATION_TOOLS = {"browser_click"}


def _success_verification_mismatch(execution_outcome: dict) -> bool:
    actions = execution_outcome.get("actions") or []
    return any(
        a.get("success") and not a.get("verified") and a.get("tool") in _SOFT_SUCCESS_VERIFICATION_TOOLS
        for a in actions
    )


def _skip_reason(execution_outcome: dict) -> str | None:
    """Decide whether this request is normal/expected noise that must never
    reach the candidate store. Returns the skip reason, or None to proceed
    to classification."""
    if execution_outcome.get("success") is True:
        if _success_verification_mismatch(execution_outcome):
            return None  # tool claims success but couldn't confirm it -- real signal, not noise
        return "verified_success" if execution_outcome.get("verified") else "success"
    actions = execution_outcome.get("actions") or []
    error_codes = {a.get("error") for a in actions if a.get("error")}
    if error_codes & _CANCELLATION_CODES:
        return "user_cancelled"
    if error_codes & _APPROVAL_CODES:
        return "approval_not_granted"
    return None


def _gather_evidence(command: str, execution_outcome: dict) -> dict:
    from brain.task_planner import segment_sequential_commands

    actions = execution_outcome.get("actions") or []
    failed_actions = [a for a in actions if not a.get("success")]
    block_errors = execution_outcome.get("block_errors") or []

    unknown_tool_names = []
    for err in block_errors:
        if isinstance(err, str) and ":unknown_tool:" in err:
            unknown_tool_names.append(err.split(":unknown_tool:", 1)[1])

    failed_error_codes = [a.get("error") for a in failed_actions if a.get("error")]
    environmental_from_actions = [c for c in failed_error_codes if c in _ENVIRONMENTAL_ERROR_CODES]

    tool_success_but_unverified_mismatch = _success_verification_mismatch(execution_outcome)

    try:
        clauses = segment_sequential_commands(command or "")
    except Exception:
        clauses = []

    tool_names = [a.get("tool") for a in actions if a.get("tool")]

    return {
        "exception_type": execution_outcome.get("exception_type"),
        "exception_message": execution_outcome.get("exception_message"),
        "block_reason": execution_outcome.get("block_reason"),
        "block_errors": block_errors,
        "unknown_tool_names": unknown_tool_names,
        "failed_error_codes": failed_error_codes,
        "failed_error_codes_environmental": environmental_from_actions,
        "unrepresented_clause": "unrepresented_clause" in block_errors,
        "tool_success_but_unverified_mismatch": tool_success_but_unverified_mismatch,
        "tool_names": tool_names,
        "clauses": clauses,
        "failed_actions": failed_actions,
    }


def _primary_tool_and_error(evidence: dict) -> tuple[str | None, str | None]:
    failed = evidence.get("failed_actions") or []
    if failed:
        return failed[0].get("tool"), failed[0].get("error")
    tool_names = evidence.get("tool_names") or []
    return (tool_names[0] if tool_names else None), None


def _fingerprint(evidence: dict, gap_type: str, subsystem: str | None, tool_involved: str | None) -> str:
    """A stable-ish identity for the *underlying* problem, deliberately
    excluding raw user wording: two differently-phrased requests that fail
    for the same structural reason (same gap type, same subsystem/tool,
    same exception type or error code, same block reason) must hash the
    same; two requests that merely sound similar but fail differently must
    not."""
    tool, primary_error = _primary_tool_and_error(evidence)
    payload = {
        "gap_type": gap_type,
        "subsystem": subsystem,
        "tool_involved": tool_involved or tool,
        "exception_type": evidence.get("exception_type"),
        "block_reason": evidence.get("block_reason"),
        "primary_error_code": primary_error,
        "unknown_tool_names": sorted(set(evidence.get("unknown_tool_names") or [])),
    }
    return stable_hash(payload)[:24]


def observe(
    *,
    command: str,
    route: dict | None,  # accepted for signature symmetry with run_agent; deliberately never read as evidence -- see module docstring
    execution_outcome: dict,
    original_user_text: str | None = None,
    interaction_id: str | None = None,
    task_id: str | None = None,
    duration_ms: float | None = None,
    resource_wait_ms: float | None = None,
) -> ImprovementCandidate | None:
    """Observe one completed request's real execution evidence and, if it
    exposes a meaningful engineering signal, persist an ImprovementCandidate.
    Returns the stored candidate, or None if nothing was recorded.

    `execution_outcome` must be the same dict `run_agent` already built from
    real ToolResults (see brain/agent.py::_build_execution_outcome) -- never
    a route or plan's intended actions.
    """
    skip = _skip_reason(execution_outcome)
    if skip is not None:
        log.info("[IMPROVEMENT] skipped reason=%s", skip)
        return None

    evidence = _gather_evidence(command, execution_outcome)
    gap_type, confidence, reason = classify(evidence)

    tool_involved, _ = _primary_tool_and_error(evidence)
    subsystem = _infer_subsystem(evidence.get("tool_names") or ([tool_involved] if tool_involved else []))
    fingerprint = _fingerprint(evidence, gap_type, subsystem, tool_involved)

    timestamp = _now()
    candidate_id = uuid.uuid4().hex
    sanitized_actions = []
    for action in execution_outcome.get("actions") or []:
        arguments = sanitize_action_arguments(action.get("tool"), action.get("arguments") or {})
        sanitized_actions.append({
            "tool": action.get("tool"),
            "arguments": arguments,
            "success": bool(action.get("success")),
            "verified": bool(action.get("verified")),
            "error": sanitize_text(action["error"])[:300] if action.get("error") else None,
            "message": sanitize_text(action["message"])[:300] if action.get("message") else None,
        })

    unmet_goal = None
    clauses = evidence.get("clauses") or []
    represented = len(evidence.get("failed_actions") or []) if execution_outcome.get("partial") else None
    if evidence.get("unrepresented_clause") and clauses:
        unmet_goal = f"{len(clauses)} requested clause(s) could not all be represented in one local plan"

    candidate = ImprovementCandidate(
        candidate_id=candidate_id,
        created_at=timestamp,
        first_seen=timestamp,
        last_seen=timestamp,
        occurrence_count=1,
        raw_request=sanitize_user_request(original_user_text or command or ""),
        normalized_goal=sanitize_user_request(command or ""),
        requested_clauses=[sanitize_user_request(c) for c in clauses],
        route_type=execution_outcome.get("route_type"),
        route_source=execution_outcome.get("route_source"),
        planner_source=execution_outcome.get("route_source"),
        executed_actions=sanitized_actions,
        executed_tool_names=[a["tool"] for a in sanitized_actions if a.get("tool")],
        success=bool(execution_outcome.get("success")),
        verified=bool(execution_outcome.get("verified")),
        partial=bool(execution_outcome.get("partial")),
        duration_ms=duration_ms,
        model_calls=int(execution_outcome.get("model_calls") or 0),
        resource_wait_ms=resource_wait_ms,
        exception_type=evidence.get("exception_type"),
        exception_message=sanitize_text(evidence["exception_message"])[:500] if evidence.get("exception_message") else None,
        verification_failure_reason=next((a["error"] for a in sanitized_actions if a.get("error")), None),
        unmet_goal=unmet_goal,
        plan_rejection_reason=evidence.get("block_reason"),
        plan_rejection_errors=[sanitize_text(str(e)) for e in (evidence.get("block_errors") or [])],
        unsupported_capability=(evidence.get("unknown_tool_names") or [None])[0],
        observations=[],
        recovery_evidence=None,
        gap_type=gap_type,
        confidence=confidence,
        classification_reason=reason,
        classifier_source=CLASSIFIER_SOURCE,
        platform="windows",
        subsystem=subsystem,
        tool_involved=tool_involved,
        environment_notes={},
        fingerprint=fingerprint,
        task_id=task_id,
        interaction_id=interaction_id,
    )

    stored, is_new = get_improvement_store().record(candidate)
    if is_new:
        log.info(
            "[IMPROVEMENT] created id=%s type=%s confidence=%.2f subsystem=%s fingerprint=%s",
            stored.candidate_id, stored.gap_type, stored.confidence, stored.subsystem, stored.fingerprint,
        )
    else:
        log.info(
            "[IMPROVEMENT] deduplicated id=%s occurrences=%d fingerprint=%s",
            stored.candidate_id, stored.occurrence_count, stored.fingerprint,
        )
    if gap_type == GapType.UNKNOWN.value:
        log.info("[IMPROVEMENT] classified type=UNKNOWN confidence=%.2f", confidence)
    return stored
