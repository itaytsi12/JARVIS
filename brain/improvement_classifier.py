"""Conservative, deterministic gap classification for improvement evidence.

No cloud/LLM call is used here by design -- this is a small, evidence-driven
rule set over structured signals that already exist in the real execution
path (exception types, structured tool error codes, plan-rejection reasons).
When the evidence doesn't clearly support one of the specific gap types,
this returns UNKNOWN with low confidence rather than guessing -- pretending
certainty here would poison the candidate store with wrong conclusions a
future improvement agent would trust.

`classify(evidence)` is the whole interface. `evidence` is a plain dict (see
`brain/improvement_observer.py::_gather_evidence`) so a future reasoning
classifier can implement the same interface -- (evidence: dict) -> (GapType,
float, str) -- and be swapped in or layered on top without touching callers.
"""
from __future__ import annotations

from brain.improvement_models import GapType

CLASSIFIER_SOURCE = "deterministic_v1"

# A handful of stable, structured signals -- not a sprawling rule table.
# Each is a substring match against a short, already-existing error code or
# exception type name, not raw free text.
_ENVIRONMENTAL_ERROR_CODES = {
    "chrome_not_found",
    "unknown_application",
}
_ENVIRONMENTAL_EXCEPTION_TYPES = {
    "ConnectionError", "ConnectionRefusedError", "ConnectionResetError",
    "TimeoutError", "URLError", "HTTPError", "OSError", "socket.gaierror",
    "BrowserUnavailable",
}
_ENVIRONMENTAL_MESSAGE_MARKERS = (
    "network", "unreachable", "getaddrinfo", "connection refused",
    "name or service not known", "temporary failure in name resolution",
    "no such host", "dns",
)
_CANCELLATION_ERROR_CODES = {"cancelled", "cancelled_while_waiting_for_resource"}
_APPROVAL_ERROR_CODES = {"human_confirmation_required"}
_AMBIGUITY_BLOCK_REASONS = {"app_identification_uncertain", "clarification_required"}
_COVERAGE_BLOCK_REASONS = {
    "generated_plan_validation_failed", "generated_route_validation_failed",
    "cloud_plan_incomplete_or_invalid",
}
# Structured tool error codes that mean "JARVIS has no mapping for this
# target at all" -- a capability/data gap, not merely "not installed".
_CAPABILITY_GAP_ERROR_CODES = {"unknown_application"}


def _is_environmental_exception(exception_type: str | None, exception_message: str | None) -> bool:
    if not exception_type:
        return False
    if exception_type in _ENVIRONMENTAL_EXCEPTION_TYPES:
        return True
    lowered = (exception_message or "").lower()
    return any(marker in lowered for marker in _ENVIRONMENTAL_MESSAGE_MARKERS)


def classify(evidence: dict) -> tuple[str, float, str]:
    """Return (gap_type, confidence, reason). Confidence is a rough 0..1
    signal-strength indicator, not a calibrated probability."""
    exception_type = evidence.get("exception_type")
    exception_message = evidence.get("exception_message") or ""
    block_reason = evidence.get("block_reason")
    block_errors = evidence.get("block_errors") or []
    unknown_tool_names = evidence.get("unknown_tool_names") or []
    environmental_error_codes = evidence.get("failed_error_codes_environmental") or []
    failed_error_codes = evidence.get("failed_error_codes") or []
    unrepresented_clause = bool(evidence.get("unrepresented_clause"))
    tool_success_but_unverified_mismatch = bool(evidence.get("tool_success_but_unverified_mismatch"))

    # 1. A real, uncaught exception during execution is the strongest signal.
    if exception_type:
        if _is_environmental_exception(exception_type, exception_message):
            return (
                GapType.ENVIRONMENTAL_FAILURE.value, 0.7,
                f"exception {exception_type} matches a known environmental signature",
            )
        return (
            GapType.EXECUTION_BUG.value, 0.85,
            f"unhandled exception {exception_type} during real execution",
        )

    # 2. Explicit ambiguity/clarification signals from the planner itself.
    if block_reason in _AMBIGUITY_BLOCK_REASONS:
        return (
            GapType.USER_INPUT_OR_AMBIGUITY.value, 0.75,
            f"planner reported '{block_reason}' before any action executed",
        )

    # 3. A tool name the router/plan validator doesn't recognize at all is a
    # concrete, high-confidence missing-capability signal.
    if unknown_tool_names:
        return (
            GapType.CODE_CAPABILITY_GAP.value, 0.85,
            f"no registered tool for: {sorted(set(unknown_tool_names))}",
        )

    # 3b. A tool ran but reported that JARVIS has no mapping for the target
    # at all (distinct from "not installed" -- the app resolution logic
    # itself has no entry for this name).
    capability_gap_codes = [c for c in failed_error_codes if c in _CAPABILITY_GAP_ERROR_CODES]
    if capability_gap_codes:
        return (
            GapType.CODE_CAPABILITY_GAP.value, 0.7,
            f"tool reported a missing-capability error: {sorted(set(capability_gap_codes))}",
        )

    # 4. A tool that ran but reported an environmental-shaped structured error.
    if environmental_error_codes:
        return (
            GapType.ENVIRONMENTAL_FAILURE.value, 0.7,
            f"tool reported environmental failure codes: {sorted(set(environmental_error_codes))}",
        )

    # 5. Tool claimed success but independent behavioral verification
    # disagreed -- this is exactly the false-success shape and always a bug.
    if tool_success_but_unverified_mismatch:
        return (
            GapType.EXECUTION_BUG.value, 0.75,
            "tool reported success but behavioral verification disagreed",
        )

    # 6. Plan/goal-coverage rejection or an unrepresented clause: the
    # generic primitives plausibly exist but planning didn't sequence them.
    # Kept at moderate confidence since a deterministic classifier can't
    # prove the request was actually achievable.
    if block_reason in _COVERAGE_BLOCK_REASONS or unrepresented_clause:
        return (
            GapType.SKILL_GAP.value, 0.55,
            "planner could not represent the full request using available primitives"
            + (f" (errors: {sorted(set(block_errors))})" if block_errors else ""),
        )

    # 7. An action ran and failed with a structured error code.
    if failed_error_codes:
        verification_shaped = [c for c in failed_error_codes if c and ("verif" in c or c.endswith("_failed"))]
        if verification_shaped:
            return (
                GapType.EXECUTION_BUG.value, 0.6,
                f"action failed a verification-shaped check: {sorted(set(verification_shaped))}",
            )
        return (
            GapType.UNKNOWN.value, 0.4,
            f"action failed with unrecognized error code(s): {sorted(set(failed_error_codes))}",
        )

    return GapType.UNKNOWN.value, 0.3, "insufficient structured evidence to classify"
