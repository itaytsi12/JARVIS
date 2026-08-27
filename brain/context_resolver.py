"""General conversational-reference resolution.

This module is the one place JARVIS decides what a pronoun, a demonstrative
("that window"), an ordinal ("the second one"), an "again"/"do it again",
or a correction ("no, I meant Telegram") actually refers to. It works
entirely from the structured, short-term state in `brain.session_context`
-- never a second parallel memory system, and never a per-example regex
for a specific sentence.

Two general mechanisms do essentially all of the work:

1. Recency by TURN, not wall-clock time. `SessionContext.turn_counter` is
   bumped once per top-level user request; every recorded entity carries
   the turn it was introduced on. "Close it" after "open Chrome" then
   "open Spotify" resolves to Spotify because turn N (Spotify) is strictly
   later than turn N-1 (Chrome) -- a decisive winner regardless of how many
   wall-clock seconds separate the two commands. "Close it" after a SINGLE
   command that opened two apps in the same turn is genuinely ambiguous,
   because both entities share the same turn and nothing else
   distinguishes them; that's when a clarification question is asked
   instead of guessed.

2. Bounded TTLs by category (`*_TTL_SECONDS` below, all overridable via
   environment variables) so a reference from a stale part of the
   conversation can never unexpectedly control something. An app referenced
   30 minutes ago no longer counts as "it"; a result list from ten minutes
   ago no longer answers "the second one".

Nothing here calls a model. `resolve_reference`/`resolve_ordinal` either
return a confident answer or say plainly that they could not -- callers
(`brain/router.py`, `brain/task_planner.py`) decide what to do with an
unresolved or ambiguous result (usually: fall through, or escalate to
AgentRuntime with the ambiguity made explicit).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from brain.session_context import SessionContext

# -- configurable TTLs (section 13: "context must not live forever") -------
APP_REFERENCE_TTL_SECONDS = float(os.getenv("JARVIS_CONTEXT_APP_TTL_SECONDS", "1800"))
RESULT_SET_TTL_SECONDS = float(os.getenv("JARVIS_CONTEXT_RESULT_SET_TTL_SECONDS", "600"))
TASK_CONTEXT_TTL_SECONDS = float(os.getenv("JARVIS_CONTEXT_TASK_TTL_SECONDS", "1800"))
ERROR_CONTEXT_TTL_SECONDS = float(os.getenv("JARVIS_CONTEXT_ERROR_TTL_SECONDS", "1800"))
COMMAND_CONTEXT_TTL_SECONDS = float(os.getenv("JARVIS_CONTEXT_COMMAND_TTL_SECONDS", "600"))
PROJECT_CONTEXT_TTL_SECONDS = float(os.getenv("JARVIS_CONTEXT_PROJECT_TTL_SECONDS", str(24 * 3600)))
FILE_CONTEXT_TTL_SECONDS = float(os.getenv("JARVIS_CONTEXT_FILE_TTL_SECONDS", "1800"))
RECENCY_HALF_LIFE_SECONDS = float(os.getenv("JARVIS_CONTEXT_RECENCY_HALF_LIFE_SECONDS", "120"))


@dataclass
class Candidate:
    entity_type: str  # "application" | "browser" | "project" | "task" | "file"
    label: str
    value: Any
    turn: int
    score: float
    source: str


@dataclass
class ReferenceResolution:
    success: bool
    method: str  # "resolved" | "ordinal" | "correction" | "replay" | "ambiguous" | "not_found"
    entity_type: str | None = None
    value: Any = None
    label: str | None = None
    confidence: float = 0.0
    candidates: list[Candidate] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None
    source: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Reference SHAPE classification -- general grammatical categories, never a
# specific example sentence. This is deliberately small: JARVIS is a
# structured resolver, not an NLP framework (section 25).
# ---------------------------------------------------------------------------

_PRONOUN_SHAPE = re.compile(r"^(?:it|this|that|them|there)$", re.I)
_DEMONSTRATIVE_NOUNS = {
    "app": "application", "application": "application", "window": "application", "program": "application",
    "project": "project", "file": "file", "document": "file", "folder": "folder",
    "browser": "browser", "page": "browser", "site": "browser", "task": "task",
}
_DEMONSTRATIVE_SHAPE = re.compile(
    r"^(?:the|this|that)\s+(" + "|".join(sorted(_DEMONSTRATIVE_NOUNS, key=len, reverse=True)) + r")$", re.I
)
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "last": -1,
}
_ORDINAL_SHAPE = re.compile(
    r"^(?:the\s+)?(" + "|".join(_ORDINAL_WORDS) + r"|\d+(?:st|nd|rd|th))\s*(?:one|result|item|file|video|entry)?$", re.I
)
# An optional verb may precede "it again" -- "run it again", "play it
# again", "open it again" are the same request as "do it again", and the
# verb is recoverable from the remembered command rather than the words.
_AGAIN_SHAPE = re.compile(r"^(?:(?:do|run|try|play|open|start|execute)\s+(?:it|that|this)\s+again|do it again|try again|again|repeat (?:that|it)|once more)$", re.I)
_CORRECTION_SHAPE = re.compile(
    r"^(?:no,?\s*i meant|no,?\s*i mean|actually,?\s*(?:i meant|use|make it)|i meant|not that one,?\s*(?:i meant|use)?\s*)\s*(.*)$",
    re.I,
)


def classify_reference_shape(phrase: str) -> str | None:
    cleaned = phrase.strip().rstrip(".?!")
    if not cleaned:
        return None
    if _PRONOUN_SHAPE.fullmatch(cleaned):
        return "pronoun"
    if _DEMONSTRATIVE_SHAPE.fullmatch(cleaned):
        return "demonstrative"
    if _ORDINAL_SHAPE.fullmatch(cleaned):
        return "ordinal"
    if _AGAIN_SHAPE.fullmatch(cleaned):
        return "again"
    match = _CORRECTION_SHAPE.match(cleaned)
    if match and match.group(1).strip():
        return "correction"
    return None


def _recency_score(age_seconds: float) -> float:
    age_seconds = max(0.0, age_seconds)
    if RECENCY_HALF_LIFE_SECONDS <= 0:
        return 1.0
    return 0.5 ** (age_seconds / RECENCY_HALF_LIFE_SECONDS)


# ---------------------------------------------------------------------------
# Candidate gathering
# ---------------------------------------------------------------------------

def gather_candidates(context: SessionContext, expected_types: set[str] | None = None) -> list[Candidate]:
    now = time.monotonic()
    candidates: list[Candidate] = []

    if not expected_types or "application" in expected_types:
        last_seen: dict[str, float] = {}
        for event in context.recent_apps:
            if event.kind == "closed":
                last_seen.pop(event.name, None)
            else:
                last_seen[event.name] = event.at
        for name, at in last_seen.items():
            age = now - at
            if age <= APP_REFERENCE_TTL_SECONDS:
                turn = context.last_turn_for_app(name) or 0
                candidates.append(Candidate("application", name, name, turn, _recency_score(age), "recent_apps"))

    if (not expected_types or "browser" in expected_types) and context.browser_active:
        age = now - (context.browser_updated_at or now)
        if age <= APP_REFERENCE_TTL_SECONDS:
            # A live browser session is itself an "application" candidate
            # for a bare "it"/"that" too -- e.g. "open Chrome" -> "close it".
            turn = context.last_turn_for_app("browser") or context.turn_counter
            candidates.append(Candidate("browser", "the browser", "browser", turn, _recency_score(age), "browser_active"))

    if (not expected_types or "project" in expected_types) and context.last_project_path:
        age = now - context.project_updated_at
        if age <= PROJECT_CONTEXT_TTL_SECONDS:
            candidates.append(Candidate(
                "project", context.last_project_name or context.last_project_path, context.last_project_path,
                context.turn_counter, _recency_score(age), "last_project",
            ))

    if (not expected_types or "task" in expected_types) and context.last_task_id:
        age = now - context.task_context_updated_at
        if age <= TASK_CONTEXT_TTL_SECONDS:
            candidates.append(Candidate(
                "task", context.last_task_goal or context.last_task_id, context.last_task_id,
                context.turn_counter, _recency_score(age), "last_task",
            ))

    if (not expected_types or "file" in expected_types) and context.last_opened_file:
        age = now - (context.last_opened_file_at or now)
        if age <= FILE_CONTEXT_TTL_SECONDS:
            candidates.append(Candidate(
                "file", context.last_opened_file, context.last_opened_file,
                context.turn_counter, _recency_score(age), "last_opened_file",
            ))

    return candidates


def _clarification_question(candidates: list[Candidate]) -> str:
    names = [candidate.label for candidate in candidates]
    if len(names) == 2:
        return f"Do you mean {names[0]} or {names[1]}, sir?"
    return "Do you mean " + ", ".join(names[:-1]) + f", or {names[-1]}, sir?"


def _resolve_from_candidates(candidates: list[Candidate]) -> ReferenceResolution:
    if not candidates:
        return ReferenceResolution(False, "not_found")
    max_turn = max(candidate.turn for candidate in candidates)
    current = [candidate for candidate in candidates if candidate.turn == max_turn]
    if len(current) == 1:
        winner = current[0]
        return ReferenceResolution(
            True, "resolved", winner.entity_type, winner.value, winner.label, winner.score,
            candidates=[winner], source=winner.source,
        )
    ranked = sorted(current, key=lambda c: c.score, reverse=True)
    return ReferenceResolution(
        False, "ambiguous", candidates=ranked, needs_clarification=True,
        clarification_question=_clarification_question(ranked),
    )


def resolve_reference(phrase: str, context: SessionContext, expected_types: set[str] | None = None) -> ReferenceResolution:
    """Resolve a pronoun/demonstrative phrase ("it", "that window", ...).

    `expected_types` narrows the candidate pool to entity types that make
    sense for the calling tool (e.g. only "application" for close/focus).
    A bare demonstrative ("the project") infers its own type from the noun
    and that takes priority over whatever the caller passed.
    """
    cleaned = phrase.strip().rstrip(".?!")
    shape = classify_reference_shape(cleaned)
    if shape not in ("pronoun", "demonstrative"):
        return ReferenceResolution(False, "not_found")
    types = expected_types
    if shape == "demonstrative":
        match = _DEMONSTRATIVE_SHAPE.fullmatch(cleaned)
        noun = match.group(1).lower() if match else None
        inferred = _DEMONSTRATIVE_NOUNS.get(noun)
        if inferred:
            types = {inferred}
    candidates = gather_candidates(context, types)
    return _resolve_from_candidates(candidates)


# ---------------------------------------------------------------------------
# Ordinal / result-set resolution (section 17)
# ---------------------------------------------------------------------------

def _ordinal_index(word: str) -> int | None:
    word = word.lower()
    if word in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[word]
    match = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", word)
    return int(match.group(1)) if match else None


def resolve_ordinal(phrase: str, context: SessionContext, kind: str | None = None) -> ReferenceResolution:
    cleaned = phrase.strip().rstrip(".?!")
    match = _ORDINAL_SHAPE.fullmatch(cleaned)
    if not match:
        return ReferenceResolution(False, "ordinal")
    index = _ordinal_index(match.group(1))
    if index is None:
        return ReferenceResolution(False, "ordinal")
    result_set = context.last_result_set
    if result_set is None:
        return ReferenceResolution(False, "ordinal", error="no_recent_result_set")
    age = time.monotonic() - result_set.created_at
    if age > RESULT_SET_TTL_SECONDS:
        return ReferenceResolution(False, "ordinal", error="result_set_expired")
    if kind is not None and result_set.kind != kind:
        return ReferenceResolution(False, "ordinal", error="result_set_type_mismatch")
    items = result_set.items
    if not items:
        return ReferenceResolution(False, "ordinal", error="result_set_empty")
    target_index = len(items) if index == -1 else index
    if target_index < 1 or target_index > len(items):
        return ReferenceResolution(False, "ordinal", error="ordinal_out_of_range")
    item = items[target_index - 1]
    return ReferenceResolution(
        True, "ordinal", result_set.kind, item.value, item.label, 0.95,
        candidates=[], source=result_set.source,
    )


# ---------------------------------------------------------------------------
# Correction / replay (section 6, 7)
# ---------------------------------------------------------------------------

def resolve_correction(phrase: str, context: SessionContext) -> ReferenceResolution:
    cleaned = phrase.strip().rstrip(".?!")
    match = _CORRECTION_SHAPE.match(cleaned)
    if not match:
        return ReferenceResolution(False, "not_found")
    new_value = match.group(1).strip()
    if not new_value:
        return ReferenceResolution(False, "not_found", error="no_correction_target")
    if not context.last_command_tool:
        return ReferenceResolution(False, "not_found", error="no_previous_command")
    age = time.monotonic() - context.last_command_time
    if age > COMMAND_CONTEXT_TTL_SECONDS:
        return ReferenceResolution(False, "not_found", error="previous_command_expired")
    return ReferenceResolution(
        True, "correction", entity_type="correction",
        value={
            "tool": context.last_command_tool,
            "previous_args": dict(context.last_command_args or {}),
            "new_value": new_value,
        },
        label=new_value, confidence=0.85, source="last_command",
    )


def resolve_replay(context: SessionContext) -> ReferenceResolution:
    """'Do it again' / 'try again' -- replay the last executed command."""
    if not context.last_command_tool:
        return ReferenceResolution(False, "not_found", error="no_previous_command")
    age = time.monotonic() - context.last_command_time
    if age > COMMAND_CONTEXT_TTL_SECONDS:
        return ReferenceResolution(False, "not_found", error="previous_command_expired")
    return ReferenceResolution(
        True, "replay", entity_type="command",
        value={"tool": context.last_command_tool, "args": dict(context.last_command_args or {})},
        label=context.last_command_text or context.last_command_tool, confidence=0.9, source="last_command",
    )


# ---------------------------------------------------------------------------
# Structured extraction from tool output (section 16/17) -- general list
# grammar, not a per-tool special case: a numbered/bulleted list, or the
# well-known pytest "FAILED <nodeid> - <reason>" convention.
# ---------------------------------------------------------------------------

_PYTEST_FAILED_LINE = re.compile(r"^FAILED\s+(\S+?)(?:\s+-\s+(.*))?$", re.M)
_NUMBERED_OR_BULLETED_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.+?)\s*$", re.M)


def extract_result_items(text: str, limit: int = 25) -> list[tuple[str, str]]:
    """Returns (label, value) pairs in original order, or [] if nothing
    list-shaped was found. Never raises on arbitrary tool output."""
    if not text:
        return []
    failed = _PYTEST_FAILED_LINE.findall(text)
    if failed:
        return [(f"{path} - {reason}" if reason else path, path) for path, reason in failed[:limit]]
    bulleted = _NUMBERED_OR_BULLETED_LINE.findall(text)
    if bulleted:
        return [(line.strip(), line.strip()) for line in bulleted[:limit] if line.strip()]
    return []


def _result_kind_for_text(text: str) -> str:
    return "test_failure" if _PYTEST_FAILED_LINE.search(text) else "line"


def observe_tool_result(context: SessionContext, tool: str, args: dict[str, Any], result: Any) -> None:
    """Feed one executed tool's structured result into short-term context.

    The single place both execution paths (AgentRuntime._update_context for
    deterministic plans, AgentLoop.run for agent-loop tool calls) route
    through, so context accumulates identically regardless of which path
    handled the request (section 16: tools emit structured entities, the
    context layer never re-parses prose it doesn't have to).  Never raises:
    a context-recording bug must not turn a successful tool call into a
    failed request.
    """
    try:
        data = result.data if hasattr(result, "data") and isinstance(result.data, dict) else {}
        success = bool(getattr(result, "success", False))
        context.record_command(tool, dict(args or {}))
        if not success:
            error = getattr(result, "error", None) or getattr(result, "message", None)
            if error:
                context.record_error(str(error), source=tool)
        if tool == "list_files" and isinstance(data.get("items"), list):
            context.record_result_set(
                [(str(name), str(name)) for name in data["items"]], source=tool, kind="file", query=data.get("path"),
            )
        elif tool == "run_command":
            output = "\n".join(str(data.get(key, "")) for key in ("stdout", "stderr") if data.get(key))
            items = extract_result_items(output)
            if items:
                context.record_result_set(items, source=tool, kind=_result_kind_for_text(output), query=str(args.get("command", "")))
        elif tool in {"open_path", "create_text_file", "write_text_file"} and data.get("path"):
            context.last_opened_file = data["path"]
            context.last_opened_file_at = time.monotonic()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Section 15: explicit, resolved context for an AgentRuntime follow-up --
# never the raw transcript.
# ---------------------------------------------------------------------------

def resolved_context_summary(context: SessionContext) -> dict[str, Any]:
    """A compact, structured snapshot of what the resolver currently
    believes is "current" -- injected into the agent's context instead of
    making it re-derive this from a long transcript (section 15).

    Uses `getattr(..., None)` throughout (matching
    `context_builder._describe_session`'s own style) rather than direct
    attribute access: callers occasionally pass a minimal duck-typed
    context object with only the fields they care about, not a full
    `SessionContext`, and this must degrade gracefully rather than crash.
    """
    summary: dict[str, Any] = {}
    last_task_id = getattr(context, "last_task_id", None)
    if last_task_id:
        summary["previous_task_id"] = last_task_id
        summary["previous_task_goal"] = getattr(context, "last_task_goal", None)
        summary["previous_task_status"] = getattr(context, "last_task_status", None)
        result_summary = getattr(context, "last_task_result_summary", None)
        if result_summary:
            summary["previous_task_result_summary"] = result_summary
        task_error = getattr(context, "last_task_error", None)
        if task_error:
            summary["previous_task_error"] = task_error
    last_error = getattr(context, "last_error", None)
    last_error_at = getattr(context, "last_error_at", 0.0) or 0.0
    if last_error and (time.monotonic() - last_error_at) <= ERROR_CONTEXT_TTL_SECONDS:
        summary["last_error"] = last_error
        summary["last_error_source"] = getattr(context, "last_error_source", None)
    project_path = getattr(context, "last_project_path", None)
    if project_path:
        summary["project_path"] = project_path
    result_set = getattr(context, "last_result_set", None)
    if result_set is not None:
        summary["last_result_set"] = {
            "kind": result_set.kind,
            "source": result_set.source,
            "items": [item.label for item in result_set.items[:10]],
        }
    return summary
