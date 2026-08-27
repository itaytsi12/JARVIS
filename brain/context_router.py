"""Turns a resolved conversational reference into a concrete route dict.

`brain/context_resolver.py` decides WHAT "it"/"that"/"the second one"
refers to (a pure, structured, model-free decision). This module is the
thin translation layer that turns a successful resolution into the same
kind of route dict every other deterministic pattern in `brain/router.py`
already returns -- so `brain/agent.py` never needs to know a request was
"contextual" versus "already fully specified".

`route_with_context(text, command, context)` returns `None` whenever
nothing here applies, and `brain/router.py` falls through to its ordinary
patterns exactly as if this module did not exist -- callers that never
pass a `SessionContext` (most existing tests, `speculative_execution.py`)
are completely unaffected.
"""
from __future__ import annotations

import re

from brain.context_resolver import (
    classify_reference_shape,
    resolve_correction,
    resolve_ordinal,
    resolve_reference,
    resolve_replay,
    resolved_context_summary,
)
from brain.models import Action
from brain.project_registry import resolve_project
from tools.registry import APP_ALIASES


def _browser_navigation_route(url: str, route_source: str) -> dict:
    """Search/navigation continuations use `browser_open_url` (the same
    Playwright-backed session `brain/task_planner.py`'s OWN "open X and
    search Y" / "go back" / "click the first result" already use) rather
    than the desktop-Chrome-controlling `open_website` tool -- so a
    continuation like "search for X instead" stays interoperable with
    "go back" and "open the first result" afterward instead of silently
    switching to a second, unrelated browser session. `browser_*` actions
    only ever execute through AgentRuntime (never the generic executor),
    hence `local_plan` here rather than a bare `tool` route."""
    return {
        "type": "local_plan",
        "actions": [Action("browser_open_url", {"url": url}, verify="url_loaded")],
        "route_source": route_source,
    }

_APP_TARGET_TOOL = {
    "close": "close_application",
    "quit": "close_application",
    "focus": "focus_application",
    "maximize": "maximize_window",
    "minimize": "minimize_window",
}
_ELLIPTICAL_APP_ACTION = re.compile(r"^(close|quit|focus|maximize|minimize)\s+(it|that|this)$", re.I)
_ELLIPTICAL_RUN_ACTION = re.compile(r"^(run|execute)\s+(it|that|this)$", re.I)
_ELLIPTICAL_OPEN_ACTION = re.compile(r"^open\s+(it|that|this)$", re.I)
_SEARCH_CONTINUATION = re.compile(r"^search(?:\s+for)?\s+(.+?)(?:\s+instead)?$", re.I)
# A query that itself names a provider ("search google for X", "search on
# youtube for X") is left untouched here -- brain/router.py's existing,
# more specific provider-aware search_patterns already handle those
# correctly, and duplicating that provider-detection logic here would risk
# the two disagreeing.
_SEARCH_NAMES_PROVIDER = re.compile(r"\b(?:google|youtube)\b", re.I)
_WHY_FAILED = re.compile(r"^why (?:did|does|didn'?t) (?:it|that|this)(?: not)? (?:fail|work)(?:ed)?\??$", re.I)
_WHAT_DOES_THAT_MEAN = re.compile(r"^what does (?:it|that|this) mean\??$", re.I)
_OPEN_PROJECT = re.compile(r"^open\s+(?:my\s+|the\s+)?(.+?)\s+project$", re.I)
_FIX_ORDINAL_PREFIX = re.compile(r"^fix\s+(.+)$", re.I)
_OPEN_ORDINAL_PREFIX = re.compile(r"^open\s+(.+)$", re.I)
_SEARCH_URLS = {
    "google": "https://www.google.com/search?q={}",
    "youtube": "https://www.youtube.com/results?search_query={}",
}


def _clarification_route(question: str, method: str) -> dict:
    return {"type": "clarification", "message": question, "route_source": f"context_{method}"}


def _app_route(tool: str, app_name: str, source: str) -> dict:
    return {"type": "tool", "tool": tool, "arguments": {"app_name": app_name}, "route_source": source}


def route_with_context(text: str, command: str, context) -> dict | None:
    stripped = text.rstrip(".?!,;:").strip()
    if not stripped:
        return None

    project_route = _route_open_project(stripped, context)
    if project_route is not None:
        return project_route

    correction_route = _route_correction(stripped, command, context)
    if correction_route is not None:
        return correction_route

    if classify_reference_shape(stripped) == "again":
        return _route_replay(context)

    app_action_route = _route_elliptical_app_action(stripped, context)
    if app_action_route is not None:
        return app_action_route

    run_route = _route_elliptical_run(stripped, command, context)
    if run_route is not None:
        return run_route

    open_it_route = _route_open_it(stripped, context)
    if open_it_route is not None:
        return open_it_route

    ordinal_route = _route_ordinal(stripped, command, context)
    if ordinal_route is not None:
        return ordinal_route

    search_route = _route_search_continuation(stripped, context)
    if search_route is not None:
        return search_route

    reasoning_route = _route_followup_reasoning(stripped, command, context)
    if reasoning_route is not None:
        return reasoning_route

    return None


#: A deictic project reference names no project at all -- "that"/"this"/"the"
#: only make sense against what the session already has open.
_DEICTIC_PROJECT = re.compile(r"^(?:that|this|the|it|my)$", re.I)


def _route_open_project(stripped: str, context=None) -> dict | None:
    match = _OPEN_PROJECT.match(stripped)
    if not match:
        return None
    spoken = match.group(1).strip()
    if _DEICTIC_PROJECT.fullmatch(spoken):
        # "open that project" / "open the project" -- resolve from session
        # state instead of the words, which name nothing. Without this the
        # phrase fell through to the generic open_application fallback and
        # tried to launch an application literally called "that project".
        path = getattr(context, "last_project_path", None)
        if not path:
            return None
        return {
            "type": "tool", "tool": "open_path", "arguments": {"path": path},
            "route_source": "context_project_open",
            "project_name": getattr(context, "last_project_name", None),
        }
    resolved = resolve_project(spoken)
    if not resolved:
        return None
    name, path = resolved
    return {
        "type": "tool", "tool": "open_path", "arguments": {"path": path},
        "route_source": "context_project_open", "project_name": name,
    }


def _route_correction(stripped: str, command: str, context) -> dict | None:
    if classify_reference_shape(stripped) != "correction":
        return None
    resolution = resolve_correction(stripped, context)
    if not resolution.success:
        return None
    tool = resolution.value["tool"]
    new_value = resolution.value["new_value"]
    if tool == "open_application":
        target = APP_ALIASES.get(new_value.lower(), new_value)
        return {**_app_route("open_application", target, "context_correction"), "corrects_previous": dict(resolution.value["previous_args"])}
    if tool == "open_website":
        from tools.registry import WEBSITE_ALIASES

        url = WEBSITE_ALIASES.get(new_value.lower())
        if url is None:
            return None
        return {"type": "tool", "tool": "open_website", "arguments": {"url": url}, "route_source": "context_correction", "corrects_previous": dict(resolution.value["previous_args"])}
    # Any other previous command: reissue it with its primary text argument
    # replaced -- the safest general behavior when the tool isn't one of the
    # two explicitly handled above is to say so rather than guess a field.
    return None


def _route_replay(context) -> dict | None:
    resolution = resolve_replay(context)
    if not resolution.success:
        return None
    return {
        "type": "tool", "tool": resolution.value["tool"], "arguments": dict(resolution.value["args"]),
        "route_source": "context_replay",
    }


def _route_elliptical_app_action(stripped: str, context) -> dict | None:
    match = _ELLIPTICAL_APP_ACTION.match(stripped)
    if not match:
        return None
    verb, reference = match.group(1).lower(), match.group(2)
    tool = _APP_TARGET_TOOL[verb]
    resolution = resolve_reference(reference, context, {"application", "browser"})
    if resolution.needs_clarification:
        return _clarification_route(resolution.clarification_question, "ambiguous")
    if not resolution.success:
        return None
    target = "browser" if resolution.entity_type == "browser" else resolution.value
    if tool in {"maximize_window", "minimize_window"}:
        # These act on the currently-focused window, not by app name -- a
        # resolved app target still tells the caller a real app was in
        # play, but the tool itself takes no arguments.
        return {"type": "tool", "tool": tool, "arguments": {}, "route_source": "context_pronoun", "resolved_target": target}
    return _app_route(tool, target, "context_pronoun")


def _route_elliptical_run(stripped: str, command: str, context) -> dict | None:
    match = _ELLIPTICAL_RUN_ACTION.match(stripped)
    if not match:
        return None
    resolution = resolve_reference(match.group(2), context, {"project", "task"})
    if resolution.needs_clarification:
        return _clarification_route(resolution.clarification_question, "ambiguous")
    if not resolution.success:
        return None
    summary = resolved_context_summary(context)
    if resolution.entity_type == "project":
        goal = f"Run the {resolution.label} project."
        summary["project_path"] = resolution.value
    else:
        goal = f"Continue/run the previous task: {resolution.label}."
    return {
        "type": "agent_task", "goal": goal, "route_source": "context_followup_run",
        "resolved_context": summary,
    }


def _route_open_it(stripped: str, context) -> dict | None:
    match = _ELLIPTICAL_OPEN_ACTION.match(stripped)
    if not match:
        return None
    resolution = resolve_reference(match.group(1), context)
    if resolution.needs_clarification:
        return _clarification_route(resolution.clarification_question, "ambiguous")
    if not resolution.success:
        return None
    if resolution.entity_type in {"application", "browser"}:
        return _app_route("open_application", resolution.value, "context_pronoun")
    if resolution.entity_type == "file":
        return {"type": "tool", "tool": "open_path", "arguments": {"path": resolution.value}, "route_source": "context_pronoun"}
    if resolution.entity_type == "project":
        return {"type": "tool", "tool": "open_path", "arguments": {"path": resolution.value}, "route_source": "context_pronoun", "project_name": resolution.label}
    return None


def _route_ordinal(stripped: str, command: str, context) -> dict | None:
    for prefix, kind_hint in ((_OPEN_ORDINAL_PREFIX, None), (_FIX_ORDINAL_PREFIX, "test_failure")):
        match = prefix.match(stripped)
        if not match:
            continue
        candidate_phrase = match.group(1).strip()
        if classify_reference_shape(candidate_phrase) != "ordinal":
            continue
        resolution = resolve_ordinal(candidate_phrase, context, kind=kind_hint)
        if not resolution.success:
            continue
        if resolution.entity_type == "file":
            return {"type": "tool", "tool": "open_path", "arguments": {"path": resolution.value}, "route_source": "context_ordinal"}
        if resolution.entity_type == "test_failure":
            summary = resolved_context_summary(context)
            return {
                "type": "agent_task",
                "goal": f"{command.strip()} ({resolution.label})",
                "route_source": "context_ordinal_followup",
                "resolved_context": summary,
            }
        return {"type": "tool", "tool": "open_path", "arguments": {"path": str(resolution.value)}, "route_source": "context_ordinal"}
    return None


def _route_search_continuation(stripped: str, context) -> dict | None:
    if not context.browser_active or _SEARCH_NAMES_PROVIDER.search(stripped):
        return None
    match = _SEARCH_CONTINUATION.match(stripped)
    if not match:
        return None
    query = match.group(1).strip()
    if not query:
        return None
    provider = context.last_search_provider or "google"
    template = _SEARCH_URLS.get(provider, _SEARCH_URLS["google"])
    import urllib.parse as _urllib_parse

    url = template.format(_urllib_parse.quote_plus(query))
    return _browser_navigation_route(url, "context_search_continuation")


def _route_followup_reasoning(stripped: str, command: str, context) -> dict | None:
    if not (_WHY_FAILED.fullmatch(stripped) or _WHAT_DOES_THAT_MEAN.fullmatch(stripped)):
        return None
    summary = resolved_context_summary(context)
    if not summary:
        return None
    return {
        "type": "agent_task", "goal": command.strip(), "route_source": "context_followup_reasoning",
        "resolved_context": summary,
    }
