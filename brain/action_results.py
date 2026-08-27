"""Letting one action consume an earlier action's result.

"Run the tests and write the failures into Notepad" is two actions where the
second needs something the first produced. Without a mechanism for that, the
only options are to make the planner inline a guess at execution time (it
cannot -- the tests have not run yet) or to have the second action re-derive
the information by scraping console text that a structured result already
contains.

A reference is ordinary validated data, never code:

    Action("write_text_file", {
        "path": "errors.txt",
        "contents": {"__from_result__": {"action": 0, "field": "summary"}},
    })

At execution time `resolve_arg_references` replaces that marker with the
named field of action 0's `ToolResult`. Nothing is evaluated; the only thing
a reference can do is read a field of a result that already exists.

`ToolResult` (brain/models.py) stays the single result type -- this module
deliberately does not introduce a second one. It provides the facets Phase 5
asks for as a *view* over the existing fields:

    status      -> "ok" / "failed"        (from .success)
    summary     -> short human sentence   (from .message / .error)
    text        -> full textual payload   (.data["text"]/["stdout"] or .message)
    data.<key>  -> structured payload     (.data, dotted path)
    artifacts   -> .data["artifacts"] or ["path"]/["paths"]
    error       -> .error
    metadata    -> .data minus the bulky/structured keys

so tools that already populate `data` gain result passing for free.
"""
from __future__ import annotations

from typing import Any

from brain.models import Action

#: Marker key identifying a reference inside an action's arguments.
REFERENCE_KEY = "__from_result__"

#: Fields a reference may name. Anything else is a planning error, reported
#: rather than guessed at.
SCALAR_FIELDS = frozenset({"status", "summary", "text", "error", "artifacts", "metadata", "success"})

#: Result `data` keys that carry the full textual payload, most specific first.
_TEXT_KEYS = ("text", "stdout", "output", "contents", "content")

#: `data` keys that are payload rather than metadata.
_NON_METADATA_KEYS = frozenset(
    {"text", "stdout", "stderr", "output", "contents", "content", "artifacts", "paths", "items", "results"}
)


class ReferenceError_(ValueError):
    """A reference could not be resolved. Named with a trailing underscore to
    avoid shadowing the builtin `ReferenceError`."""


def is_reference(value: Any) -> bool:
    return isinstance(value, dict) and REFERENCE_KEY in value and len(value) == 1


def _spec(value: dict) -> dict:
    spec = value[REFERENCE_KEY]
    if not isinstance(spec, dict):
        raise ReferenceError_(f"reference payload must be an object, got {type(spec).__name__}")
    if "action" not in spec:
        raise ReferenceError_("reference is missing 'action'")
    index = spec["action"]
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ReferenceError_(f"reference 'action' must be a non-negative integer, got {index!r}")
    return spec


def reference_targets(action: Action) -> set[int]:
    """Every action index this action's arguments refer to.

    Used to derive implied dependencies so a plan that references a result
    without also declaring `depends_on` still executes in a correct order
    rather than reading a result that does not exist yet.
    """
    targets: set[int] = set()

    def walk(value: Any) -> None:
        if is_reference(value):
            try:
                targets.add(_spec(value)["action"])
            except ReferenceError_:
                return
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(action.args or {})
    return targets


def with_reference_dependencies(actions: list[Action]) -> list[Action]:
    """Return `actions` with every referenced index added to `depends_on`.

    Mutates nothing the caller owns: each affected action is replaced by a
    copy. Applied by the scheduler so referencing a result is by itself
    enough to order two actions correctly.
    """
    updated: list[Action] = []
    for index, action in enumerate(actions):
        targets = {target for target in reference_targets(action) if 0 <= target < len(actions) and target != index}
        missing = sorted(targets - set(action.depends_on))
        if not missing:
            updated.append(action)
            continue
        clone = Action(
            tool=action.tool,
            args=action.args,
            depends_on=list(action.depends_on) + missing,
            optional=action.optional,
            verify=action.verify,
            stop_condition=action.stop_condition,
            risk=action.risk,
            sensitive_fields=set(action.sensitive_fields),
            max_attempts=action.max_attempts,
        )
        updated.append(clone)
    return updated


def result_status(result) -> str:
    return "ok" if result.success else "failed"


def result_text(result) -> str:
    data = result.data if isinstance(result.data, dict) else {}
    for key in _TEXT_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return result.message or ""


def result_summary(result) -> str:
    """One short sentence describing the outcome.

    Prefers a summary the tool supplied itself, then its message, then its
    error -- never a fabricated description of what "probably" happened.
    """
    data = result.data if isinstance(result.data, dict) else {}
    for key in ("summary", "description"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if result.message:
        return result.message.strip()
    if result.error:
        return str(result.error)
    return result_status(result)


def result_artifacts(result) -> list[str]:
    data = result.data if isinstance(result.data, dict) else {}
    artifacts = data.get("artifacts")
    if isinstance(artifacts, list):
        return [str(item) for item in artifacts]
    paths = data.get("paths")
    if isinstance(paths, list):
        return [str(item) for item in paths]
    path = data.get("path")
    return [str(path)] if path else []


def result_metadata(result) -> dict:
    data = result.data if isinstance(result.data, dict) else {}
    return {key: value for key, value in data.items() if key not in _NON_METADATA_KEYS}


def result_field(result, field: str) -> Any:
    """Read one named field from a `ToolResult`.

    `field` is either one of `SCALAR_FIELDS` or a dotted path into `.data`
    (e.g. `data.failures`, `data.exit_code`). An unknown name raises rather
    than returning None, so a planning mistake surfaces as a reported error
    instead of an action silently receiving nothing.
    """
    if field in ("status",):
        return result_status(result)
    if field == "success":
        return bool(result.success)
    if field == "summary":
        return result_summary(result)
    if field == "text":
        return result_text(result)
    if field == "error":
        return result.error
    if field == "artifacts":
        return result_artifacts(result)
    if field == "metadata":
        return result_metadata(result)
    if field == "data" or field.startswith("data."):
        cursor: Any = result.data if isinstance(result.data, dict) else {}
        for part in field.split(".")[1:]:
            if isinstance(cursor, dict) and part in cursor:
                cursor = cursor[part]
            elif isinstance(cursor, list) and part.isdigit() and int(part) < len(cursor):
                cursor = cursor[int(part)]
            else:
                raise ReferenceError_(f"result has no field '{field}'")
        return cursor
    raise ReferenceError_(
        f"unknown result field '{field}'; expected one of {sorted(SCALAR_FIELDS)} or a 'data.*' path"
    )


def _render(value: Any) -> Any:
    """Reference values are substituted as-is, except that a list destined for
    a text argument renders as newline-joined lines -- the shape a caller
    almost always wants when writing failures into a file."""
    if isinstance(value, list) and all(isinstance(item, (str, int, float)) for item in value):
        return "\n".join(str(item) for item in value)
    return value


def resolve_arg_references(args: dict, results_by_index: dict) -> dict:
    """Return `args` with every reference replaced by the value it names.

    Raises `ReferenceError_` when a reference names an action that has not
    produced a result, or a field that does not exist. The caller turns that
    into a failed `ToolResult` for the referring action -- the failure is
    attributed to the action that made the bad reference, and is never
    silently ignored.
    """

    def walk(value: Any) -> Any:
        if is_reference(value):
            spec = _spec(value)
            index = spec["action"]
            if index not in results_by_index:
                raise ReferenceError_(f"action {index} has not produced a result yet")
            field = spec.get("field", "summary")
            if not isinstance(field, str):
                raise ReferenceError_(f"reference 'field' must be a string, got {field!r}")
            return _render(result_field(results_by_index[index], field))
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return walk(dict(args or {}))


def plan_has_references(actions: list[Action]) -> bool:
    return any(reference_targets(action) for action in actions)


__all__ = [
    "REFERENCE_KEY",
    "SCALAR_FIELDS",
    "ReferenceError_",
    "is_reference",
    "plan_has_references",
    "reference_targets",
    "resolve_arg_references",
    "result_artifacts",
    "result_field",
    "result_metadata",
    "result_status",
    "result_summary",
    "result_text",
    "with_reference_dependencies",
]
