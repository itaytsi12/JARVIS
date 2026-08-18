"""Safe speculative execution of deterministic, reversible actions from
unstable partial realtime-STT transcripts (Parts B/C of the overlapping
voice pipeline).

This module never calls an LLM and never invents a second router: every
candidate route comes from the SAME `brain.router.route_command` used for
final commands, restricted to a small explicit allowlist of tools that are
self-evidently safe to start before the user has finished speaking (see
`SAFE_PARTIAL_TOOLS`). A candidate must also resolve to the identical route
across `MIN_STABLE_PARTIALS` consecutive partial-transcript updates before
it is allowed to fire -- a single noisy partial is never enough.

`PartialActionLedger` is per-interaction (construct one per wake/listen
cycle, discard it afterwards). It also answers Part C's duplicate-action
question: once the final, committed transcript resolves to a route, the
caller reconciles it against whatever this ledger already fired via
`reconcile_final_route` so the same open/mute/volume action never runs
twice. If the final intent turns out to differ from what was speculatively
started, this module never attempts to undo the earlier action -- undoing a
possibly-unsafe-to-undo action is explicitly out of scope (Part C).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from brain.router import route_command
from brain.safe_tools import CONTEXT_INDEPENDENT_TOOLS

# Same safety classification `brain.agent_runtime` uses for Part H's
# parallel-independent-action execution -- see brain/safe_tools.py for the
# full rationale and the MAY/MUST-NOT examples it was built against.
SAFE_PARTIAL_TOOLS = CONTEXT_INDEPENDENT_TOOLS

MIN_STABLE_PARTIALS = 2


def classify_partial_route(text: str) -> dict | None:
    """Return a deterministic route dict IFF `text` resolves, via the real
    router, to a single tool call in `SAFE_PARTIAL_TOOLS`. Any other route
    type (question/plan/tools/ai/local_plan/None) is not eligible for
    speculative execution and returns None."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        route = route_command(text)
    except Exception:
        return None
    if not isinstance(route, dict) or route.get("type") != "tool":
        return None
    if route.get("tool") not in SAFE_PARTIAL_TOOLS:
        return None
    return route


def route_signature(route: dict) -> tuple:
    arguments = route.get("arguments") or {}
    return (route.get("tool"), tuple(sorted(arguments.items())))


@dataclass
class SpeculativeAction:
    route: dict
    signature: tuple
    partial_text: str = ""
    fired_at: float = 0.0
    result: object = None
    acknowledged: bool = False


class PartialActionLedger:
    """Tracks partial-transcript observations for ONE interaction and
    decides when a candidate becomes stable enough to execute early."""

    def __init__(self, min_stable: int = MIN_STABLE_PARTIALS, clock=None):
        import time as _time
        self.min_stable = max(1, min_stable)
        self._clock = clock or _time.monotonic
        self._recent_signatures: list[tuple | None] = []
        self._fired: dict[tuple, SpeculativeAction] = {}
        self._lock = threading.Lock()

    def observe_partial(self, text: str) -> SpeculativeAction | None:
        """Call for every partial-transcript update. Returns a NEW
        `SpeculativeAction` the caller must execute (and may acknowledge)
        immediately if, and only if, this update just made a candidate
        stable for the first time; otherwise None."""
        route = classify_partial_route(text)
        with self._lock:
            signature = route_signature(route) if route is not None else None
            self._recent_signatures.append(signature)
            if len(self._recent_signatures) > self.min_stable:
                self._recent_signatures = self._recent_signatures[-self.min_stable:]
            if signature is None:
                return None
            if signature in self._fired:
                return None
            if len(self._recent_signatures) < self.min_stable:
                return None
            if any(item != signature for item in self._recent_signatures[-self.min_stable:]):
                return None
            action = SpeculativeAction(route=route, signature=signature, partial_text=text, fired_at=self._clock())
            self._fired[signature] = action
            return action

    def already_fired(self, route: dict) -> SpeculativeAction | None:
        if not isinstance(route, dict):
            return None
        return self._fired.get(route_signature(route))

    def already_fired_tool(self, tool: str | None, arguments: dict | None) -> SpeculativeAction | None:
        if not tool:
            return None
        return self._fired.get(route_signature({"tool": tool, "arguments": arguments or {}}))

    def fired_actions(self) -> list[SpeculativeAction]:
        with self._lock:
            return list(self._fired.values())

    def has_fired_anything(self) -> bool:
        with self._lock:
            return bool(self._fired)


def reconcile_final_route(ledger: PartialActionLedger, final_route: dict | None) -> tuple[dict | None, SpeculativeAction | None]:
    """Given the FINAL committed route, decide whether it was already
    (speculatively) satisfied.

    Returns `(route_to_execute_or_None, matched_speculative_action_or_None)`.
    Only an exact single-tool route match counts -- this never guesses at
    partial equivalence between a `local_plan`/`plan`/`tools` route and a
    single fired action; callers that build multi-step plans should
    reconcile their own first step separately using `already_fired`. If the
    final route differs from anything fired, it is returned unchanged so it
    still executes normally (Part C: never try to undo an already-started
    action, and never silently drop the user's actual final request)."""
    if ledger is None or not isinstance(final_route, dict) or final_route.get("type") != "tool":
        return final_route, None
    matched = ledger.already_fired(final_route)
    if matched is not None:
        return None, matched
    return final_route, None


def reconcile_local_plan_actions(ledger: PartialActionLedger | None, actions: list) -> list:
    """Drop LEADING actions from a `local_plan`'s action list that a
    speculative partial-transcript execution already completed (Part C's
    "open Spotify and play my liked songs" example). Only ever trims from
    the front, and only actions with no unmet dependency wiring
    (`depends_on` empty) -- exactly what
    `brain.local_planner.create_local_plan` always produces, since its
    actions are a flat sequential list with no dependency graph. Stops at
    the first action that wasn't already fired, so a later step already
    matching a fired signature (unusual, but possible) is never skipped
    just because it isn't first."""
    if ledger is None or not actions:
        return actions
    trimmed = list(actions)
    while trimmed:
        first = trimmed[0]
        if getattr(first, "depends_on", None):
            break
        matched = ledger.already_fired_tool(getattr(first, "tool", None), getattr(first, "args", None))
        if matched is None:
            break
        trimmed = trimmed[1:]
    return trimmed
