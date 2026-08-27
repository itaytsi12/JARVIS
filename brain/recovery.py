"""Bounded recovery for a failed action.

The tool layer already carries its own fallbacks -- `tools/applications.py`
tries VS Code aliases, a direct path, `shutil.which`, the Windows start-app
command and the application index before giving up. This module is the layer
above that: what the RUNTIME does when a tool has exhausted its own options
and still failed.

It is deliberately small and deliberately boring:

- A strategy may only propose actions, never run anything itself.
- It proposes at most `MAX_RECOVERY_ACTIONS`, and the runtime runs each once.
- Recovery is never recursive: an action produced by a strategy is executed
  with recovery disabled, so a failing recovery cannot generate more
  recovery. This is what makes an infinite retry loop structurally
  impossible rather than merely unlikely.
- Only `ActionRisk.SAFE` actions are eligible, and a fixed set of failures is
  never recovered at all (a cancellation is a decision, not a fault; a
  confirmation request is waiting on a human; a dependency failure means the
  real error is elsewhere).
- A strategy may instead return a CLARIFICATION when the honest answer is
  that JARVIS cannot choose -- an ambiguous application name is a question
  for the user, not something to guess at.

Adding a strategy is adding one entry to `_STRATEGIES`. It receives the
failed action and its result and returns a `Recovery`; returning None means
"nothing sensible to try", which is the correct answer most of the time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from brain.models import Action, ActionRisk

#: Hard cap on how many alternative actions any single failure may produce.
MAX_RECOVERY_ACTIONS = 2

#: Failures that must never trigger recovery. These are not faults to work
#: around -- they are decisions, or they point at a different action.
NEVER_RECOVER_ERRORS = frozenset(
    {
        "cancelled",
        "human_confirmation_required",
        "dependency_failure",
        "resource_timeout",
    }
)

#: A failed reference is a planning error; retrying it would fail identically.
NEVER_RECOVER_PREFIXES = ("unresolved_reference",)


@dataclass
class Recovery:
    """What to try, or what to ask, after a failure.

    `actions` are executed in order, each exactly once, with recovery itself
    disabled. `clarification` is spoken to the user instead of a raw tool
    error and stops the plan.
    """

    reason: str
    actions: list[Action] = field(default_factory=list)
    clarification: str | None = None

    def is_empty(self) -> bool:
        return not self.actions and not self.clarification


def _recover_unverified_window(action: Action, result) -> Recovery | None:
    """The process started but no window was seen yet.

    Waiting again is read-only and cannot make anything worse: either the
    window turns up (the app was simply slow to paint) or it genuinely never
    appears and the original failure stands.
    """
    app_name = action.args.get("app_name")
    if not app_name:
        return None
    return Recovery(
        reason="window_not_seen_yet",
        actions=[Action("wait_for_window", {"app_name": app_name}, verify="window_exists", max_attempts=3)],
    )


def _ask_which_application(action: Action, result) -> Recovery | None:
    """Several applications matched. Guessing one would be worse than asking."""
    candidates = (result.data or {}).get("candidates") or []
    listed = ", ".join(str(candidate) for candidate in candidates[:5])
    target = action.args.get("app_name", "that")
    if listed:
        return Recovery(reason="ambiguous_application", clarification=f"I found more than one match for {target}, sir: {listed}. Which one did you mean?")
    return Recovery(reason="ambiguous_application", clarification=f"I found more than one application matching {target}, sir. Which one did you mean?")


#: (tool, error) -> strategy. A tool of None matches any tool.
_STRATEGIES: dict[tuple[str | None, str], object] = {
    ("open_application", "application_window_unverified"): _recover_unverified_window,
    (None, "ambiguous_application"): _ask_which_application,
}


def _eligible(action: Action, result) -> bool:
    if result is None or result.success:
        return False
    if action.risk is not ActionRisk.SAFE:
        return False
    error = str(result.error or "")
    if error in NEVER_RECOVER_ERRORS:
        return False
    return not any(error.startswith(prefix) for prefix in NEVER_RECOVER_PREFIXES)


def plan_recovery(action: Action, result) -> Recovery | None:
    """The bounded recovery for this failure, or None if there is nothing to try.

    Never raises: a strategy that misbehaves yields "no recovery" rather than
    turning a recoverable tool failure into a crashed plan.
    """
    if not _eligible(action, result):
        return None
    error = str(result.error or "")
    strategy = _STRATEGIES.get((action.tool, error)) or _STRATEGIES.get((None, error))
    if strategy is None:
        return None
    try:
        recovery = strategy(action, result)
    except Exception:
        return None
    if recovery is None or recovery.is_empty():
        return None
    if len(recovery.actions) > MAX_RECOVERY_ACTIONS:
        recovery.actions = recovery.actions[:MAX_RECOVERY_ACTIONS]
    # A recovery action must itself be safe and cheap; anything else is a
    # strategy bug and is dropped rather than executed.
    recovery.actions = [candidate for candidate in recovery.actions if candidate.risk is ActionRisk.SAFE]
    return None if recovery.is_empty() else recovery


__all__ = ["MAX_RECOVERY_ACTIONS", "NEVER_RECOVER_ERRORS", "Recovery", "plan_recovery"]
