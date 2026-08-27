"""Dependency-aware scheduling for a `Plan`'s actions.

`brain/models.py::Action` has always carried `depends_on`, but nothing ever
scheduled from it: `brain/agent_runtime.py` either ran a plan strictly one
action at a time, or -- only when EVERY action was simultaneously
dependency-free and context-independent -- ran the whole plan at once
(`_all_actions_independent`). That is an all-or-nothing choice, and it loses
on exactly the requests users actually make: "open Chrome and Spotify, then
lower the volume" has one ordering edge at the end, so the two independent
launches were serialized behind each other for no reason.

This module turns `depends_on` into a real schedule:

- `build_waves` levels the dependency graph (Kahn). Wave *n* contains every
  action whose dependencies all completed in waves < *n*.
- `partition_wave` splits one wave into the subset that may genuinely run at
  the same time and the remainder that must not.

Two properties keep this a strict generalization of the previous behavior
rather than a replacement:

- A plan that is a pure chain (`task_planner` emits these: every action
  depends on the one before it) levels into N waves of one action each, so
  it executes in exactly the original order with exactly the original
  semantics.
- A plan whose actions are all independent and context-independent levels
  into a single wave that partitions entirely into the parallel group --
  the previous `_execute_plan_parallel` case, unchanged.

Everything in between -- the mixed plans that were previously forced fully
sequential -- is where the new capability lives.

Parallel safety is NOT re-derived here. It reuses the two classifications
this codebase already maintains:

- `brain/safe_tools.py::CONTEXT_INDEPENDENT_TOOLS` -- the tool neither reads
  state a sibling writes nor writes state a sibling reads.
- `brain/resource_locks.py::resource_for_tool` -- two actions needing the
  same exclusive resource (the keyboard, a browser page, the speaker) would
  only serialize inside the lock anyway, and holding a plan's worth of
  worker threads on a lock queue is worse than not starting them.
"""
from __future__ import annotations

from brain.models import Action, ActionRisk
from brain.resource_locks import resource_for_tool
from brain.safe_tools import CONTEXT_INDEPENDENT_TOOLS


class CyclicPlanError(ValueError):
    """A plan's `depends_on` edges form a cycle, so no order can satisfy it.

    Raised rather than silently dropping an edge: a cyclic plan is a planner
    bug, and executing an arbitrary subset of it would produce a confidently
    wrong outcome.
    """


def _dependencies(action: Action, count: int) -> set[int]:
    """`action.depends_on` reduced to the edges that can actually be waited on.

    Out-of-range indices are ignored here instead of raising. They are already
    a validation error (`brain/plan_validator.py` rejects them before a plan
    reaches execution), and a scheduler that crashed on one would turn a
    recoverable planning mistake into a lost request.
    """
    return {dep for dep in action.depends_on if isinstance(dep, int) and 0 <= dep < count}


def build_waves(actions: list[Action]) -> list[list[int]]:
    """Group action indices into dependency levels, earliest first.

    Every index appears exactly once. An action is placed in the first wave
    strictly after all of its dependencies, so executing wave by wave (and
    waiting for each wave to finish) satisfies every `depends_on` edge.
    """
    count = len(actions)
    if count == 0:
        return []
    remaining = {index: _dependencies(actions[index], count) for index in range(count)}
    done: set[int] = set()
    waves: list[list[int]] = []
    while remaining:
        ready = sorted(index for index, deps in remaining.items() if deps <= done)
        if not ready:
            raise CyclicPlanError(
                "plan dependencies form a cycle among actions "
                + ", ".join(str(index) for index in sorted(remaining))
            )
        waves.append(ready)
        done.update(ready)
        for index in ready:
            del remaining[index]
    return waves


def is_parallel_candidate(action: Action) -> bool:
    """May this action ever share a wave with another one?

    Deliberately conservative, and deliberately the same test the sequential
    speculative-execution path uses: the tool must be self-contained
    (`CONTEXT_INDEPENDENT_TOOLS`), must not be a high-impact or
    confirmation-gated action, and must not be `optional` -- an optional
    action's whole point is that the plan continues past its failure, and the
    previous parallel path excluded it too.
    """
    return (
        action.tool in CONTEXT_INDEPENDENT_TOOLS
        and not action.optional
        and action.risk is ActionRisk.SAFE
    )


def _signature(action: Action) -> tuple:
    """Identity of an action for duplicate detection within one wave."""
    return (action.tool, tuple(sorted((str(k), repr(v)) for k, v in (action.args or {}).items())))


def partition_wave(actions: list[Action], wave: list[int]) -> tuple[list[int], list[int]]:
    """Split one wave into `(parallel, sequential)` index lists.

    An index joins the parallel group only when the action is a parallel
    candidate, needs no exclusive resource already claimed by another member
    of the group, and is not a repeat of one already in the group. Anything
    rejected keeps its relative order in `sequential` and is run one at a
    time, exactly as before.

    A parallel group of fewer than two members is pointless -- the thread
    pool would cost more than it saves -- so it is folded back into the
    sequential list, preserving the original order of the whole wave.
    """
    parallel: list[int] = []
    sequential: list[int] = []
    claimed_resources: set[str] = set()
    seen: set[tuple] = set()
    for index in wave:
        action = actions[index]
        resource = resource_for_tool(action.tool)
        signature = _signature(action)
        if (
            is_parallel_candidate(action)
            and (resource is None or resource not in claimed_resources)
            and signature not in seen
        ):
            parallel.append(index)
            seen.add(signature)
            if resource is not None:
                claimed_resources.add(resource)
        else:
            sequential.append(index)
    if len(parallel) < 2:
        return [], sorted(parallel + sequential)
    return parallel, sequential


def describe_schedule(actions: list[Action]) -> list[dict]:
    """A compact, loggable description of how a plan will be scheduled.

    Used by the runtime's trace output and by the interaction dataset, so the
    recorded plan shows the concurrency that was actually available rather
    than a flat list that hides it.
    """
    schedule = []
    for wave_number, wave in enumerate(build_waves(actions)):
        parallel, sequential = partition_wave(actions, wave)
        schedule.append(
            {
                "wave": wave_number,
                "parallel": [{"index": i, "tool": actions[i].tool} for i in parallel],
                "sequential": [{"index": i, "tool": actions[i].tool} for i in sequential],
            }
        )
    return schedule


def plan_is_chain(actions: list[Action]) -> bool:
    """Is this plan a strict chain (every wave holding exactly one action)?

    The runtime uses this to keep the previous, well-tested sequential code
    path for the plans that gain nothing from scheduling.
    """
    return all(len(wave) == 1 for wave in build_waves(actions))


__all__ = [
    "CyclicPlanError",
    "build_waves",
    "describe_schedule",
    "is_parallel_candidate",
    "partition_wave",
    "plan_is_chain",
]
