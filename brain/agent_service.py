"""The agent path, wired together.

`run_agent_task(goal)` is the single entry point for a request that needs
real reasoning. It:

  1. retrieves relevant memory (not all of it),
  2. picks the skills that fit the goal (no model call),
  3. builds a budgeted context,
  4. runs the agent loop against whichever provider is configured,
  5. records a complete episode (the training record) and any durable
     memory the exchange produced,
  6. returns a short, speech-safe answer.

Everything except step 4 works with no API key at all, which is why the
no-provider case still produces a real episode and a truthful answer
rather than an exception.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from brain.agent_loop import AgentLoop, AgentRun
from brain.context_builder import ContextBuilder
from brain.models import ToolResult
from brain.tool_catalog import ToolCatalog
from config import get_config
from memory.agent_memory import AgentMemory, get_agent_memory
from memory.episodic import Episode
from providers.registry import get_agent_provider
from providers.usage import TrackedProvider
from skills import get_skill_registry
from skills.base import Skill
from tasks.manager import TaskManager, get_task_manager
from tasks.models import Task, TaskKind

log = logging.getLogger("jarvis.agent")

# Skills whose tools drive the real keyboard/mouse/foreground window. A
# task using one of these must not run concurrently with another such
# task -- see `tasks/manager.py`'s EXCLUSIVE_UI scheduling.
UI_SKILLS = frozenset({"computer_control", "browser"})


@dataclass
class AgentOutcome:
    """What the caller needs, without the internals."""

    answer: str
    success: bool
    verified: bool
    task_id: str
    episode_id: str
    stop_reason: str
    run: AgentRun

    @property
    def used_model(self) -> bool:
        return self.run.model_calls > 0

    def describe(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "success": self.success,
            "verified": self.verified,
            "stop_reason": self.stop_reason,
            **self.run.describe(),
        }


def build_memory_handlers(memory: AgentMemory, task_id: str | None = None) -> dict[str, Callable[[dict], ToolResult]]:
    """In-process handlers for the memory tools.

    They need the LIVE memory object (this session, this task), which a
    module-level dispatch table cannot provide -- hence handlers rather
    than another entry in `brain/tool_router.py`.
    """

    def remember_fact(arguments: dict) -> ToolResult:
        text = str(arguments.get("text") or "").strip()
        if not text:
            return ToolResult(False, "remember_fact", "There was nothing to remember.", {}, "empty_memory")
        record = memory.long_term.remember(
            text,
            kind=str(arguments.get("kind") or "fact"),
            importance=int(arguments.get("importance") or 4),
            source="explicit",
            session_id=memory.session_id,
            task_id=task_id,
        )
        return ToolResult(
            True,
            "remember_fact",
            f"Remembered: {record.text}",
            {"verified": True, "memory_id": record.memory_id, "kind": record.kind},
        )

    def recall_memory(arguments: dict) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        limit = int(arguments.get("limit") or 5)
        found = memory.search_memories(query, limit=limit)
        if not found:
            return ToolResult(True, "recall_memory", "I have nothing stored about that.", {"verified": True, "results": []})
        lines = [f"({item.memory.kind}) {item.memory.text}" for item in found]
        return ToolResult(
            True,
            "recall_memory",
            "\n".join(lines),
            {"verified": True, "results": lines, "scores": [item.score for item in found]},
        )

    return {"remember_fact": remember_fact, "recall_memory": recall_memory}


#: Verbs that mean the task CHANGES something and has to be got right --
#: writing code, fixing a bug, running and interpreting tests, debugging.
#: These get the deeper reasoning budget. Read-only inspection ("list the
#: files", "what does this do", "run git status and tell me") does not need
#: it, and paying for it on every step is a large part of why a simple
#: question felt slow.
_DEMANDING = re.compile(
    r"\b(fix|repair|debug|diagnose|implement|refactor|rewrite|patch|migrate|optimi[sz]e|"
    r"troubleshoot|root\s+cause|why\s+(?:is|does|did|are)|failing|broken|crash\w*|"
    r"crashes|regression|crash|test\s+again|make\s+it\s+work)\b",
    re.I,
)


def select_effort(goal: str, skills: Iterable[Skill] = ()) -> str:
    """How much reasoning effort this task deserves.

    `output_config.effort` controls thinking depth and total token spend per
    turn. The API default is "high", which is right for genuinely hard work
    and wasteful for "tell me what files are in this folder" -- it costs
    thinking tokens, output tokens and wall-clock time on EVERY step.

    So: "high" whenever the goal names work that changes something or has to
    be reasoned out (see `_DEMANDING`), the configured interactive default
    otherwise. Deterministic, offline and logged, so the choice is
    observable rather than a hidden knob -- and it never lowers effort for
    the tasks that need it.
    """
    config = get_config()
    if _DEMANDING.search(goal or ""):
        return config.agent_effort_complex
    return config.agent_effort


def _classify_kind(skills: list[Skill]) -> TaskKind:
    return TaskKind.EXCLUSIVE_UI if any(skill.name in UI_SKILLS for skill in skills) else TaskKind.CONCURRENT


def run_agent_task(
    goal: str,
    *,
    provider: Any | None = None,
    memory: AgentMemory | None = None,
    runtime: Any | None = None,
    catalog: ToolCatalog | None = None,
    task: Task | None = None,
    cancellation_token: Any = None,
    session_context: Any = None,
    progress: Callable[[str, dict], None] | None = None,
    on_answer_text: Callable[[str], None] | None = None,
    record_turns: bool = True,
) -> AgentOutcome:
    """Handle one goal end to end, synchronously.

    Use `submit_agent_task` instead when the caller must not block (voice
    input, or a second task running alongside).
    """
    started = time.perf_counter()
    memory = memory or get_agent_memory()
    config = get_config()
    task_id = task.task_id if task is not None else uuid.uuid4().hex

    if record_turns:
        memory.record_user(goal, task_id=task_id)

    registry = get_skill_registry()
    skills = registry.select(goal)
    retrieved = memory.retrieve(goal)

    catalog = catalog or ToolCatalog(runtime=runtime)
    for name, handler in build_memory_handlers(memory, task_id).items():
        catalog.register_handler(name, handler)

    if skills:
        wanted = {definition.name for skill in skills for definition in skill.tools(catalog)}
        # The memory tools are always offered: any task may discover a
        # durable fact, and any task may need one recalled.
        wanted |= {"remember_fact", "recall_memory"}
        specs = catalog.specs(names=wanted)
    else:
        specs = catalog.specs()

    context = ContextBuilder().build(
        goal,
        retrieved=retrieved,
        skills=skills,
        tool_names=[spec.name for spec in specs],
        task_goal=task.goal if task is not None else None,
        session_context=session_context,
    )

    base_provider = provider if provider is not None else get_agent_provider()
    tracked = (
        TrackedProvider(base_provider, session_id=memory.session_id, task_id=task_id, operation="agent_loop")
        if base_provider is not None
        else None
    )

    if tracked is None:
        run = AgentRun(goal=goal, context=context, skills=[skill.name for skill in skills])
        run.stop_reason = "no_provider"
        run.answer = (
            "I can't reason about that yet, sir. Add ANTHROPIC_API_KEY to enable the agent; "
            "everything local still works."
        )
        run.duration_ms = (time.perf_counter() - started) * 1000
    else:
        effort = select_effort(goal, skills)
        log.info("Agent effort for this task: %s (skills=%s)", effort, [skill.name for skill in skills])
        loop = AgentLoop(tracked, catalog, progress=progress, effort=effort, on_answer_text=on_answer_text)
        run = loop.run(
            goal,
            context=context,
            skills=skills,
            tool_specs=specs,
            cancellation_token=cancellation_token,
            task=task,
        )

    if record_turns and run.answer:
        memory.record_assistant(run.answer, task_id=task_id)
    written = memory.observe_exchange(goal, run.answer, task_id=task_id, record_turns=False)

    episode = Episode(
        episode_id=uuid.uuid4().hex,
        user_request=goal,
        session_id=memory.session_id,
        task_id=task_id,
        route="agent",
        model_used=run.model,
        provider=run.provider,
        context_summary=_context_summary(context),
        plan=[skill.name for skill in skills],
        steps=list(run.steps),
        errors=list(run.errors),
        retries=run.retries,
        final_result=run.answer,
        success=run.success,
        verified=run.verified,
        stop_reason=run.stop_reason,
        duration_ms=run.duration_ms,
        token_usage={**run.usage.to_dict(), "model_calls": run.model_calls},
        estimated_cost_usd=run.estimated_cost_usd,
        memories_written=[record.memory_id for record in written],
        metadata={
            "context": context.describe(),
            "retrieval": retrieved.describe(),
            "cost_tracking_enabled": config.cost_tracking_enabled,
        },
    )
    memory.record_episode(episode)

    outcome = AgentOutcome(
        answer=run.answer,
        success=run.success,
        verified=run.verified,
        task_id=task_id,
        episode_id=episode.episode_id,
        stop_reason=run.stop_reason,
        run=run,
    )
    log.info("Agent task finished: %s", outcome.describe())
    return outcome


def submit_agent_task(
    goal: str,
    *,
    manager: TaskManager | None = None,
    parent_task_id: str | None = None,
    **kwargs: Any,
):
    """Run a goal as a managed background task and return its handle.

    The caller is never blocked. UI-driving goals are submitted as
    `EXCLUSIVE_UI` so two of them can never fight over the keyboard,
    while research/coding goals run concurrently.
    """
    manager = manager or get_task_manager()
    skills = get_skill_registry().select(goal)
    kind = _classify_kind(skills)

    def body(task: Task) -> str:
        outcome = run_agent_task(goal, task=task, cancellation_token=task.token, **kwargs)
        task.metadata["episode_id"] = outcome.episode_id
        task.metadata["stop_reason"] = outcome.stop_reason
        task.metadata["success"] = outcome.success
        return outcome.answer

    return manager.submit(
        goal,
        body,
        kind=kind,
        parent_task_id=parent_task_id,
        plan=[skill.name for skill in skills],
        skills=[skill.name for skill in skills],
    )


def _context_summary(context) -> str:
    described = context.describe()
    return (
        f"skills={described['skills']} tools={described['tool_count']} "
        f"context_chars={described['used_chars']}/{described['budget_chars']} dropped={described['dropped']}"
    )
