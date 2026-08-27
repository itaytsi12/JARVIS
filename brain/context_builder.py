"""Assemble the smallest context that is still sufficient.

The failure mode this module exists to prevent is sending Claude the
whole world: every stored memory, the entire transcript, every prior
episode. That is slow, expensive, and actively worse -- relevant
information gets buried.

So the context is BUDGETED. Sections are added in priority order and the
budget (`JARVIS_CONTEXT_MAX_CHARS`) is enforced as they are added; when a
section does not fit it is truncated or dropped, and the result records
exactly what was included and what was cut, so the decision is
observable rather than silent.

Priority order, highest first:
  1. the request itself and the active task state
  2. skill guidance (how to do this kind of work)
  3. retrieved long-term memories (durable facts about the user)
  4. relevant past episodes (what happened last time)
  5. recent conversation turns (immediate continuity)
  6. system/application context (what is open right now)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from config import get_config
from memory.agent_memory import RetrievedContext
from skills.base import Skill

log = logging.getLogger("jarvis.context")


@dataclass
class ContextSection:
    name: str
    text: str
    priority: int
    included: bool = True
    truncated: bool = False

    @property
    def size(self) -> int:
        return len(self.text)


@dataclass
class BuiltContext:
    system_prompt: str
    user_prompt: str
    sections: list[ContextSection] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    budget_chars: int = 0
    used_chars: int = 0
    dropped: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "budget_chars": self.budget_chars,
            "used_chars": self.used_chars,
            "sections": [
                {"name": section.name, "size": section.size, "included": section.included, "truncated": section.truncated}
                for section in self.sections
            ],
            "dropped": list(self.dropped),
            "skills": list(self.skills),
            "tool_count": len(self.tool_names),
        }


BASE_SYSTEM_PROMPT = """You are JARVIS, a Windows desktop assistant that operates a real computer for one user.

How you work:
- You act by calling the tools you were given. You cannot do anything you have no tool for.
- When several things you need are INDEPENDENT of each other, ask for them in the SAME turn --
  listing a directory, reading a file and checking git status do not depend on one another, and
  requesting them together gets you all the answers in one round trip. Anything whose arguments
  depend on a previous result must wait for that result.
- Otherwise take one step at a time and read the result before deciding the next. Results are real:
  a tool that reports failure genuinely failed, and repeating it unchanged will fail again.
- Start with the most useful action rather than a written plan. A short note about what you are
  about to do is fine; a paragraph of planning before the first tool call is not.
- Verify before you claim. Never report something as done unless a tool result actually showed
  it happened. If you could not verify it, say exactly that.
- Prefer the smallest sequence of steps that achieves the goal. Do not take extra screenshots,
  re-read files you already read, or re-run commands that already succeeded.
- When you are finished, reply with a short, plain answer for the user -- it is going to be read
  aloud. A few sentences; only go longer when the user asked for detail. No JSON, no markdown,
  no tool names, no internal reasoning, no URLs unless the user asked for one.
- If the task cannot be completed, say what you established, what blocked you, and stop.
  A precise account of a failure is more useful than a fabricated success."""


class ContextBuilder:
    def __init__(self, budget_chars: int | None = None):
        config = get_config()
        self.budget_chars = budget_chars or config.context_max_chars
        self.max_observation_chars = config.context_max_observation_chars

    def build(
        self,
        request: str,
        *,
        retrieved: RetrievedContext | None = None,
        skills: Iterable[Skill] = (),
        tool_names: Iterable[str] = (),
        task_goal: str | None = None,
        session_context: Any = None,
        extra: dict[str, str] | None = None,
    ) -> BuiltContext:
        skills = list(skills)
        tool_names = list(tool_names)
        sections: list[ContextSection] = []

        if skills:
            guidance = "\n\n".join(f"## Skill: {skill.name}\n{skill.guidance}" for skill in skills)
            criteria = [skill.completion_criteria for skill in skills if skill.completion_criteria]
            if criteria:
                guidance += "\n\n## What counts as done\n" + "\n".join(f"- {item}" for item in criteria)
            sections.append(ContextSection("skills", guidance, priority=2))

        if retrieved and retrieved.memories:
            lines = [
                f"- ({item.memory.kind}) {item.memory.text}"
                for item in retrieved.memories
            ]
            sections.append(
                ContextSection(
                    "memories",
                    "## What you know about this user\n" + "\n".join(lines),
                    priority=3,
                )
            )

        if retrieved and retrieved.episodes:
            lines = []
            for item in retrieved.episodes:
                episode = item.episode
                outcome = "succeeded" if episode.success else "failed"
                detail = (episode.final_result or "")[:300]
                lines.append(f"- {episode.user_request!r} -> {outcome}. {detail}".rstrip())
            sections.append(
                ContextSection(
                    "episodes",
                    "## Related things you did before\n" + "\n".join(lines),
                    priority=4,
                )
            )

        if retrieved and retrieved.recent_turns:
            lines = [f"{turn.role}: {turn.text[:400]}" for turn in retrieved.recent_turns]
            sections.append(
                ContextSection("conversation", "## Recent conversation\n" + "\n".join(lines), priority=5)
            )

        system_text = _describe_session(session_context)
        if system_text:
            sections.append(ContextSection("system_state", "## Current desktop state\n" + system_text, priority=6))

        for name, text in (extra or {}).items():
            if text:
                sections.append(ContextSection(name, text, priority=7))

        # Reserve room for the base prompt and the request itself; they are
        # never dropped, because without them there is no task at all.
        reserved = len(BASE_SYSTEM_PROMPT) + len(request) + 200
        remaining = max(0, self.budget_chars - reserved)
        dropped: list[str] = []
        kept: list[str] = []
        for section in sorted(sections, key=lambda item: item.priority):
            if section.size <= remaining:
                remaining -= section.size
                kept.append(section.text)
                continue
            if remaining < 200:
                section.included = False
                dropped.append(section.name)
                continue
            section.text = section.text[: remaining - 20].rstrip() + "\n... [truncated]"
            section.truncated = True
            kept.append(section.text)
            remaining = 0

        system_prompt = BASE_SYSTEM_PROMPT
        if tool_names:
            system_prompt += "\n\nTools available to you right now: " + ", ".join(sorted(tool_names)) + "."
        if kept:
            system_prompt += "\n\n" + "\n\n".join(kept)

        user_prompt = request if not task_goal or task_goal == request else f"{task_goal}\n\nCurrent request: {request}"
        built = BuiltContext(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            sections=sections,
            skills=[skill.name for skill in skills],
            tool_names=tool_names,
            budget_chars=self.budget_chars,
            used_chars=len(system_prompt) + len(user_prompt),
            dropped=dropped,
        )
        log.debug("Context built: %s", built.describe())
        return built

    def bound_observation(self, text: str) -> str:
        """Trim one tool observation to fit the per-observation budget.

        Keeps the head AND the tail: a stack trace's most useful lines are
        usually at the end, and a listing's at the start.
        """
        limit = self.max_observation_chars
        if len(text) <= limit:
            return text
        half = limit // 2
        return f"{text[:half]}\n... [{len(text) - limit} characters omitted] ...\n{text[-half:]}"


def _describe_session(session_context: Any) -> str:
    if session_context is None:
        return ""
    parts = []
    for label, attribute in (
        ("Active application", "active_app"),
        ("Last opened application", "last_opened_app"),
        ("Current URL", "current_url"),
        ("Last opened file", "last_opened_file"),
    ):
        value = getattr(session_context, attribute, None)
        if value:
            parts.append(f"- {label}: {value}")
    return "\n".join(parts)
