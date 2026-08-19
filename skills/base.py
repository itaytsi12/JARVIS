"""Skills: reusable capabilities that sit above individual tools.

A tool is one operation. A skill is *how to accomplish a kind of task*:
which tools belong together, in what order they are usually useful, and
what counts as done. Skills exist so the agent is told "here is how you
debug a project" once, instead of the runtime growing a giant if/else
tree of special cases.

A skill contributes three things to a request:

- **tools**: the subset of the catalog it needs, so the model is offered
  a coherent, small toolset instead of everything;
- **guidance**: a short instruction block appended to the system prompt;
- **selection signals**: keywords and an optional predicate, used to pick
  the relevant skills for a goal without a model call.

Skills never execute anything themselves -- execution stays in the tool
catalog and the agent runtime, so there is exactly one execution path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from brain.tool_catalog import ToolCatalog, ToolDefinition, get_tool_catalog


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    guidance: str
    tool_categories: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    completion_criteria: str = ""
    matcher: Callable[[str], bool] | None = field(default=None, repr=False)

    def tools(self, catalog: ToolCatalog | None = None) -> list[ToolDefinition]:
        catalog = catalog or get_tool_catalog()
        found: dict[str, ToolDefinition] = {}
        if self.tool_categories:
            # Only when categories are declared. Passing an empty tuple
            # through as `None` would mean "no filter" and silently hand
            # the skill the ENTIRE catalog, which is the opposite of the
            # point: a skill exists to offer a small, coherent toolset.
            for definition in catalog.select(categories=self.tool_categories):
                found[definition.name] = definition
        for name in self.tool_names:
            definition = catalog.get(name)
            if definition is not None:
                found[name] = definition
        return list(found.values())

    def relevance(self, goal: str) -> float:
        """How well this skill fits `goal`, in [0, 1]. No model call."""
        text = (goal or "").lower()
        if self.matcher is not None and self.matcher(text):
            return 1.0
        if not self.keywords:
            return 0.0
        hits = sum(1 for keyword in self.keywords if re.search(rf"\b{re.escape(keyword)}\b", text))
        return min(1.0, hits / 2) if hits else 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tool_categories": list(self.tool_categories),
            "tool_names": list(self.tool_names),
            "keywords": list(self.keywords),
            "completion_criteria": self.completion_criteria,
        }


class SkillRegistry:
    """Discovery and selection over the available skills."""

    def __init__(self, skills: Iterable[Skill] = ()):
        self._skills: dict[str, Skill] = {skill.name: skill for skill in skills}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def names(self) -> list[str]:
        return sorted(self._skills)

    def select(self, goal: str, *, limit: int = 2, min_relevance: float = 0.5) -> list[Skill]:
        """The skills relevant to `goal`, best first.

        Returns an empty list when nothing clearly applies; the caller
        then falls back to a general toolset rather than being forced
        into a skill that does not fit.
        """
        scored = [(skill.relevance(goal), skill) for skill in self._skills.values()]
        scored = [(score, skill) for score, skill in scored if score >= min_relevance]
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [skill for _, skill in scored[: max(0, limit)]]

    def catalog_summary(self) -> str:
        """A compact description of every skill, for the system prompt."""
        return "\n".join(f"- {skill.name}: {skill.description}" for skill in sorted(self._skills.values(), key=lambda s: s.name))
