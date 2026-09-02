"""Skills: reusable knowledge about HOW to do something, stored as notes.

A Job says WHAT kind of work this is. A Skill says how the work is
actually performed -- and, crucially, which methods have already been
tried and found to fail. One Skill supports many Jobs: "Test
Verification" belongs to every Job that changes code.

This is a knowledge layer, deliberately separate from the pre-existing
code-level `skills/` package (`skills/base.py`, `skills/builtin.py`),
which selects TOOLSETS. Both survive and both are used: the code skill
decides which tools the model is offered, the vault Skill decides what
the model is told about using them. They are not merged because they
answer different questions and only one of them should be editable by a
correction at runtime.

The most valuable section in a Skill note is `Known Working Method`. It is
where "method A failed, method B failed, method C works" is recorded, and
reading it is how JARVIS avoids paying the discovery tax twice.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from vault.index import NoteSummary, VaultIndex, get_index
from vault.manager import VaultManager
from vault.note import SKILL, Note, extract_section, replace_section, utc_now
from vault.retrieval import VaultRetriever, get_retriever

log = logging.getLogger("jarvis.vault.skills")

SKILL_SECTIONS = ("When To Use", "Procedure", "Known Working Method", "Known Problems", "Lessons Learned")

#: The heading a discovered working method is recorded under.
WORKING_METHOD_HEADING = "Known Working Method"
FAILED_APPROACH_HEADING = "Known Problems"


@dataclass
class VaultSkill:
    """One Skill note, read in full."""

    note: Note
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.note.title

    @property
    def relative_path(self) -> str:
        return self.note.relative_path

    @property
    def summary(self) -> str:
        return self.note.summary

    def section(self, heading: str) -> str:
        return self.note.section(heading)

    @property
    def known_working_method(self) -> str:
        return self.section(WORKING_METHOD_HEADING)

    def guidance(self, *, max_chars: int = 2000) -> str:
        """The Skill rendered for the model's system prompt.

        `Known Working Method` comes FIRST, before the general procedure.
        It is the part that is expensive to rediscover, and putting it
        last is how it ends up being the section a budget truncation
        drops.
        """
        parts = [f"## Skill: {self.title}", self.note.summary]
        for heading in (WORKING_METHOD_HEADING, "Procedure", FAILED_APPROACH_HEADING, "Lessons Learned"):
            text = self.section(heading).strip()
            if not text or text.startswith("_Nothing recorded"):
                continue
            parts.append(f"### {heading}\n{text}")
        rendered = "\n\n".join(parts)
        if len(rendered) > max_chars:
            rendered = rendered[:max_chars].rstrip() + "\n... [Skill note truncated to fit the context budget]"
        return rendered

    def describe(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "path": self.relative_path,
            "score": round(self.score, 3),
            "has_working_method": bool(self.known_working_method and not self.known_working_method.startswith("_Nothing")),
            "why": "; ".join(self.reasons),
        }


class SkillLibrary:
    """Discovery, loading and improvement of the Skill notes."""

    def __init__(self, index: VaultIndex | None = None, vault: VaultManager | None = None, retriever: VaultRetriever | None = None):
        self.index = index or get_index(vault)
        self.vault = vault or self.index.vault
        self.retriever = retriever or get_retriever(index=self.index, vault=self.vault)

    def summaries(self) -> list[NoteSummary]:
        return self.index.by_type(SKILL)

    def titles(self) -> list[str]:
        return sorted(item.title for item in self.summaries())

    def catalog(self) -> str:
        return "\n".join(
            f"- {item.title}: {item.summary}"
            for item in sorted(self.summaries(), key=lambda entry: entry.title.lower())
        )

    def load(self, title_or_path: str) -> VaultSkill | None:
        """One Skill by note title, filename or relative path.

        Title-first, because a Job declares its Skills as wikilinks --
        `[[Python Debugging]]` -- and a wikilink names a title.
        """
        if not title_or_path:
            return None
        summary = self.index.find_by_title(title_or_path)
        if summary is not None and summary.note_type == SKILL:
            note = self.vault.read(summary.relative_path)
            if note is not None:
                return VaultSkill(note=note)
        if self.vault.note_exists(title_or_path):
            note = self.vault.read(title_or_path)
            if note is not None and note.note_type == SKILL:
                return VaultSkill(note=note)
        return None

    def load_all(self, titles: Iterable[str]) -> tuple[list[VaultSkill], list[str]]:
        """Load every named Skill, reporting the ones that do not exist.

        The missing list matters: a Job naming a Skill that was never
        written is a real gap, and silently loading four of five Skills
        would hide it. The Clipping placeholder Job names nine Skills that
        do not exist yet, and this is how that stays visible.
        """
        found: list[VaultSkill] = []
        missing: list[str] = []
        for title in titles:
            skill = self.load(title)
            if skill is None:
                missing.append(title)
            elif skill.relative_path not in {item.relative_path for item in found}:
                found.append(skill)
        return found, missing

    def select(self, request: str, *, limit: int = 3, min_score: float = 1.0) -> list[VaultSkill]:
        """The Skills that fit `request` on their own, best first.

        Used when no Job was selected (so nothing declared its Skills for
        it) and as a supplement when one was.
        """
        candidates, _, _ = self.retriever.scan(request, types=[SKILL])
        skills: list[VaultSkill] = []
        for candidate in candidates:
            if candidate.score < min_score or len(skills) >= limit:
                break
            note = self.vault.read(candidate.relative_path)
            if note is None:
                continue
            skills.append(VaultSkill(note=note, score=candidate.score, reasons=list(candidate.reasons)))
        return skills

    # ------------------------------------------------------ improvement
    def record_working_method(
        self,
        title_or_path: str,
        *,
        method: str,
        failed_attempts: Iterable[str] = (),
        source: str = "",
    ) -> VaultSkill | None:
        """Record a method that was observed to WORK, and what did not.

        This is Milestone 12 -- "do not pay the discovery tax twice" -- and
        it is deliberately append-with-dedup rather than replace: a Skill
        may legitimately have several working methods for different
        situations, and overwriting the previous one loses the situation
        it applied to.
        """
        skill = self.load(title_or_path)
        if skill is None:
            return None
        stamp = utc_now()[:10]
        entry_lines = [f"- **{method.strip()}**", f"  - Confirmed working {stamp}" + (f" ({source})" if source else "") + "."]
        for failure in failed_attempts:
            text = str(failure).strip()
            if text:
                entry_lines.append(f"  - Does NOT work: {text}")
        entry = "\n".join(entry_lines)

        def mutate(note: Note) -> Note:
            existing = extract_section(note.body, WORKING_METHOD_HEADING)
            if method.strip() and method.strip() in existing:
                # Already known. Recording it again would grow the note
                # without adding anything, and a Skill note that grows on
                # every run stops being cheap to read.
                return note
            if not existing or existing.startswith("_Nothing recorded"):
                merged = entry
            else:
                merged = f"{existing.rstrip()}\n{entry}"
            note.body = replace_section(note.body, WORKING_METHOD_HEADING, merged)
            return note

        updated = self.vault.update_note(skill.relative_path, mutate)
        self.index.invalidate()
        self.index.refresh()
        if updated is not None:
            log.info("Skill %s gained a known working method.", skill.title)
            return VaultSkill(note=updated)
        return None

    def record_failed_approach(self, title_or_path: str, *, approach: str, reason: str = "") -> VaultSkill | None:
        """Record an approach that did NOT work, and why."""
        skill = self.load(title_or_path)
        if skill is None or not approach.strip():
            return None
        entry = f"- {approach.strip()}" + (f" -- {reason.strip()}" if reason.strip() else "")

        def mutate(note: Note) -> Note:
            existing = extract_section(note.body, FAILED_APPROACH_HEADING)
            if approach.strip() in existing:
                return note
            merged = entry if not existing or existing.startswith("_Nothing recorded") else f"{existing.rstrip()}\n{entry}"
            note.body = replace_section(note.body, FAILED_APPROACH_HEADING, merged)
            return note

        updated = self.vault.update_note(skill.relative_path, mutate)
        self.index.invalidate()
        self.index.refresh()
        return VaultSkill(note=updated) if updated is not None else None

    def create(
        self,
        title: str,
        *,
        summary: str,
        tags: Iterable[str] = (),
        quick_summary: Iterable[str] = (),
        sections: dict[str, str] | None = None,
    ) -> VaultSkill:
        from vault.paths import SKILLS_DIR

        provided = {key.lower(): value for key, value in (sections or {}).items()}
        ordered = [(heading, provided.get(heading.lower(), "")) for heading in SKILL_SECTIONS]
        path = self.vault.unique_path(SKILLS_DIR, title, fallback="skill")
        note = self.vault.create_note(
            path,
            title=title,
            note_type=SKILL,
            summary=summary,
            tags=sorted({"skill", *[str(tag).lower() for tag in tags]}),
            quick_summary=quick_summary or [summary],
            sections=ordered,
        )
        self.index.invalidate()
        self.index.refresh()
        return VaultSkill(note=note)


_LIBRARY: SkillLibrary | None = None


def get_skill_library(index: VaultIndex | None = None, vault: VaultManager | None = None) -> SkillLibrary:
    global _LIBRARY
    if index is not None or vault is not None:
        return SkillLibrary(index=index, vault=vault)
    if _LIBRARY is None:
        _LIBRARY = SkillLibrary()
    return _LIBRARY


def reset_skill_library() -> None:
    global _LIBRARY
    _LIBRARY = None
