"""Priming: onboarding JARVIS before it starts each piece of real work.

This is the heart of the architecture. Given a mission, and BEFORE any
difficult work begins:

     1. understand the mission
     2. scan the vault index (summaries only)
     3. identify the relevant Job
     4. identify the relevant project
     5. identify the relevant Skills -- the Job's own, plus any that fit
     6. identify the relevant preferences
     7. identify the relevant Lessons
     8. deep-read only the selected notes
     9. build a BOUNDED working context
    10. hand it to the existing agent loop

Two things make this cheap. First, steps 2-7 read no note bodies at all;
they rank metadata. Second, step 9 has a hard character budget, and the
sections are added in priority order so that when the budget runs out it
is the least important knowledge that is dropped -- and the result records
exactly what was dropped, so the decision is observable.

What comes back is a `PrimedContext`: text blocks keyed by name, ready to
be handed to `brain/context_builder.py::ContextBuilder.build(extra=...)`.
That is deliberate -- vault knowledge goes through the SAME budgeting,
truncation and reporting machinery as memories, episodes and the
conversation, rather than being a second, unbudgeted context path.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from vault.daily import DailyJournal, get_journal
from vault.index import VaultIndex, get_index
from vault.jobs import Job, JobRegistry, get_job_registry
from vault.manager import VaultManager
from vault.note import DAILY, IDENTITY, LESSON, MISSION, PROJECT, USER, Note
from vault.retrieval import RetrievalTrace, VaultRetriever, get_retriever
from vault.skills import SkillLibrary, VaultSkill, get_skill_library

log = logging.getLogger("jarvis.vault.priming")

#: Notes read on every primed mission. They are what makes JARVIS itself
#: rather than a search engine over Markdown, and they are small.
ALWAYS_READ = ("identity/core_rules.md", "user/preferences.md")

#: Priority order for the priming sections, lowest number first. This is
#: the order the budget is spent in: identity and the Job survive a tight
#: budget; recent-day continuity is the first thing dropped.
_SECTION_PRIORITY = (
    "vault_identity",
    "vault_job",
    "vault_skills",
    "vault_project",
    "vault_preferences",
    "vault_lessons",
    "vault_mission",
    "vault_continuity",
)


@dataclass
class PrimedContext:
    """The bounded knowledge one mission starts with."""

    request: str
    job: Job | None = None
    skills: list[VaultSkill] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    project: Note | None = None
    notes_read: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)
    trace: RetrievalTrace | None = None
    budget_chars: int = 0
    used_chars: int = 0
    dropped: list[str] = field(default_factory=list)
    scanned: int = 0

    @property
    def job_title(self) -> str:
        return self.job.title if self.job else ""

    @property
    def skill_titles(self) -> list[str]:
        return [skill.title for skill in self.skills]

    def extra_sections(self) -> dict[str, str]:
        """The payload for `ContextBuilder.build(extra=...)`."""
        return {name: text for name, text in self.sections.items() if text.strip()}

    def describe(self) -> dict[str, Any]:
        return {
            "job": self.job_title or None,
            "skills": self.skill_titles,
            "missing_skills": list(self.missing_skills),
            "project": self.project.title if self.project else None,
            "notes_read": list(self.notes_read),
            "scanned_notes": self.scanned,
            "used_chars": self.used_chars,
            "budget_chars": self.budget_chars,
            "dropped": list(self.dropped),
            "sections": sorted(self.sections),
        }

    def explain(self) -> str:
        """The human-readable account, logged once per primed mission."""
        lines = [f'Priming for: "{self.request[:120]}"']
        if self.trace is not None:
            lines.append(self.trace.explain())
        lines.append(f"Job selected: {self.job_title or '(none -- no Job fitted this request)'}")
        lines.append(f"Skills loaded: {', '.join(self.skill_titles) or '(none)'}")
        if self.missing_skills:
            lines.append(f"Skills named but NOT found in the vault: {', '.join(self.missing_skills)}")
        lines.append(f"Project: {self.project.title if self.project else '(none)'}")
        lines.append(f"Notes read in full: {', '.join(self.notes_read) or '(none)'}")
        lines.append(f"Context used: {self.used_chars}/{self.budget_chars} characters" + (f"; dropped {', '.join(self.dropped)}" if self.dropped else ""))
        return "\n".join(lines)


class Primer:
    """Builds a `PrimedContext` for one request."""

    def __init__(
        self,
        vault: VaultManager | None = None,
        index: VaultIndex | None = None,
        retriever: VaultRetriever | None = None,
        jobs: JobRegistry | None = None,
        skills: SkillLibrary | None = None,
        journal: DailyJournal | None = None,
    ):
        self.index = index or get_index(vault)
        self.vault = vault or self.index.vault
        self.retriever = retriever or get_retriever(index=self.index, vault=self.vault)
        self.jobs = jobs or get_job_registry(index=self.index, vault=self.vault)
        self.skills = skills or get_skill_library(index=self.index, vault=self.vault)
        self.journal = journal or get_journal(vault=self.vault, index=self.index)

    def prime(
        self,
        request: str,
        *,
        budget_chars: int = 6000,
        include_continuity: bool = True,
        mission_brief: str = "",
    ) -> PrimedContext:
        primed = PrimedContext(request=request, budget_chars=budget_chars)

        # -- stage 1: scan everything's metadata, read nothing ----------
        #
        # Missions and daily notes are RECORDS of past work, not knowledge
        # about how to do it. A completed mission note is close to a
        # verbatim copy of the request, so it outscores every genuine
        # Skill on a repeat of that request (48.6 against 22.0, observed)
        # and would crowd the real knowledge out of a tight budget. They
        # stay fully searchable through `vault_search` and are how the
        # continuity section is built -- they are simply not candidates
        # for "what does JARVIS need to know to do this".
        candidates, scanned, scan_ms = self.retriever.scan(request, exclude_types=(MISSION, DAILY))
        trace = RetrievalTrace(query=request, scanned=scanned, candidates=candidates[:20], scan_ms=scan_ms, budget_chars=budget_chars)
        primed.scanned = scanned

        # -- stage 2: decide what is worth reading ---------------------
        job = self.jobs.select(request)
        primed.job = job

        skill_titles: list[str] = []
        if job is not None:
            skill_titles = job.required_skills
        loaded, missing = self.skills.load_all(skill_titles)
        if len(loaded) < 2:
            # Either no Job was selected, or the Job named few Skills. A
            # direct Skill scan is the fallback -- a mission with no Skill
            # loaded is one performing from first principles, which is the
            # exact cost this system exists to avoid.
            for extra in self.skills.select(request, limit=3 - len(loaded)):
                if extra.relative_path not in {item.relative_path for item in loaded}:
                    loaded.append(extra)
        primed.skills = loaded
        primed.missing_skills = missing

        project = self._select_project(candidates)
        primed.project = project

        lessons = self._select(candidates, LESSON, limit=2)
        preferences = self._select(candidates, USER, limit=2)

        # -- stage 3: deep-read, in priority order, within the budget ---
        blocks: list[tuple[str, str, str]] = []  # (section, path, text)

        identity = self.vault.read("identity/core_rules.md")
        if identity is not None:
            blocks.append(("vault_identity", identity.relative_path, f"## JARVIS core rules\n{identity.section('Rules') or identity.body.strip()}"))

        if job is not None:
            blocks.append(("vault_job", job.relative_path, job.guidance()))
            for candidate in trace.candidates:
                if candidate.relative_path == job.relative_path:
                    candidate.selected = True

        for skill in loaded:
            blocks.append(("vault_skills", skill.relative_path, skill.guidance()))

        if project is not None:
            blocks.append(("vault_project", project.relative_path, _project_block(project)))

        preference_note = self.vault.read("user/preferences.md")
        if preference_note is not None:
            text = preference_note.section("Preferences") or preference_note.quick_summary
            if text:
                blocks.append(("vault_preferences", preference_note.relative_path, f"## What the user prefers\n{text}"))
        for note in preferences:
            if note.relative_path != "user/preferences.md":
                blocks.append(("vault_preferences", note.relative_path, f"## {note.title}\n{note.quick_summary or note.summary}"))

        for note in lessons:
            blocks.append(("vault_lessons", note.relative_path, f"## Lesson: {note.title}\n{note.quick_summary or note.summary}"))

        if mission_brief:
            blocks.append(("vault_mission", "", f"## This mission so far\n{mission_brief}"))

        if include_continuity:
            continuity = self._continuity()
            if continuity:
                blocks.append(("vault_continuity", "", continuity))

        order = {name: position for position, name in enumerate(_SECTION_PRIORITY)}
        blocks.sort(key=lambda item: order.get(item[0], 99))

        used = 0
        sections: dict[str, list[str]] = {}
        for name, path, text in blocks:
            if not text.strip():
                continue
            cost = len(text)
            if used + cost > budget_chars:
                remaining = budget_chars - used
                if remaining < 400:
                    primed.dropped.append(path or name)
                    continue
                text = text[: remaining - 40].rstrip() + "\n... [truncated to fit the vault priming budget]"
                cost = len(text)
            sections.setdefault(name, []).append(text)
            used += cost
            if path:
                primed.notes_read.append(path)

        primed.sections = {name: "\n\n".join(parts) for name, parts in sections.items()}
        primed.used_chars = used
        trace.selected = list(primed.notes_read)
        trace.deep_read_chars = used
        trace.dropped_for_budget = list(primed.dropped)
        primed.trace = trace

        log.info("Vault priming: %s", primed.describe())
        return primed

    # ------------------------------------------------------------ parts
    def _select(self, candidates, note_type: str, *, limit: int, min_score: float = 1.0) -> list[Note]:
        found: list[Note] = []
        for candidate in candidates:
            if len(found) >= limit:
                break
            if candidate.summary.note_type != note_type or candidate.score < min_score:
                continue
            note = self.vault.read(candidate.relative_path)
            if note is not None:
                found.append(note)
                candidate.selected = True
        return found

    def _select_project(self, candidates) -> Note | None:
        found = self._select(candidates, PROJECT, limit=1, min_score=0.9)
        return found[0] if found else None

    def _continuity(self) -> str:
        """Today's and the previous day's context, summaries only.

        This is what makes "carry on with what we were doing yesterday" a
        question JARVIS can answer. It reads the Quick Summary and the
        unfinished-work section, never the whole day.
        """
        parts: list[str] = []
        today = self.journal.existing(self.journal.today().date)
        if today is not None:
            brief = today.brief(max_chars=700)
            if brief:
                parts.append(brief)
        previous = self.journal.yesterday()
        if previous is not None:
            brief = previous.brief(max_chars=700)
            if brief:
                parts.append(brief)
        return ("## Recent days\n" + "\n\n".join(parts)) if parts else ""


_PRIMER: Primer | None = None


def get_primer(vault: VaultManager | None = None, index: VaultIndex | None = None) -> Primer:
    global _PRIMER
    if vault is not None or index is not None:
        return Primer(vault=vault, index=index)
    if _PRIMER is None:
        _PRIMER = Primer()
    return _PRIMER


def reset_primer() -> None:
    global _PRIMER
    _PRIMER = None


def _project_block(note: Note) -> str:
    parts = [f"## Project: {note.title}", note.summary]
    for heading in ("Current State", "Known Problems", "Successful Commands", "Environment", "Important Files"):
        text = note.section(heading).strip()
        if text and not text.startswith("_Nothing recorded"):
            parts.append(f"### {heading}\n{text}")
    return "\n\n".join(part for part in parts if part.strip())
