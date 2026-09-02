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

import re
from pathlib import Path

from vault.daily import DailyJournal, get_journal
from vault.index import VaultIndex, get_index
from vault.jobs import Job, JobRegistry, get_job_registry
from vault.manager import VaultManager
from vault.note import DAILY, IDENTITY, LESSON, MISSION, PROJECT, USER, Note
from vault.retrieval import RetrievalTrace, VaultRetriever, get_retriever
from vault.preferences import PreferenceStore, ResolvedPreferences, get_preferences
from vault.skills import SkillLibrary, VaultSkill, get_skill_library

log = logging.getLogger("jarvis.vault.priming")

#: Notes read on every primed mission. They are what makes JARVIS itself
#: rather than a search engine over Markdown, and they are small.
ALWAYS_READ = ("identity/core_rules.md",)

#: Phrases that reach BACKWARDS in time. Recent Daily Notes are loaded
#: only when one of these appears, or when a mission is already running.
#:
#: A Daily Note is a record of what happened, not knowledge about how to
#: work, so loading yesterday into an unrelated task spends budget on
#: noise and invites the model to act on it. "Continue what we were doing
#: yesterday" needs it; "fix the login bug" does not.
_REFERENCES_EARLIER_WORK = re.compile(
    r"\b(yesterday|earlier|before|last (?:time|night|session|week)|previous(?:ly)?|"
    r"continue|carry on|resume|pick up|finish|where (?:we|i|you) (?:left|got|stopped)|"
    r"what (?:we|i|you) (?:were|was) (?:doing|working)|still|again|the other day|"
    r"this morning|recent(?:ly)?|so far|update me|catch me up)\b",
    re.I,
)


def references_earlier_work(request: str) -> bool:
    """Does this request reach back to previous work?

    Deterministic and free -- it runs on every primed mission.
    """
    return bool(_REFERENCES_EARLIER_WORK.search(request or ""))

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
    #: Every project note loaded, in order. A request naming two projects
    #: loads both, and `project` is simply the first of them.
    projects: list[Note] = field(default_factory=list)
    preferences: ResolvedPreferences | None = None
    #: True when recent Daily Notes were deep-read, and why.
    continuity_reason: str = ""
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
            "projects": [note.title for note in self.projects],
            "preferences": self.preferences.describe() if self.preferences else None,
            "continuity": self.continuity_reason or None,
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
        lines.append(f"Projects: {', '.join(note.title for note in self.projects) or '(none)'}")
        if self.preferences is not None:
            described = self.preferences.describe()
            lines.append(
                f"Preferences: {described['global_rules']} global"
                + (f" + {described['job_rules']} for this Job" if described["job_rules"] else "")
                + (f"; {len(described['overridden'])} global rule(s) overridden by the Job" if described["overridden"] else "")
            )
        lines.append(f"Recent daily notes: {self.continuity_reason or 'not loaded (nothing referenced earlier work)'}")
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
        self.preferences = get_preferences(vault=self.vault, index=self.index)

    def prime(
        self,
        request: str,
        *,
        budget_chars: int = 6000,
        include_continuity: bool = True,
        mission_brief: str = "",
        select_job: bool = True,
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
        job = self.jobs.select(request) if select_job else None
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

        # A project named EXPLICITLY is loaded whether or not the task
        # looks like it needs one -- "what's the run command for JARVIS"
        # is a simple request whose answer lives in one specific note.
        # Naming two projects loads both, kept as separate blocks so one
        # project's facts can never be reported as the other's.
        projects = self._mentioned_projects(request)
        if not projects:
            found = self._select_project(candidates)
            projects = [found] if found is not None else []
        primed.projects = projects
        project = projects[0] if projects else None
        primed.project = project

        lessons = self._select(candidates, LESSON, limit=2)
        # Preferences are NOT discovered by scanning. The global note is
        # loaded by policy, because it must apply whether or not its words
        # happen to match the request; the Job's own note is loaded only
        # HERE, after the Job has been selected.
        resolved = self.preferences.resolve(job.title if job is not None else "")
        primed.preferences = resolved

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

        for note in projects:
            blocks.append(("vault_project", note.relative_path, _project_block(note)))

        rendered_preferences = resolved.render()
        if rendered_preferences:
            # Attributed to the note that supplied most of it; every
            # contributing note is recorded below so the mission's own
            # record names the files that shaped the behaviour.
            anchor = resolved.paths[0] if resolved.paths else ""
            blocks.append(("vault_preferences", anchor, rendered_preferences))
            for extra in resolved.paths[1:]:
                primed.notes_read.append(extra)

        for note in lessons:
            blocks.append(("vault_lessons", note.relative_path, f"## Lesson: {note.title}\n{note.quick_summary or note.summary}"))

        if mission_brief:
            blocks.append(("vault_mission", "", f"## This mission so far\n{mission_brief}"))

        # Recent Daily Notes: only when the request actually reaches
        # backwards, or a mission is already in flight. `include_continuity`
        # stays as the caller's veto (a light request never gets it).
        wants_history = references_earlier_work(request) or bool(mission_brief)
        if include_continuity and wants_history:
            continuity = self._continuity()
            if continuity:
                primed.continuity_reason = (
                    "the request refers to earlier work"
                    if references_earlier_work(request)
                    else "a mission is already in progress"
                )
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

    def _mentioned_projects(self, request: str, *, limit: int = 3) -> list[Note]:
        """Project notes whose name the request actually says.

        Matched against each project's TITLE, its filename stem and any
        `aliases` in its frontmatter -- an exact word-boundary match, not
        a fuzzy one, because this bypasses the relevance threshold
        entirely. "Fix the login bug in Northwind" must load Northwind's
        note even though the request is short and mentions no Job.

        Every match is returned, so a request naming two projects loads
        both. They stay separate blocks in the primed context: merging
        them is how Project A's run command gets reported as Project B's.
        """
        text = (request or "").lower()
        if not text:
            return []
        found: list[Note] = []
        for summary in self.index.by_type(PROJECT):
            names = {summary.title.lower(), Path(summary.relative_path).stem.replace("-", " ").lower()}
            note = None
            for alias in self._aliases(summary.relative_path):
                names.add(alias.lower())
            for name in names:
                cleaned = name.replace(" project", "").strip()
                if len(cleaned) < 3:
                    continue
                if re.search(rf"\b{re.escape(cleaned)}\b", text):
                    note = self.vault.read(summary.relative_path)
                    break
            if note is not None and note.relative_path not in {item.relative_path for item in found}:
                found.append(note)
            if len(found) >= limit:
                break
        return found

    def _aliases(self, relative_path: str) -> list[str]:
        note = self.vault.read(relative_path)
        if note is None:
            return []
        value = note.metadata.get("aliases")
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value] if isinstance(value, list) else []

    def _continuity(self) -> str:
        """Today's and the previous day's context, summaries only.

        This is what makes "carry on with what we were doing yesterday" a
        question JARVIS can answer. It reads the Quick Summary and the
        unfinished-work section, never the whole day -- and, per
        `references_earlier_work`, only when the request actually reaches
        backwards. Loading yesterday's work into an unrelated task spends
        budget on noise and invites the model to act on it.
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
