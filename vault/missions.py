"""Missions: one concrete piece of work, recorded in Markdown as it happens.

A Mission is not a plan object held in memory. It is a note under
`missions/active/`, written to disk BEFORE the work starts and appended to
as the work proceeds. That is what makes "Jarvis, I'm going to sleep, run
the Clipping Job tonight" survivable: if the process dies at 3am, the
mission note on disk still says what was done, what failed, what was
discovered and which step was next, and `resumable_missions()` finds it
on the next start.

Why Markdown rather than a table in SQLite: the user has to be able to
open a running mission in Obsidian at 3am and read what JARVIS is doing.
A row in a database is not that. The mission note is also where the
mission's OWN discoveries live before they are promoted into a Skill.

Lifecycle:

    missions/active/2026-09-02-fix-the-import-error.md     status: active
                    |
                    +-- appended to throughout execution
                    v
    missions/completed/2026-09-02-fix-the-import-error.md  status: completed
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from vault.index import VaultIndex, get_index
from vault.manager import VaultManager
from vault.note import (
    MISSION,
    Note,
    extract_section,
    replace_section,
    today_stamp,
    utc_now,
)
from vault.paths import MISSIONS_ACTIVE_DIR, MISSIONS_COMPLETED_DIR, slugify

log = logging.getLogger("jarvis.vault.missions")

ACTIVE = "active"
COMPLETED = "completed"
FAILED = "failed"
ABANDONED = "abandoned"
#: A mission whose process died. Distinct from `failed`, which means the
#: work was attempted and did not succeed -- this one means nobody knows.
INTERRUPTED = "interrupted"

TERMINAL_STATUSES = frozenset({COMPLETED, FAILED, ABANDONED})

#: The section contract of a mission note.
SECTION_REQUEST = "Original Request"
SECTION_KNOWLEDGE = "Knowledge Loaded"
SECTION_PLAN = "Plan"
SECTION_PROGRESS = "Progress"
SECTION_FAILURES = "Failures And Retries"
SECTION_DISCOVERIES = "Discoveries"
SECTION_ARTIFACTS = "Artifacts"
SECTION_OUTCOME = "Outcome"

MISSION_SECTIONS = (
    SECTION_REQUEST,
    SECTION_KNOWLEDGE,
    SECTION_PLAN,
    SECTION_PROGRESS,
    SECTION_FAILURES,
    SECTION_DISCOVERIES,
    SECTION_ARTIFACTS,
    SECTION_OUTCOME,
)


@dataclass
class Mission:
    """A handle on one mission note. Every mutation writes to disk."""

    mission_id: str
    relative_path: str
    vault: VaultManager
    index: VaultIndex | None = None
    _note: Note | None = field(default=None, repr=False)

    # -- reads ---------------------------------------------------------
    def note(self, *, refresh: bool = True) -> Note | None:
        if refresh or self._note is None:
            self._note = self.vault.read(self.relative_path)
        return self._note

    @property
    def title(self) -> str:
        note = self.note(refresh=False)
        return note.title if note else self.mission_id

    @property
    def status(self) -> str:
        note = self.note()
        return (note.status if note else "") or ACTIVE

    @property
    def goal(self) -> str:
        note = self.note(refresh=False)
        if note is None:
            return ""
        return str(note.metadata.get("goal") or extract_section(note.body, SECTION_REQUEST)).strip()

    @property
    def job(self) -> str:
        note = self.note(refresh=False)
        return str((note.metadata.get("job") if note else "") or "")

    @property
    def skills(self) -> list[str]:
        note = self.note(refresh=False)
        value = note.metadata.get("skills") if note else None
        return [str(item) for item in value] if isinstance(value, list) else []

    @property
    def current_step(self) -> str:
        note = self.note(refresh=False)
        return str((note.metadata.get("current_step") if note else "") or "")

    def section(self, heading: str) -> str:
        note = self.note(refresh=False)
        return note.section(heading) if note else ""

    # -- writes --------------------------------------------------------
    def _mutate(self, mutate, **metadata: Any) -> Note | None:
        def apply(note: Note) -> Note:
            changed = mutate(note) if mutate is not None else note
            if metadata:
                merged = dict(changed.metadata)
                merged.update({key: value for key, value in metadata.items() if value is not None})
                changed.metadata = merged
            return changed

        self._note = self.vault.update_note(self.relative_path, apply)
        return self._note

    def set_plan(self, steps: Iterable[str]) -> None:
        lines = [f"{position}. {str(step).strip()}" for position, step in enumerate(steps, start=1) if str(step).strip()]
        if not lines:
            return
        self._mutate(lambda note: _set(note, SECTION_PLAN, "\n".join(lines)))

    def record_knowledge(self, *, job: str = "", skills: Iterable[str] = (), notes: Iterable[str] = (), rationale: str = "") -> None:
        """What the priming step actually loaded, as wikilinks.

        Recorded on the mission itself so "which notes did JARVIS read for
        this?" is answerable a week later from the vault alone, without a
        log file.
        """
        lines: list[str] = []
        if job:
            lines.append(f"- Job: [[{job}]]")
        skill_list = [str(item) for item in skills if str(item).strip()]
        if skill_list:
            lines.append("- Skills: " + ", ".join(f"[[{item}]]" for item in skill_list))
        note_list = [str(item) for item in notes if str(item).strip()]
        if note_list:
            lines.append("- Notes read in full: " + ", ".join(f"`{item}`" for item in note_list))
        if rationale:
            lines.append(f"- Selection: {rationale}")
        if not lines:
            return
        self._mutate(
            lambda note: _set(note, SECTION_KNOWLEDGE, "\n".join(lines)),
            job=job or None,
            skills=skill_list or None,
        )

    def append_progress(self, text: str, *, step: str | None = None) -> None:
        """Append one timestamped line of progress, and update `current_step`.

        Called as the work happens, never batched to the end -- a crash
        must not erase what was already done.
        """
        entry = f"- `{datetime.now().strftime('%H:%M:%S')}` {text.strip()}"
        self._mutate(lambda note: _append(note, SECTION_PROGRESS, entry), current_step=step)

    def append_failure(self, text: str) -> None:
        entry = f"- `{datetime.now().strftime('%H:%M:%S')}` {text.strip()}"
        self._mutate(lambda note: _append(note, SECTION_FAILURES, entry))

    def append_discovery(self, text: str) -> None:
        """Something learned during this mission.

        Discoveries land here first and are only promoted into a Skill or
        a Lesson afterwards, deliberately: not every observation during a
        mission is durable knowledge, and a Skill note that absorbed every
        one of them would stop being worth reading.
        """
        entry = f"- {text.strip()}"
        self._mutate(lambda note: _append(note, SECTION_DISCOVERIES, entry))

    def append_artifact(self, path: str, *, description: str = "") -> None:
        entry = f"- `{path}`" + (f" -- {description}" if description else "")
        self._mutate(lambda note: _append(note, SECTION_ARTIFACTS, entry))

    def complete(self, *, success: bool, outcome: str, verified: bool = False) -> Note | None:
        """Finish the mission and move it to `missions/completed/`.

        A failed mission is moved too. A record of what did NOT work is
        exactly as valuable as one of what did, and leaving it in
        `active/` would make the active list meaningless.
        """
        status = COMPLETED if success else FAILED
        text = (
            f"**{'Succeeded' if success else 'Did not succeed'}"
            f"{' (verified)' if verified else ''}** at {utc_now()}.\n\n{outcome.strip()}"
        )
        self._mutate(
            lambda note: _set(note, SECTION_OUTCOME, text),
            status=status,
            completed=utc_now(),
            verified=verified,
            success=success,
        )
        destination = f"{MISSIONS_COMPLETED_DIR}/{self.relative_path.rsplit('/', 1)[-1]}"
        moved = self.vault.move_note(self.relative_path, destination)
        if moved is not None:
            self.relative_path = moved.relative_path
            self._note = moved
        if self.index is not None:
            self.index.invalidate()
            self.index.refresh()
        log.info("Mission %s -> %s (%s)", self.mission_id, status, self.relative_path)
        return self._note

    def mark_interrupted(self, reason: str = "The process stopped before this mission finished.") -> None:
        """Mark a mission whose process died, without moving it.

        It stays in `active/` on purpose: it is a candidate for resumption,
        and moving it to `completed/` would claim an outcome nobody
        observed.
        """
        self._mutate(lambda note: _append(note, SECTION_FAILURES, f"- {reason}"), status=INTERRUPTED)

    def resume(self) -> None:
        self._mutate(
            lambda note: _append(note, SECTION_PROGRESS, f"- `{datetime.now().strftime('%H:%M:%S')}` Mission resumed."),
            status=ACTIVE,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "path": self.relative_path,
            "title": self.title,
            "status": self.status,
            "job": self.job,
            "skills": self.skills,
            "current_step": self.current_step,
        }

    def brief(self) -> str:
        """A short account of this mission, for a resuming session."""
        progress = self.section(SECTION_PROGRESS).strip()
        tail = "\n".join(progress.splitlines()[-6:]) if progress and not progress.startswith("_Nothing") else "(nothing recorded yet)"
        return (
            f"Mission {self.mission_id} ({self.status}): {self.goal}\n"
            f"Job: {self.job or '(none)'}\n"
            f"Current step: {self.current_step or '(unknown)'}\n"
            f"Recent progress:\n{tail}"
        )


def _set(note: Note, heading: str, text: str) -> Note:
    note.body = replace_section(note.body, heading, text)
    return note


def _append(note: Note, heading: str, entry: str) -> Note:
    existing = extract_section(note.body, heading)
    merged = entry if not existing or existing.startswith("_Nothing recorded") else f"{existing.rstrip()}\n{entry}"
    note.body = replace_section(note.body, heading, merged)
    return note


class MissionStore:
    """Create, find and resume missions."""

    def __init__(self, vault: VaultManager | None = None, index: VaultIndex | None = None):
        self.index = index or get_index(vault)
        self.vault = vault or self.index.vault

    def create(
        self,
        request: str,
        *,
        job: str = "",
        skills: Iterable[str] = (),
        project: str = "",
        mission_id: str | None = None,
        task_id: str | None = None,
    ) -> Mission:
        mission_id = mission_id or uuid.uuid4().hex[:12]
        headline = _headline(request)
        filename = f"{today_stamp()}-{slugify(headline, fallback='mission')}"
        path = f"{MISSIONS_ACTIVE_DIR}/{filename}.md"
        counter = 2
        while self.vault.note_exists(path):
            path = f"{MISSIONS_ACTIVE_DIR}/{filename}-{counter}.md"
            counter += 1

        skill_list = [str(item) for item in skills if str(item).strip()]
        self.vault.create_note(
            path,
            title=f"Mission: {headline}",
            note_type=MISSION,
            summary=f"Mission {mission_id}: {headline}",
            tags=["mission", *([slugify(job)] if job else [])],
            quick_summary=[
                f"Status: active. Started {utc_now()}.",
                f"Job: {job or '(none selected)'}.",
                f"Request: {request.strip()[:200]}",
            ],
            sections=[
                (SECTION_REQUEST, request.strip()),
                (SECTION_KNOWLEDGE, ""),
                (SECTION_PLAN, ""),
                (SECTION_PROGRESS, ""),
                (SECTION_FAILURES, ""),
                (SECTION_DISCOVERIES, ""),
                (SECTION_ARTIFACTS, ""),
                (SECTION_OUTCOME, ""),
            ],
            extra_metadata={
                "status": ACTIVE,
                "mission_id": mission_id,
                "goal": request.strip()[:300],
                "job": job,
                "skills": skill_list,
                "project": project,
                "task_id": task_id or "",
                "started": utc_now(),
            },
        )
        self.index.refresh()
        log.info("Mission %s created at %s", mission_id, path)
        return Mission(mission_id=mission_id, relative_path=path, vault=self.vault, index=self.index)

    def load(self, mission_id_or_path: str) -> Mission | None:
        if not mission_id_or_path:
            return None
        if self.vault.note_exists(mission_id_or_path):
            note = self.vault.read(mission_id_or_path)
            if note is not None and note.note_type == MISSION:
                return Mission(
                    mission_id=str(note.metadata.get("mission_id") or ""),
                    relative_path=note.relative_path,
                    vault=self.vault,
                    index=self.index,
                    _note=note,
                )
        for note in self.vault.iter_notes(MISSIONS_ACTIVE_DIR):
            if str(note.metadata.get("mission_id") or "") == mission_id_or_path:
                return Mission(mission_id=mission_id_or_path, relative_path=note.relative_path, vault=self.vault, index=self.index, _note=note)
        for note in self.vault.iter_notes(MISSIONS_COMPLETED_DIR):
            if str(note.metadata.get("mission_id") or "") == mission_id_or_path:
                return Mission(mission_id=mission_id_or_path, relative_path=note.relative_path, vault=self.vault, index=self.index, _note=note)
        return None

    def active(self) -> list[Mission]:
        missions: list[Mission] = []
        for note in self.vault.iter_notes(MISSIONS_ACTIVE_DIR):
            if note.note_type != MISSION:
                continue
            missions.append(
                Mission(
                    mission_id=str(note.metadata.get("mission_id") or ""),
                    relative_path=note.relative_path,
                    vault=self.vault,
                    index=self.index,
                    _note=note,
                )
            )
        missions.sort(key=lambda item: str((item.note(refresh=False) or Note(path=item.vault.root, relative_path="")).metadata.get("started") or ""), reverse=True)
        return missions

    def completed(self, *, limit: int = 20) -> list[Mission]:
        notes = [note for note in self.vault.iter_notes(MISSIONS_COMPLETED_DIR) if note.note_type == MISSION]
        notes.sort(key=lambda note: str(note.metadata.get("completed") or note.updated), reverse=True)
        return [
            Mission(
                mission_id=str(note.metadata.get("mission_id") or ""),
                relative_path=note.relative_path,
                vault=self.vault,
                index=self.index,
                _note=note,
            )
            for note in notes[:limit]
        ]

    def resumable(self) -> list[Mission]:
        """Active missions left behind by a previous process.

        Every mission still in `active/` at startup is by definition one
        that was never completed -- the process that owned it is gone.
        """
        return [mission for mission in self.active() if mission.status in {ACTIVE, INTERRUPTED}]

    def mark_orphans_interrupted(self) -> list[Mission]:
        """At startup, say plainly that these missions were interrupted.

        Called once by the startup recovery. It records the fact rather
        than guessing at an outcome, which is what makes a resumed mission
        trustworthy afterwards.
        """
        orphans = [mission for mission in self.active() if mission.status == ACTIVE]
        for mission in orphans:
            mission.mark_interrupted("The JARVIS process stopped while this mission was still active.")
        return orphans


def _headline(request: str) -> str:
    """A short, human title for a mission, taken from the request itself."""
    text = re.sub(r"\s+", " ", (request or "").strip())
    text = re.sub(r"^(hey\s+)?jarvis[,\s]+", "", text, flags=re.I)
    if len(text) <= 70:
        return text or "Untitled mission"
    cut = text[:70].rsplit(" ", 1)[0]
    return (cut or text[:70]).rstrip(",.;:") + "..."


_STORE: MissionStore | None = None


def get_mission_store(vault: VaultManager | None = None, index: VaultIndex | None = None) -> MissionStore:
    global _STORE
    if vault is not None or index is not None:
        return MissionStore(vault=vault, index=index)
    if _STORE is None:
        _STORE = MissionStore()
    return _STORE


def reset_mission_store() -> None:
    global _STORE
    _STORE = None
