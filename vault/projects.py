"""Project memory: what JARVIS knows about one body of work.

A project note is the durable half of "what were we doing". It holds the
goal, the architecture, the environment, the commands that actually work
on this machine, the known problems, and what state the work was left in.

The rule that keeps it useful is restraint: **not every event becomes
permanent.** A project note that recorded every command run against a
repository would be a log, and nobody -- including JARVIS -- would read it.
Only these are written:

- a command that was OBSERVED to work (so the next session does not have
  to rediscover the invocation),
- a problem that was hit and is likely to recur,
- a change to the current state worth resuming from,
- an explicit decision.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from vault.consolidation import integrate_rule
from vault.index import VaultIndex, get_index
from vault.manager import VaultManager
from vault.note import PROJECT, Note, extract_section, replace_section
from vault.paths import PROJECTS_DIR, slugify

log = logging.getLogger("jarvis.vault.projects")

SECTION_STATE = "Current State"
SECTION_PROBLEMS = "Known Problems"
SECTION_COMMANDS = "Successful Commands"
SECTION_DECISIONS = "Decisions"
SECTION_FILES = "Important Files"
SECTION_ENVIRONMENT = "Environment"
SECTION_UNRESOLVED = "Unresolved Tasks"
SECTION_RECENT = "Recent Work"

PROJECT_SECTIONS = (
    "Goal",
    "Architecture",
    "Technologies",
    SECTION_STATE,
    SECTION_FILES,
    SECTION_ENVIRONMENT,
    SECTION_COMMANDS,
    SECTION_PROBLEMS,
    SECTION_DECISIONS,
    SECTION_UNRESOLVED,
    SECTION_RECENT,
    "Related Jobs",
)


class ProjectMemory:
    """Find, create and carefully update project notes."""

    def __init__(self, vault: VaultManager | None = None, index: VaultIndex | None = None):
        self.index = index or get_index(vault)
        self.vault = vault or self.index.vault

    def all(self) -> list[Note]:
        return [note for note in self.vault.iter_notes(PROJECTS_DIR) if note.note_type == PROJECT]

    def find(self, name_or_path: str) -> Note | None:
        """Resolve a project by note title, filename, or filesystem path.

        The filesystem-path case is what makes this usable from an agent
        run: the mission knows it is working in
        `C:\\Users\\Ori\\Desktop\\jarvis`, not that the note is called
        "JARVIS Project".
        """
        if not name_or_path:
            return None
        summary = self.index.find_by_title(name_or_path)
        if summary is not None and summary.note_type == PROJECT:
            return self.vault.read(summary.relative_path)
        needle = str(name_or_path).replace("\\", "/").rstrip("/").lower()
        stem = Path(needle).name
        for note in self.all():
            recorded = str(note.metadata.get("path") or "").replace("\\", "/").rstrip("/").lower()
            if recorded and (recorded == needle or needle.startswith(recorded + "/")):
                return note
            if stem and stem in note.title.lower():
                return note
        return None

    def ensure(self, name: str, *, path: str = "", summary: str = "") -> Note:
        existing = self.find(path or name)
        if existing is not None:
            return existing
        relative = self.vault.unique_path(PROJECTS_DIR, name, fallback="project")
        note = self.vault.create_note(
            relative,
            title=name,
            note_type=PROJECT,
            summary=summary or f"Knowledge about the {name} project: its goal, state, environment and known problems.",
            tags=["project", slugify(name)],
            quick_summary=[summary or f"Project note for {name}.", f"Location: {path}" if path else "Location not recorded."],
            sections=[(heading, "") for heading in PROJECT_SECTIONS],
            extra_metadata={"path": path} if path else None,
        )
        self.index.invalidate()
        self.index.refresh()
        log.info("Project note created: %s", relative)
        return note

    # ------------------------------------------------------ focused edits
    def record_working_command(self, project: str, command: str, *, note_text: str = "") -> Note | None:
        """A command that was OBSERVED to succeed on this machine."""
        target = self.find(project)
        if target is None or not command.strip():
            return None
        entry = f"`{command.strip()}`" + (f" -- {note_text.strip()}" if note_text.strip() else "")
        result = integrate_rule(self.vault, target.relative_path, SECTION_COMMANDS, entry)
        self.index.invalidate()
        self.index.refresh()
        return result.note

    def record_problem(self, project: str, problem: str) -> Note | None:
        target = self.find(project)
        if target is None or not problem.strip():
            return None
        result = integrate_rule(self.vault, target.relative_path, SECTION_PROBLEMS, problem.strip())
        self.index.invalidate()
        self.index.refresh()
        return result.note

    def record_decision(self, project: str, decision: str) -> Note | None:
        target = self.find(project)
        if target is None or not decision.strip():
            return None
        result = integrate_rule(self.vault, target.relative_path, SECTION_DECISIONS, decision.strip())
        self.index.invalidate()
        self.index.refresh()
        return result.note

    def set_state(self, project: str, state: str) -> Note | None:
        """Replace the current-state section.

        Replaced, not appended: "where the work stands" has exactly one
        current answer, and a growing list of past states is what the
        Daily Notes are for.
        """
        target = self.find(project)
        if target is None:
            return None

        def mutate(note: Note) -> Note:
            note.body = replace_section(note.body, SECTION_STATE, state.strip())
            return note

        updated = self.vault.update_note(target.relative_path, mutate)
        self.index.invalidate()
        self.index.refresh()
        return updated

    def record_recent_work(self, project: str, text: str, *, keep: int = 8) -> Note | None:
        """Append to a BOUNDED recent-work list.

        Bounded on purpose: the point of this section is "what has been
        happening lately", and an unbounded list stops answering that.
        The full history is in the Daily Notes, which is where it belongs.
        """
        target = self.find(project)
        if target is None or not text.strip():
            return None

        def mutate(note: Note) -> Note:
            existing = extract_section(note.body, SECTION_RECENT)
            lines = [line for line in existing.splitlines() if line.strip().startswith("- ")]
            entry = f"- {text.strip()}"
            if entry in lines:
                return note
            lines.append(entry)
            note.body = replace_section(note.body, SECTION_RECENT, "\n".join(lines[-keep:]))
            return note

        updated = self.vault.update_note(target.relative_path, mutate)
        self.index.invalidate()
        self.index.refresh()
        return updated

    def record_unresolved(self, project: str, text: str) -> Note | None:
        target = self.find(project)
        if target is None or not text.strip():
            return None
        result = integrate_rule(self.vault, target.relative_path, SECTION_UNRESOLVED, text.strip())
        self.index.invalidate()
        self.index.refresh()
        return result.note


_MEMORY: ProjectMemory | None = None


def get_project_memory(vault: VaultManager | None = None, index: VaultIndex | None = None) -> ProjectMemory:
    global _MEMORY
    if vault is not None or index is not None:
        return ProjectMemory(vault=vault, index=index)
    if _MEMORY is None:
        _MEMORY = ProjectMemory()
    return _MEMORY


def reset_project_memory() -> None:
    global _MEMORY
    _MEMORY = None
