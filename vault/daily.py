"""Daily Notes: the chronological record of what actually happened.

The Daily Note is what makes "continue what we were doing yesterday" a
real question with a real answer. It is not a summary written at
shutdown -- it is appended to THROUGHOUT the day, immediately after each
meaningful piece of work, because a crash must not erase the day's
memory. There is no flush, no buffer and no shutdown hook: every append
is an atomic write to disk.

Structure (each heading is appended to independently):

    ## Quick Summary          <- rewritten as the day progresses
    ## Timeline               <- "### 09:42 - <what happened>"
    ## Decisions Made
    ## User Corrections / Preferences Learned
    ## Problems Encountered
    ## Working Methods Discovered
    ## Projects Updated
    ## Files / Artifacts Created
    ## Unfinished Work
    ## Suggested Next Actions

What is deliberately NOT written here: credentials of any kind, and
verbatim transcripts. The note has to be detailed enough that a future
session understands what happened, and that is a different thing from a
recording of everything said.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable

from vault.index import VaultIndex, get_index
from vault.manager import VaultManager
from vault.note import (
    DAILY,
    Note,
    extract_section,
    replace_section,
    today_stamp,
)
from vault.paths import DAILY_DIR

log = logging.getLogger("jarvis.vault.daily")

SECTION_TIMELINE = "Timeline"
SECTION_DECISIONS = "Decisions Made"
SECTION_CORRECTIONS = "User Corrections / Preferences Learned"
SECTION_PROBLEMS = "Problems Encountered"
SECTION_METHODS = "Working Methods Discovered"
SECTION_PROJECTS = "Projects Updated"
SECTION_ARTIFACTS = "Files / Artifacts Created"
SECTION_UNFINISHED = "Unfinished Work"
SECTION_NEXT = "Suggested Next Actions"

DAILY_SECTIONS = (
    SECTION_TIMELINE,
    SECTION_DECISIONS,
    SECTION_CORRECTIONS,
    SECTION_PROBLEMS,
    SECTION_METHODS,
    SECTION_PROJECTS,
    SECTION_ARTIFACTS,
    SECTION_UNFINISHED,
    SECTION_NEXT,
)

_EMPTY = "_Nothing recorded yet._"

#: Patterns that look like a credential. The Daily Note records what
#: happened; it must never become the place a key leaks into plain text on
#: disk, and this is the last line of defence before the write.
_SECRET = re.compile(
    r"""(sk-[A-Za-z0-9_\-]{16,}|xox[baprs]-[A-Za-z0-9-]{10,}|gh[pousr]_[A-Za-z0-9]{20,}|
        AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}|
        \b[A-Za-z0-9_\-]{0,10}(?:api[_\-]?key|secret|password|passwd|token|bearer)\s*[:=]\s*\S{6,})""",
    re.I | re.X,
)


def redact(text: str) -> str:
    """Replace anything that looks like a credential before it is written."""
    return _SECRET.sub("<REDACTED>", text or "")


@dataclass
class DailyNote:
    """A handle on one day's note. Every append writes straight to disk."""

    date: str
    relative_path: str
    vault: VaultManager
    index: VaultIndex | None = None

    def note(self) -> Note | None:
        return self.vault.read(self.relative_path)

    def section(self, heading: str) -> str:
        note = self.note()
        return note.section(heading) if note else ""

    # -- appends -------------------------------------------------------
    def _append(self, heading: str, entry: str) -> None:
        clean = redact(entry).strip()
        if not clean:
            return

        def mutate(note: Note) -> Note:
            existing = extract_section(note.body, heading)
            if existing.strip() == _EMPTY:
                existing = ""
            if clean in existing:
                # Ordinary repeated work ("ran the tests") is worth
                # recording twice; an identical LINE is not, and a note
                # that accumulates duplicates stops being readable.
                return note
            merged = f"{existing.rstrip()}\n{clean}".strip() if existing.strip() else clean
            note.body = replace_section(note.body, heading, merged)
            return note

        self.vault.update_note(self.relative_path, mutate)

    def add_event(
        self,
        headline: str,
        *,
        request: str = "",
        did: str = "",
        result: str = "",
        files: Iterable[str] = (),
        lesson: str = "",
        when: str | None = None,
    ) -> None:
        """One timeline entry, written the moment the work finishes.

        This is the detailed half of the Daily Note. A bare "worked on
        JARVIS" line is exactly what this exists to avoid, so an entry
        carries what was asked, what was done, what came of it, and what
        was learned.
        """
        stamp = when or datetime.now().strftime("%H:%M")
        lines = [f"### {stamp} - {headline.strip()}", ""]
        if request:
            lines += [f"**Asked:** {request.strip()}", ""]
        if did:
            lines += [f"**Did:** {did.strip()}", ""]
        if result:
            lines += [f"**Result:** {result.strip()}", ""]
        file_list = [str(item) for item in files if str(item).strip()]
        if file_list:
            lines += ["**Files:** " + ", ".join(f"`{item}`" for item in file_list), ""]
        if lesson:
            lines += [f"**Lesson:** {lesson.strip()}", ""]
        self._append(SECTION_TIMELINE, "\n".join(lines).rstrip())
        # The Quick Summary is what a LATER session reads first, and the
        # whole continuity story depends on it being true. Refreshing it
        # here (rather than at shutdown) is what makes "what did we do
        # yesterday" answerable after a crash, and costs one more atomic
        # write on an event that has already done several.
        self.refresh_quick_summary()

    def add_decision(self, text: str) -> None:
        self._append(SECTION_DECISIONS, f"- {text.strip()}")

    def add_correction(self, text: str) -> None:
        self._append(SECTION_CORRECTIONS, f"- {text.strip()}")

    def add_problem(self, text: str) -> None:
        self._append(SECTION_PROBLEMS, f"- {text.strip()}")

    def add_method(self, text: str) -> None:
        self._append(SECTION_METHODS, f"- {text.strip()}")

    def add_project_update(self, project: str, text: str = "") -> None:
        self._append(SECTION_PROJECTS, f"- [[{project}]]" + (f" -- {text.strip()}" if text.strip() else ""))

    def add_artifact(self, path: str, description: str = "") -> None:
        self._append(SECTION_ARTIFACTS, f"- `{path}`" + (f" -- {description.strip()}" if description.strip() else ""))

    def add_unfinished(self, text: str) -> None:
        self._append(SECTION_UNFINISHED, f"- {text.strip()}")

    def add_next_action(self, text: str) -> None:
        self._append(SECTION_NEXT, f"- {text.strip()}")

    # -- summary -------------------------------------------------------
    def refresh_quick_summary(self, extra: Iterable[str] = ()) -> None:
        """Rewrite the Quick Summary from what the day actually contains.

        The Quick Summary is what a future session reads FIRST, and the
        whole architecture depends on it being accurate -- so it is
        derived from the note's own sections rather than written once at
        creation and left to go stale.
        """
        note = self.note()
        if note is None:
            return
        events = len(re.findall(r"^### ", extract_section(note.body, SECTION_TIMELINE), re.M))
        bullets = [f"{events} recorded pieces of work today." if events else "No work recorded yet today."]
        for label, heading in (
            ("Corrections learned", SECTION_CORRECTIONS),
            ("Working methods discovered", SECTION_METHODS),
            ("Problems encountered", SECTION_PROBLEMS),
        ):
            body = extract_section(note.body, heading)
            count = len([line for line in body.splitlines() if line.strip().startswith("- ")]) if body.strip() != _EMPTY else 0
            if count:
                bullets.append(f"{label}: {count}.")
        unfinished = extract_section(note.body, SECTION_UNFINISHED)
        if unfinished.strip() and unfinished.strip() != _EMPTY:
            first = next((line.strip()[2:] for line in unfinished.splitlines() if line.strip().startswith("- ")), "")
            if first:
                bullets.append(f"Unfinished: {first}")
        bullets += [str(item) for item in extra if str(item).strip()]

        def mutate(target: Note) -> Note:
            target.body = replace_section(target.body, "Quick Summary", "\n".join(f"- {item}" for item in bullets))
            summary = f"{events} pieces of work recorded"
            corrections = extract_section(target.body, SECTION_CORRECTIONS)
            if corrections.strip() and corrections.strip() != _EMPTY:
                summary += ", including user corrections that changed stored knowledge"
            target.metadata = {**target.metadata, "summary": f"Daily record for {self.date}: {summary}."}
            return target

        self.vault.update_note(self.relative_path, mutate)

    def brief(self, *, max_chars: int = 1500) -> str:
        """This day, condensed, for a later session's startup context."""
        note = self.note()
        if note is None:
            return ""
        parts = [f"## Daily Note {self.date}", note.quick_summary]
        # The headlines of what actually happened -- the question a later
        # session is really asking is "what were we doing", and the Quick
        # Summary counts events without naming any of them.
        headlines = re.findall(r"^###\s+(.+)$", extract_section(note.body, SECTION_TIMELINE), re.M)
        if headlines:
            bullets = "\n".join(f"- {line.strip()}" for line in headlines[-6:])
            parts.append(f"### What happened\n{bullets}")
        for heading in (SECTION_UNFINISHED, SECTION_NEXT, SECTION_CORRECTIONS):
            text = extract_section(note.body, heading).strip()
            if text and text != _EMPTY:
                parts.append(f"### {heading}\n{text}")
        rendered = "\n\n".join(part for part in parts if part.strip())
        return rendered[:max_chars]

    def describe(self) -> dict[str, Any]:
        note = self.note()
        return {
            "date": self.date,
            "path": self.relative_path,
            "exists": note is not None,
            "sections": note.sections() if note else [],
        }


class DailyJournal:
    """Creates and finds Daily Notes."""

    def __init__(self, vault: VaultManager | None = None, index: VaultIndex | None = None):
        self.index = index or get_index(vault)
        self.vault = vault or self.index.vault

    def path_for(self, date: str) -> str:
        return f"{DAILY_DIR}/{date}.md"

    def today(self) -> DailyNote:
        return self.for_date(today_stamp())

    def for_date(self, date: str, *, create: bool = True) -> DailyNote:
        path = self.path_for(date)
        if create and not self.vault.note_exists(path):
            pretty = _pretty_date(date)
            self.vault.create_note(
                path,
                title=f"Daily Note - {date}",
                note_type=DAILY,
                summary=f"Daily record for {date}: nothing recorded yet.",
                tags=["daily"],
                quick_summary=[f"Daily record for {pretty}.", "Appended to throughout the day as work happens."],
                sections=[(heading, "") for heading in DAILY_SECTIONS],
                extra_metadata={"date": date},
            )
            self.index.invalidate()
            self.index.refresh()
            log.info("Daily note created for %s", date)
        return DailyNote(date=date, relative_path=path, vault=self.vault, index=self.index)

    def existing(self, date: str) -> DailyNote | None:
        path = self.path_for(date)
        return DailyNote(date=date, relative_path=path, vault=self.vault, index=self.index) if self.vault.note_exists(path) else None

    def yesterday(self) -> DailyNote | None:
        """The most recent PREVIOUS day that has a note.

        Literally-yesterday is usually right and sometimes wrong -- JARVIS
        is not used every day. Walking back to the last day that actually
        has a note is what makes "continue what we were doing yesterday"
        work after a weekend.
        """
        today = datetime.now().date()
        for offset in range(1, 15):
            stamp = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
            found = self.existing(stamp)
            if found is not None:
                return found
        return None

    def recent(self, *, limit: int = 5) -> list[DailyNote]:
        notes = [note for note in self.vault.iter_notes(DAILY_DIR) if note.note_type == DAILY]
        notes.sort(key=lambda note: note.relative_path, reverse=True)
        return [
            DailyNote(date=str(note.metadata.get("date") or note.path.stem), relative_path=note.relative_path, vault=self.vault, index=self.index)
            for note in notes[:limit]
        ]


def _pretty_date(date: str) -> str:
    try:
        return datetime.strptime(date, "%Y-%m-%d").strftime("%B %d, %Y")
    except ValueError:
        return date


_JOURNAL: DailyJournal | None = None


def get_journal(vault: VaultManager | None = None, index: VaultIndex | None = None) -> DailyJournal:
    global _JOURNAL
    if vault is not None or index is not None:
        return DailyJournal(vault=vault, index=index)
    if _JOURNAL is None:
        _JOURNAL = DailyJournal()
    return _JOURNAL


def reset_journal() -> None:
    global _JOURNAL
    _JOURNAL = None
