"""The archive: superseded knowledge, kept but never consulted by accident.

Nothing in this system deletes knowledge. A rule the user replaced, a
method that stopped working, a note that was folded into another -- all of
it moves here with the date and, when known, the reason. That is what
makes a change auditable: "why does JARVIS behave like this now" is
answerable from the vault a year later.

The hard rule is the other half:

    ARCHIVED CONTENT NEVER PARTICIPATES IN NORMAL RETRIEVAL.

An archived rule that could still be retrieved is an archived rule that
can still change behaviour, which would defeat the point of superseding
it. Enforcement is not scattered through the callers -- `archive/` is
excluded once, in `vault/paths.py::exclusion_reason`, and applied once, in
`VaultIndex.refresh`, so the retriever, priming, Job selection, Skill
selection and the generated `VAULT_INDEX.md` all inherit it.

The archive is reachable in exactly two ways, both deliberate:

- `ArchiveStore.search(...)` / `history_for(...)`, when an explicit
  history-facing caller asks what used to be true;
- opening `archive/ARCHIVE_INDEX.md` in Obsidian.

Layout:

    archive/
      ARCHIVE_INDEX.md          generated; not part of priming
      preferences/              superseded preference rules
      methods/                  methods that were replaced
      notes/                    whole notes that were retired
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from vault.index import NoteSummary, VaultIndex, get_index
from vault.manager import VaultManager
from vault.note import Note, build_note_text, extract_section, replace_section, utc_now
from vault.paths import (
    ARCHIVE_DIR,
    ARCHIVE_INDEX_FILE,
    ARCHIVE_METHODS_DIR,
    ARCHIVE_NOTES_DIR,
    ARCHIVE_PREFERENCES_DIR,
    EXCLUDED_ARCHIVE,
    slugify,
)

log = logging.getLogger("jarvis.vault.archive")

#: What kind of thing was archived. Also the subdirectory it lands in.
KIND_PREFERENCE = "preference"
KIND_METHOD = "method"
KIND_NOTE = "note"

_DIRECTORIES = {
    KIND_PREFERENCE: ARCHIVE_PREFERENCES_DIR,
    KIND_METHOD: ARCHIVE_METHODS_DIR,
    KIND_NOTE: ARCHIVE_NOTES_DIR,
}

#: The heading each archived entry records its history under.
SECTION_ENTRIES = "Superseded Entries"


@dataclass
class ArchivedItem:
    """One superseded thing, as the archive records it."""

    kind: str
    text: str
    source_path: str
    reason: str = ""
    archived_at: str = ""
    relative_path: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "source": self.source_path,
            "reason": self.reason,
            "archived_at": self.archived_at,
            "path": self.relative_path,
        }


class ArchiveStore:
    """Move superseded knowledge out of the way, and find it again on request."""

    def __init__(self, vault: VaultManager | None = None, index: VaultIndex | None = None):
        self.index = index or get_index(vault)
        self.vault = vault or self.index.vault

    # ------------------------------------------------------------ write
    def archive_rule(
        self,
        *,
        kind: str,
        text: str,
        source_path: str,
        reason: str = "",
        replaced_by: str = "",
    ) -> ArchivedItem | None:
        """Record one superseded rule or method.

        Entries for the same source note accumulate in ONE archive note,
        so a preference the user has changed four times reads as a
        history rather than as four unrelated files.
        """
        text = (text or "").strip()
        if not text:
            return None
        kind = kind if kind in _DIRECTORIES else KIND_NOTE
        directory = _DIRECTORIES[kind]
        stem = slugify(source_path.rsplit("/", 1)[-1].removesuffix(".md"), fallback=kind)
        path = f"{directory}/{stem}.md"
        stamp = utc_now()
        entry_lines = [f"- `{stamp[:10]}` **Was:** {text}"]
        if replaced_by.strip():
            entry_lines.append(f"  - **Now:** {replaced_by.strip()}")
        if reason.strip():
            entry_lines.append(f"  - **Why:** {reason.strip()}")
        entry_lines.append(f"  - **From:** `{source_path}`")
        entry = "\n".join(entry_lines)

        if not self.vault.note_exists(path):
            self.vault.create_note(
                path,
                title=f"Archived {kind}s from {source_path.rsplit('/', 1)[-1].removesuffix('.md').replace('-', ' ')}",
                note_type="archive",
                summary=(
                    f"Superseded {kind} rules that came from {source_path}. "
                    "Kept for history only -- never used to decide current behaviour."
                ),
                tags=["archive", kind, EXCLUDED_ARCHIVE],
                quick_summary=[
                    "ARCHIVED. This note is excluded from JARVIS's normal memory scan.",
                    f"It records what {source_path} used to say, and why that changed.",
                ],
                sections=[(SECTION_ENTRIES, entry)],
                extra_metadata={"archived_from": source_path, "archive_kind": kind},
            )
        else:
            def mutate(note: Note) -> Note:
                existing = extract_section(note.body, SECTION_ENTRIES)
                merged = f"{existing.rstrip()}\n{entry}" if existing.strip() and not existing.startswith("_Nothing") else entry
                note.body = replace_section(note.body, SECTION_ENTRIES, merged)
                return note

            self.vault.update_note(path, mutate)

        self.index.invalidate()
        self.index.refresh()
        log.info("Archived a superseded %s from %s", kind, source_path)
        return ArchivedItem(
            kind=kind,
            text=text,
            source_path=source_path,
            reason=reason,
            archived_at=stamp,
            relative_path=path,
        )

    def archive_note(self, relative_path: str, *, reason: str = "") -> str | None:
        """Retire a whole note into `archive/notes/`, preserving it intact.

        The note keeps its content; only its `status` and location change,
        and the reason is written into it so the retirement explains
        itself. Nothing is deleted, here or anywhere else in this system.
        """
        note = self.vault.read(relative_path)
        if note is None:
            return None
        stamp = datetime.now().strftime("%Y%m%d")
        destination = f"{ARCHIVE_NOTES_DIR}/{stamp}-{relative_path.rsplit('/', 1)[-1]}"

        def mutate(target: Note) -> Note:
            target.metadata = {
                **target.metadata,
                "status": "archived",
                "archived_at": utc_now(),
                "archived_from": relative_path,
                "archive_reason": reason or "superseded",
            }
            banner = (
                "> [!warning] Archived\n"
                f"> Retired on {utc_now()[:10]}"
                + (f": {reason}" if reason else ".")
                + "\n> JARVIS does not read this note when deciding what to do."
            )
            target.body = f"{banner}\n\n{target.body.lstrip()}"
            return target

        self.vault.update_note(relative_path, mutate)
        moved = self.vault.move_note(relative_path, destination)
        self.index.invalidate()
        self.index.refresh()
        if moved is None:
            return None
        log.info("Archived note %s -> %s", relative_path, moved.relative_path)
        return moved.relative_path

    # ------------------------------------------------------------- read
    def entries(self) -> list[NoteSummary]:
        """Every archive note's metadata. Explicit access only."""
        return self.index.excluded(EXCLUDED_ARCHIVE)

    def search(self, query: str, *, limit: int = 8) -> list[tuple[NoteSummary, float]]:
        """Rank the ARCHIVE for `query`.

        A separate ranking pass over a separate set, deliberately -- the
        ordinary retriever must never see these notes, so this cannot be
        implemented by relaxing a flag on it.
        """
        from vault.retrieval import _field_score, tokenize

        terms = tokenize(query)
        scored: list[tuple[NoteSummary, float]] = []
        for item in self.entries():
            score = (
                _field_score(terms, item.title, 3.0)
                + _field_score(terms, item.summary, 2.0)
                + _field_score(terms, " ".join(item.tags), 1.6)
                + _field_score(terms, item.relative_path.replace("/", " ").replace("-", " "), 0.6)
            )
            # The BODY is searched here, unlike in ordinary retrieval. An
            # archived rule's actual wording -- "always use emojis in
            # emails" -- lives in the entry list, not in the metadata, and
            # finding an old rule by what it SAID is the entire purpose of
            # this search. It is affordable because the archive is only
            # ever read on an explicit request, and it is bounded by how
            # much has actually been superseded.
            note = self.vault.read(item.relative_path)
            if note is not None:
                score += _field_score(terms, extract_section(note.body, SECTION_ENTRIES), 1.4)
            if score > 0:
                scored.append((item, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].relative_path))
        return scored[: max(1, limit)]

    def history_for(self, source_path: str) -> list[str]:
        """What one note used to say, newest last."""
        lines: list[str] = []
        for item in self.entries():
            note = self.vault.read(item.relative_path)
            if note is None:
                continue
            if str(note.metadata.get("archived_from") or "") != source_path:
                continue
            body = extract_section(note.body, SECTION_ENTRIES)
            lines.extend(line for line in body.splitlines() if line.strip())
        return lines

    # ------------------------------------------------------- generation
    def write_index(self) -> str | None:
        """(Re)generate `archive/ARCHIVE_INDEX.md`.

        Deliberately a separate file from `VAULT_INDEX.md`, and just as
        deliberately not part of priming: it exists so the user can browse
        what was superseded in Obsidian, not so JARVIS can stumble into it.
        """
        items = sorted(self.entries(), key=lambda entry: entry.relative_path)
        rows = ["| Archived note | What it holds | Originally from |", "| --- | --- | --- |"]
        for item in items:
            note = self.vault.read(item.relative_path)
            source = str((note.metadata.get("archived_from") if note else "") or "-")
            rows.append(f"| `{item.relative_path}` | {(item.summary or '-')[:110]} | `{source}` |")
        text = build_note_text(
            title="Archive Index",
            note_type="archive",
            summary=(
                f"Generated list of the {len(items)} archived notes. Archived knowledge is kept for "
                "history and is never used to decide current behaviour."
            ),
            tags=["archive", "index", "generated"],
            quick_summary=[
                "ARCHIVED CONTENT. Excluded from JARVIS's normal memory scan by design.",
                "Reachable only by explicitly asking for history, or by browsing here.",
                f"{len(items)} archived notes.",
            ],
            sections=[("Archived", "\n".join(rows) if items else "_Nothing archived yet._")],
        )
        self.vault.write_text(ARCHIVE_INDEX_FILE, text)
        return ARCHIVE_INDEX_FILE


_STORE: ArchiveStore | None = None


def get_archive(vault: VaultManager | None = None, index: VaultIndex | None = None) -> ArchiveStore:
    global _STORE
    if vault is not None or index is not None:
        return ArchiveStore(vault=vault, index=index)
    if _STORE is None:
        _STORE = ArchiveStore()
    return _STORE


def reset_archive() -> None:
    global _STORE
    _STORE = None
