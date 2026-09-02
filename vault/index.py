"""The vault index: the cheap map JARVIS reads before it reads anything.

Stage 1 of retrieval scans this. One `NoteSummary` is roughly 150-250
characters; a full skill note is several thousand. Scanning a thousand
summaries therefore costs about what deep-reading three notes costs,
which is exactly the trade this system is built on.

Three artefacts, one source of truth:

- **The `NoteSummary` objects in memory** -- what ranking actually reads.
- **`VAULT_INDEX.md`** at the vault root, and an `INDEX.md` per browsable
  collection -- generated Markdown, so the map is visible in Obsidian and
  navigable by wikilink. These are OUTPUT: nothing reads them back.
- **A JSON cache outside the vault** -- so a cold start does not have to
  re-parse every note. It is keyed on each file's `(mtime, size)`, and a
  miss simply re-reads that one file. The cache can be deleted at any
  time with no loss; Markdown is canonical.

The index refreshes itself lazily. `VaultIndex.summaries()` compares the
directory listing against what it holds and re-parses only what actually
changed, so a note edited by hand in Obsidian is picked up on the next
request without anything having to watch the filesystem.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from config.settings import env_float
from vault.manager import VaultManager, get_vault
from vault.note import INDEX, Note, build_note_text, utc_now
from vault import paths as vault_paths
from vault.paths import (
    ARCHIVE_INDEX_FILE,
    DIRECTORY_INDEX_FILE,
    INDEXED_DIRECTORIES,
    VAULT_INDEX_FILE,
    exclusion_reason,
)

log = logging.getLogger("jarvis.vault.index")

#: Index entries are never the thing being searched FOR.
_GENERATED = {
    VAULT_INDEX_FILE.lower(),
    DIRECTORY_INDEX_FILE.lower(),
    ARCHIVE_INDEX_FILE.rsplit("/", 1)[-1].lower(),
}


@dataclass
class NoteSummary:
    """Everything the SCAN stage is allowed to see about one note.

    Deliberately not the body. If ranking needed the body, the two-stage
    design would be pointless.
    """

    relative_path: str
    title: str
    note_type: str
    summary: str
    tags: list[str] = field(default_factory=list)
    updated: str = ""
    status: str = ""
    links: list[str] = field(default_factory=list)
    #: The first few lines of the Quick Summary, which is the one piece of
    #: BODY text cheap enough to carry: it is bounded by the format and it
    #: is what makes an ambiguous one-line `summary` decidable.
    quick_summary: str = ""
    size: int = 0
    mtime: float = 0.0
    malformed: bool = False
    #: Why this note is kept OUT of the ordinary scan ("archive",
    #: "job_preference"), or "" when it is active. Set by the index from
    #: the note's path -- see `vault/paths.py::exclusion_reason`.
    excluded_reason: str = ""

    @property
    def scannable(self) -> bool:
        return not self.excluded_reason

    def digest(self) -> str:
        parts = [f"- {self.title} [{self.note_type}] ({self.relative_path})"]
        if self.summary:
            parts.append(f"  {self.summary}")
        if self.tags:
            parts.append(f"  tags: {', '.join(self.tags)}")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoteSummary":
        known = {key: data.get(key) for key in cls.__dataclass_fields__}
        known["tags"] = list(known.get("tags") or [])
        known["links"] = list(known.get("links") or [])
        return cls(**{key: value for key, value in known.items() if value is not None})

    @classmethod
    def from_note(cls, note: Note, *, quick_summary_chars: int = 400) -> "NoteSummary":
        quick = note.quick_summary
        if len(quick) > quick_summary_chars:
            quick = quick[:quick_summary_chars].rstrip() + " ..."
        return cls(
            relative_path=note.relative_path,
            title=note.title,
            note_type=note.note_type,
            summary=note.summary,
            tags=note.tags,
            updated=note.updated,
            status=note.status,
            links=note.links[:12],
            quick_summary=quick,
            size=note.size,
            mtime=note.mtime,
            malformed=note.malformed,
        )


class VaultIndex:
    """A lazily-refreshed map of every note's metadata."""

    def __init__(
        self,
        vault: VaultManager | None = None,
        cache_path: Path | None = None,
        *,
        use_cache: bool = True,
        refresh_interval: float | None = None,
    ):
        self.vault = vault or get_vault()
        # Through the module, not a name imported from it -- see
        # `VaultManager.__init__` for why that distinction matters.
        self.cache_path = cache_path if cache_path is not None else vault_paths.default_cache_path()
        self.use_cache = use_cache
        #: How long a refresh is considered still current, in seconds.
        #:
        #: Refreshing is a `stat` per note -- cheap per call, but one
        #: request asks several times (the primer scans, then the Job
        #: registry scans, then the Skill library scans), and on a
        #: thousand-note vault on Windows that added up to about a second
        #: of pure `stat` per request. Within this window a refresh is a
        #: no-op, so one request costs one scan. A note edited by hand in
        #: Obsidian is still picked up on the next request.
        self.refresh_interval = (
            refresh_interval if refresh_interval is not None else env_float("JARVIS_VAULT_INDEX_TTL", 2.0)
        )
        self._summaries: dict[str, NoteSummary] = {}
        self._signatures: dict[str, tuple[float, int]] = {}
        self._lock = threading.RLock()
        self._loaded = False
        self._last_refresh: float = 0.0
        self.last_scan_ms: float = 0.0
        self.last_reparsed: int = 0

    # ----------------------------------------------------------- cache
    def _load_cache(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.use_cache or self.cache_path is None:
            return
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict) or raw.get("root") != str(self.vault.root):
            # A cache built for a different vault is not wrong, it is
            # irrelevant. Rebuild rather than mixing two vaults' notes.
            return
        for entry in raw.get("notes", []):
            try:
                summary = NoteSummary.from_dict(entry)
            except (TypeError, ValueError):
                continue
            self._summaries[summary.relative_path] = summary
            self._signatures[summary.relative_path] = (summary.mtime, summary.size)

    def _save_cache(self) -> None:
        if not self.use_cache or self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "root": str(self.vault.root),
                "generated": utc_now(),
                "notes": [summary.to_dict() for summary in self._summaries.values()],
            }
            self.cache_path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - a cache failure is not fatal
            log.debug("Could not write the vault index cache: %s", exc)

    # ------------------------------------------------------------ scan
    def refresh(self, *, force: bool = False) -> list[NoteSummary]:
        """Bring the index in line with what is on disk.

        Only notes whose `(mtime, size)` changed are re-parsed, so an
        ordinary refresh over an unchanged vault is a `stat` per file.
        """
        started = time.perf_counter()
        with self._lock:
            self._load_cache()
            if (
                not force
                and self._last_refresh
                and (time.monotonic() - self._last_refresh) < self.refresh_interval
            ):
                return list(self._summaries.values())
            if force:
                self._summaries.clear()
                self._signatures.clear()
            seen: set[str] = set()
            reparsed = 0
            for path in self.vault.iter_note_paths():
                relative = self.vault.relative(path)
                if Path(relative).name.lower() in _GENERATED:
                    continue
                seen.add(relative)
                try:
                    stat = path.stat()
                    signature = (stat.st_mtime, stat.st_size)
                except OSError:  # pragma: no cover
                    continue
                if not force and self._signatures.get(relative) == signature:
                    continue
                note = self.vault.read(relative)
                if note is None:
                    continue
                summary = NoteSummary.from_note(note)
                summary.excluded_reason = exclusion_reason(relative)
                self._summaries[relative] = summary
                self._signatures[relative] = signature
                reparsed += 1
            for missing in set(self._summaries) - seen:
                self._summaries.pop(missing, None)
                self._signatures.pop(missing, None)
            self.last_reparsed = reparsed
            self.last_refresh_at = time.monotonic()
            self._last_refresh = self.last_refresh_at
            self.last_scan_ms = (time.perf_counter() - started) * 1000
            if reparsed or force:
                self._save_cache()
            return list(self._summaries.values())

    def invalidate(self) -> None:
        """Force the next refresh to actually run.

        Called after JARVIS writes a note. Without it, the throttle above
        would hide a note JARVIS had JUST created from its own next scan
        -- which is exactly the sequence the end-to-end test exercises.
        """
        with self._lock:
            self._last_refresh = 0.0

    def summaries(self, *, refresh: bool = True, include_excluded: bool = False) -> list[NoteSummary]:
        """The ACTIVE notes, unless `include_excluded` says otherwise.

        This is the one place the archive and the per-Job preference notes
        are filtered out, so every consumer -- the retriever, priming, Job
        and Skill selection, the generated index -- inherits the rule
        rather than each remembering to apply it.
        """
        if refresh:
            everything = self.refresh()
        else:
            with self._lock:
                self._load_cache()
                everything = list(self._summaries.values())
        return everything if include_excluded else [item for item in everything if item.scannable]

    def excluded(self, reason: str = "", *, refresh: bool = True) -> list[NoteSummary]:
        """The notes deliberately kept out of the scan.

        `reason` narrows to one kind ("archive", "job_preference"). This is
        how the explicit archive/history tools reach superseded knowledge
        without a second filesystem walk.
        """
        found = [item for item in self.summaries(refresh=refresh, include_excluded=True) if item.excluded_reason]
        return [item for item in found if item.excluded_reason == reason] if reason else found

    def by_type(self, note_type: str, *, refresh: bool = True) -> list[NoteSummary]:
        return [item for item in self.summaries(refresh=refresh) if item.note_type == note_type]

    def get(self, relative_path: str, *, refresh: bool = False) -> NoteSummary | None:
        # Addressed by path, so an excluded note IS returned: naming a note
        # exactly is a deliberate lookup, not a search.
        for item in self.summaries(refresh=refresh, include_excluded=True):
            if item.relative_path == relative_path:
                return item
        return None

    def find_by_title(self, title: str, *, refresh: bool = True) -> NoteSummary | None:
        """Resolve a wikilink target (`[[Python Debugging]]`) to a note.

        Title first, then filename stem -- the two things an Obsidian
        wikilink can legitimately mean.
        """
        wanted = (title or "").strip().lower()
        if not wanted:
            return None
        # Excluded notes are included here on purpose: a Job note pointing
        # at `[[Preferences - Send Email]]` is naming one specific note,
        # which is a reference, not a search. Refusing to resolve it would
        # break the very mechanism that keeps preferences out of the scan.
        candidates = self.summaries(refresh=refresh, include_excluded=True)
        for item in candidates:
            if item.title.strip().lower() == wanted:
                return item
        for item in candidates:
            if Path(item.relative_path).stem.lower() == wanted:
                return item
        for item in candidates:
            if Path(item.relative_path).stem.replace("-", " ").lower() == wanted:
                return item
        return None

    def statistics(self) -> dict[str, Any]:
        summaries = self.summaries()
        by_type: dict[str, int] = {}
        for item in summaries:
            by_type[item.note_type] = by_type.get(item.note_type, 0) + 1
        summary_chars = sum(len(item.summary) + len(item.quick_summary) for item in summaries)
        body_chars = sum(item.size for item in summaries)
        return {
            "notes": len(summaries),
            "by_type": dict(sorted(by_type.items())),
            "scan_ms": round(self.last_scan_ms, 2),
            "reparsed": self.last_reparsed,
            "summary_chars": summary_chars,
            "full_chars": body_chars,
            #: What the two-stage design actually buys, measured rather
            #: than asserted: the ratio of a full-vault scan to a
            #: full-vault read.
            "scan_fraction_of_full": round(summary_chars / body_chars, 4) if body_chars else 0.0,
        }

    # ------------------------------------------------------- generation
    def write_markdown_index(self) -> Path:
        """(Re)generate `VAULT_INDEX.md` and each collection's `INDEX.md`.

        Generated files carry `type: index` so retrieval can exclude them:
        an index that ranked highly for every query would crowd out the
        knowledge it points at.
        """
        summaries = sorted(self.summaries(), key=lambda item: (item.note_type, item.title.lower()))
        groups: dict[str, list[NoteSummary]] = {}
        for item in summaries:
            groups.setdefault(item.note_type, []).append(item)

        lines: list[str] = []
        for note_type in sorted(groups):
            lines.append(f"### {note_type}")
            lines.append("")
            lines.append("| Note | Summary | Tags | Updated |")
            lines.append("| --- | --- | --- | --- |")
            for item in groups[note_type]:
                lines.append(
                    f"| [[{item.title}]] <br><sub>{item.relative_path}</sub> "
                    f"| {_cell(item.summary)} | {_cell(', '.join(item.tags))} | {_cell(item.updated[:10])} |"
                )
            lines.append("")

        text = build_note_text(
            title="Vault Index",
            note_type=INDEX,
            summary=(
                f"Generated map of all {len(summaries)} notes in this vault: title, path, type, "
                "summary, tags and last update, so JARVIS can triage without reading anything."
            ),
            tags=["index", "generated"],
            quick_summary=[
                "Generated by JARVIS. Edits here are overwritten -- change the notes themselves.",
                f"{len(summaries)} notes: " + ", ".join(f"{count} {name}" for name, count in sorted(
                    ((name, len(items)) for name, items in groups.items()), key=lambda pair: -pair[1]
                )) + ".",
                "Stage 1 of retrieval reads this kind of metadata; only the selected notes are read in full.",
            ],
            sections=[("Notes", "\n".join(lines).strip() or "_The vault is empty._")],
        )
        path = self.vault.write_text(VAULT_INDEX_FILE, text)

        for directory in INDEXED_DIRECTORIES:
            self._write_directory_index(directory)
        return path

    def _write_directory_index(self, directory: str) -> Path | None:
        prefix = directory.rstrip("/") + "/"
        items = sorted(
            (item for item in self.summaries(refresh=False) if item.relative_path.startswith(prefix)),
            key=lambda item: item.title.lower(),
        )
        if not (self.vault.root / directory).is_dir():
            return None
        rows = ["| Note | Summary | Tags |", "| --- | --- | --- |"]
        rows += [f"| [[{item.title}]] | {_cell(item.summary)} | {_cell(', '.join(item.tags))} |" for item in items]
        name = directory.rstrip("/").split("/")[-1]
        text = build_note_text(
            title=f"{name.capitalize()} Index",
            note_type=INDEX,
            summary=f"Generated list of the {len(items)} {name} notes in this vault, with their one-line summaries.",
            tags=["index", "generated", name],
            quick_summary=[
                f"{len(items)} {name} notes.",
                "Generated by JARVIS -- edit the notes themselves, not this file.",
            ],
            sections=[(name.capitalize(), "\n".join(rows) if items else "_None yet._")],
        )
        return self.vault.write_text(f"{directory.rstrip('/')}/{DIRECTORY_INDEX_FILE}", text)


def _cell(text: str) -> str:
    """Make one value safe inside a Markdown table cell."""
    return (text or "").replace("|", "\\|").replace("\n", " ").strip() or "-"


_INDEX: VaultIndex | None = None
_INDEX_LOCK = threading.Lock()


def get_index(vault: VaultManager | None = None) -> VaultIndex:
    global _INDEX
    if vault is not None:
        return VaultIndex(vault)
    if _INDEX is None:
        with _INDEX_LOCK:
            if _INDEX is None:
                _INDEX = VaultIndex()
    return _INDEX


def reset_index() -> None:
    global _INDEX
    with _INDEX_LOCK:
        _INDEX = None
