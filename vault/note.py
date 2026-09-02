"""The JARVIS note: what every piece of persistent knowledge looks like.

The single most important property of a note is that JARVIS can decide
whether it is relevant WITHOUT reading it. A 5,000-word skill note and a
one-line preference note must cost the same to triage. That is what the
frontmatter and the Quick Summary are for, and why they are a hard part
of the format rather than a convention:

    ---
    title: Apple Music Control
    type: skill
    summary: How JARVIS opens, reuses, searches and controls Apple Music.
    tags:
      - music
      - apple-music
    updated: 2026-09-02T14:03:11+00:00
    ---

    # Apple Music Control

    ## Quick Summary

    - Reuse the existing window if one is already open.
    - Launch only when no instance exists.

    ...full detail...

Everything here is plain, Obsidian-compatible Markdown. Obsidian never
has to be running; when it IS running the user sees and edits exactly the
files JARVIS reads.

## Why the frontmatter parser is written here

The vault is edited by a human as well as by JARVIS, so a note WILL
eventually contain frontmatter that is slightly wrong -- a stray tab, an
unquoted colon, a truncated write. A malformed note must degrade to "a
note with no metadata", never take down the scan that was trying to
triage it. So parsing is tolerant by construction: PyYAML is used when it
is installed AND the block parses, and a small, deliberately narrow
scalar/list parser is used otherwise. Writing never goes through PyYAML
at all -- output is emitted by `dump_frontmatter` so the byte-level shape
of a note JARVIS wrote is stable, diffable and predictable in a git
history.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

#: The note types the system understands. `other` is deliberately allowed:
#: a human-created note of an unknown type must still be scannable, not
#: rejected.
IDENTITY = "identity"
USER = "user"
PROJECT = "project"
JOB = "job"
SKILL = "skill"
LESSON = "lesson"
MISSION = "mission"
DAILY = "daily"
STATE = "state"
SYSTEM = "system"
INDEX = "index"
OTHER = "other"

KNOWN_TYPES = frozenset(
    {IDENTITY, USER, PROJECT, JOB, SKILL, LESSON, MISSION, DAILY, STATE, SYSTEM, INDEX, OTHER}
)

#: The heading that introduces the human-readable digest near the top.
QUICK_SUMMARY_HEADING = "Quick Summary"

_BOM = "﻿"
_FRONTMATTER = re.compile(r"\A[﻿]?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)
_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$", re.M)


def utc_now() -> str:
    """A stable, sortable, timezone-explicit timestamp.

    Seconds resolution: a note's `updated` field is read by a human in
    Obsidian, and microseconds are noise there.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def today_stamp() -> str:
    """Today's local date, which is what a Daily Note is named after."""
    return datetime.now().strftime("%Y-%m-%d")


# ---------------------------------------------------------------- parsing


def _coerce_scalar(raw: str) -> Any:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "~", ""}:
        return None
    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"-?\d+\.\d+", text):
        try:
            return float(text)
        except ValueError:
            return text
    return text


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are not inside quotes or brackets."""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            current.append(char)
            continue
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current))
    return [part for part in (item.strip() for item in parts) if part]


def _parse_inline_list(text: str) -> list[Any]:
    inner = text.strip()[1:-1].strip()
    if not inner:
        return []
    return [_coerce_scalar(part) for part in _split_top_level(inner)]


def _parse_simple_frontmatter(block: str) -> dict[str, Any]:
    """The fallback parser: top-level `key: value` and `- item` lists only.

    Deliberately narrow. It is not a YAML implementation and does not try
    to be one -- it exists so a note whose frontmatter PyYAML rejects
    still yields whatever fields are individually readable, instead of
    yielding nothing at all.
    """
    data: dict[str, Any] = {}
    key: str | None = None
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if re.match(r"^\s*-\s", line) and key is not None:
            value = _coerce_scalar(line.split("-", 1)[1])
            if value is None:
                continue
            existing = data.get(key)
            if isinstance(existing, list):
                existing.append(value)
            else:
                data[key] = [value]
            continue
        match = re.match(r"^([A-Za-z0-9_\-]+)\s*:\s*(.*)$", line)
        if not match:
            continue
        key, raw = match.group(1), match.group(2).strip()
        if raw.startswith("[") and raw.endswith("]"):
            data[key] = _parse_inline_list(raw)
        elif raw == "":
            data[key] = []
        else:
            data[key] = _coerce_scalar(raw)
    return data


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split `text` into (metadata, body).

    Never raises. A note with no frontmatter, or with frontmatter nothing
    can read, returns `({}, text)` -- the body is always preserved
    verbatim, so a bad parse can never lose the user's content.
    """
    source = text or ""
    match = _FRONTMATTER.match(source)
    if not match:
        return {}, source.lstrip(_BOM)
    block = match.group(1)
    body = source[match.end():]
    data: dict[str, Any] = {}
    try:  # pragma: no cover - the PyYAML branch, exercised when installed
        import yaml

        loaded = yaml.safe_load(block)
        if isinstance(loaded, dict):
            data = loaded
    except Exception:
        data = {}
    if not data:
        data = _parse_simple_frontmatter(block)
    return {str(key): _normalize_value(value) for key, value in data.items() if key is not None}, body


def _normalize_value(value: Any) -> Any:
    """Undo PyYAML's helpful-but-lossy scalar typing.

    PyYAML turns `updated: 2026-09-02T14:03:11+00:00` into a `datetime`,
    whose `str()` is space-separated rather than ISO -- so a note that was
    merely read and written back came out byte-different from the one
    JARVIS had just written. Timestamps are strings in this format; the
    only thing that reads them is a lexicographic sort and a human.
    """
    if isinstance(value, datetime):
        return value.replace(microsecond=0).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


_NEEDS_QUOTING = re.compile(
    r"""^[\s>|&*!%@`{}\[\]#-]|:\s|\s\#|[\n\r]|^(?:true|false|yes|no|null|~)$""",
    re.I,
)


def _quote_if_needed(value: str) -> str:
    if value == "":
        return '""'
    if _NEEDS_QUOTING.search(value):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def dump_frontmatter(data: dict[str, Any]) -> str:
    """Serialize metadata to the exact frontmatter shape JARVIS writes.

    Field ORDER is fixed for the known keys so a note edited a hundred
    times still produces a small, readable diff, and so a human scanning
    the vault sees the same fields in the same place every time. Unknown
    keys are preserved and emitted afterwards, sorted -- a field a human
    added by hand is never silently dropped.
    """
    preferred = ["title", "type", "summary", "tags", "updated", "created", "status", "aliases"]
    keys = [key for key in preferred if key in data]
    keys += sorted(str(key) for key in data if key not in preferred)
    lines = ["---"]
    for key in keys:
        value = data[key]
        if isinstance(value, (list, tuple, set)):
            items = [str(item) for item in value]
            if not items:
                lines.append(f"{key}: []")
                continue
            lines.append(f"{key}:")
            lines.extend(f"  - {_quote_if_needed(str(item))}" for item in items)
            continue
        if isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
            continue
        if value is None:
            lines.append(f"{key}:")
            continue
        if isinstance(value, (int, float)):
            lines.append(f"{key}: {value}")
            continue
        lines.append(f"{key}: {_quote_if_needed(str(value))}")
    lines.append("---")
    return "\n".join(lines)


def _as_tag_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,\s]+", value.strip())
        return [part.strip().lstrip("#") for part in parts if part.strip()]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_as_tag_list(item))
        return out
    return [str(value)]


def extract_quick_summary(body: str) -> str:
    """The text under the `## Quick Summary` heading, if there is one.

    Returns "" rather than guessing when the heading is absent -- an
    inferred summary that is wrong is worse than no summary, because the
    retrieval layer would trust it.
    """
    match = re.search(rf"^#{{1,6}}\s+{re.escape(QUICK_SUMMARY_HEADING)}\s*$", body or "", re.M | re.I)
    if not match:
        return ""
    rest = body[match.end():]
    end = re.search(r"^#{1,6}\s+\S", rest, re.M)
    return (rest[: end.start()] if end else rest).strip()


def extract_section(body: str, heading: str) -> str:
    """One `## Heading` section's text, or "" if that heading is absent."""
    match = re.search(rf"^(#{{1,6}})\s+{re.escape(heading)}\s*$", body or "", re.M | re.I)
    if not match:
        return ""
    level = len(match.group(1))
    rest = body[match.end():]
    end = re.search(rf"^#{{1,{level}}}\s+\S", rest, re.M)
    return (rest[: end.start()] if end else rest).strip()


def replace_section(body: str, heading: str, content: str) -> str:
    """Return `body` with one section's content replaced.

    The heading itself, its level, and everything around it are preserved;
    only the text between this heading and the next one of the same or
    higher level changes. A heading that does not exist is APPENDED, so
    "update the Procedure" works whether or not the note already had one.
    """
    text = content.strip()
    match = re.search(rf"^(#{{1,6}})\s+{re.escape(heading)}\s*$", body or "", re.M | re.I)
    if not match:
        return (body or "").rstrip() + f"\n\n## {heading}\n\n{text}\n"
    level = len(match.group(1))
    rest = body[match.end():]
    end = re.search(rf"^#{{1,{level}}}\s+\S", rest, re.M)
    tail = rest[end.start():] if end else ""
    head = body[: match.end()]
    rebuilt = f"{head}\n\n{text}\n\n{tail}".rstrip() + "\n"
    # A note is read by a human in Obsidian, and this function runs on it
    # dozens of times over its life. Without this, every edit leaves one
    # more blank line behind a heading and the note slowly fills with gaps.
    return re.sub(r"\n{3,}", "\n\n", rebuilt)


def section_names(body: str) -> list[str]:
    return [match.group(2).strip() for match in _HEADING.finditer(body or "")]


def extract_wikilinks(text: str) -> list[str]:
    """Every `[[Target]]` / `[[Target|alias]]` in order, de-duplicated."""
    seen: list[str] = []
    for match in _WIKILINK.finditer(text or ""):
        target = match.group(1).strip()
        if target and target not in seen:
            seen.append(target)
    return seen


def extract_list_items(section_text: str) -> list[str]:
    """The `- item` bullets of a section, with wikilink brackets stripped."""
    items: list[str] = []
    for line in (section_text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith(("- ", "* ", "+ ")):
            continue
        value = stripped[2:].strip()
        links = extract_wikilinks(value)
        items.append(links[0] if links else value)
    return [item for item in items if item]


# ------------------------------------------------------------------ note


@dataclass
class Note:
    """One Markdown note, parsed but not interpreted.

    `path` is absolute; `relative_path` is what the index, the wikilinks
    and every log line use, so a vault can be moved without invalidating
    anything JARVIS wrote about it.
    """

    path: Path
    relative_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    body: str = ""
    #: True when the frontmatter block was absent or unreadable. Surfaced
    #: rather than hidden, so a broken note shows up in diagnostics
    #: instead of quietly never being retrieved.
    malformed: bool = False
    mtime: float = 0.0
    size: int = 0

    # -- metadata views ------------------------------------------------
    @property
    def title(self) -> str:
        value = self.metadata.get("title")
        if isinstance(value, str) and value.strip():
            return value.strip()
        heading = re.search(r"^#\s+(.+)$", self.body or "", re.M)
        if heading:
            return heading.group(1).strip()
        return self.path.stem.replace("_", " ").replace("-", " ").strip()

    @property
    def note_type(self) -> str:
        value = str(self.metadata.get("type") or "").strip().lower()
        return value or OTHER

    @property
    def summary(self) -> str:
        value = self.metadata.get("summary")
        return value.strip() if isinstance(value, str) else ""

    @property
    def tags(self) -> list[str]:
        return _as_tag_list(self.metadata.get("tags"))

    @property
    def updated(self) -> str:
        value = self.metadata.get("updated")
        return str(value).strip() if value not in (None, "") else ""

    @property
    def status(self) -> str:
        value = self.metadata.get("status")
        return str(value).strip().lower() if value not in (None, "") else ""

    @property
    def quick_summary(self) -> str:
        return extract_quick_summary(self.body)

    @property
    def links(self) -> list[str]:
        return extract_wikilinks(self.body)

    def section(self, heading: str) -> str:
        return extract_section(self.body, heading)

    def sections(self) -> list[str]:
        return section_names(self.body)

    def list_items(self, heading: str) -> list[str]:
        return extract_list_items(self.section(heading))

    @property
    def has_summary(self) -> bool:
        """Does this note expose its purpose without being read in full?

        Both halves are required: the machine-readable `summary` field
        (what the scan ranks on) and the human `## Quick Summary` section
        (what a person -- and the first lines of a deep read -- see).
        """
        return bool(self.summary) and bool(self.quick_summary)

    # -- serialization -------------------------------------------------
    def to_markdown(self) -> str:
        body = (self.body or "").strip("\n")
        front = dump_frontmatter(self.metadata)
        return f"{front}\n\n{body}\n" if body else f"{front}\n"

    def digest(self) -> str:
        """The cheap view: what the SCAN stage sees instead of the body."""
        parts = [f"[{self.note_type}] {self.title} ({self.relative_path})"]
        if self.summary:
            parts.append(f"  summary: {self.summary}")
        if self.tags:
            parts.append(f"  tags: {', '.join(self.tags)}")
        if self.updated:
            parts.append(f"  updated: {self.updated}")
        return "\n".join(parts)

    def with_metadata(self, **updates: Any) -> "Note":
        merged = dict(self.metadata)
        merged.update(updates)
        return replace(self, metadata=merged)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        path: Path,
        relative_path: str,
        mtime: float = 0.0,
        size: int | None = None,
    ) -> "Note":
        metadata, body = parse_frontmatter(text)
        return cls(
            path=path,
            relative_path=relative_path,
            metadata=metadata,
            body=body,
            malformed=not metadata,
            mtime=mtime,
            size=len(text) if size is None else size,
        )


def build_note_text(
    *,
    title: str,
    note_type: str,
    summary: str,
    tags: Iterable[str] = (),
    quick_summary: Iterable[str] | str = (),
    sections: Iterable[tuple[str, str]] = (),
    extra_metadata: dict[str, Any] | None = None,
    body_prefix: str = "",
) -> str:
    """Compose a note that satisfies the standard by construction.

    Every note JARVIS creates goes through here, which is why the format
    is a guarantee rather than a hope: there is no code path that writes a
    new knowledge note without a type, a summary, tags, a timestamp and a
    Quick Summary.
    """
    metadata: dict[str, Any] = {
        "title": title,
        "type": note_type,
        "summary": (summary or "").strip(),
        "tags": sorted({str(tag).strip().lower() for tag in tags if str(tag).strip()}),
        "updated": utc_now(),
    }
    metadata.update(extra_metadata or {})
    metadata.setdefault("created", metadata["updated"])

    if isinstance(quick_summary, str):
        quick_block = quick_summary.strip()
    else:
        bullets = [str(item).strip() for item in quick_summary if str(item).strip()]
        quick_block = "\n".join(f"- {item}" for item in bullets)
    if not quick_block:
        quick_block = f"- {(summary or title).strip()}"

    parts = [f"# {title}", "", f"## {QUICK_SUMMARY_HEADING}", "", quick_block]
    if body_prefix.strip():
        parts += ["", body_prefix.strip()]
    for heading, content in sections:
        parts += ["", f"## {heading}", "", (content or "").strip() or "_Nothing recorded yet._"]
    body = "\n".join(parts)
    return f"{dump_frontmatter(metadata)}\n\n{body.strip()}\n"
