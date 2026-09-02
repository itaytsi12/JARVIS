"""The VaultManager: JARVIS's read/write access to its own long-term brain.

This is the only module that touches the vault's files. Everything above
it (index, retrieval, jobs, skills, missions, daily notes, learning)
works in terms of `Note` objects and relative paths, so there is exactly
one place where a write can go wrong -- and exactly one place that has to
get durability right.

Invariants:

- **Writes are atomic.** A note is written to a temporary file in the same
  directory and then `os.replace`d over the target, so a crash (or a
  half-flushed disk) can never leave a truncated note behind. The vault is
  the memory; a corrupted note is amnesia.
- **Reads never raise.** A note that cannot be decoded, or whose
  frontmatter is nonsense, comes back as a `Note` with `malformed=True`
  and its body intact. One bad file must never stop a scan of a thousand
  good ones.
- **Writes are serialized within a process.** A single `RLock` guards
  every mutation, because the daily note is appended to from the voice
  thread, agent worker threads and background tasks at the same time.
- **Nothing outside the vault is ever touched.** Every path is resolved
  and checked against the vault root before use, so a relative path with
  `..` in it -- from a model-authored tool call, say -- fails loudly
  instead of writing somewhere else on the disk.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from vault.note import (
    Note,
    build_note_text,
    dump_frontmatter,
    parse_frontmatter,
    utc_now,
)
from vault import paths as vault_paths
from vault.paths import IGNORED_DIRECTORIES, slugify

log = logging.getLogger("jarvis.vault")


class VaultError(Exception):
    """A vault operation that could not be performed safely."""


class OutsideVault(VaultError):
    """A path escaped the vault root. Never silently corrected."""


class VaultManager:
    """Open, read, create, modify, move and archive notes."""

    def __init__(self, root: Path | str | None = None):
        # Resolved through the MODULE, not through a name imported from it.
        # `from vault.paths import default_vault_path` binds the function
        # once at import time, so a later `patch("vault.paths.default_vault_path")`
        # -- what a test does to redirect the vault to a temporary
        # directory -- would be silently ignored and the test would run
        # against the real vault. The same late-binding rule
        # `voice/speech_coordinator.py` follows, for the same reason.
        self.root = Path(root).expanduser().resolve() if root else vault_paths.default_vault_path().resolve()
        self._lock = threading.RLock()

    # ------------------------------------------------------------ paths
    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"VaultManager(root={self.root!s})"

    def exists(self) -> bool:
        return self.root.is_dir()

    def ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def resolve(self, relative_path: str | Path) -> Path:
        """An absolute path inside the vault, or `OutsideVault`.

        `..` traversal, an absolute path pointing elsewhere, and a
        symlinked escape all fail here rather than being "helpfully"
        clamped -- a caller that asked for the wrong place must be told,
        not quietly redirected.
        """
        candidate = Path(str(relative_path).replace("\\", "/"))
        target = candidate if candidate.is_absolute() else (self.root / candidate)
        try:
            resolved = target.resolve()
        except OSError as exc:  # pragma: no cover - unusual filesystem states
            raise VaultError(f"Could not resolve {relative_path!r}: {exc}") from exc
        root = self.root.resolve()
        if resolved != root and root not in resolved.parents:
            raise OutsideVault(f"{relative_path!r} is outside the vault at {root}")
        return resolved

    def relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def note_path(self, relative_path: str | Path) -> Path:
        """Resolve, adding the `.md` extension when it was left off."""
        text = str(relative_path).replace("\\", "/")
        if not text.lower().endswith(".md"):
            text += ".md"
        return self.resolve(text)

    # ------------------------------------------------------------ reads
    def read_text(self, relative_path: str | Path) -> str | None:
        path = self.note_path(relative_path)
        return self._read_file(path)

    @staticmethod
    def _read_file(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except UnicodeDecodeError:
            # A note saved in another encoding is still the user's memory.
            # Recover what is readable rather than pretending it is absent.
            try:
                return path.read_bytes().decode("utf-8", errors="replace")
            except OSError as exc:  # pragma: no cover
                log.warning("Could not read %s: %s", path, exc)
                return None
        except OSError as exc:
            log.warning("Could not read %s: %s", path, exc)
            return None

    def read(self, relative_path: str | Path) -> Note | None:
        """One note, fully parsed. `None` only when the file is absent."""
        path = self.note_path(relative_path)
        text = self._read_file(path)
        if text is None:
            return None
        try:
            stat = path.stat()
            mtime, size = stat.st_mtime, stat.st_size
        except OSError:  # pragma: no cover
            mtime, size = 0.0, len(text)
        return Note.from_text(text, path=path, relative_path=self.relative(path), mtime=mtime, size=size)

    def note_exists(self, relative_path: str | Path) -> bool:
        try:
            return self.note_path(relative_path).is_file()
        except VaultError:
            return False

    def iter_note_paths(self, subdirectory: str | None = None) -> Iterator[Path]:
        """Every `.md` file in the vault, ignored folders pruned.

        `os.walk` with in-place `dirnames` pruning, for the same reason
        `tools/code.py::walk_source_files` uses it: descending into a
        folder only to discard its results costs orders of magnitude more
        than the answer. `.obsidian/` in particular holds hundreds of
        files that are configuration, not knowledge.
        """
        start = self.resolve(subdirectory) if subdirectory else self.root
        if not start.is_dir():
            return
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = [name for name in dirnames if name not in IGNORED_DIRECTORIES and not name.startswith(".")]
            for filename in sorted(filenames):
                if filename.lower().endswith(".md"):
                    yield Path(dirpath) / filename

    def iter_notes(self, subdirectory: str | None = None) -> Iterator[Note]:
        for path in self.iter_note_paths(subdirectory):
            text = self._read_file(path)
            if text is None:
                continue
            try:
                stat = path.stat()
                mtime, size = stat.st_mtime, stat.st_size
            except OSError:  # pragma: no cover
                mtime, size = 0.0, len(text)
            yield Note.from_text(text, path=path, relative_path=self.relative(path), mtime=mtime, size=size)

    def notes(self, subdirectory: str | None = None) -> list[Note]:
        return list(self.iter_notes(subdirectory))

    def count_notes(self) -> int:
        return sum(1 for _ in self.iter_note_paths())

    # ----------------------------------------------------------- writes
    def write_text(self, relative_path: str | Path, text: str) -> Path:
        """Atomically write `text`, creating parent directories.

        Returns the absolute path written.
        """
        path = self.note_path(relative_path)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write(path, text)
        return path

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=str(path.parent),
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def create_note(
        self,
        relative_path: str | Path,
        *,
        title: str,
        note_type: str,
        summary: str,
        tags: Iterable[str] = (),
        quick_summary: Iterable[str] | str = (),
        sections: Iterable[tuple[str, str]] = (),
        extra_metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> Note:
        """Create a note that satisfies the summary standard.

        There is deliberately no way to create a knowledge note without a
        summary: `build_note_text` requires one, and this is the only
        creation path.
        """
        if not str(summary or "").strip():
            raise VaultError(f"A {note_type} note needs a one-sentence summary; {title!r} was given none.")
        path = self.note_path(relative_path)
        with self._lock:
            if path.exists() and not overwrite:
                existing = self.read(self.relative(path))
                if existing is not None:
                    return existing
            text = build_note_text(
                title=title,
                note_type=note_type,
                summary=summary,
                tags=tags,
                quick_summary=quick_summary,
                sections=sections,
                extra_metadata=extra_metadata,
            )
            self.write_text(self.relative(path), text)
        log.info("Vault note created: %s", self.relative(path))
        return self.read(self.relative(path))  # type: ignore[return-value]

    def write_note(self, note: Note, *, touch: bool = True) -> Note:
        """Persist a `Note` object, refreshing its `updated` timestamp.

        `touch=False` exists for the index writer, which rewrites a
        generated file whose timestamp should reflect the generation, not
        a knowledge change.
        """
        metadata = dict(note.metadata)
        if touch:
            metadata["updated"] = utc_now()
        updated = Note(
            path=note.path,
            relative_path=note.relative_path,
            metadata=metadata,
            body=note.body,
            malformed=note.malformed,
        )
        self.write_text(updated.relative_path, updated.to_markdown())
        return self.read(updated.relative_path)  # type: ignore[return-value]

    def update_note(
        self,
        relative_path: str | Path,
        mutate: Callable[[Note], Note],
        *,
        touch: bool = True,
    ) -> Note | None:
        """Read-modify-write one note under the vault lock.

        The lock is held across the whole read/modify/write so two threads
        appending to the same note (the daily note, most often) cannot
        each read the same version and one lose the other's text.
        """
        with self._lock:
            note = self.read(relative_path)
            if note is None:
                return None
            changed = mutate(note)
            if changed is None:
                return note
            return self.write_note(changed, touch=touch)

    def append_to_section(
        self,
        relative_path: str | Path,
        heading: str,
        text: str,
        *,
        create_missing: bool = True,
    ) -> Note | None:
        """Append text under one heading, creating the heading if needed."""
        from vault.note import extract_section, replace_section

        def mutate(note: Note) -> Note:
            existing = extract_section(note.body, heading)
            if not existing and not create_missing:
                return note
            merged = f"{existing}\n{text.strip()}".strip() if existing and existing != "_Nothing recorded yet._" else text.strip()
            note.body = replace_section(note.body, heading, merged)
            return note

        return self.update_note(relative_path, mutate)

    # --------------------------------------------------------- lifecycle
    def move_note(self, source: str | Path, destination: str | Path) -> Note | None:
        """Move a note within the vault (mission active -> completed).

        The destination directory is created; an existing destination is
        never silently clobbered -- a numeric suffix is added instead,
        because a mission record is evidence and overwriting one loses it.
        """
        with self._lock:
            source_path = self.note_path(source)
            if not source_path.is_file():
                return None
            target = self.note_path(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                stem, suffix = target.stem, target.suffix
                counter = 2
                while target.exists():
                    target = target.with_name(f"{stem}-{counter}{suffix}")
                    counter += 1
            shutil.move(str(source_path), str(target))
            log.info("Vault note moved: %s -> %s", self.relative(source_path), self.relative(target))
            return self.read(self.relative(target))

    def archive_note(self, relative_path: str | Path, archive_dir: str = "archive") -> Note | None:
        """Move a note out of the way without ever deleting it.

        Nothing in this system deletes a note. Knowledge that turned out
        to be wrong is superseded and archived, so the provenance of a
        change survives -- which is the whole point of keeping history.
        """
        source = self.note_path(relative_path)
        stamp = datetime.now().strftime("%Y%m%d")
        return self.move_note(relative_path, f"{archive_dir}/{stamp}-{source.name}")

    def unique_path(self, directory: str, title: str, *, fallback: str = "note") -> str:
        """A free `<directory>/<slug>.md`, suffixed if the slug is taken."""
        slug = slugify(title, fallback=fallback)
        candidate = f"{directory.rstrip('/')}/{slug}.md"
        counter = 2
        while self.note_exists(candidate):
            candidate = f"{directory.rstrip('/')}/{slug}-{counter}.md"
            counter += 1
        return candidate

    # ------------------------------------------------------- diagnostics
    def describe(self) -> dict[str, Any]:
        """A cheap, log-safe account of the vault's state."""
        notes = self.notes()
        by_type: dict[str, int] = {}
        malformed: list[str] = []
        without_summary: list[str] = []
        for note in notes:
            by_type[note.note_type] = by_type.get(note.note_type, 0) + 1
            if note.malformed:
                malformed.append(note.relative_path)
            elif not note.has_summary and note.note_type != "index":
                without_summary.append(note.relative_path)
        return {
            "root": str(self.root),
            "exists": self.exists(),
            "notes": len(notes),
            "by_type": dict(sorted(by_type.items())),
            "malformed": malformed[:20],
            "without_summary": without_summary[:20],
        }


_MANAGER: VaultManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_vault(root: Path | str | None = None) -> VaultManager:
    """The process-wide vault.

    Passing an explicit `root` always builds a fresh manager (tests do
    this against a temporary directory); the cached singleton is only
    used for the configured vault.
    """
    global _MANAGER
    if root is not None:
        return VaultManager(root)
    if _MANAGER is None:
        with _MANAGER_LOCK:
            if _MANAGER is None:
                _MANAGER = VaultManager()
    return _MANAGER


def reset_vault() -> None:
    """Drop the cached manager. For tests and an explicit reconfiguration."""
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = None
