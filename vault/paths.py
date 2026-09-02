"""Where the vault lives, and what its folders are called.

One module owns the layout so nothing else ever hard-codes a folder name.
The location follows the project's existing configuration rule: it is a
`config/settings.py` setting (`JARVIS_VAULT_PATH`), resolved relative to
the repository root -- never to the process's working directory, which is
the mistake that once made `.env` loading depend on where JARVIS was
started from.
"""
from __future__ import annotations

from pathlib import Path

from config.settings import PROJECT_ROOT, _text

#: Folder names, relative to the vault root. These are also the strings
#: that appear in every `relative_path` JARVIS writes into a note, so
#: renaming one is a migration, not a rename.
IDENTITY_DIR = "identity"
USER_DIR = "user"
JOBS_DIR = "jobs"
SKILLS_DIR = "skills"
PROJECTS_DIR = "projects"
LESSONS_DIR = "lessons"
MISSIONS_DIR = "missions"
MISSIONS_ACTIVE_DIR = "missions/active"
MISSIONS_COMPLETED_DIR = "missions/completed"
DAILY_DIR = "daily"
STATE_DIR = "state"
SYSTEM_DIR = "system"

#: Every directory the bootstrap creates, in display order.
VAULT_DIRECTORIES: tuple[str, ...] = (
    IDENTITY_DIR,
    USER_DIR,
    JOBS_DIR,
    SKILLS_DIR,
    PROJECTS_DIR,
    LESSONS_DIR,
    MISSIONS_ACTIVE_DIR,
    MISSIONS_COMPLETED_DIR,
    DAILY_DIR,
    STATE_DIR,
    SYSTEM_DIR,
)

#: The generated map of the whole vault, at the vault root.
VAULT_INDEX_FILE = "VAULT_INDEX.md"
#: A per-directory index, generated for the browsable collections.
DIRECTORY_INDEX_FILE = "INDEX.md"
#: Directories that get their own `INDEX.md`.
INDEXED_DIRECTORIES: tuple[str, ...] = (JOBS_DIR, SKILLS_DIR, PROJECTS_DIR, LESSONS_DIR)

#: The machine cache. It lives OUTSIDE the vault on purpose: the vault is
#: what the user opens in Obsidian, and a binary/JSON cache sitting in it
#: is clutter that Obsidian would try to index. Markdown stays canonical;
#: this file can be deleted at any time and is simply rebuilt.
INDEX_CACHE_FILE = "vault_index_cache.json"

#: Files and folders inside the vault that are never notes.
IGNORED_DIRECTORIES = frozenset({".obsidian", ".trash", ".git", "__pycache__", ".jarvis"})


def default_vault_path() -> Path:
    """The configured vault root, as an absolute path.

    `JARVIS_VAULT_PATH` may be absolute (a real Obsidian vault anywhere on
    the machine -- the normal case once the user has one) or relative, in
    which case it is resolved against the repository root.
    """
    raw = _text("JARVIS_VAULT_PATH", "")
    if raw:
        candidate = Path(raw).expanduser()
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
    return PROJECT_ROOT / "data" / "vault"


def default_cache_path() -> Path:
    raw = _text("JARVIS_VAULT_CACHE_PATH", "")
    if raw:
        candidate = Path(raw).expanduser()
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
    return PROJECT_ROOT / "data" / INDEX_CACHE_FILE


def slugify(text: str, *, fallback: str = "note") -> str:
    """A filename a human can read in Obsidian's file list.

    Wikilinks resolve on note TITLE, not filename, so the slug only has to
    be stable and legible -- it never has to encode the title exactly.
    """
    import re

    cleaned = re.sub(r"[^\w\s-]", "", (text or "").strip().lower(), flags=re.UNICODE)
    cleaned = re.sub(r"[\s_-]+", "-", cleaned).strip("-")
    return cleaned[:80] or fallback
