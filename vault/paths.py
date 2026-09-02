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
#: Two-level preferences. `preferences/global.md` applies to every
#: mission; `preferences/jobs/<slug>.md` applies only to its own Job and
#: is loaded only after that Job has been selected.
PREFERENCES_DIR = "preferences"
JOB_PREFERENCES_DIR = "preferences/jobs"
#: Superseded knowledge. Never deleted, never scanned.
ARCHIVE_DIR = "archive"
ARCHIVE_PREFERENCES_DIR = "archive/preferences"
ARCHIVE_METHODS_DIR = "archive/methods"
ARCHIVE_NOTES_DIR = "archive/notes"

#: Every directory the bootstrap creates, in display order.
VAULT_DIRECTORIES: tuple[str, ...] = (
    IDENTITY_DIR,
    USER_DIR,
    PREFERENCES_DIR,
    JOB_PREFERENCES_DIR,
    JOBS_DIR,
    SKILLS_DIR,
    PROJECTS_DIR,
    LESSONS_DIR,
    MISSIONS_ACTIVE_DIR,
    MISSIONS_COMPLETED_DIR,
    DAILY_DIR,
    STATE_DIR,
    SYSTEM_DIR,
    ARCHIVE_DIR,
    ARCHIVE_PREFERENCES_DIR,
    ARCHIVE_METHODS_DIR,
    ARCHIVE_NOTES_DIR,
)

#: The global preference note, loaded for every full mission by policy --
#: never discovered by a scan, because it must apply whether or not its
#: words happen to match the request.
GLOBAL_PREFERENCES_NOTE = f"{PREFERENCES_DIR}/global.md"

# ---------------------------------------------------------------- scanning
#
# Why two kinds of note are kept OUT of the ordinary summary scan.
#
# ARCHIVE: superseded knowledge is kept so a decision's history survives,
# but an archived rule that could still be retrieved would be an archived
# rule that can still change behaviour -- which defeats the point of
# superseding it. Archive is reachable only through the explicit
# history/archive tools.
#
# JOB PREFERENCES: there is one per Job, so they scale with the Jobs and
# would come to dominate a scan while carrying no signal about WHICH Job
# fits a request -- they say how to do a job, not when it applies. They
# are loaded deterministically, after their Job has been selected.
#
# Neither is hidden from an explicit lookup: resolving `[[Preferences -
# Send Email]]` from a Job note is a deliberate reference, not a search.

#: Why a note is excluded from the ordinary scan. "" means it is active.
EXCLUDED_ARCHIVE = "archive"
EXCLUDED_JOB_PREFERENCE = "job_preference"


def exclusion_reason(relative_path: str) -> str:
    """Why `relative_path` is kept out of the ordinary scan, or "".

    One predicate, consulted once per note by `VaultIndex.refresh`, so
    every consumer of the index inherits the rule instead of each
    re-deriving it.
    """
    path = (relative_path or "").replace("\\", "/").lstrip("./")
    if path == ARCHIVE_DIR or path.startswith(ARCHIVE_DIR + "/"):
        return EXCLUDED_ARCHIVE
    if path.startswith(JOB_PREFERENCES_DIR + "/"):
        return EXCLUDED_JOB_PREFERENCE
    return ""

#: The generated map of the ACTIVE vault, at the vault root.
VAULT_INDEX_FILE = "VAULT_INDEX.md"
#: The generated map of the archive. Deliberately a SEPARATE file, and
#: deliberately not part of priming: it exists so the user can browse
#: what was superseded, not so JARVIS can retrieve it by accident.
ARCHIVE_INDEX_FILE = "archive/ARCHIVE_INDEX.md"
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
