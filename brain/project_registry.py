"""Known projects JARVIS can open/reference by name ("my JARVIS project").

"My JARVIS project" always resolves to JARVIS's own repository root -- no
configuration required, since that path is trivially knowable at runtime.
Additional real projects are configured via `JARVIS_KNOWN_PROJECTS`, a
comma-separated `name:path` list (e.g.
`"website:C:/dev/site,api:C:/dev/api"`), so referencing another project by
name never requires a code change.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])


def _known_projects() -> dict[str, str]:
    projects = {"jarvis": _REPO_ROOT}
    raw = os.getenv("JARVIS_KNOWN_PROJECTS", "")
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        name, path = entry.split(":", 1)
        name = " ".join(name.strip().lower().split())
        path = path.strip()
        if name and path:
            projects[name] = path
    return projects


def resolve_project(name: str) -> tuple[str, str] | None:
    """Returns (canonical_name, path) for a spoken project name, or None.

    Normalizes away the "my "/"the " prefix and a trailing " project" that
    the caller (`brain/context_router.py`) typically hasn't already
    stripped, so "jarvis", "my jarvis project", and "the jarvis project"
    all resolve identically.
    """
    key = " ".join(name.strip().lower().split())
    for prefix in ("my ", "the "):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.endswith(" project"):
        key = key[: -len(" project")]
    key = key.strip()
    if not key:
        return None
    projects = _known_projects()
    if key in projects:
        return key, projects[key]
    return None
