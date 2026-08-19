"""Code-oriented tools: project inspection, targeted reading, and edits.

These sit ON TOP of `tools/files.py` and `tools/terminal.py` rather than
re-implementing file or process handling. What they add is the shape a
coding agent actually needs:

- a compact repository overview instead of a raw recursive listing;
- reading a bounded slice of a file with line numbers, so a large file
  does not blow the model's context;
- an exact-match replacement edit that reports whether it applied and
  can be reverted, instead of blind whole-file overwrites;
- a syntax check, so an obviously broken edit is caught before a test
  run is wasted on it.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

log = logging.getLogger("jarvis.code")

# Directories that are never source code worth showing an agent.
IGNORED_DIRECTORIES = {
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "dist", "build", ".idea", ".vscode", ".cache", "site-packages",
}
IGNORED_PREFIXES = (".venv", "venv", ".jarvis-improvement-worktrees")

SOURCE_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".go", ".rs", ".rb",
    ".cs", ".c", ".h", ".cpp", ".hpp", ".php", ".swift", ".scala", ".sh", ".ps1",
    ".sql", ".html", ".css", ".scss", ".json", ".yaml", ".yml", ".toml", ".ini", ".md",
}

PROJECT_MARKERS = (
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "pytest.ini", "tox.ini",
    "package.json", "tsconfig.json", "go.mod", "Cargo.toml", "pom.xml", "build.gradle",
    "Makefile", "CLAUDE.md", "README.md",
)


def _skip(path: Path) -> bool:
    return any(
        part in IGNORED_DIRECTORIES or part.startswith(IGNORED_PREFIXES)
        for part in path.parts
    )


def inspect_project(path: str, max_files: int = 200) -> dict:
    """Describe a project: markers, entry points, layout, and file counts.

    Bounded on purpose -- `max_files` caps how many source paths are
    returned, and the reply says so honestly (`truncated`) instead of
    silently showing a partial picture.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        return {"success": False, "message": "That project directory does not exist.", "error": "path_not_found", "path": str(root)}

    markers = [name for name in PROJECT_MARKERS if (root / name).is_file()]
    files: list[str] = []
    by_suffix: dict[str, int] = {}
    total = 0
    for item in sorted(root.rglob("*")):
        if item.is_dir() or _skip(item.relative_to(root)):
            continue
        suffix = item.suffix.lower()
        if suffix not in SOURCE_SUFFIXES:
            continue
        total += 1
        by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
        if len(files) < max_files:
            files.append(str(item.relative_to(root)).replace("\\", "/"))

    top_level = sorted(
        item.name for item in root.iterdir()
        if item.is_dir() and not _skip(Path(item.name))
    )
    entry_points = [
        name for name in ("main.py", "app.py", "manage.py", "__main__.py", "index.js", "server.js")
        if (root / name).is_file()
    ]
    language = max(by_suffix, key=by_suffix.get) if by_suffix else None
    return {
        "success": True,
        "verified": True,
        "message": f"Inspected {root.name}: {total} source files across {len(top_level)} top-level directories.",
        "path": str(root),
        "project_markers": markers,
        "entry_points": entry_points,
        "top_level_directories": top_level,
        "file_counts_by_suffix": dict(sorted(by_suffix.items(), key=lambda kv: -kv[1])),
        "primary_language": language,
        "source_file_count": total,
        "files": files,
        "truncated": total > len(files),
    }


def read_code(path: str, start_line: int = 1, end_line: int | None = None, max_lines: int = 400) -> dict:
    """Read a bounded, line-numbered slice of a source file."""
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        return {"success": False, "message": "That file does not exist.", "error": "file_not_found", "path": str(target)}
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"success": False, "message": "The file could not be read.", "error": str(exc), "path": str(target)}

    start = max(1, int(start_line))
    stop = len(lines) if end_line is None else min(len(lines), int(end_line))
    stop = min(stop, start + max_lines - 1)
    slice_lines = lines[start - 1 : stop]
    numbered = "\n".join(f"{number:>5}| {text}" for number, text in enumerate(slice_lines, start))
    return {
        "success": True,
        "verified": True,
        "message": numbered or "(empty selection)",
        "path": str(target),
        "contents": "\n".join(slice_lines),
        "numbered_contents": numbered,
        "start_line": start,
        "end_line": stop,
        "total_lines": len(lines),
        "truncated": stop < len(lines),
    }


def edit_code(path: str, old_text: str, new_text: str, expect_unique: bool = True) -> dict:
    """Replace an exact substring in a file.

    Refuses ambiguity: by default the anchor must occur exactly once, so
    an edit can never silently land in the wrong place. Returns the
    previous file content so a caller can revert.
    """
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        return {"success": False, "message": "That file does not exist.", "error": "file_not_found", "path": str(target)}
    if not old_text:
        return {"success": False, "message": "The text to replace must not be empty.", "error": "empty_anchor", "path": str(target)}
    try:
        original = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {"success": False, "message": "The file could not be read.", "error": str(exc), "path": str(target)}

    occurrences = original.count(old_text)
    if occurrences == 0:
        return {"success": False, "message": "The text to replace was not found in that file.", "error": "anchor_not_found", "path": str(target)}
    if occurrences > 1 and expect_unique:
        return {
            "success": False,
            "message": f"The text to replace appears {occurrences} times; I need a unique anchor.",
            "error": "anchor_not_unique",
            "occurrences": occurrences,
            "path": str(target),
        }

    updated = original.replace(old_text, new_text, 1 if expect_unique else -1)
    target.write_text(updated, encoding="utf-8")
    applied = target.read_text(encoding="utf-8") == updated
    return {
        "success": applied,
        "verified": applied,
        "message": f"Edited {target.name}." if applied else "The edit did not persist.",
        "error": None if applied else "write_verification_failed",
        "path": str(target),
        "occurrences_replaced": 1 if expect_unique else occurrences,
        "previous_contents": original,
        "bytes": target.stat().st_size,
    }


def check_syntax(path: str) -> dict:
    """Parse a Python file and report a syntax error precisely.

    Only Python is genuinely checked. For any other language this
    reports `checked=False` instead of claiming a file is valid it never
    actually parsed.
    """
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        return {"success": False, "message": "That file does not exist.", "error": "file_not_found", "path": str(target)}
    if target.suffix.lower() not in {".py", ".pyi"}:
        return {
            "success": True,
            "verified": False,
            "checked": False,
            "message": f"No syntax checker for {target.suffix or 'this file type'}; it was not validated.",
            "path": str(target),
        }
    source = target.read_text(encoding="utf-8", errors="replace")
    try:
        ast.parse(source, filename=str(target))
    except SyntaxError as exc:
        return {
            "success": False,
            "verified": True,
            "checked": True,
            "message": f"Syntax error in {target.name} at line {exc.lineno}: {exc.msg}",
            "error": "syntax_error",
            "line": exc.lineno,
            "offset": exc.offset,
            "detail": exc.msg,
            "path": str(target),
        }
    return {
        "success": True,
        "verified": True,
        "checked": True,
        "message": f"{target.name} parses cleanly.",
        "path": str(target),
    }


def search_code(path: str, query: str, max_results: int = 60, suffixes: list[str] | None = None) -> dict:
    """Case-insensitive substring search across a project's source files."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return {"success": False, "message": "That path does not exist.", "error": "path_not_found", "path": str(root)}
    wanted = {s.lower() if s.startswith(".") else f".{s.lower()}" for s in (suffixes or [])} or SOURCE_SUFFIXES
    needle = query.lower()
    matches: list[dict] = []
    scanned = 0
    targets = [root] if root.is_file() else sorted(root.rglob("*"))
    for item in targets:
        if item.is_dir():
            continue
        relative = item.relative_to(root) if item != root else Path(item.name)
        if _skip(relative) or item.suffix.lower() not in wanted:
            continue
        scanned += 1
        try:
            for number, line in enumerate(item.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if needle in line.lower():
                    matches.append({"path": str(item), "line": number, "text": line.strip()[:200]})
                    if len(matches) >= max_results:
                        return {
                            "success": True, "verified": True, "matches": matches, "truncated": True,
                            "files_scanned": scanned, "path": str(root),
                            "message": f"Found {len(matches)} matches (truncated).",
                        }
        except OSError:
            continue
    return {
        "success": True,
        "verified": True,
        "matches": matches,
        "truncated": False,
        "files_scanned": scanned,
        "path": str(root),
        "message": f"Found {len(matches)} matches in {scanned} files.",
    }
