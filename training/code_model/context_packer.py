"""Compact repository-context packer (Phase 4).

Real, evidence-driven selection of relevant files/snippets for a
multi-file coding task -- not a full semantic code-search engine, and
deliberately not a blind whole-repository dump. Matches this codebase's
existing "small, deterministic, evidence-driven" style
(`brain/improvement_classifier.py`, `brain/improvement_evaluator.py`):
selection is based on structural evidence already available (which files a
diff touched, which files a Python import graph connects, which files a
keyword/path search surfaces), not an ML embedding index.

Used by:
  - `training/code_model/dataset_formatting.py`, to build the repository
    context block shown to the model during SFT (Phase 3).
  - `training/code_model/student_adapter.py`, to build the repository
    context block shown to the model at inference time in the benchmark
    harness / real coding-agent loop (Phase 11/14).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_FROM_IMPORT_RE = re.compile(r"^\s*from\s+([.\w]+)\s+import\s+(.+)$", re.MULTILINE)
_PLAIN_IMPORT_RE = re.compile(r"^\s*import\s+([.\w]+)", re.MULTILINE)
_PY_EXCLUDED_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".jarvis-improvement-worktrees", ".jarvis-learning-variants"}

MAX_FILE_CHARS = 4000
MAX_TOTAL_CHARS = 16000
MAX_FILES = 8


@dataclass
class ContextFile:
    path: str
    content: str
    reason: str  # why this file was selected -- kept for debuggability, not shown to the model


@dataclass
class RepositoryContext:
    files: list[ContextFile] = field(default_factory=list)
    truncated: bool = False

    def render(self) -> str:
        """Render as a compact text block for a prompt. Files are truncated
        (never silently dropped mid-file) so the model always sees a
        complete, if shortened, view of each selected file."""
        parts = []
        for f in self.files:
            parts.append(f"--- {f.path} ---\n{f.content}")
        text = "\n\n".join(parts)
        if self.truncated:
            text += "\n\n[additional relevant files omitted for length]"
        return text


def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in _PY_EXCLUDED_DIRS for part in path.parts):
            continue
        yield path


def _local_imports(content: str, package_hint: str | None) -> set[str]:
    """Module names a file imports that plausibly resolve within the same
    repository (relative imports, or absolute imports sharing the given
    top-level package hint). For `from X import Y[, Z...]`, also yields
    `X.Y` (and `X.Z`, ...) candidates -- `Y` is commonly a submodule
    (`pkg/helper.py`), not only a symbol defined inside `X/__init__.py`."""
    modules: set[str] = set()

    def relevant(name: str) -> bool:
        return bool(name) and (name.startswith(".") or (package_hint and name.split(".")[0] == package_hint))

    for from_module, imported in _FROM_IMPORT_RE.findall(content):
        stripped_module = from_module.lstrip(".")
        if relevant(from_module):
            modules.add(stripped_module)
            for name in imported.split(","):
                name = name.strip().split(" as ")[0].strip().strip("()")
                if name and name.isidentifier():
                    modules.add(f"{stripped_module}.{name}" if stripped_module else name)

    for absolute in _PLAIN_IMPORT_RE.findall(content):
        if relevant(absolute):
            modules.add(absolute.lstrip("."))

    return modules


def _module_to_candidate_paths(module: str, root: Path) -> list[Path]:
    parts = module.split(".")
    candidates = [root.joinpath(*parts).with_suffix(".py"), root.joinpath(*parts, "__init__.py")]
    return [c for c in candidates if c.exists()]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


def pack_repository_context(
    repository_root: str | Path,
    *,
    seed_files: list[str] | None = None,
    keywords: list[str] | None = None,
    test_files: list[str] | None = None,
    max_files: int = MAX_FILES,
    max_total_chars: int = MAX_TOTAL_CHARS,
) -> RepositoryContext:
    """Build a compact, evidence-selected repository context.

    Selection order (most to least direct evidence):
    1. `seed_files` -- files evidence already names directly (e.g. the
       files a diff touched, or files a failure's traceback mentions).
    2. Files those seed files import (one hop of the local import graph),
       so the model sees the call-chain/dependency neighborhood, not just
       the isolated file.
    3. `test_files` -- test files evidence names, so the model sees the
       acceptance criteria.
    4. `keywords` -- a plain substring search across the repo's Python
       files, for cases where no seed file is known at all.
    """
    root = Path(repository_root)
    selected: dict[str, ContextFile] = {}

    def add(relative: Path, reason: str) -> None:
        if len(selected) >= max_files:
            return
        key = str(relative).replace("\\", "/")
        if key in selected:
            return
        full = root / relative
        if not full.is_file():
            return
        try:
            content = full.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return
        selected[key] = ContextFile(path=key, content=_truncate(content, MAX_FILE_CHARS), reason=reason)

    package_hint = next((p.name for p in root.iterdir() if p.is_dir() and (p / "__init__.py").exists()), None) if root.is_dir() else None

    for raw in seed_files or []:
        add(Path(raw), "seed file named by evidence")

    for key in list(selected.keys()):
        if len(selected) >= max_files:
            break
        full_content = selected[key].content
        for module in _local_imports(full_content, package_hint):
            for candidate in _module_to_candidate_paths(module, root):
                add(candidate.relative_to(root), f"imported by {key}")

    for raw in test_files or []:
        add(Path(raw), "acceptance/regression test named by evidence")

    if keywords and root.is_dir():
        for path in _iter_python_files(root):
            if len(selected) >= max_files:
                break
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if any(kw.lower() in content.lower() for kw in keywords):
                add(path.relative_to(root), f"matched keyword search")

    total = sum(len(f.content) for f in selected.values())
    truncated = False
    files = list(selected.values())
    if total > max_total_chars:
        truncated = True
        kept, running = [], 0
        for f in files:
            if running + len(f.content) > max_total_chars:
                break
            kept.append(f)
            running += len(f.content)
        files = kept

    return RepositoryContext(files=files, truncated=truncated)
