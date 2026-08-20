"""Discovers installed/launchable Windows applications and caches an index.

This module owns discovery and caching only -- it never launches anything.
`tools/app_resolver.py` resolves a spoken name against the index built here,
and `tools/applications.py` is the only place that actually launches a
resolved target.

Sources (each individually defensive -- a failure in one source must never
prevent the others from contributing entries):

* Start Menu shortcuts (user + system, recursive ``*.lnk``).
* The Windows "App Paths" registry key (HKCU/HKLM, 32 and 64-bit views).
* ``Get-StartApps`` (covers UWP/Store apps via their AppUserModelID, reusing
  `tools.applications`'s existing cached catalog rather than a second
  PowerShell round trip).
* The "installed programs" (Uninstall) registry keys, used only as a
  low-confidence supplemental source for display names / install locations.

Each raw Windows-facing fetch is split from the pure "build AppEntry list"
step specifically so tests can substitute the raw fetch and never need a
real registry, filesystem, or PowerShell call.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

log = logging.getLogger("jarvis.app_index")

try:
    import winreg
except ImportError:  # pragma: no cover - this module only ever runs on Windows
    winreg = None


CACHE_PATH = Path(os.getenv("JARVIS_APP_INDEX_CACHE_PATH", "data/app_index_cache.json"))
CACHE_TTL_SECONDS = max(0.0, float(os.getenv("JARVIS_APP_INDEX_CACHE_TTL", str(24 * 3600))))

_TRAILING_WORDS = {"app", "application"}


def normalize_app_name(name: str) -> str:
    """Normalize a spoken/typed app name for identity comparison.

    Lower-cases, strips a trailing ``.exe``, folds punctuation/underscores/
    dashes to spaces, collapses whitespace, and drops a trailing filler word
    such as "app" (but never down to an empty string).
    """
    text = (name or "").strip().casefold()
    text = re.sub(r"\.exe$", "", text)
    text = re.sub(r"[-_.]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split(" ") if text else []
    while len(tokens) > 1 and tokens[-1] in _TRAILING_WORDS:
        tokens = tokens[:-1]
    return " ".join(tokens)


@dataclass
class AppEntry:
    display_name: str
    normalized_name: str
    aliases: list[str] = field(default_factory=list)
    launch_type: str = "exe"  # "exe" | "shortcut" | "uwp" | "path"
    launch_target: str = ""
    arguments: list[str] = field(default_factory=list)
    executable_name: str | None = None
    source: str = "unknown"
    source_path: str | None = None
    app_user_model_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def token_set(self) -> set[str]:
        return set(self.normalized_name.split()) if self.normalized_name else set()


@dataclass
class AppIndex:
    entries: list[AppEntry] = field(default_factory=list)
    built_at: float = 0.0
    merge_report: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "built_at": self.built_at,
            "entries": [asdict(e) for e in self.entries],
            "merge_report": self.merge_report,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "AppIndex":
        entries = [AppEntry(**e) for e in payload["entries"]]
        return cls(entries=entries, built_at=float(payload.get("built_at", 0.0)), merge_report=payload.get("merge_report", []))


# ---------------------------------------------------------------------------
# Source 1: Start Menu shortcuts
# ---------------------------------------------------------------------------

def _start_menu_dirs() -> list[Path]:
    dirs = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        dirs.append(Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    program_data = os.environ.get("PROGRAMDATA")
    if program_data:
        dirs.append(Path(program_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return dirs


def _iter_lnk_files(dirs: list[Path]) -> list[Path]:
    found: list[Path] = []
    for base in dirs:
        try:
            if not base.is_dir():
                continue
            found.extend(p for p in base.rglob("*.lnk") if p.is_file())
        except OSError:
            log.debug("Start Menu scan failed for %s", base, exc_info=True)
    return found


_LNK_HEADER_MAGIC = b"L\x00\x00\x00"


def _resolve_shortcut_target(lnk_path: Path) -> dict | None:
    """Best-effort MS-SHLLINK parse extracting the LinkInfo LocalBasePath.

    Only reads enough of the format to recover the target executable's
    path for indexing/alias purposes -- launching a shortcut always goes
    through the shell (``explorer.exe <path.lnk>``), never this parsed
    value, so a parse failure here only means a slightly less specific
    index entry, never a broken launch.
    """
    try:
        data = lnk_path.read_bytes()
    except OSError:
        return None

    if len(data) < 76 or data[:4] != _LNK_HEADER_MAGIC:
        return None

    flags = int.from_bytes(data[20:24], "little")
    has_idlist = bool(flags & 0x1)
    has_link_info = bool(flags & 0x2)

    offset = 76
    try:
        if has_idlist:
            if offset + 2 > len(data):
                return None
            idlist_size = int.from_bytes(data[offset:offset + 2], "little")
            offset += 2 + idlist_size

        if not has_link_info or offset + 4 > len(data):
            return None

        link_info_start = offset
        link_info_size = int.from_bytes(data[offset:offset + 4], "little")
        if link_info_size < 20 or link_info_start + link_info_size > len(data):
            return None

        link_info_flags = int.from_bytes(data[offset + 8:offset + 12], "little")
        local_base_path_offset = int.from_bytes(data[offset + 16:offset + 20], "little")

        if not (link_info_flags & 0x1) or not local_base_path_offset:
            return None

        abs_offset = link_info_start + local_base_path_offset
        end = data.find(b"\x00", abs_offset)
        if end == -1 or abs_offset >= len(data):
            return None

        target = data[abs_offset:end].decode("mbcs", errors="replace")
    except (IndexError, ValueError):
        return None

    return {"target": target} if target else None


def _build_start_menu_entry(lnk_path: Path) -> AppEntry:
    display_name = lnk_path.stem
    target_info = _resolve_shortcut_target(lnk_path)
    executable_name = None
    aliases = {display_name}
    if target_info and target_info.get("target"):
        executable_name = Path(target_info["target"]).name
        aliases.add(Path(executable_name).stem)
    return AppEntry(
        display_name=display_name,
        normalized_name=normalize_app_name(display_name),
        aliases=sorted({normalize_app_name(a) for a in aliases if a}),
        launch_type="shortcut",
        launch_target=str(lnk_path),
        arguments=[],
        executable_name=executable_name,
        source="start_menu",
        source_path=str(lnk_path),
        app_user_model_id=None,
        metadata={},
    )


def discover_start_menu_shortcuts(dirs: list[Path] | None = None) -> list[AppEntry]:
    lnk_files = _iter_lnk_files(dirs if dirs is not None else _start_menu_dirs())
    entries = []
    for lnk in lnk_files:
        try:
            entries.append(_build_start_menu_entry(lnk))
        except Exception:
            log.debug("Failed to build Start Menu entry for %s", lnk, exc_info=True)
    return entries


# ---------------------------------------------------------------------------
# Source 2: Windows "App Paths" registry
# ---------------------------------------------------------------------------

_APP_PATHS_KEY = r"Software\Microsoft\Windows\CurrentVersion\App Paths"


def _enumerate_app_paths_raw() -> list[tuple[str, str, str]]:
    """Returns (hive_label, subkey_name, default_value) tuples. Real winreg I/O."""
    if winreg is None:
        return []
    roots = [
        ("HKCU", winreg.HKEY_CURRENT_USER, 0),
        ("HKLM", winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_64KEY),
        ("HKLM", winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_32KEY),
    ]
    results: list[tuple[str, str, str]] = []
    for label, hive, view_flag in roots:
        try:
            with winreg.OpenKey(hive, _APP_PATHS_KEY, 0, winreg.KEY_READ | view_flag) as key:
                index = 0
                while True:
                    try:
                        subkey_name = winreg.EnumKey(key, index)
                    except OSError:
                        break
                    index += 1
                    try:
                        with winreg.OpenKey(hive, f"{_APP_PATHS_KEY}\\{subkey_name}", 0, winreg.KEY_READ | view_flag) as subkey:
                            default_value, _ = winreg.QueryValueEx(subkey, None)
                    except OSError:
                        continue
                    if default_value:
                        results.append((label, subkey_name, str(default_value)))
        except OSError:
            log.debug("App Paths registry scan failed for %s", label, exc_info=True)
    return results


def discover_app_paths_registry(raw_entries: list[tuple[str, str, str]] | None = None) -> list[AppEntry]:
    entries = []
    for label, subkey_name, default_value in (raw_entries if raw_entries is not None else _enumerate_app_paths_raw()):
        target = default_value.strip().strip('"')
        display_name = Path(subkey_name).stem
        executable_name = subkey_name if subkey_name.lower().endswith(".exe") else f"{subkey_name}.exe"
        entries.append(AppEntry(
            display_name=display_name,
            normalized_name=normalize_app_name(display_name),
            aliases=sorted({normalize_app_name(display_name), normalize_app_name(executable_name)}),
            launch_type="exe",
            launch_target=target,
            arguments=[],
            executable_name=executable_name,
            source="app_paths",
            source_path=f"{label}\\{_APP_PATHS_KEY}\\{subkey_name}",
            app_user_model_id=None,
            metadata={},
        ))
    return entries


# ---------------------------------------------------------------------------
# Source 3: UWP / Store apps, via the existing Get-StartApps catalog
# ---------------------------------------------------------------------------

def _uwp_raw_entries() -> list[dict]:
    """Reuses tools.applications's existing cached Get-StartApps catalog.

    Imported lazily to avoid a module-load-time circular import (applications
    -> app_resolver -> app_index -> applications).
    """
    try:
        from tools import applications as _applications
        return _applications._start_apps_catalog()
    except Exception:
        log.debug("Get-StartApps catalog fetch failed", exc_info=True)
        return []


def discover_uwp_apps(raw_entries: list[dict] | None = None) -> list[AppEntry]:
    entries = []
    for item in (raw_entries if raw_entries is not None else _uwp_raw_entries()):
        name = str(item.get("Name") or "").strip()
        app_id = str(item.get("AppID") or "").strip()
        if not name or not app_id:
            continue
        entries.append(AppEntry(
            display_name=name,
            normalized_name=normalize_app_name(name),
            aliases=sorted({normalize_app_name(name)}),
            launch_type="uwp",
            launch_target=f"shell:AppsFolder\\{app_id}",
            arguments=[],
            executable_name=None,
            source="start_apps",
            source_path=None,
            app_user_model_id=app_id,
            metadata={},
        ))
    return entries


# ---------------------------------------------------------------------------
# Source 4: installed programs (Uninstall registry) -- supplemental only
# ---------------------------------------------------------------------------

_UNINSTALL_KEYS = [
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall",
    r"Software\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
]


def _enumerate_uninstall_raw() -> list[tuple[str, str, str]]:
    """Returns (display_name, install_location, source_path) tuples."""
    if winreg is None:
        return []
    roots = [
        ("HKCU", winreg.HKEY_CURRENT_USER, 0),
        ("HKLM", winreg.HKEY_LOCAL_MACHINE, 0),
    ]
    results: list[tuple[str, str, str]] = []
    for label, hive, view_flag in roots:
        for base_key in _UNINSTALL_KEYS:
            try:
                with winreg.OpenKey(hive, base_key, 0, winreg.KEY_READ | view_flag) as key:
                    index = 0
                    while True:
                        try:
                            subkey_name = winreg.EnumKey(key, index)
                        except OSError:
                            break
                        index += 1
                        try:
                            with winreg.OpenKey(hive, f"{base_key}\\{subkey_name}", 0, winreg.KEY_READ | view_flag) as subkey:
                                try:
                                    display_name, _ = winreg.QueryValueEx(subkey, "DisplayName")
                                except OSError:
                                    continue
                                try:
                                    install_location, _ = winreg.QueryValueEx(subkey, "InstallLocation")
                                except OSError:
                                    install_location = ""
                                try:
                                    system_component, _ = winreg.QueryValueEx(subkey, "SystemComponent")
                                except OSError:
                                    system_component = 0
                                if system_component:
                                    continue
                        except OSError:
                            continue
                        if display_name:
                            results.append((str(display_name), str(install_location or ""), f"{label}\\{base_key}\\{subkey_name}"))
            except OSError:
                log.debug("Uninstall registry scan failed for %s\\%s", label, base_key, exc_info=True)
    return results


def discover_installed_programs(raw_entries: list[tuple[str, str, str]] | None = None) -> list[AppEntry]:
    """Low-confidence supplemental entries: a display name + (maybe) a lone top-level exe.

    Never trusts an install location with more than one plausible top-level
    executable -- ambiguity there means "add no launch target", not "guess".
    """
    entries = []
    for display_name, install_location, source_path in (raw_entries if raw_entries is not None else _enumerate_uninstall_raw()):
        launch_target = ""
        launch_type = "unknown"
        executable_name = None
        if install_location:
            try:
                install_dir = Path(install_location)
                if install_dir.is_dir():
                    top_level_exes = [p for p in install_dir.iterdir() if p.is_file() and p.suffix.lower() == ".exe"]
                    if len(top_level_exes) == 1:
                        launch_target = str(top_level_exes[0])
                        launch_type = "exe"
                        executable_name = top_level_exes[0].name
            except OSError:
                pass
        if launch_type == "unknown":
            # Alias-only contribution: no safe launch target could be inferred.
            continue
        entries.append(AppEntry(
            display_name=display_name,
            normalized_name=normalize_app_name(display_name),
            aliases=sorted({normalize_app_name(display_name)}),
            launch_type=launch_type,
            launch_target=launch_target,
            arguments=[],
            executable_name=executable_name,
            source="installed_programs",
            source_path=source_path,
            app_user_model_id=None,
            metadata={"install_location": install_location},
        ))
    return entries


# ---------------------------------------------------------------------------
# Merge / dedup
# ---------------------------------------------------------------------------

_SOURCE_PRIORITY = {"start_menu": 4, "app_paths": 3, "start_apps": 2, "installed_programs": 1, "unknown": 0}


class _UnionFind:
    def __init__(self, size: int):
        self._parent = list(range(size))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _merge_entries(entries: list[AppEntry]) -> tuple[list[AppEntry], list[dict]]:
    """Groups entries that are almost certainly the same app.

    Two entries are the same identity if they share a normalized display
    name (the common case -- Start Menu + App Paths + Get-StartApps all
    tend to agree on the human-readable name) OR share an executable name
    (catches cases like a generically-named App Paths key "chrome" next to
    a Start Menu shortcut "Google Chrome" -- same exe, different label).
    Using either signal alone under-merges: many real UWP entries carry no
    executable_name at all, so name-only matching is required for those to
    ever join their Start Menu counterpart.
    """
    if not entries:
        return [], []

    union_find = _UnionFind(len(entries))
    by_name: dict[str, list[int]] = {}
    by_exe: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        by_name.setdefault(entry.normalized_name, []).append(index)
        if entry.executable_name:
            by_exe.setdefault(entry.executable_name.casefold(), []).append(index)

    for group in list(by_name.values()) + list(by_exe.values()):
        for index in group[1:]:
            union_find.union(group[0], index)

    groups: dict[int, list[int]] = {}
    for index in range(len(entries)):
        groups.setdefault(union_find.find(index), []).append(index)

    merged: list[AppEntry] = []
    report: list[dict] = []
    for indices in groups.values():
        group_entries = [entries[i] for i in indices]
        primary = max(group_entries, key=lambda e: _SOURCE_PRIORITY.get(e.source, 0))
        aliases = sorted(set().union(*(set(e.aliases) for e in group_entries)))
        executable_name = primary.executable_name or next((e.executable_name for e in group_entries if e.executable_name), None)
        app_user_model_id = primary.app_user_model_id or next((e.app_user_model_id for e in group_entries if e.app_user_model_id), None)
        merged.append(AppEntry(
            display_name=primary.display_name,
            normalized_name=primary.normalized_name,
            aliases=aliases,
            launch_type=primary.launch_type,
            launch_target=primary.launch_target,
            arguments=primary.arguments,
            executable_name=executable_name,
            source=primary.source,
            source_path=primary.source_path,
            app_user_model_id=app_user_model_id,
            metadata=primary.metadata,
        ))
        if len(group_entries) > 1:
            report.append({
                "display_name": primary.display_name,
                "merged_from": [e.source_path or e.source for e in group_entries],
            })
    return merged, report


# ---------------------------------------------------------------------------
# Build / cache lifecycle
# ---------------------------------------------------------------------------

_INDEX_LOCK = threading.Lock()
_INDEX_CACHE: AppIndex | None = None


def build_index() -> AppIndex:
    entries: list[AppEntry] = []
    sources = (
        ("start_menu", discover_start_menu_shortcuts),
        ("app_paths", discover_app_paths_registry),
        ("start_apps", discover_uwp_apps),
        ("installed_programs", discover_installed_programs),
    )
    for label, discover in sources:
        try:
            entries.extend(discover())
        except Exception:
            log.warning("Application discovery source failed: %s", label, exc_info=True)
    merged, report = _merge_entries(entries)
    return AppIndex(entries=merged, built_at=time.time(), merge_report=report)


def _load_cache_file() -> AppIndex | None:
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return AppIndex.from_json(payload)
    except (OSError, ValueError, KeyError, TypeError):
        log.info("No usable app index cache at %s; will rebuild.", CACHE_PATH)
        return None


def _save_cache_file(index: AppIndex) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = CACHE_PATH.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(index.to_json()), encoding="utf-8")
        tmp_path.replace(CACHE_PATH)
    except OSError:
        log.warning("Failed to persist app index cache to %s", CACHE_PATH, exc_info=True)


def get_index(force_refresh: bool = False) -> AppIndex:
    global _INDEX_CACHE
    with _INDEX_LOCK:
        if not force_refresh and _INDEX_CACHE is not None:
            return _INDEX_CACHE

        if not force_refresh:
            cached = _load_cache_file()
            if cached is not None and (time.time() - cached.built_at) < CACHE_TTL_SECONDS:
                _INDEX_CACHE = cached
                return _INDEX_CACHE

        index = build_index()
        _INDEX_CACHE = index
        _save_cache_file(index)
        return index


def refresh_index() -> AppIndex:
    return get_index(force_refresh=True)


def search_index(query: str, index: AppIndex | None = None) -> list[AppEntry]:
    idx = index or get_index()
    normalized = normalize_app_name(query)
    if not normalized:
        return []
    tokens = set(normalized.split())
    return [
        e for e in idx.entries
        if normalized in e.normalized_name or normalized in e.aliases or tokens & e.token_set()
    ]


# ---------------------------------------------------------------------------
# Developer CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect/refresh JARVIS's installed-application index.")
    parser.add_argument("--refresh", action="store_true", help="Force a full rescan and rewrite the cache.")
    parser.add_argument("--search", metavar="NAME", help="Search the index for a name (does not launch anything).")
    parser.add_argument("--duplicates", action="store_true", help="Show entries that were merged from multiple sources.")
    args = parser.parse_args(argv)

    index = refresh_index() if args.refresh else get_index()

    if args.search:
        matches = search_index(args.search, index=index)
        print(f"{len(matches)} match(es) for {args.search!r}:")
        for entry in matches:
            print(f"  - {entry.display_name!r} source={entry.source} launch_type={entry.launch_type} target={entry.launch_target!r} aliases={entry.aliases}")
        return

    if args.duplicates:
        print(f"{len(index.merge_report)} merged (multi-source) entries:")
        for item in index.merge_report:
            print(f"  - {item['display_name']!r} <- {item['merged_from']}")
        return

    print(f"Indexed {len(index.entries)} applications (built {time.ctime(index.built_at)})")
    from collections import Counter
    for source, count in Counter(e.source for e in index.entries).most_common():
        print(f"  {source}: {count}")


if __name__ == "__main__":
    _cli()
