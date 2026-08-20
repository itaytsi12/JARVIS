"""Resolves a natural spoken/typed application name against the app index.

Pure with respect to the index it's given (`resolve_app_name(name, index=...)`
never mutates or launches anything) -- `tools/app_index.py` builds/caches the
`AppIndex`, and `tools/applications.py` is the only place that turns a
resolution into an actual process launch.

Resolution order (conservative -- confidence matters more than coverage):

1. exact normalized display-name match
2. exact known alias match
3. exact executable-name match
4. strong prefix/whole-word match (unique candidate only)
5. high-confidence fuzzy match (unique, clear-margin winner)

A name that matches several plausible apps about equally well is reported as
*ambiguous* rather than guessed -- callers should surface the candidate list
to the user instead of picking one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from tools.app_index import AppEntry, AppIndex, get_index, normalize_app_name

FUZZY_ACCEPT_THRESHOLD = 0.82
FUZZY_AMBIGUOUS_THRESHOLD = 0.6
FUZZY_MARGIN = 0.08


@dataclass
class AppResolution:
    success: bool
    requested_name: str
    resolved_name: str | None = None
    confidence: float = 0.0
    resolution_method: str = "not_found"
    launch_type: str | None = None
    launch_target: str | None = None
    arguments: list[str] = field(default_factory=list)
    executable_name: str | None = None
    app_user_model_id: str | None = None
    source: str | None = None
    candidates: list[str] = field(default_factory=list)
    error: str | None = None


def _not_found(requested_name: str) -> AppResolution:
    return AppResolution(success=False, requested_name=requested_name, resolution_method="not_found", error="unknown_application")


def _ambiguous(requested_name: str, candidates: list[AppEntry]) -> AppResolution:
    names = sorted({c.display_name for c in candidates})
    return AppResolution(success=False, requested_name=requested_name, resolution_method="ambiguous", candidates=names, error="ambiguous_application")


def _success(requested_name: str, entry: AppEntry, method: str, confidence: float) -> AppResolution:
    return AppResolution(
        success=True,
        requested_name=requested_name,
        resolved_name=entry.display_name,
        confidence=confidence,
        resolution_method=method,
        launch_type=entry.launch_type,
        launch_target=entry.launch_target,
        arguments=list(entry.arguments),
        executable_name=entry.executable_name,
        app_user_model_id=entry.app_user_model_id,
        source=entry.source,
    )


def _best_fuzzy_score(normalized_query: str, entry: AppEntry) -> float:
    candidates = [entry.normalized_name, *entry.aliases]
    return max((SequenceMatcher(None, normalized_query, c).ratio() for c in candidates if c), default=0.0)


def resolve_app_name(name: str, index: AppIndex | None = None) -> AppResolution:
    requested = name
    normalized = normalize_app_name(name)
    if not normalized:
        return _not_found(requested)

    idx = index or get_index()
    entries = idx.entries
    if not entries:
        return _not_found(requested)

    # 1. exact normalized display-name match
    exact_name = [e for e in entries if e.normalized_name == normalized]
    if len(exact_name) == 1:
        return _success(requested, exact_name[0], "exact_display_name", 1.0)
    if len(exact_name) > 1:
        return _ambiguous(requested, exact_name)

    # 2. exact known alias match
    exact_alias = [e for e in entries if normalized in e.aliases]
    if len(exact_alias) == 1:
        return _success(requested, exact_alias[0], "exact_alias", 0.97)
    if len(exact_alias) > 1:
        return _ambiguous(requested, exact_alias)

    # 3. exact executable-name match
    exact_exe = [e for e in entries if e.executable_name and normalize_app_name(e.executable_name) == normalized]
    if len(exact_exe) == 1:
        return _success(requested, exact_exe[0], "exact_executable", 0.95)
    if len(exact_exe) > 1:
        return _ambiguous(requested, exact_exe)

    # 4. strong prefix / whole-word match
    query_tokens = set(normalized.split())
    word_matches = [e for e in entries if query_tokens and query_tokens <= e.token_set()]
    if len(word_matches) == 1:
        return _success(requested, word_matches[0], "prefix_match", 0.9)
    if len(word_matches) > 1:
        return _ambiguous(requested, word_matches)

    # 5. high-confidence fuzzy match
    scored = sorted(((_best_fuzzy_score(normalized, e), e) for e in entries), key=lambda pair: pair[0], reverse=True)
    if not scored or scored[0][0] < FUZZY_AMBIGUOUS_THRESHOLD:
        return _not_found(requested)

    top_score, top_entry = scored[0]
    close_runners_up = [e for score, e in scored[1:] if score >= top_score - FUZZY_MARGIN and e is not top_entry]

    if top_score >= FUZZY_ACCEPT_THRESHOLD and not close_runners_up:
        return _success(requested, top_entry, "fuzzy_match", top_score)

    plausible = [e for score, e in scored if score >= FUZZY_AMBIGUOUS_THRESHOLD]
    if len(plausible) > 1:
        return _ambiguous(requested, plausible[:5])

    return _not_found(requested)
