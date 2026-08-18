"""Local cache of the signed-in user's Apple Music playlist names (Part 9).

Safe metadata only -- a playlist's display name and its Apple Music href,
nothing about the user's account, no auth tokens/cookies. Refreshed when
stale (default 6 hours) or when a requested playlist isn't found in the
current cache, so a fuzzy match like "play gym" doesn't have to re-scan the
whole library sidebar on every single request (Part 23: avoid full page
rediscovery for every command).
"""
from __future__ import annotations

import difflib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

CACHE_PATH = Path(os.getenv("JARVIS_MUSIC_PLAYLIST_CACHE", "data/music_cache/apple_music_playlists.json"))
DEFAULT_TTL_SECONDS = 6 * 3600

# Unicode-aware (`\w` matches Hebrew -- and any other script's -- letters,
# not just ASCII): an ASCII-only `[a-z0-9]+` version used to normalize
# every Hebrew playlist name down to an empty string, making Hebrew
# playlist names unmatchable (confirmed live).
_WORD = re.compile(r"\w+", re.UNICODE)


def _normalize(name: str) -> str:
    return " ".join(_WORD.findall(name.lower()))


def _strip_playlist_word(name: str) -> str:
    normalized = _normalize(name)
    return re.sub(r"\bplaylist\b", "", normalized).strip()


@dataclass
class PlaylistMatch:
    name: str
    href: str
    score: float


class PlaylistCache:
    def __init__(self, path: Path | str | None = None, ttl_seconds: float = DEFAULT_TTL_SECONDS):
        self.path = Path(path) if path else CACHE_PATH
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()

    def load(self) -> dict | None:
        with self._lock:
            if not self.path.exists():
                return None
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return None

    def save(self, playlists: list[dict[str, str]]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"cached_at": time.time(), "playlists": playlists}
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def is_stale(self) -> bool:
        payload = self.load()
        if payload is None:
            return True
        return (time.time() - float(payload.get("cached_at", 0))) > self.ttl_seconds

    def playlists(self) -> list[dict[str, str]]:
        payload = self.load()
        return list(payload.get("playlists", [])) if payload else []

    def find(self, query: str, min_score: float = 0.55) -> list[PlaylistMatch]:
        """Return candidate playlists ranked by match confidence, best
        first. An empty query-normalization or empty cache returns []."""
        target = _strip_playlist_word(query)
        if not target:
            return []
        matches: list[PlaylistMatch] = []
        for item in self.playlists():
            name = item.get("name", "")
            candidate = _strip_playlist_word(name)
            if not candidate:
                continue
            if candidate == target:
                score = 1.0
            elif target in candidate or candidate in target:
                score = 0.9
            else:
                target_tokens, candidate_tokens = set(target.split()), set(candidate.split())
                overlap = len(target_tokens & candidate_tokens) / max(1, len(target_tokens | candidate_tokens))
                ratio = difflib.SequenceMatcher(None, target, candidate).ratio()
                score = max(overlap, ratio)
            if score >= min_score:
                matches.append(PlaylistMatch(name=item.get("name", ""), href=item.get("href", ""), score=score))
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches
