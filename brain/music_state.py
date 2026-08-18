"""Lightweight, structured music state + local playback history (Part 19 /
Part 11-A of the Alexa-like music feature).

Same SQLite-store shape as the rest of this codebase's persistence
(`brain/experience_store.py`, `brain/improvement_store.py`,
`brain/learning_store.py`): a dataclass, a thin `to_dict`/`from_dict` pair,
and a small store class guarded by a single process-wide lock. Never stores
authentication data, cookies, or tokens -- only safe playback metadata
(provider name, song/artist/album/playlist titles, timestamps).

Two concerns live here:

- `MusicState`: the CURRENT session snapshot (what's playing right now, the
  last track/playlist, shuffle/repeat state). Singleton row, upserted after
  every observed/verified playback change. This is what resolves
  contextual commands ("pause it", "what song is this?", "add this to my
  library") and "continue where I left off" when a live player session no
  longer exists.
- `TrackRecord` / history: an append-only local log of tracks JARVIS itself
  started or observed, used to answer "play the last song I listened to"
  fast, without depending on Apple Music's own Recently Played surface.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_DB_PATH = Path(os.getenv("JARVIS_MUSIC_STATE_DB", "data/music_state.db"))
SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrackRecord:
    track_id: str
    timestamp: str
    provider: str
    song: str | None = None
    artist: str | None = None
    album: str | None = None
    playlist: str | None = None
    context_type: str | None = None  # "playlist" | "album" | "library" | "search" | None
    identifier: str | None = None  # provider-native id/url, if safely available -- never a token
    verified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrackRecord":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class MusicState:
    provider: str | None = None
    is_playing: bool = False
    current_song: str | None = None
    current_artist: str | None = None
    current_album: str | None = None
    current_playlist: str | None = None
    shuffle: bool | None = None
    repeat: bool | None = None
    last_track: dict | None = None
    last_playlist: str | None = None
    recent_tracks: list[dict] = field(default_factory=list)
    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MusicState":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


class MusicStateStore:
    """Process-wide (and, via SQLite, cross-process-safe-enough for a
    single-user desktop assistant) store for the current music session
    snapshot plus a bounded local playback history."""

    def __init__(self, db_path: Path | str | None = None, max_history: int = 500):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.max_history = max_history
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._init_db()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def _init_db(self) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS music_state ("
                " id INTEGER PRIMARY KEY CHECK (id = 1),"
                " payload TEXT NOT NULL,"
                " schema_version INTEGER NOT NULL"
                ")"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS music_history ("
                " track_id TEXT PRIMARY KEY,"
                " timestamp TEXT NOT NULL,"
                " payload TEXT NOT NULL"
                ")"
            )
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_music_history_timestamp ON music_history(timestamp DESC)")

    # ------------------------------------------------------------------
    # Current state
    # ------------------------------------------------------------------

    def get_state(self) -> MusicState:
        with self._lock:
            row = self.connection.execute("SELECT payload FROM music_state WHERE id = 1").fetchone()
        if row is None:
            return MusicState()
        return MusicState.from_dict(json.loads(row["payload"]))

    def update_state(self, **changes: Any) -> MusicState:
        with self._lock, self.connection:
            row = self.connection.execute("SELECT payload FROM music_state WHERE id = 1").fetchone()
            state = MusicState.from_dict(json.loads(row["payload"])) if row else MusicState()
            for key, value in changes.items():
                if key in MusicState.__dataclass_fields__:
                    setattr(state, key, value)
            state.last_updated = _now()
            self.connection.execute(
                "INSERT INTO music_state (id, payload, schema_version) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload, schema_version = excluded.schema_version",
                (json.dumps(state.to_dict()), SCHEMA_VERSION),
            )
            return state

    def clear_state(self) -> None:
        with self._lock, self.connection:
            self.connection.execute("DELETE FROM music_state WHERE id = 1")

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def record_track(
        self,
        provider: str,
        song: str | None = None,
        artist: str | None = None,
        album: str | None = None,
        playlist: str | None = None,
        context_type: str | None = None,
        identifier: str | None = None,
        verified: bool = True,
        is_playing: bool = True,
    ) -> TrackRecord:
        """Append a track to local history AND refresh the current-state
        snapshot. Called both after JARVIS itself starts playback
        (`is_playing=True`, the default) AND when JARVIS merely OBSERVES a
        track already playing -- e.g. one the user started manually
        (`context_type="observed"`; still `is_playing=True`, since it was
        observed while playing) -- or a track observed at some other known
        play state (`is_playing` should reflect what was actually seen,
        never assumed)."""
        record = TrackRecord(
            track_id=uuid.uuid4().hex, timestamp=_now(), provider=provider,
            song=song, artist=artist, album=album, playlist=playlist,
            context_type=context_type, identifier=identifier, verified=verified,
        )
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT INTO music_history (track_id, timestamp, payload) VALUES (?, ?, ?)",
                (record.track_id, record.timestamp, json.dumps(record.to_dict())),
            )
            self.connection.execute(
                "DELETE FROM music_history WHERE track_id NOT IN ("
                " SELECT track_id FROM music_history ORDER BY timestamp DESC LIMIT ?)",
                (self.max_history,),
            )
        recent = self.recent_tracks(limit=10)
        self.update_state(
            provider=provider, is_playing=is_playing,
            current_song=song, current_artist=artist, current_album=album,
            current_playlist=playlist,
            last_track=record.to_dict(), last_playlist=playlist or self.get_state().last_playlist,
            recent_tracks=[item.to_dict() for item in recent],
        )
        return record

    def recent_tracks(self, limit: int = 10) -> list[TrackRecord]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT payload FROM music_history ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [TrackRecord.from_dict(json.loads(row["payload"])) for row in rows]

    def last_track(self) -> TrackRecord | None:
        tracks = self.recent_tracks(limit=1)
        return tracks[0] if tracks else None


_STORE: MusicStateStore | None = None
_STORE_LOCK = threading.Lock()


def get_music_state_store() -> MusicStateStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = MusicStateStore()
        return _STORE


def reset_music_state_store_for_tests(db_path: Path | str) -> MusicStateStore:
    """Test-only helper: point the process-wide singleton at an isolated
    database file so tests never share state with a real run."""
    global _STORE
    with _STORE_LOCK:
        _STORE = MusicStateStore(db_path)
        return _STORE
