"""Small provider abstraction (Part 24) so the music intent layer never has
to know which backend actually plays a song. Only `AppleMusicProvider`
(`tools/music/apple_music_provider.py`) is a full implementation right now
-- Apple Music Web is the default provider (see CLAUDE.md's music
section). Spotify/YouTube are handled separately, upstream, by
`brain.music_intent.route_music_command` reusing the existing generic
browser-search-and-click-first-result plan rather than a second bespoke
integration; they don't need a class here.

Deliberately minimal: a `Protocol`, not an ABC hierarchy with unused hooks
-- adding `SpotifyProvider`/`YouTubeMusicProvider` later means implementing
this same shape, not touching the intent parser.
"""
from __future__ import annotations

from typing import Any, Protocol


class MusicProvider(Protocol):
    name: str

    def handle(self, intent: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute one music intent (PLAY_SONG, PAUSE, NEXT, ...) and return
        a result dict with at least `success`/`verified`/`message` keys,
        matching the rest of this codebase's tool-result convention."""
        ...
