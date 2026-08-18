"""Direct, no-voice-needed diagnostics for the Apple Music pipeline.

Every command is safe/read-only except `play`, which genuinely starts
playback -- that IS the diagnostic for "does starting a song actually
work" (the primary live bug this module was added to debug). No secrets
(cookies/tokens/headers/passwords) are ever printed.

Usage:
    python -m tools.music.diagnose_cli now-playing
    python -m tools.music.diagnose_cli playlists
    python -m tools.music.diagnose_cli history
    python -m tools.music.diagnose_cli recently-played
    python -m tools.music.diagnose_cli search "Starboy The Weeknd"
    python -m tools.music.diagnose_cli play "Starboy" "The Weeknd"
"""
from __future__ import annotations

import json
import sys


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def now_playing() -> None:
    """A: current now-playing metadata."""
    from tools.music.apple_music_browser import get_apple_music_controller
    controller = get_apple_music_controller()
    controller.ensure_music_tab()
    _print(controller.current_track_info())


def playlists() -> None:
    """B: discovered library playlists."""
    from tools.music.apple_music_browser import get_apple_music_controller
    controller = get_apple_music_controller()
    _print(controller.list_library_playlists())


def history() -> None:
    """C: latest local (JARVIS-observed) playback history."""
    from brain.music_state import get_music_state_store
    store = get_music_state_store()
    _print([record.to_dict() for record in store.recent_tracks(limit=10)])


def recently_played() -> None:
    """D: Apple Music's own Recently Played shelf."""
    from tools.music.apple_music_browser import get_apple_music_controller
    controller = get_apple_music_controller()
    _print(controller.get_recently_played())


def search(query: str) -> None:
    """E: sanitized candidate search results (no secrets -- these are
    just catalog titles/types/hrefs, already public)."""
    from tools.music.apple_music_browser import get_apple_music_controller
    controller = get_apple_music_controller()
    controller.ensure_music_tab()
    _print(controller.search(query))


def play(song: str, artist: str | None = None) -> None:
    """F: attempt to play a song and report every step -- the result
    selected, the action used, observed player state/title/artist, and
    the verification result."""
    from tools.music.apple_music_browser import AppleMusicSignInRequired, AppleMusicUnavailable
    from tools.music.apple_music_provider import _best_search_match, _ensure_ready, _title_matches

    try:
        controller, _page = _ensure_ready()
    except AppleMusicSignInRequired:
        _print({"error": "sign_in_required"})
        return
    except AppleMusicUnavailable as exc:
        _print({"error": str(exc)})
        return

    query = f"{song} {artist}" if artist else song
    match = _best_search_match(controller, query, song, prefer_types=("song", "album"))
    if match is None:
        _print({"error": "not_found", "query": query})
        return
    report: dict = {"selected_result": match}

    opened = controller.open_result(match["href"])
    report["opened"] = opened
    if not opened:
        _print(report)
        return

    if match.get("type") == "song":
        action = "play_specific_track"
        started = controller.play_specific_track(match.get("title", ""), artist)
    else:
        action = "play_from_current_page"
        started = controller.play_from_current_page()
    report["action_used"] = action
    report["play_click_started"] = started
    if not started:
        _print(report)
        return

    playing = controller.wait_for_playing(timeout=6.0)
    info = controller.current_track_info() if playing else {}
    report["observed_playing"] = playing
    report["observed_song"] = info.get("song")
    report["observed_artist"] = info.get("artist")
    report["verified_song"] = bool(info.get("observed") and _title_matches(song, info.get("song") or ""))
    if artist:
        report["verified_artist"] = bool(info.get("observed") and _title_matches(artist, info.get("artist") or "", threshold=0.5))
    _print(report)


_COMMANDS = {
    "now-playing": lambda args: now_playing(),
    "playlists": lambda args: playlists(),
    "history": lambda args: history(),
    "recently-played": lambda args: recently_played(),
    "search": lambda args: search(args[0]),
    "play": lambda args: play(args[0], args[1] if len(args) > 1 else None),
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS or (sys.argv[1] in {"search", "play"} and len(sys.argv) < 3):
        print("Usage: python -m tools.music.diagnose_cli <now-playing|playlists|history|recently-played>")
        print('       python -m tools.music.diagnose_cli search "QUERY"')
        print('       python -m tools.music.diagnose_cli play "SONG" ["ARTIST"]')
        sys.exit(1)
    _COMMANDS[sys.argv[1]](sys.argv[2:])
