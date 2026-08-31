"""High-level Apple Music Web provider: turns a classified music intent
(`brain/music_intent.py`) into real browser actions against the persistent
`AppleMusicWebController` session, with search-result scoring, playlist
matching, local history/state updates, and honest verification.

Every public function here matches one `music_*` / `open_music` tool name
registered in `brain/tool_router.py` and returns the same
`{"success", "verified", "message", "error", ...}` shape every other tool
in this codebase returns -- so recording, verification, and the speculative
partial-action ledger all work unmodified (see `brain/agent.py` /
`brain/speculative_execution.py`).

`_get_controller()` / `_get_playlist_cache()` / `_get_state_store()` are
indirection points tests monkeypatch to inject fakes -- nothing here talks
to a real browser or real disk during unit tests.
"""
from __future__ import annotations

import difflib
import logging
import random
import re
import time
from typing import Any

from brain.music_state import get_music_state_store
from tools.music.apple_music_browser import AppleMusicSignInRequired, AppleMusicUnavailable, get_apple_music_controller
from tools.music.playlist_cache import PlaylistCache
from tools.music import media_keys

log = logging.getLogger("jarvis.music.provider")

PROVIDER_NAME = "apple_music"
_playlist_cache = PlaylistCache()


def _get_controller():
    return get_apple_music_controller()


def _get_playlist_cache() -> PlaylistCache:
    return _playlist_cache


def _get_state_store():
    return get_music_state_store()


_HEBREW_CHAR = re.compile(r"[֐-׿]")


def _contains_hebrew(text: str) -> bool:
    return bool(_HEBREW_CHAR.search(text or ""))


def _norm(text: str) -> str:
    # Unicode-aware: `\w` matches Hebrew (and any other script) letters,
    # not just ASCII -- an `[^a-z0-9 ]` ASCII-only version used to strip
    # Hebrew titles down to nothing before fuzzy-scoring them, making
    # every Hebrew search result score identically (confirmed live: this
    # is exactly the kind of "fallback that strips Hebrew characters" a
    # Hebrew voice mode cannot tolerate). `.lower()` is a no-op on Hebrew
    # (no case), so this stays a correct no-op change for English.
    return re.sub(r"[^\w ]", "", (text or ""), flags=re.UNICODE).lower().strip()


def _fuzzy(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _title_matches(expected: str, observed: str, threshold: float = 0.55) -> bool:
    """True when `observed` is a confident match for `expected`. Handles
    the extremely common real-catalog pattern where the actual title
    carries a "(feat. X)" / "(with X)" / similar suffix the user's spoken
    request never mentions -- e.g. the real Apple Music song title for
    "Starboy" is "Starboy (feat. Daft Punk)" (confirmed live), which a
    plain fuzzy ratio scores only ~0.48, well under any sane threshold. A
    clean prefix match (either direction, normalized) is treated as
    confident regardless of the ratio; used both for search-result
    selection scoring and post-playback verification so the two never
    disagree with each other."""
    if not expected or not observed:
        return False
    norm_expected, norm_observed = _norm(expected), _norm(observed)
    if not norm_expected or not norm_observed:
        return False
    if norm_observed.startswith(norm_expected) or norm_expected.startswith(norm_observed):
        return True
    return _fuzzy(expected, observed) >= threshold


def _result(success: bool, message: str, verified: bool = False, error: str | None = None, **data: Any) -> dict[str, Any]:
    payload = {"success": success, "verified": verified, "message": message, "error": error}
    payload.update(data)
    return payload


def _unavailable_result(exc: Exception) -> dict[str, Any]:
    # `AppleMusicUnavailable`'s message is already a short, actionable,
    # speakable sentence (see tools/music/apple_music_browser.py and
    # tools/browser_authenticated.py) -- e.g. "Authenticated Chrome is not
    # running. Start the JARVIS browser session first." -- so it's spoken
    # directly rather than papered over with a generic line (Part 25: no
    # silent fallback to a separate, unauthenticated browser).
    message = str(exc).strip() or "Apple Music isn't available right now, sir."
    return _result(False, message, error=f"apple_music_unavailable: {exc}")


def _sign_in_required_result() -> dict[str, Any]:
    return _result(
        False,
        "Apple Music needs you to sign in, sir. I opened it so you can sign in manually.",
        error="sign_in_required",
    )


def _ensure_ready():
    """Ensure the persistent tab exists and is signed in. Raises
    AppleMusicSignInRequired (caller turns this into an honest, non-fatal
    response -- Part 25) rather than ever attempting to fill the sign-in
    form itself."""
    controller = _get_controller()
    page = controller.ensure_music_tab()
    if not controller.is_signed_in(page):
        raise AppleMusicSignInRequired("Apple Music Web is showing a sign-in page.")
    return controller, page


# ---------------------------------------------------------------------
# Simple transport controls (fast path -- Part 6)
# ---------------------------------------------------------------------

def open_music() -> dict[str, Any]:
    try:
        controller = _get_controller()
        page = controller.ensure_music_tab()
    except AppleMusicUnavailable as exc:
        return _unavailable_result(exc)
    if not controller.is_signed_in(page):
        return _sign_in_required_result()
    return _result(True, "Opened Apple Music, sir.", verified=True)


def _maybe_record_observed_playback(info: dict[str, Any] | None) -> None:
    """Update local history from OBSERVED playback state -- even when the
    user started the track manually rather than through a JARVIS PLAY_*
    action (Part 7: so "play the last song I listened to" still works
    later). Never records a duplicate entry while the same track
    continues: only when the observed song differs from the most
    recently recorded one. Never fabricates: a `None`/unobserved `info`
    is a silent no-op, not a guess."""
    if not info or not info.get("observed") or not info.get("song"):
        return
    store = _get_state_store()
    last = store.last_track()
    if last is not None and _norm(last.song or "") == _norm(info.get("song") or "") and _norm(last.artist or "") == _norm(info.get("artist") or ""):
        return
    store.record_track(
        provider=PROVIDER_NAME,
        song=info.get("song"),
        artist=info.get("artist"),
        context_type="observed",
        verified=True,
        is_playing=bool(info.get("is_playing")),
    )


def _fast_transport(media_key_fn, click_fallback_fn, expect_playing: bool | None, action_label: str) -> dict[str, Any]:
    media_key_fn()
    controller = _get_controller()
    if not controller.is_session_live():
        return _result(True, f"{action_label}.", verified=False, note="media_key_sent_no_session_to_verify")
    page = controller.page
    if page is None:
        return _result(True, f"{action_label}.", verified=False, note="media_key_sent_no_tab_to_verify")
    verified = False
    if expect_playing is True:
        verified = controller.wait_for_playing(timeout=1.5)
    elif expect_playing is False:
        verified = controller.wait_for_paused(timeout=1.5)
    if not verified and click_fallback_fn is not None:
        try:
            click_fallback_fn()
        except Exception:
            pass
        if expect_playing is True:
            verified = controller.wait_for_playing(timeout=2.5)
        elif expect_playing is False:
            verified = controller.wait_for_paused(timeout=2.5)
    if expect_playing is None:
        verified = True  # no player-state expectation to confirm (e.g. next/previous handled by caller)
    _maybe_record_observed_playback(controller.current_track_info())
    message = f"{action_label}." if verified else f"{action_label} (sent, but I couldn't confirm the player state)."
    return _result(True, message, verified=verified)


def music_pause() -> dict[str, Any]:
    controller = _get_controller()
    return _fast_transport(media_keys.press_play_pause, controller.press_pause, expect_playing=False, action_label="Paused")


def music_resume() -> dict[str, Any]:
    controller = _get_controller()
    return _fast_transport(media_keys.press_play_pause, controller.press_play, expect_playing=True, action_label="Resumed")


def music_stop() -> dict[str, Any]:
    controller = _get_controller()
    return _fast_transport(media_keys.press_play_pause, controller.press_pause, expect_playing=False, action_label="Stopped the music")


def music_next() -> dict[str, Any]:
    controller = _get_controller()
    before = controller.current_track_info().get("song") if controller.is_session_live() else None
    media_keys.press_next()
    if not controller.is_session_live() or controller.page is None:
        return _result(True, "Skipped to the next song.", verified=False, note="media_key_sent_no_tab_to_verify")
    info = controller.wait_for_track_change(before, timeout=2.0)
    if not info.get("observed") or info.get("song") == before:
        controller.next_track()
        info = controller.wait_for_track_change(before, timeout=3.0)
    verified = bool(info.get("observed") and info.get("song") != before)
    _maybe_record_observed_playback(info)
    return _result(True, "Skipped to the next song." if verified else "Sent skip, but I couldn't confirm the track changed.", verified=verified, current=info)


def music_previous() -> dict[str, Any]:
    controller = _get_controller()
    before = controller.current_track_info().get("song") if controller.is_session_live() else None
    media_keys.press_previous()
    if not controller.is_session_live() or controller.page is None:
        return _result(True, "Went back to the previous song.", verified=False, note="media_key_sent_no_tab_to_verify")
    info = controller.wait_for_track_change(before, timeout=2.0)
    if not info.get("observed") or info.get("song") == before:
        controller.previous_track()
        info = controller.wait_for_track_change(before, timeout=3.0)
    verified = bool(info.get("observed") and info.get("song") != before)
    _maybe_record_observed_playback(info)
    return _result(True, "Went back to the previous song." if verified else "Sent previous, but I couldn't confirm the track changed.", verified=verified, current=info)


def music_restart_track() -> dict[str, Any]:
    controller = _get_controller()
    if not controller.is_session_live() or controller.page is None:
        return _result(False, "There's no active Apple Music session to restart.", error="no_active_session")
    ok = controller.restart_track()
    return _result(ok, "Restarted the song." if ok else "I couldn't restart the song.", verified=ok, error=None if ok else "control_not_found")


def music_shuffle_on() -> dict[str, Any]:
    return _set_shuffle(True)


def music_shuffle_off() -> dict[str, Any]:
    return _set_shuffle(False)


def _set_shuffle(on: bool) -> dict[str, Any]:
    controller = _get_controller()
    if not controller.is_session_live() or controller.page is None:
        return _result(False, "There's no active Apple Music session to shuffle.", error="no_active_session")
    ok = controller.set_shuffle(on)
    store = _get_state_store()
    if ok:
        store.update_state(shuffle=on)
    message = ("Shuffle on." if on else "Shuffle off.") if ok else "I couldn't change shuffle."
    return _result(ok, message, verified=ok, error=None if ok else "control_not_found")


def music_repeat_on() -> dict[str, Any]:
    return _set_repeat(True)


def music_repeat_off() -> dict[str, Any]:
    return _set_repeat(False)


def _set_repeat(on: bool) -> dict[str, Any]:
    controller = _get_controller()
    if not controller.is_session_live() or controller.page is None:
        return _result(False, "There's no active Apple Music session to repeat.", error="no_active_session")
    ok = controller.set_repeat(on)
    store = _get_state_store()
    if ok:
        store.update_state(repeat=on)
    message = ("Repeating this song." if on else "Repeat off.") if ok else "I couldn't change repeat."
    return _result(ok, message, verified=ok, error=None if ok else "control_not_found")


def music_add_to_library() -> dict[str, Any]:
    controller = _get_controller()
    if not controller.is_session_live() or controller.page is None:
        return _result(False, "There's nothing currently playing to add.", error="no_active_session")
    ok = controller.add_current_to_library()
    return _result(ok, "Added to your library." if ok else "I couldn't add that to your library.", verified=ok, error=None if ok else "control_not_found")


def music_add_to_favorites() -> dict[str, Any]:
    controller = _get_controller()
    if not controller.is_session_live() or controller.page is None:
        return _result(False, "There's nothing currently playing to favorite.", error="no_active_session")
    ok = controller.add_current_to_favorites()
    return _result(ok, "Added to your favorites." if ok else "I couldn't favorite that.", verified=ok, error=None if ok else "control_not_found")


def music_now_playing(aspect: str = "song") -> dict[str, Any]:
    # "What's playing" must reflect the REAL, live player DOM at this
    # exact moment -- never the last song JARVIS itself requested, a
    # search query, or locally-remembered state, all of which can be
    # stale or simply wrong the instant a human changes the track another
    # way. `current_track_info()` is the single source of truth here;
    # honest failure ("I can't tell what's currently playing, sir.") beats
    # ever answering from stale local state.
    controller = _get_controller()
    info: dict[str, Any] = {}
    if controller.is_session_live() and controller.page is not None:
        info = controller.current_track_info()
        _maybe_record_observed_playback(info)
    if not info.get("observed"):
        return _result(False, "I can't tell what's currently playing, sir.", error="now_playing_unavailable")
    if aspect == "artist":
        if not info.get("artist"):
            return _result(False, "I can see a song is playing, but not the artist.", error="artist_unavailable")
        return _result(True, f"That's {info['artist']}, sir.", verified=True, artist=info["artist"], song=info.get("song"))
    message = f"That's {info['song']} by {info['artist']}, sir." if info.get("artist") else f"That's {info['song']}, sir."
    return _result(True, message, verified=True, song=info.get("song"), artist=info.get("artist"))


def music_artist_more() -> dict[str, Any]:
    controller = _get_controller()
    info = controller.current_track_info() if controller.is_session_live() else {}
    artist = info.get("artist") or _get_state_store().get_state().current_artist
    if not artist:
        return _result(False, "I don't know the current artist, sir.", error="artist_unknown")
    return _play_artist(artist)


def music_queue_add(song: str | None, contextual: bool = False) -> dict[str, Any]:
    return _queue_action(song, contextual, up_next=False)


def music_queue_next(song: str | None, contextual: bool = False) -> dict[str, Any]:
    return _queue_action(song, contextual, up_next=True)


def _queue_action(song: str | None, contextual: bool, up_next: bool) -> dict[str, Any]:
    if contextual or not song:
        return _result(False, "There's nothing specific for me to queue, sir.", error="nothing_to_queue")
    try:
        controller, _ = _ensure_ready()
    except AppleMusicSignInRequired:
        return _sign_in_required_result()
    except AppleMusicUnavailable as exc:
        return _unavailable_result(exc)
    match = _best_search_match(controller, song, song, prefer_types=("song", "album", "artist"))
    if match is None:
        return _result(False, f"I couldn't find {song} to queue, sir.", error="not_found")
    ok = controller.queue_result(match["href"], up_next=up_next) if hasattr(controller, "queue_result") else False
    label = "next in your queue" if up_next else "added to your queue"
    return _result(ok, f"{match['title']} is {label}." if ok else f"I found {match['title']} but couldn't add it to the queue.", verified=ok, error=None if ok else "queue_control_not_found")


# ---------------------------------------------------------------------
# Search + scoring (Part 8)
# ---------------------------------------------------------------------

#: Minimum raw title similarity for a NAMED item (a specific song, album
#: or artist the user actually said) to be usable. A result that passes
#: `_title_matches` is accepted regardless of this, so
#: "Starboy" -> "Starboy (feat. Daft Punk)" still works.
#:
#: 0.5 rather than something lower because `difflib` is generous on short
#: strings: "Save Your Tears" vs "Stressed Out" scores 0.44 on shared
#: letters alone. Confirmed live -- that pair is exactly the wrong answer
#: this floor exists to reject.
TITLE_FLOOR = 0.5
#: A free-text or mood request ("play something upbeat") is NOT a named
#: item: there is no title it should resemble, so the floor is disabled
#: for those callers rather than applied and fudged.
NO_TITLE_FLOOR = 0.0


def _best_search_match(controller, search_query: str, score_against: str, prefer_types: tuple[str, ...], min_score: float = 0.45, title_floor: float = TITLE_FLOOR) -> dict[str, Any] | None:
    """Search for `search_query` (which may combine song+artist/album+artist
    for a better catalog hit rate) but SCORE each candidate's title against
    `score_against` specifically -- e.g. the song title alone, not "song
    artist" combined, or an artist result with the artist's own name in it
    would frequently out-score the actual song (Part 8: never blindly pick
    the first/highest-recall result)."""
    results = controller.search(search_query)
    if not results:
        return None
    norm_query = _norm(score_against)
    scored = []
    for item in results:
        title = item.get("title", "")
        score = _fuzzy(score_against, title)
        # Confirmed live: real catalog titles routinely carry a "(feat. X)"
        # suffix the user never says (Apple's actual song title for
        # "Starboy" is "Starboy (feat. Daft Punk)") -- a clean prefix
        # match is a strong, confident signal a plain ratio underscores.
        if norm_query and _norm(title).startswith(norm_query):
            score = max(score, 0.9)
        item_type = item.get("type")
        # Type correctness matters more than small fuzzy-score noise: an
        # exact-title ALBUM must not outrank a decent-but-imperfect-title
        # SONG for a song request (confirmed live -- see module changelog
        # in the class docstring). The single most-preferred type gets a
        # decisive bonus; any other listed type only a small one.
        if prefer_types and item_type == prefer_types[0]:
            score += 0.35
        elif item_type in prefer_types:
            score += 0.1
        else:
            score -= 0.15
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best = scored[0]
    if best_score < min_score:
        return None
    # A TITLE floor, applied separately from the combined score. The type
    # bonus above is decisive by design (an exact-title album must not beat
    # the right song), but it is large enough to carry a completely
    # unrelated title over `min_score` on its own: confirmed live, a search
    # for "Save Your Tears" whose results genuinely did not contain that
    # song returned "Stressed Out" -- fuzzy ~0.2 plus the 0.35 song bonus
    # clears 0.45. Requiring the title itself to be at least plausible is
    # what turns "the catalog does not have it" into an honest miss instead
    # of a confident wrong answer.
    if title_floor and _fuzzy(score_against, best.get("title", "")) < title_floor and not _title_matches(score_against, best.get("title", "")):
        log.info(
            "Apple Music: best match %r is too far from the request %r; reporting no match.",
            best.get("title", ""),
            score_against,
        )
        return None
    return best


def _play_and_record(controller, match: dict[str, Any], expected_song: str | None, expected_artist: str | None, context_type: str | None, playlist: str | None = None) -> dict[str, Any]:
    if not controller.open_result(match["href"]):
        return _result(False, "I found it, but couldn't open it, sir.", error="open_failed")
    # For an exact SONG match, target that specific track's own row control
    # (confirmed live to reliably start THAT track; the page-level hero
    # PLAY button plays the whole album/context instead and was also
    # observed live to sometimes render disabled). Album/playlist/artist/
    # mood contexts have no single target track, so they use the page's
    # primary Play affordance (which itself now prefers the first track
    # row over a possibly-disabled hero button -- see
    # AppleMusicWebController.play_from_current_page).
    if match.get("type") == "song":
        started = controller.play_specific_track(match.get("title", ""), expected_artist)
    else:
        started = controller.play_from_current_page()
    if not started:
        return _result(False, f"I found {match['title']}, but couldn't start playback.", error="play_control_not_found")
    playing = controller.wait_for_playing(timeout=6.0)
    info = controller.current_track_info() if playing else {}
    song_match = bool(expected_song) and info.get("observed") and _title_matches(expected_song, info.get("song") or "")
    artist_match = not expected_artist or (info.get("observed") and _title_matches(expected_artist, info.get("artist") or "", threshold=0.5))
    # Observed live: the player-bar's marquee/scroller label can very
    # briefly report inconsistent text right after a fresh play trigger
    # (once observed an artist button read as "Song by <artist>" instead
    # of "<artist>" alone, immediately after starting playback, though not
    # reliably reproduced on repeated fresh triggers). The SONG matched
    # correctly every time this was observed -- only the artist label
    # looked transiently stale -- so a short, bounded re-read specifically
    # covers a failed artist_match without re-doing song verification or
    # adding latency to the common case (playing/song_match already true
    # and artist_match already true skips this entirely).
    if playing and song_match and not artist_match and expected_artist:
        for _ in range(3):
            time.sleep(0.3)
            info = controller.current_track_info()
            artist_match = info.get("observed") and _title_matches(expected_artist, info.get("artist") or "", threshold=0.5)
            if artist_match:
                break
    # Known limitation (already documented): an artist whose real Apple
    # Music catalog metadata is Latin-script (e.g. "Omer Adam") can never
    # fuzzy-match a Hebrew-spoken artist name ("עומר אדם") -- no shared
    # characters at all, so no threshold fixes it, and transliterating
    # here to force a match would violate the "never transliterate
    # Hebrew" rule. Once the SONG itself is a confirmed, exact match, an
    # unbridgeable script mismatch on the artist alone must not turn a
    # genuinely correct play into a reported failure -- it's the search
    # query dispatch (`_play_song`'s `f"{song} {artist}"` combined query)
    # that already used the Hebrew artist name to FIND the right result;
    # this is purely a verification-time comparison limit.
    if not artist_match and expected_artist and _contains_hebrew(expected_artist) and info.get("artist") and not _contains_hebrew(info["artist"]):
        artist_match = bool(song_match)
    verified = bool(playing and (song_match or (not expected_song and playing)) and artist_match)
    # Preview-vs-full detection (confirmed live, see
    # AppleMusicWebController.playback_type's docstring for the full
    # investigation): song/artist metadata matching is NOT sufficient
    # proof of full playback -- Apple's short instant-preview clip shows
    # the exact same "now playing" song/artist/is_playing state as real
    # full-track streaming. A verified song/artist match downgrades to
    # unverified (never claimed as confirmed success) when the actual
    # audio is a detected preview.
    playback = controller.playback_type() if playing else {"observed": False}
    is_preview = bool(playback.get("observed") and playback.get("is_preview"))
    if is_preview:
        verified = False
    store = _get_state_store()
    if playing:
        store.record_track(
            provider=PROVIDER_NAME,
            song=info.get("song") or expected_song or match.get("title"),
            artist=info.get("artist") or expected_artist,
            playlist=playlist,
            context_type=context_type,
            identifier=match.get("href"),
            verified=verified,
        )
    if not playing:
        return _result(False, f"I found {match['title']}, but playback didn't start.", error="playback_did_not_start")
    song_name = info.get("song") or match.get("title")
    artist_name = info.get("artist") or expected_artist
    if is_preview:
        message = "I could only start a short preview, not the full track, sir."
    elif verified:
        message = f"Playing {song_name} by {artist_name}." if artist_name else f"Playing {song_name}."
    else:
        # False-success rule: a row click / Play click succeeding is NOT
        # itself proof the right track is playing (search returning a
        # result, or a click landing, are exactly the false signals this
        # must never trust) -- only observed player metadata matching the
        # requested song/artist counts as confirmed. `success=False` here
        # (not the previous hardcoded `True`) is what makes this honest
        # hedge actually reach the user instead of being silently treated
        # as a normal success with nothing further spoken.
        message = "I started the request, but I couldn't confirm the track, sir."
    return _result(verified, message, verified=verified, song=song_name, artist=artist_name, is_preview=is_preview)


def _play_song(song: str, artist: str | None) -> dict[str, Any]:
    try:
        controller, _ = _ensure_ready()
    except AppleMusicSignInRequired:
        return _sign_in_required_result()
    except AppleMusicUnavailable as exc:
        return _unavailable_result(exc)
    query = f"{song} {artist}" if artist else song
    match = _best_search_match(controller, query, song, prefer_types=("song", "album"))
    if match is None:
        return _result(False, f"I couldn't find {song}{' by ' + artist if artist else ''}, sir.", error="not_found")
    return _play_and_record(controller, match, expected_song=song, expected_artist=artist, context_type="search")


def _play_artist(artist: str) -> dict[str, Any]:
    try:
        controller, _ = _ensure_ready()
    except AppleMusicSignInRequired:
        return _sign_in_required_result()
    except AppleMusicUnavailable as exc:
        return _unavailable_result(exc)
    match = _best_search_match(controller, artist, artist, prefer_types=("artist",))
    if match is None:
        return _result(False, f"I couldn't find the artist {artist}, sir.", error="not_found")
    return _play_and_record(controller, match, expected_song=None, expected_artist=artist, context_type="artist")


def _play_album(album: str, artist: str | None) -> dict[str, Any]:
    try:
        controller, _ = _ensure_ready()
    except AppleMusicSignInRequired:
        return _sign_in_required_result()
    except AppleMusicUnavailable as exc:
        return _unavailable_result(exc)
    query = f"{album} {artist}" if artist else album
    match = _best_search_match(controller, query, album, prefer_types=("album", "song"))
    if match is None:
        return _result(False, f"I couldn't find the album {album}, sir.", error="not_found")
    return _play_and_record(controller, match, expected_song=None, expected_artist=artist, context_type="album")


def _play_query(query: str, contextual: bool) -> dict[str, Any]:
    if contextual:
        state = _get_state_store().get_state()
        if not state.current_song:
            return _result(False, "I don't have anything to play there, sir.", error="no_context")
        return _play_song(state.current_song, state.current_artist)
    try:
        controller, _ = _ensure_ready()
    except AppleMusicSignInRequired:
        return _sign_in_required_result()
    except AppleMusicUnavailable as exc:
        return _unavailable_result(exc)
    # Free text: the user did not name a title, so there is nothing for a
    # title floor to compare against.
    match = _best_search_match(controller, query, query, prefer_types=("song", "album", "artist", "playlist"), title_floor=NO_TITLE_FLOOR)
    if match is None:
        return _result(False, f"I couldn't find {query}, sir.", error="not_found")
    expected_song = query if match["type"] in {"song", "album", "playlist"} else None
    return _play_and_record(controller, match, expected_song=expected_song, expected_artist=None, context_type=match["type"])


# ---------------------------------------------------------------------
# Playlists (Part 9/10)
# ---------------------------------------------------------------------

def _refresh_playlist_cache(controller) -> list[dict[str, str]]:
    playlists = controller.list_library_playlists()
    if playlists:
        _get_playlist_cache().save(playlists)
    return playlists


def _play_playlist(playlist: str | None, scope: str | None, shuffle: bool) -> dict[str, Any]:
    try:
        controller, _ = _ensure_ready()
    except AppleMusicSignInRequired:
        return _sign_in_required_result()
    except AppleMusicUnavailable as exc:
        return _unavailable_result(exc)
    cache = _get_playlist_cache()

    if scope == "random_user_playlist":
        if cache.is_stale() or not cache.playlists():
            _refresh_playlist_cache(controller)
        playlists = cache.playlists()
        if not playlists:
            return _result(False, "You don't seem to have any playlists in your library, sir.", error="no_playlists")
        last_playlist = _get_state_store().get_state().last_playlist
        candidates = [p for p in playlists if p.get("name") != last_playlist] or playlists
        choice = random.choice(candidates)
        return _play_and_record(controller, {"href": choice["href"], "title": choice["name"], "type": "playlist"},
                                  expected_song=None, expected_artist=None, context_type="playlist", playlist=choice["name"])

    if not playlist:
        return _result(False, "I don't know which playlist you mean, sir.", error="playlist_not_specified")

    matches = cache.find(playlist)
    if not matches and cache.is_stale():
        _refresh_playlist_cache(controller)
        matches = cache.find(playlist)
    if not matches:
        # One retry: the requested playlist may simply not have been in a
        # not-yet-stale cache (Part 9: "refresh cache ... when a playlist
        # is not found").
        _refresh_playlist_cache(controller)
        matches = cache.find(playlist)
    if not matches:
        return _result(False, "I couldn't find that playlist in your library, sir.", error="playlist_not_found")
    if len(matches) > 1 and (matches[0].score - matches[1].score) < 0.1:
        options = ", ".join(m.name for m in matches[:3])
        return _result(False, f"I found a few playlists that could match: {options}. Which one did you mean, sir?", error="ambiguous_playlist")
    best = matches[0]
    result = _play_and_record(controller, {"href": best.href, "title": best.name, "type": "playlist"},
                               expected_song=None, expected_artist=None, context_type="playlist", playlist=best.name)
    if result["success"] and shuffle:
        if controller.set_shuffle(True):
            _get_state_store().update_state(shuffle=True)
            result["message"] = f"Shuffling your {best.name} playlist."
    return result


# ---------------------------------------------------------------------
# History / favorites / library / mood / resume (Part 11-15)
# ---------------------------------------------------------------------

def _play_last_played() -> dict[str, Any]:
    store = _get_state_store()
    last = store.last_track()
    try:
        controller, _ = _ensure_ready()
    except AppleMusicSignInRequired:
        return _sign_in_required_result()
    except AppleMusicUnavailable as exc:
        return _unavailable_result(exc)
    if last is not None:
        if last.identifier:
            return _play_and_record(controller, {"href": last.identifier, "title": last.song or "that track", "type": "song"},
                                      expected_song=last.song, expected_artist=last.artist, context_type="history")
        if last.song:
            return _play_song(last.song, last.artist)
    recent = controller.get_recently_played() if hasattr(controller, "get_recently_played") else []
    if recent:
        item = recent[0]
        return _play_and_record(controller, item, expected_song=item.get("title"), expected_artist=None, context_type="apple_music_recent")
    return _result(False, "I don't have a record of what you last played, sir.", error="no_history")


def _play_recent() -> dict[str, Any]:
    return _play_last_played()


def _resume_last_session() -> dict[str, Any]:
    try:
        controller, _ = _ensure_ready()
    except AppleMusicSignInRequired:
        return _sign_in_required_result()
    except AppleMusicUnavailable as exc:
        return _unavailable_result(exc)
    info = controller.current_track_info()
    if info.get("observed") and not info.get("is_playing"):
        if controller.press_play():
            playing = controller.wait_for_playing(timeout=3.0)
            return _result(playing, "Resuming your music." if playing else "I tried to resume, but couldn't confirm playback started.", verified=playing)
    if info.get("observed") and info.get("is_playing"):
        return _result(True, "Your music is already playing, sir.", verified=True)
    state = _get_state_store().get_state()
    if state.last_playlist:
        return _play_playlist(state.last_playlist, scope=None, shuffle=False)
    if state.last_track and state.last_track.get("song"):
        # No live player session survived -- reconstruct from local history,
        # preferring the stored provider-native identifier (fast, exact)
        # over a fresh search (see `_play_last_played`).
        return _play_last_played()
    return _result(False, "I don't have anything to resume, sir.", error="no_session")


def _play_favorites() -> dict[str, Any]:
    try:
        controller, _ = _ensure_ready()
    except AppleMusicSignInRequired:
        return _sign_in_required_result()
    except AppleMusicUnavailable as exc:
        return _unavailable_result(exc)
    cache = _get_playlist_cache()
    if cache.is_stale() or not cache.playlists():
        _refresh_playlist_cache(controller)
    matches = cache.find("favorites") or cache.find("favorite songs")
    if not matches:
        return _result(False, "I couldn't find a favorites playlist in your library, sir.", error="favorites_not_found")
    best = matches[0]
    return _play_and_record(controller, {"href": best.href, "title": best.name, "type": "playlist"},
                              expected_song=None, expected_artist=None, context_type="favorites", playlist=best.name)


def _play_library() -> dict[str, Any]:
    try:
        controller, _ = _ensure_ready()
    except AppleMusicSignInRequired:
        return _sign_in_required_result()
    except AppleMusicUnavailable as exc:
        return _unavailable_result(exc)
    if not controller.open_result("/library/songs"):
        return _result(False, "I couldn't open your library, sir.", error="open_failed")
    time.sleep(0.3)
    if not controller.play_from_current_page():
        return _result(False, "I opened your library, but couldn't start playback.", error="play_control_not_found")
    playing = controller.wait_for_playing(timeout=5.0)
    info = controller.current_track_info() if playing else {}
    if playing:
        _get_state_store().record_track(provider=PROVIDER_NAME, song=info.get("song"), artist=info.get("artist"), context_type="library", verified=True)
    return _result(playing, "Playing your library." if playing else "I opened your library, but playback didn't start.", verified=playing)


def _play_mood(mood: str) -> dict[str, Any]:
    try:
        controller, _ = _ensure_ready()
    except AppleMusicSignInRequired:
        return _sign_in_required_result()
    except AppleMusicUnavailable as exc:
        return _unavailable_result(exc)
    # A mood is not a title either -- "workout" should reach "Pure Workout".
    match = _best_search_match(controller, f"{mood} music", mood, prefer_types=("playlist", "album", "song"), min_score=0.3, title_floor=NO_TITLE_FLOOR)
    if match is None:
        return _result(False, f"I couldn't find anything for {mood}, sir.", error="not_found")
    return _play_and_record(controller, match, expected_song=None, expected_artist=None, context_type="mood")


def _play_generic() -> dict[str, Any]:
    try:
        controller, _ = _ensure_ready()
    except AppleMusicSignInRequired:
        return _sign_in_required_result()
    except AppleMusicUnavailable as exc:
        return _unavailable_result(exc)
    info = controller.current_track_info()
    if info.get("observed") and not info.get("is_playing"):
        return _resume_last_session()
    state = _get_state_store().get_state()
    if state.last_track and state.last_track.get("song"):
        return _play_last_played()
    if state.last_playlist:
        return _play_playlist(state.last_playlist, scope=None, shuffle=False)
    favorites = _play_favorites()
    if favorites["success"]:
        return favorites
    return _result(True, "Opened Apple Music. What would you like to hear, sir?", verified=True)


_PLAY_DISPATCH = {
    "PLAY_SONG": lambda a: _play_song(a["song"], a.get("artist")),
    "PLAY_ARTIST": lambda a: _play_artist(a["artist"]),
    "PLAY_ALBUM": lambda a: _play_album(a["album"], a.get("artist")),
    "PLAY_PLAYLIST": lambda a: _play_playlist(a.get("playlist"), a.get("scope"), bool(a.get("shuffle"))),
    # A PLAY_QUERY with an artist attached (e.g. Hebrew "X של Y" / "X מאת Y"
    # -- see brain/music_intent.py::_split_hebrew_song_artist) is really a
    # song+artist request, not an ambiguous bare query -- reuse _play_song
    # so the artist is both used for a better catalog search AND for
    # post-playback verification, instead of being silently dropped.
    "PLAY_QUERY": lambda a: (
        _play_song(a["song"], a["artist"]) if a.get("song") and a.get("artist")
        else _play_query(a.get("song") or a.get("raw_text") or "", bool(a.get("contextual")))
    ),
    "PLAY_LAST_PLAYED": lambda a: _play_last_played(),
    "PLAY_RECENT": lambda a: _play_recent(),
    "RESUME_LAST_SESSION": lambda a: _resume_last_session(),
    "PLAY_FAVORITES": lambda a: _play_favorites(),
    "PLAY_LIBRARY": lambda a: _play_library(),
    "PLAY_MOOD": lambda a: _play_mood(a["mood"]),
    "PLAY_GENERIC": lambda a: _play_generic(),
}


def music_play(intent: str, song: str | None = None, artist: str | None = None, album: str | None = None,
               playlist: str | None = None, mood: str | None = None, scope: str | None = None,
               contextual: bool = False, shuffle: bool = False) -> dict[str, Any]:
    handler = _PLAY_DISPATCH.get(intent)
    if handler is None:
        return _result(False, "I don't know how to handle that music request, sir.", error="unsupported_intent")
    arguments = {"song": song, "artist": artist, "album": album, "playlist": playlist, "mood": mood,
                 "scope": scope, "contextual": contextual, "shuffle": shuffle}
    return handler(arguments)


# ---------------------------------------------------------------------------
# Playlist authoring (create / add). Part of the same provider so it shares
# `_ensure_ready`'s sign-in and availability handling rather than growing a
# second error vocabulary.
# ---------------------------------------------------------------------------
#: How many songs one `music_create_playlist` call will look up. A spoken
#: request naming a dozen songs is realistic; a hundred is a runaway, and
#: each one costs a real catalog search.
MAX_PLAYLIST_SONGS = 25


def _locate_song(controller, song: str, artist: str | None) -> dict[str, Any] | None:
    """Find one song in the catalog and open its page.

    Reuses `_best_search_match` with the SAME `prefer_types` as
    `_play_song`, so a song added to a playlist is chosen exactly like a
    song that gets played rather than by a second, divergent rule.

    Then it applies one extra, stricter gate that playback does not need:
    the matched title has to actually match what was asked for. Playing the
    wrong song is an annoyance the user hears immediately and can correct;
    adding the wrong song to a playlist is a silent, persistent edit to
    their library. Confirmed live -- a request for "Save Your Tears"
    scored a result titled "Stressed Out" highest and was added, and the
    verification then passed because it checked for the MATCHED title
    instead of the REQUESTED one. Both halves are fixed here: an
    unconvincing match is refused outright, and verification upstream
    compares against the request.
    """
    query = f"{song} {artist}".strip() if artist else song
    match = _best_search_match(controller, query, song, prefer_types=("song", "album"))
    if match is None:
        return None
    if not _title_matches(song, match.get("title", "")):
        log.info(
            "Apple Music: refusing to add %r -- the best match was %r, which is a different song.",
            song,
            match.get("title", ""),
        )
        return None
    if not controller.open_result(match["href"]):
        return None
    return match


def _resolve_playlist_name(controller, playlist: str) -> dict[str, Any]:
    """Match a spoken playlist name to one the account actually has.

    Exact (case-insensitive) first, then an unambiguous prefix -- one
    candidate, never "the closest of several". Returns the real name so
    everything downstream uses Apple's spelling rather than the user's.
    """
    wanted = (playlist or "").strip().lower()
    try:
        available = [item.get("name", "") for item in controller.list_library_playlists() if item.get("name")]
    except Exception:
        log.debug("Listing playlists for name resolution failed", exc_info=True)
        available = []
    exact = [name for name in available if name.strip().lower() == wanted]
    if exact:
        return {"name": exact[0], "available": available}
    prefix = [name for name in available if name.strip().lower().startswith(wanted)] if wanted else []
    if len(prefix) == 1:
        return {"name": prefix[0], "available": available}
    return {
        "error": "playlist_ambiguous" if len(prefix) > 1 else "playlist_not_found",
        "available": available,
        "candidates": prefix,
    }


def music_add_to_playlist(song: str, playlist: str, artist: str | None = None) -> dict[str, Any]:
    """Add one song to an existing playlist, and verify it landed.

    Verification is a real re-read: after the menu accepts the entry, the
    playlist itself is opened and its track titles are checked for the
    song. Apple's menu closing is not treated as proof -- the same
    "success is not verification" rule the rest of this module follows.
    """
    if not (song or "").strip():
        return _result(False, "I need a song to add, sir.", error="missing_song")
    if not (playlist or "").strip():
        return _result(False, "I need to know which playlist, sir.", error="missing_playlist")
    try:
        controller, _page = _ensure_ready()
    except AppleMusicSignInRequired:
        return _sign_in_required_result()
    except Exception as exc:
        return _unavailable_result(exc)

    # Resolve the playlist name FIRST, against the account's real library.
    # It is the cheap half of the work, and doing it first means an
    # unknown name produces the useful "you have: ..." answer instead of
    # whatever the row menu happened to fail with after a pointless
    # catalog search -- confirmed live, where a bad name surfaced as
    # `row_menu_unavailable`, which tells the user nothing.
    resolved = _resolve_playlist_name(controller, playlist)
    if resolved.get("error"):
        available = resolved.get("available") or []
        if resolved["error"] == "playlist_ambiguous":
            listed = ", ".join(resolved.get("candidates") or [])
            return _result(False, f"Several playlists match {playlist}, sir: {listed}. Which one?", error="playlist_ambiguous", candidates=resolved.get("candidates"))
        listed = ", ".join(available[:6])
        return _result(
            False,
            f"I couldn't find a playlist called {playlist}, sir." + (f" You have: {listed}." if listed else ""),
            error="playlist_not_found",
            available=available,
        )
    playlist = resolved["name"]

    match = _locate_song(controller, song, artist)
    if match is None:
        return _result(False, f"I couldn't find {song} on Apple Music, sir.", error="song_not_found", song=song)

    outcome = controller.add_track_to_playlist(playlist, match.get("title") or song, artist)
    if not outcome.get("added"):
        return _result(False, f"I couldn't add {song} to {playlist}, sir.", error=outcome.get("error") or "add_failed")

    target_playlist = outcome.get("playlist") or playlist
    # Deliberately the REQUESTED song, never the matched title: verifying
    # that "whatever I added is in the playlist" is trivially true and
    # proves nothing.
    verified = _playlist_contains(controller, target_playlist, song)
    return _result(
        True if verified else False,
        f"Added {match.get('title') or song} to {target_playlist}." if verified
        else f"I asked Apple Music to add {song} to {target_playlist}, but I couldn't confirm it, sir.",
        verified=verified,
        error=None if verified else "verification_failed",
        song=match.get("title") or song,
        playlist=target_playlist,
    )


def _playlist_contains(controller, playlist: str, song: str) -> bool:
    """Is `song` genuinely listed in `playlist` right now?

    Re-opens the playlist and reads its real track titles. Uses the same
    `_title_matches` prefix-aware comparison the playback verification
    uses, so a catalog title carrying a "(feat. X)" suffix the request
    never mentioned still counts as a match here exactly as it does there.
    """
    try:
        opened = controller.open_library_playlist(playlist)
        if not opened.get("opened"):
            return False
        titles = controller.current_page_track_titles()
    except Exception:
        log.debug("Playlist verification failed", exc_info=True)
        return False
    return any(_title_matches(song, title) for title in titles)


#: How long to wait for a just-created playlist to appear in the library
#: listing. Confirmed live: `create_playlist_from_track` genuinely created
#: "JARVIS Smoke Test", but the very next read of `/library/all-playlists`
#: still returned the previous 13 names -- Apple's library view is
#: eventually consistent, and a single eager read reported a real success
#: as `verification_failed`. Bounded and short: this waits for a fact to
#: become observable, it never invents one.
PLAYLIST_SYNC_ATTEMPTS = 4
PLAYLIST_SYNC_DELAY = 1.5


def _playlist_exists(controller, name: str, attempts: int = 1) -> bool:
    """Is a playlist with this exact name really in the account's library?

    Re-reads `/library/all-playlists` rather than trusting that the create
    flow's UI acknowledgement meant a playlist now exists -- the same
    "success is not verification" rule the playback path follows. With
    `attempts > 1`, re-reads a few times before concluding it is absent,
    because the listing lags a creation by a second or two.
    """
    wanted = name.strip().lower()
    for attempt in range(max(1, int(attempts))):
        try:
            if any(item.get("name", "").strip().lower() == wanted for item in controller.list_library_playlists()):
                return True
        except Exception:
            log.debug("Playlist existence check failed", exc_info=True)
        if attempt + 1 < attempts:
            time.sleep(PLAYLIST_SYNC_DELAY)
    return False


def music_create_playlist(name: str, songs: list[str] | str | None = None, artist: str | None = None) -> dict[str, Any]:
    """Create a playlist called `name`, optionally filled with `songs`.

    Apple Music Web offers no "create an empty playlist called X" control
    that can be driven reliably, so the playlist is created around its
    first song and renamed -- which is also why at least one song is
    required, and why that is reported as a clear error rather than
    silently creating something called "New Playlist".

    Every song after the first is added through the same verified
    `music_add_to_playlist` path, so a partial success is reported
    honestly: the result says which songs made it and which did not.
    """
    if not (name or "").strip():
        return _result(False, "I need a name for the playlist, sir.", error="missing_name")
    if isinstance(songs, str):
        songs = [songs]
    songs = [item for item in (songs or []) if str(item).strip()]
    if not songs:
        return _result(
            False,
            "I need at least one song to start the playlist with, sir -- Apple Music can't create an empty one from the web player.",
            error="no_songs",
        )
    if len(songs) > MAX_PLAYLIST_SONGS:
        return _result(False, f"That's more than {MAX_PLAYLIST_SONGS} songs, sir; ask me in smaller batches.", error="too_many_songs")

    try:
        controller, _page = _ensure_ready()
    except AppleMusicSignInRequired:
        return _sign_in_required_result()
    except Exception as exc:
        return _unavailable_result(exc)

    # An existing playlist of the same name would make "add the rest"
    # ambiguous, so say so rather than creating a second one with an
    # identical name.
    try:
        existing = [item.get("name", "").strip().lower() for item in controller.list_library_playlists()]
    except Exception:
        existing = []
    if name.strip().lower() in existing:
        return _result(False, f"You already have a playlist called {name}, sir.", error="playlist_exists", playlist=name)

    first = str(songs[0])
    match = _locate_song(controller, first, artist)
    if match is None:
        return _result(False, f"I couldn't find {first} on Apple Music, sir.", error="song_not_found", song=first)

    created = controller.create_playlist_from_track(name, match.get("title") or first, artist)
    if not created.get("created"):
        return _result(False, "I couldn't create the playlist, sir.", error=created.get("error") or "create_failed")

    # Apple names the playlist at creation, so "did it work" is one
    # question, not two: does a playlist with this name now exist?
    # Verified against the account's real library rather than against the
    # UI's own acknowledgement.
    if not _playlist_exists(controller, name, attempts=PLAYLIST_SYNC_ATTEMPTS):
        return _result(
            False,
            f"I asked Apple Music to create {name} but couldn't confirm it exists, sir.",
            error="verification_failed",
            playlist=name,
        )

    added = [match.get("title") or first]
    failed: list[str] = []
    for song in songs[1:]:
        outcome = music_add_to_playlist(str(song), name, artist)
        (added if outcome.get("success") else failed).append(str(song))

    if failed:
        return _result(
            True,
            f"Created {name} with {len(added)} song{'' if len(added) == 1 else 's'}, sir. I couldn't add: {', '.join(failed)}.",
            verified=True,
            playlist=name,
            added=added,
            failed=failed,
        )
    return _result(
        True,
        f"Created {name} with {len(added)} song{'' if len(added) == 1 else 's'}, sir.",
        verified=True,
        playlist=name,
        added=added,
        failed=[],
    )


def music_list_playlists() -> dict[str, Any]:
    """The account's own playlists, by name.

    Read-only, and the thing the agent needs before it can sensibly act on
    "add this to my workout playlist" -- otherwise it has to guess a name
    and find out by failing.
    """
    try:
        controller, _page = _ensure_ready()
    except AppleMusicSignInRequired:
        return _sign_in_required_result()
    except Exception as exc:
        return _unavailable_result(exc)
    try:
        playlists = controller.list_library_playlists()
    except Exception as exc:
        return _unavailable_result(exc)
    names = [item.get("name", "") for item in playlists if item.get("name")]
    return _result(
        True,
        f"You have {len(names)} playlist{'' if len(names) == 1 else 's'}, sir." if names else "I couldn't find any playlists in your library, sir.",
        verified=True,
        playlists=names,
        count=len(names),
    )
