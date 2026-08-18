"""Unit tests for tools/music/apple_music_provider.py against a fake
Apple Music controller -- no real browser, no network. Exercises search
scoring, playlist matching, history/resume behavior, contextual commands,
and false-success prevention (Part 18/25/26)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brain.music_state import MusicStateStore
from tools.music import apple_music_provider as provider
from tools.music.apple_music_browser import AppleMusicSignInRequired
from tools.music.playlist_cache import PlaylistCache


class FakePage:
    def __init__(self):
        self.url = "https://music.apple.com"
        self.closed = False

    def is_closed(self):
        return self.closed


class FakeController:
    """Minimal stand-in for AppleMusicWebController exposing the exact
    surface tools/music/apple_music_provider.py calls."""

    def __init__(self, signed_in=True, session_live=True):
        self._signed_in = signed_in
        self._session_live = session_live
        self.page = FakePage() if session_live else None
        self.search_results: list[dict] = []
        self.playlists: list[dict] = []
        self.recently_played: list[dict] = []
        self.opened_hrefs: list[str] = []
        self.played = False
        self.play_from_page_ok = True
        self.now_playing = {"song": None, "artist": None, "is_playing": False, "observed": False}
        self.next_now_playing = None  # what current_track_info() returns AFTER a control action
        self.shuffle_state = False
        self.repeat_state = False
        self.library_added = False
        self.favorite_added = False
        self.queue_calls: list[tuple[str, bool]] = []
        self.control_calls: list[str] = []
        self.play_specific_track_calls: list[tuple[str, str | None]] = []
        self.search_queries: list[str] = []

    # lifecycle -------------------------------------------------------
    def ensure_music_tab(self, focus=True):
        if self.page is None:
            self.page = FakePage()
        return self.page

    def is_session_live(self):
        return self._session_live

    def is_signed_in(self, page=None):
        return self._signed_in

    # now playing -------------------------------------------------------
    def current_track_info(self, page=None):
        return dict(self.now_playing)

    def wait_for_playing(self, timeout=6.0):
        if self.next_now_playing:
            self.now_playing = dict(self.next_now_playing)
            self.now_playing["is_playing"] = True
        return bool(self.now_playing.get("is_playing"))

    def wait_for_paused(self, timeout=4.0):
        return not self.now_playing.get("is_playing", False)

    def wait_for_track_change(self, previous_song, timeout=6.0):
        if self.next_now_playing:
            self.now_playing = dict(self.next_now_playing)
        return dict(self.now_playing)

    # transport -------------------------------------------------------
    def press_play(self):
        self.control_calls.append("press_play")
        self.now_playing["is_playing"] = True
        return True

    def press_pause(self):
        self.control_calls.append("press_pause")
        self.now_playing["is_playing"] = False
        return True

    def play_pause(self):
        return self.press_play()

    def next_track(self):
        self.control_calls.append("next_track")
        return True

    def previous_track(self):
        self.control_calls.append("previous_track")
        return True

    def restart_track(self):
        self.control_calls.append("restart_track")
        return True

    def set_shuffle(self, on):
        self.shuffle_state = on
        return True

    def set_repeat(self, on):
        self.repeat_state = on
        return True

    def add_current_to_library(self):
        self.library_added = True
        return True

    def add_current_to_favorites(self):
        self.favorite_added = True
        return True

    # search / playback -------------------------------------------------
    def search(self, query):
        self.search_queries.append(query)
        return list(self.search_results)

    def open_result(self, href):
        self.opened_hrefs.append(href)
        return True

    def play_from_current_page(self):
        self.played = self.play_from_page_ok
        return self.play_from_page_ok

    def play_specific_track(self, title, artist=None):
        self.play_specific_track_calls.append((title, artist))
        return self.play_from_current_page()

    def list_library_playlists(self):
        return list(self.playlists)

    def get_recently_played(self):
        return list(self.recently_played)

    def queue_result(self, href, up_next=False):
        self.queue_calls.append((href, up_next))
        return True


class ProviderTestCase(unittest.TestCase):
    def setUp(self):
        self.controller = FakeController()
        self._tmp = tempfile.TemporaryDirectory()
        self.state_store = MusicStateStore(Path(self._tmp.name) / "music_state.db")
        self.playlist_cache = PlaylistCache(Path(self._tmp.name) / "playlists.json", ttl_seconds=3600)

        self._controller_patch = patch.object(provider, "_get_controller", lambda: self.controller)
        self._state_patch = patch.object(provider, "_get_state_store", lambda: self.state_store)
        self._cache_patch = patch.object(provider, "_get_playlist_cache", lambda: self.playlist_cache)
        self._controller_patch.start()
        self._state_patch.start()
        self._cache_patch.start()

    def tearDown(self):
        self._controller_patch.stop()
        self._state_patch.stop()
        self._cache_patch.stop()
        self.state_store.close()
        self._tmp.cleanup()


class OpenMusicTests(ProviderTestCase):
    def test_open_music_success(self):
        result = provider.open_music()
        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])

    def test_open_music_sign_in_required_reported_honestly(self):
        self.controller._signed_in = False
        result = provider.open_music()
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "sign_in_required")


class FastTransportTests(ProviderTestCase):
    def test_pause_verifies_via_page_when_session_live(self):
        self.controller.now_playing = {"song": "X", "artist": "Y", "is_playing": True, "observed": True}
        result = provider.music_pause()
        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])

    def test_pause_unverified_without_a_tracked_session(self):
        self.controller._session_live = False
        self.controller.page = None
        result = provider.music_pause()
        self.assertTrue(result["success"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["note"], "media_key_sent_no_session_to_verify")

    def test_next_verifies_track_change(self):
        self.controller.now_playing = {"song": "Old Song", "artist": "A", "is_playing": True, "observed": True}
        self.controller.next_now_playing = {"song": "New Song", "artist": "A", "is_playing": True, "observed": True}
        result = provider.music_next()
        self.assertTrue(result["verified"])
        self.assertEqual(result["current"]["song"], "New Song")

    def test_next_does_not_claim_false_success_when_track_unchanged(self):
        self.controller.now_playing = {"song": "Same Song", "artist": "A", "is_playing": True, "observed": True}
        self.controller.next_now_playing = None  # track never actually changes
        result = provider.music_next()
        self.assertTrue(result["success"])  # the key press was sent
        self.assertFalse(result["verified"])  # but never confirmed -- no false success


class SearchAndPlayTests(ProviderTestCase):
    def test_play_song_picks_best_matching_song_over_decoys(self):
        self.controller.search_results = [
            {"type": "artist", "title": "The Weeknd", "href": "/artist/1"},
            {"type": "song", "title": "Starboy (Live)", "href": "/song/2"},
            {"type": "song", "title": "Starboy", "href": "/song/3"},
        ]
        self.controller.next_now_playing = {"song": "Starboy", "artist": "The Weeknd", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_SONG", song="Starboy", artist="The Weeknd")
        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])
        self.assertEqual(self.controller.opened_hrefs[-1], "/song/3")

    def test_play_song_not_found_is_honest_failure(self):
        self.controller.search_results = []
        result = provider.music_play("PLAY_SONG", song="Nonexistent Track", artist="Nobody")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "not_found")

    def test_play_verifies_song_and_artist_before_claiming_success_text(self):
        self.controller.search_results = [{"type": "song", "title": "Starboy", "href": "/song/1"}]
        # Simulate a wrong track actually loading (playback started, but not
        # the requested song) -- verification must catch this.
        self.controller.next_now_playing = {"song": "Completely Different Song", "artist": "Someone Else", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_SONG", song="Starboy", artist="The Weeknd")
        self.assertTrue(result["success"])
        self.assertFalse(result["verified"])
        self.assertIn("couldn't confirm", result["message"])

    def test_play_artist_prefers_artist_type_result(self):
        self.controller.search_results = [
            {"type": "song", "title": "Some Random Song mentioning The Weeknd", "href": "/song/1"},
            {"type": "artist", "title": "The Weeknd", "href": "/artist/9"},
        ]
        self.controller.next_now_playing = {"song": "Some Hit", "artist": "The Weeknd", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_ARTIST", artist="The Weeknd")
        self.assertTrue(result["success"])
        self.assertEqual(self.controller.opened_hrefs[-1], "/artist/9")

    def test_play_records_local_history_after_verified_playback(self):
        self.controller.search_results = [{"type": "song", "title": "Starboy", "href": "/song/1"}]
        self.controller.next_now_playing = {"song": "Starboy", "artist": "The Weeknd", "is_playing": True, "observed": True}
        provider.music_play("PLAY_SONG", song="Starboy", artist="The Weeknd")
        last = self.state_store.last_track()
        self.assertIsNotNone(last)
        self.assertEqual(last.song, "Starboy")

    def test_sign_in_required_is_reported_not_crashed(self):
        self.controller._signed_in = False
        result = provider.music_play("PLAY_SONG", song="Starboy", artist="The Weeknd")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "sign_in_required")


class PlaylistTests(ProviderTestCase):
    def test_exact_playlist_match(self):
        self.playlist_cache.save([{"name": "Gym", "href": "/playlist/1"}])
        self.controller.next_now_playing = {"song": "Pump It", "artist": "X", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_PLAYLIST", playlist="Gym")
        self.assertTrue(result["success"])
        self.assertEqual(self.controller.opened_hrefs[-1], "/playlist/1")

    def test_fuzzy_playlist_match(self):
        self.playlist_cache.save([{"name": "Gym Motivation", "href": "/playlist/2"}])
        self.controller.next_now_playing = {"song": "Pump It", "artist": "X", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_PLAYLIST", playlist="gym")
        self.assertTrue(result["success"])
        self.assertEqual(self.controller.opened_hrefs[-1], "/playlist/2")

    def test_ambiguous_playlist_asks_for_clarification(self):
        self.playlist_cache.save([
            {"name": "Chill Vibes", "href": "/playlist/3"},
            {"name": "Chill Vibes 2", "href": "/playlist/4"},
        ])
        result = provider.music_play("PLAY_PLAYLIST", playlist="chill vibes")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "ambiguous_playlist")

    def test_missing_playlist_refreshes_cache_then_fails_honestly(self):
        self.controller.playlists = []  # nothing in the real library either
        result = provider.music_play("PLAY_PLAYLIST", playlist="Does Not Exist")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "playlist_not_found")

    def test_missing_playlist_found_after_cache_refresh(self):
        self.controller.playlists = [{"name": "Road Trip", "href": "/playlist/5"}]
        self.controller.next_now_playing = {"song": "Drive", "artist": "X", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_PLAYLIST", playlist="Road Trip")
        self.assertTrue(result["success"])

    def test_one_of_my_playlists_avoids_immediate_repeat(self):
        self.playlist_cache.save([
            {"name": "A", "href": "/playlist/a"},
            {"name": "B", "href": "/playlist/b"},
        ])
        self.state_store.update_state(last_playlist="A")
        self.controller.next_now_playing = {"song": "Song", "artist": "X", "is_playing": True, "observed": True}
        previous_href = "/playlist/a"
        for _ in range(5):
            result = provider.music_play("PLAY_PLAYLIST", scope="random_user_playlist")
            self.assertTrue(result["success"])
            chosen_href = self.controller.opened_hrefs[-1]
            self.assertNotEqual(chosen_href, previous_href, "must never immediately repeat the last-played playlist")
            previous_href = chosen_href

    def test_shuffle_playlist_turns_shuffle_on(self):
        self.playlist_cache.save([{"name": "Gym", "href": "/playlist/1"}])
        self.controller.next_now_playing = {"song": "Pump It", "artist": "X", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_PLAYLIST", playlist="Gym", shuffle=True)
        self.assertTrue(result["success"])
        self.assertTrue(self.controller.shuffle_state)


class HistoryAndResumeTests(ProviderTestCase):
    def test_last_played_resolves_from_local_history(self):
        self.state_store.record_track(provider="apple_music", song="Starboy", artist="The Weeknd", identifier="/song/1")
        self.controller.next_now_playing = {"song": "Starboy", "artist": "The Weeknd", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_LAST_PLAYED")
        self.assertTrue(result["success"])
        self.assertEqual(self.controller.opened_hrefs[-1], "/song/1")

    def test_last_played_falls_back_to_apple_music_recently_played(self):
        self.controller.recently_played = [{"type": "song", "title": "Some Recent Song", "href": "/song/9"}]
        self.controller.next_now_playing = {"song": "Some Recent Song", "artist": "Someone", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_LAST_PLAYED")
        self.assertTrue(result["success"])

    def test_last_played_with_no_history_at_all_is_honest(self):
        result = provider.music_play("PLAY_LAST_PLAYED")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "no_history")

    def test_resume_uses_active_paused_player_before_searching(self):
        self.controller.now_playing = {"song": "Paused Song", "artist": "X", "is_playing": False, "observed": True}
        result = provider.music_play("RESUME_LAST_SESSION")
        self.assertTrue(result["success"])
        self.assertEqual(self.controller.control_calls, ["press_play"])
        self.assertEqual(self.controller.search_results, [])  # never searched

    def test_resume_reconstructs_from_history_when_no_active_session(self):
        self.controller._session_live = False
        self.controller.page = None
        self.state_store.record_track(provider="apple_music", song="Starboy", artist="The Weeknd", identifier="/song/1")
        self.controller.next_now_playing = {"song": "Starboy", "artist": "The Weeknd", "is_playing": True, "observed": True}
        result = provider.music_play("RESUME_LAST_SESSION")
        self.assertTrue(result["success"])


class ContextualCommandTests(ProviderTestCase):
    def test_now_playing_reports_observed_song_and_artist(self):
        self.controller.now_playing = {"song": "Starboy", "artist": "The Weeknd", "is_playing": True, "observed": True}
        result = provider.music_now_playing("song")
        self.assertTrue(result["success"])
        self.assertIn("Starboy", result["message"])

    def test_now_playing_artist_aspect(self):
        self.controller.now_playing = {"song": "Starboy", "artist": "The Weeknd", "is_playing": True, "observed": True}
        result = provider.music_now_playing("artist")
        self.assertIn("The Weeknd", result["message"])

    def test_now_playing_unavailable_is_honest(self):
        result = provider.music_now_playing("song")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "now_playing_unavailable")

    def test_queue_add_contextual_it_without_current_song_is_rejected(self):
        result = provider.music_queue_add("it", contextual=True)
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "nothing_to_queue")

    def test_queue_add_specific_song(self):
        self.controller.search_results = [{"type": "song", "title": "Starboy", "href": "/song/1"}]
        result = provider.music_queue_add("Starboy", contextual=False)
        self.assertTrue(result["success"])
        self.assertEqual(self.controller.queue_calls, [("/song/1", False)])

    def test_play_next_queues_up_next_not_add_to_queue(self):
        self.controller.search_results = [{"type": "song", "title": "Starboy", "href": "/song/1"}]
        result = provider.music_queue_next("Starboy", contextual=False)
        self.assertTrue(result["success"])
        self.assertEqual(self.controller.queue_calls, [("/song/1", True)])

    def test_artist_more_uses_current_artist(self):
        self.controller.now_playing = {"song": "Starboy", "artist": "The Weeknd", "is_playing": True, "observed": True}
        self.controller.search_results = [{"type": "artist", "title": "The Weeknd", "href": "/artist/1"}]
        self.controller.next_now_playing = {"song": "Some Hit", "artist": "The Weeknd", "is_playing": True, "observed": True}
        result = provider.music_artist_more()
        self.assertTrue(result["success"])


class RealCatalogTitleScoringTests(ProviderTestCase):
    """Confirmed live: Apple's real catalog title for "Starboy" is
    "Starboy (feat. Daft Punk)" -- an exact-title ALBUM used to outrank
    the correct but suffix-carrying SONG (0.483 fuzzy vs 1.0 fuzzy). These
    pin the fix (prefix-match boost + a decisive type-preference bonus)."""

    def test_song_with_feat_suffix_beats_exact_title_album(self):
        self.controller.search_results = [
            {"type": "album", "title": "Starboy", "href": "/album/starboy"},
            {"type": "song", "title": "Starboy (feat. Daft Punk)", "href": "/song/starboy-feat"},
        ]
        self.controller.next_now_playing = {"song": "Starboy (feat. Daft Punk)", "artist": "The Weeknd", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_SONG", song="Starboy", artist="The Weeknd")
        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])
        self.assertEqual(self.controller.opened_hrefs[-1], "/song/starboy-feat")

    def test_verification_tolerates_feat_suffix_via_title_matches(self):
        from tools.music.apple_music_provider import _title_matches
        self.assertTrue(_title_matches("Starboy", "Starboy (feat. Daft Punk)"))
        self.assertTrue(_title_matches("Starboy (feat. Daft Punk)", "Starboy"))
        self.assertFalse(_title_matches("Starboy", "Blinding Lights"))

    def test_song_type_selected_uses_play_specific_track_not_generic(self):
        self.controller.search_results = [{"type": "song", "title": "Starboy (feat. Daft Punk)", "href": "/song/1"}]
        self.controller.next_now_playing = {"song": "Starboy (feat. Daft Punk)", "artist": "The Weeknd", "is_playing": True, "observed": True}
        provider.music_play("PLAY_SONG", song="Starboy", artist="The Weeknd")
        self.assertEqual(self.controller.play_specific_track_calls, [("Starboy (feat. Daft Punk)", "The Weeknd")])

    def test_album_type_selected_uses_generic_play_not_specific_track(self):
        self.controller.search_results = [{"type": "album", "title": "After Hours", "href": "/album/1"}]
        self.controller.next_now_playing = {"song": "Alone Again", "artist": "The Weeknd", "is_playing": True, "observed": True}
        provider.music_play("PLAY_ALBUM", album="After Hours", artist="The Weeknd")
        self.assertEqual(self.controller.play_specific_track_calls, [])
        self.assertTrue(self.controller.played)


class ObservedPlaybackHistoryTests(ProviderTestCase):
    """Part 7: JARVIS must learn from playback it merely OBSERVES (the
    user started it manually), not only playback it started itself."""

    def test_now_playing_records_a_manually_started_track(self):
        self.assertIsNone(self.state_store.last_track())
        self.controller.now_playing = {"song": "גברת אגו", "artist": "Omer Adam", "is_playing": True, "observed": True}
        provider.music_now_playing("song")
        last = self.state_store.last_track()
        self.assertIsNotNone(last)
        self.assertEqual(last.song, "גברת אגו")
        self.assertEqual(last.context_type, "observed")

    def test_does_not_duplicate_while_the_same_track_continues(self):
        self.controller.now_playing = {"song": "Swim", "artist": "Chase Atlantic", "is_playing": True, "observed": True}
        provider.music_now_playing("song")
        provider.music_pause()
        provider.music_now_playing("song")
        history = self.state_store.recent_tracks(limit=10)
        self.assertEqual(len(history), 1)

    def test_pause_and_resume_also_observe_and_record(self):
        self.controller.now_playing = {"song": "Six Feet Under", "artist": "The Weeknd", "is_playing": True, "observed": True}
        provider.music_pause()
        last = self.state_store.last_track()
        self.assertIsNotNone(last)
        self.assertEqual(last.song, "Six Feet Under")

    def test_next_track_change_updates_history_to_the_new_track(self):
        self.controller.now_playing = {"song": "Old Song", "artist": "X", "is_playing": True, "observed": True}
        self.controller.next_now_playing = {"song": "New Song", "artist": "Y", "is_playing": True, "observed": True}
        provider.music_next()
        last = self.state_store.last_track()
        self.assertEqual(last.song, "New Song")

    def test_unobserved_state_never_fabricates_a_history_entry(self):
        self.controller._session_live = False
        self.controller.page = None
        provider.music_now_playing("song")
        self.assertIsNone(self.state_store.last_track())


class PlaylistNeverUsesGlobalSearchTests(ProviderTestCase):
    """Bug B: "play my <playlist>" must resolve from the user's OWN
    library only -- never fall through to a general catalog search, even
    when the library lookup fails."""

    def test_playlist_request_never_calls_catalog_search(self):
        self.controller.playlists = []  # nothing in the real library either
        provider.music_play("PLAY_PLAYLIST", playlist="Does Not Exist")
        self.assertEqual(self.controller.search_results, [])  # search() was never given anything to find

    def test_playlist_not_found_message_matches_library_scoped_wording(self):
        self.controller.playlists = []
        result = provider.music_play("PLAY_PLAYLIST", playlist="Gym")
        self.assertFalse(result["success"])
        self.assertIn("library", result["message"].lower())

    def test_playlist_resolution_uses_list_library_playlists_not_search(self):
        # A FakeController with a populated `search_results` list (as if a
        # public catalog playlist of the same name existed) must still be
        # ignored -- only `list_library_playlists`/the cache are consulted.
        self.controller.search_results = [{"type": "playlist", "title": "Gym", "href": "/catalog/playlist/gym"}]
        self.playlist_cache.save([{"name": "Gym", "href": "/library/playlist/real-gym"}])
        self.controller.next_now_playing = {"song": "Pump It", "artist": "X", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_PLAYLIST", playlist="Gym")
        self.assertTrue(result["success"])
        self.assertEqual(self.controller.opened_hrefs[-1], "/library/playlist/real-gym")


class HebrewSearchAndScoringTests(ProviderTestCase):
    """VOICE_LANGUAGE=he: Apple Music search must receive the exact
    original Hebrew entity -- never translated, never transliterated --
    and Hebrew-titled results must be scored/matched correctly (the old
    ASCII-only `_norm` silently stripped Hebrew to nothing before
    fuzzy-scoring it)."""

    def test_search_receives_the_exact_hebrew_song_text(self):
        self.controller.search_results = [{"type": "song", "title": "שני משוגעים", "href": "/song/1"}]
        self.controller.next_now_playing = {"song": "שני משוגעים", "artist": "Omer Adam", "is_playing": True, "observed": True}
        provider.music_play("PLAY_QUERY", song="שני משוגעים")
        self.assertIn("שני משוגעים", self.controller.search_queries)

    def test_hebrew_title_is_never_translated_or_transliterated_in_the_response(self):
        self.controller.search_results = [{"type": "song", "title": "שני משוגעים", "href": "/song/1"}]
        self.controller.next_now_playing = {"song": "שני משוגעים", "artist": "Omer Adam", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_QUERY", song="שני משוגעים")
        self.assertEqual(result["song"], "שני משוגעים")
        self.assertIn("שני משוגעים", result["message"])

    def test_hebrew_fuzzy_scoring_distinguishes_candidates_correctly(self):
        # Before the Unicode-aware _norm fix, every Hebrew title normalized
        # to an empty string and scored identically -- this pins the fix
        # by giving the wrong-title decoy a real chance to win if scoring
        # regresses back to ASCII-only stripping.
        self.controller.search_results = [
            {"type": "song", "title": "שיר אחר לגמרי", "href": "/song/wrong"},
            {"type": "song", "title": "שני משוגעים", "href": "/song/right"},
        ]
        self.controller.next_now_playing = {"song": "שני משוגעים", "artist": "Omer Adam", "is_playing": True, "observed": True}
        provider.music_play("PLAY_QUERY", song="שני משוגעים")
        self.assertEqual(self.controller.opened_hrefs[-1], "/song/right")

    def test_hebrew_playlist_name_matches_exactly(self):
        self.playlist_cache.save([{"name": "ישראלי", "href": "/library/playlist/israeli"}])
        self.controller.next_now_playing = {"song": "איך שהיא רוקדת", "artist": "Eden Hason", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_PLAYLIST", playlist="ישראלי")
        self.assertTrue(result["success"])
        self.assertEqual(self.controller.opened_hrefs[-1], "/library/playlist/israeli")


if __name__ == "__main__":
    unittest.main()
