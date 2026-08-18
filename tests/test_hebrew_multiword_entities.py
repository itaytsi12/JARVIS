# -*- coding: utf-8 -*-
"""Regression coverage for the Hebrew multi-word music entity extraction
bug: song titles with two or more words (and song+artist qualification)
must be preserved as one whole Unicode entity, never truncated to a
single token, and the false-success playback-verification bug (a search
hit / row click / Play click was being reported as unconditional success
even when observed player metadata never confirmed the requested
song/artist)."""
import unittest
from unittest.mock import patch

from brain.music_intent import classify_music_intent, MusicIntentType


class MultiWordSongExtractionTests(unittest.TestCase):
    CASES = [
        ("נגן שני משוגעים", "שני משוגעים"),
        ("נגן יום חדש", "יום חדש"),
        ("נגן דרך השלום", "דרך השלום"),
        ("נגן את השיר שני משוגעים", "שני משוגעים"),
        ("שים לי שני משוגעים", "שני משוגעים"),
        ("שים את השיר שני משוגעים", "שני משוגעים"),
    ]

    def test_multiword_song_preserved_whole_never_truncated(self):
        for phrase, expected_song in self.CASES:
            with self.subTest(phrase=phrase):
                intent = classify_music_intent(phrase)
                self.assertIsNotNone(intent, f"{phrase!r} should classify as a music intent")
                self.assertIn(intent.intent, (MusicIntentType.PLAY_QUERY, MusicIntentType.PLAY_SONG))
                self.assertEqual(intent.song, expected_song)
                # Explicitly guard against the exact failure modes named in
                # the bug report: only the first word, only the last word,
                # or nothing at all.
                first_word, last_word = expected_song.split(" ", 1)[0], expected_song.rsplit(" ", 1)[-1]
                self.assertNotEqual(intent.song, first_word)
                self.assertNotEqual(intent.song, last_word)

    def test_single_word_hebrew_song_still_works(self):
        intent = classify_music_intent("נגן משוגעים")
        self.assertEqual(intent.song, "משוגעים")

    def test_english_song_parsing_unchanged(self):
        intent = classify_music_intent("play Starboy")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_QUERY)
        self.assertEqual(intent.song, "Starboy")

    def test_hebrew_playlist_parsing_unchanged(self):
        intent = classify_music_intent("נגן את הפלייליסט ישראלי")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_PLAYLIST)
        self.assertEqual(intent.playlist, "ישראלי")

    def test_last_played_not_misclassified_as_a_song_title(self):
        intent = classify_music_intent("נגן את השיר האחרון ששמעתי")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_LAST_PLAYED)
        self.assertIsNone(intent.song)


class HebrewSongArtistSplitTests(unittest.TestCase):
    def test_song_and_artist_split_on_shel(self):
        intent = classify_music_intent("נגן שני משוגעים של עומר אדם")
        self.assertEqual(intent.song, "שני משוגעים")
        self.assertEqual(intent.artist, "עומר אדם")

    def test_song_and_artist_split_on_meet(self):
        intent = classify_music_intent("נגן שני משוגעים מאת עומר אדם")
        self.assertEqual(intent.song, "שני משוגעים")
        self.assertEqual(intent.artist, "עומר אדם")

    def test_multiword_song_and_multiword_artist(self):
        intent = classify_music_intent("נגן דרך השלום של פאר טסי")
        self.assertEqual(intent.song, "דרך השלום")
        self.assertEqual(intent.artist, "פאר טסי")

    def test_song_without_artist_qualifier_is_not_split(self):
        # "השלום" (with the ה attached) must never be mistaken for a
        # standalone "של" token.
        intent = classify_music_intent("נגן דרך השלום")
        self.assertEqual(intent.song, "דרך השלום")
        self.assertIsNone(intent.artist)


class PlayQueryArtistDispatchTests(unittest.TestCase):
    """A PLAY_QUERY with both song and artist extracted must actually use
    both when dispatched to the Apple Music provider, not silently drop
    the artist."""

    def test_route_carries_artist_through_to_tool_arguments(self):
        from brain.router import route_command
        route = route_command("נגן שני משוגעים של עומר אדם")
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route["tool"], "music_play")
        self.assertEqual(route["arguments"]["song"], "שני משוגעים")
        self.assertEqual(route["arguments"]["artist"], "עומר אדם")

    def test_play_query_with_artist_dispatches_to_play_song_not_play_query(self):
        from tools.music import apple_music_provider as provider
        with patch.object(provider, "_play_song", return_value={"success": True}) as fake_play_song, \
             patch.object(provider, "_play_query") as fake_play_query:
            provider.music_play("PLAY_QUERY", song="שני משוגעים", artist="עומר אדם")
        fake_play_song.assert_called_once_with("שני משוגעים", "עומר אדם")
        fake_play_query.assert_not_called()

    def test_play_query_without_artist_still_uses_play_query(self):
        from tools.music import apple_music_provider as provider
        with patch.object(provider, "_play_song") as fake_play_song, \
             patch.object(provider, "_play_query", return_value={"success": True}) as fake_play_query:
            provider.music_play("PLAY_QUERY", song="שני משוגעים", artist=None)
        fake_play_query.assert_called_once()
        fake_play_song.assert_not_called()


class FalseSuccessAndVerificationTests(unittest.TestCase):
    """False-success rule: a search hit / row click / Play click landing
    is never itself proof of success -- only observed player metadata
    matching the requested song/artist counts. Also covers the
    now-playing honesty rule and the cross-script (Hebrew artist name vs.
    Latin-script catalog metadata) verification softening."""

    def setUp(self):
        from tests.test_apple_music_provider import FakeController, ProviderTestCase
        self._case = ProviderTestCase()
        self._case.setUp()
        self.controller = self._case.controller

    def tearDown(self):
        self._case.tearDown()

    def test_unconfirmed_playback_reports_success_false(self):
        from tools.music import apple_music_provider as provider
        self.controller.search_results = [{"type": "song", "title": "Starboy", "href": "/song/1"}]
        self.controller.next_now_playing = {"song": "Something Else Entirely", "artist": "Nobody", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_SONG", song="Starboy", artist=None)
        self.assertFalse(result["success"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["message"], "I started the request, but I couldn't confirm the track, sir.")

    def test_confirmed_playback_still_reports_success_true(self):
        from tools.music import apple_music_provider as provider
        self.controller.search_results = [{"type": "song", "title": "Starboy", "href": "/song/1"}]
        self.controller.next_now_playing = {"song": "Starboy", "artist": "The Weeknd", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_SONG", song="Starboy", artist="The Weeknd")
        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])

    def test_now_playing_never_falls_back_to_local_requested_state(self):
        from tools.music import apple_music_provider as provider
        # Local state store has a recorded song, but the live DOM has
        # nothing observable right now -- must report honest failure, not
        # the stale locally-remembered song.
        self._case.state_store.record_track(provider="apple_music", song="Old Song", artist="Old Artist", identifier="/song/1")
        self.controller.now_playing = {"song": None, "artist": None, "is_playing": False, "observed": False}
        result = provider.music_now_playing("song")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "now_playing_unavailable")

    def test_now_playing_reads_live_dom_not_local_state(self):
        from tools.music import apple_music_provider as provider
        self._case.state_store.record_track(provider="apple_music", song="Old Song", artist="Old Artist", identifier="/song/1")
        self.controller.now_playing = {"song": "Currently Playing", "artist": "Real Artist", "is_playing": True, "observed": True}
        result = provider.music_now_playing("song")
        self.assertIn("Currently Playing", result["message"])
        self.assertNotIn("Old Song", result["message"])

    def test_cross_script_hebrew_artist_does_not_block_verification_when_song_matches(self):
        from tools.music import apple_music_provider as provider
        self.controller.search_results = [{"type": "song", "title": "שני משוגעים", "href": "/song/1"}]
        # Real catalog artist metadata is Latin-script; the spoken/parsed
        # artist is Hebrew -- an unbridgeable script mismatch that must not
        # block verification once the song itself is an exact match.
        self.controller.next_now_playing = {"song": "שני משוגעים", "artist": "Omer Adam", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_QUERY", song="שני משוגעים", artist="עומר אדם")
        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])

    def test_genuinely_wrong_artist_in_same_script_still_fails_verification(self):
        from tools.music import apple_music_provider as provider
        self.controller.search_results = [{"type": "song", "title": "Starboy", "href": "/song/1"}]
        self.controller.next_now_playing = {"song": "Starboy", "artist": "Some Other Artist", "is_playing": True, "observed": True}
        result = provider.music_play("PLAY_SONG", song="Starboy", artist="The Weeknd")
        self.assertFalse(result["verified"])


class PreviewVsFullPlaybackTests(unittest.TestCase):
    """Confirmed live: Apple Music Web can serve a short instant-preview
    clip (a plain <audio> element whose src is on Apple's AudioPreview
    CDN path, ~90s long) while the player-bar UI reports the exact same
    "now playing" song/artist/is_playing state as real full-track
    streaming -- song/artist metadata matching alone is NOT sufficient
    proof of full playback."""

    def setUp(self):
        from tests.test_apple_music_provider import ProviderTestCase
        self._case = ProviderTestCase()
        self._case.setUp()
        self.controller = self._case.controller

    def tearDown(self):
        self._case.tearDown()

    def test_detected_preview_downgrades_verified_and_success_even_with_matching_metadata(self):
        from tools.music import apple_music_provider as provider
        self.controller.search_results = [{"type": "song", "title": "שני משוגעים", "href": "/song/1"}]
        self.controller.next_now_playing = {"song": "שני משוגעים", "artist": "Omer Adam", "is_playing": True, "observed": True}
        self.controller.next_playback_type = {"observed": True, "is_preview": True, "duration": 89.98}
        result = provider.music_play("PLAY_QUERY", song="שני משוגעים", artist=None)
        self.assertFalse(result["success"])
        self.assertFalse(result["verified"])
        self.assertTrue(result["is_preview"])
        self.assertEqual(result["message"], "I could only start a short preview, not the full track, sir.")

    def test_full_playback_not_flagged_as_preview(self):
        from tools.music import apple_music_provider as provider
        self.controller.search_results = [{"type": "song", "title": "Starboy", "href": "/song/1"}]
        self.controller.next_now_playing = {"song": "Starboy", "artist": "The Weeknd", "is_playing": True, "observed": True}
        self.controller.next_playback_type = {"observed": True, "is_preview": False, "duration": 230.5}
        result = provider.music_play("PLAY_SONG", song="Starboy", artist="The Weeknd")
        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])
        self.assertFalse(result["is_preview"])

    def test_no_audio_element_observable_does_not_assume_preview(self):
        # observed=False (no <audio> element found at all) must not itself
        # be treated as evidence of either preview or full playback --
        # real DRM'd streaming may use a mechanism this can't introspect.
        from tools.music import apple_music_provider as provider
        self.controller.search_results = [{"type": "song", "title": "Starboy", "href": "/song/1"}]
        self.controller.next_now_playing = {"song": "Starboy", "artist": "The Weeknd", "is_playing": True, "observed": True}
        self.controller.next_playback_type = {"observed": False}
        result = provider.music_play("PLAY_SONG", song="Starboy", artist="The Weeknd")
        self.assertTrue(result["success"])
        self.assertFalse(result["is_preview"])


if __name__ == "__main__":
    unittest.main()
