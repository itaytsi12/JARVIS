"""Apple Music playlist authoring, and the search-quality fixes behind it.

Driven entirely through a fake controller: no browser, no account, no
network. What is asserted is the behaviour that was found and fixed by
running this against the real signed-in account:

- a search result whose title has nothing to do with the request must
  never be used ("Save Your Tears" resolved to "Stressed Out"),
- a track must never be reported as added unless it is really in the
  playlist afterwards, and that check must be against what the USER asked
  for, not against whatever happened to match,
- a playlist that does not exist must produce the names that do.
"""
import unittest
from unittest.mock import patch

from tools.music import apple_music_provider as provider


class FakeController:
    """Enough of `AppleMusicWebController` to drive the provider."""

    def __init__(self, results=None, playlists=None, tracks=None):
        self._results = results or []
        self._playlists = list(playlists or [])
        # playlist name -> track titles currently in it
        self._tracks = dict(tracks or {})
        self.opened = []
        self.created = []
        self.added = []
        self.current_playlist = None

    # -- search / navigation
    def search(self, query):
        return list(self._results)

    def open_result(self, href):
        self.opened.append(href)
        return True

    def list_library_playlists(self):
        return [{"name": name, "href": f"/library/playlist/{name}"} for name in self._playlists]

    def open_library_playlist(self, name):
        match = [p for p in self._playlists if p.lower() == name.lower()]
        if not match:
            return {"opened": False, "error": "playlist_not_found", "available": list(self._playlists)}
        self.current_playlist = match[0]
        return {"opened": True, "name": match[0]}

    def current_page_track_titles(self, limit=60):
        return list(self._tracks.get(self.current_playlist, []))

    # -- authoring
    def add_track_to_playlist(self, playlist, title=None, artist=None):
        if playlist not in self._playlists:
            return {"added": False, "error": "playlist_not_found", "available": list(self._playlists)}
        self.added.append((playlist, title))
        self._tracks.setdefault(playlist, []).append(title)
        return {"added": True, "playlist": playlist}

    def create_playlist_from_track(self, name, title=None, artist=None):
        self.created.append((name, title))
        self._playlists.append(name)
        self._tracks[name] = [title]
        return {"created": True, "name": name}


def _song(title, kind="song"):
    return {"type": kind, "title": title, "href": f"/il/album/x?i={title}"}


class MatchQualityTests(unittest.TestCase):
    """`_best_search_match`'s type bonus is decisive by design, and was
    large enough to carry an unrelated title over the score threshold."""

    def test_an_unrelated_title_is_refused_rather_than_returned_confidently(self):
        controller = FakeController(results=[_song("Stressed Out"), _song("thank u, next", "album")])
        self.assertIsNone(
            provider._best_search_match(controller, "Save Your Tears", "Save Your Tears", prefer_types=("song", "album"))
        )

    def test_a_real_match_still_wins(self):
        controller = FakeController(results=[_song("Stressed Out"), _song("Blinding Lights")])
        match = provider._best_search_match(controller, "Blinding Lights", "Blinding Lights", prefer_types=("song", "album"))
        self.assertEqual(match["title"], "Blinding Lights")

    def test_a_feat_suffix_is_still_the_same_song(self):
        """Apple's real title for "Starboy" is "Starboy (feat. Daft Punk)".
        A prefix match must survive the floor."""
        controller = FakeController(results=[_song("Starboy (feat. Daft Punk)")])
        match = provider._best_search_match(controller, "Starboy", "Starboy", prefer_types=("song", "album"))
        self.assertEqual(match["title"], "Starboy (feat. Daft Punk)")

    def test_a_free_text_request_is_not_held_to_a_title_floor(self):
        """"play something upbeat" names no title, so there is nothing for
        the floor to compare against."""
        controller = FakeController(results=[_song("Good Vibes Mix", "playlist")])
        match = provider._best_search_match(
            controller, "upbeat", "upbeat", prefer_types=("playlist",), title_floor=provider.NO_TITLE_FLOOR
        )
        self.assertIsNotNone(match)


class LocateSongTests(unittest.TestCase):
    def test_a_wrong_song_is_never_opened_for_a_playlist_edit(self):
        """Playing the wrong song is audible and correctable; adding it to
        a playlist is a silent, persistent edit."""
        controller = FakeController(results=[_song("Stressed Out")])
        self.assertIsNone(provider._locate_song(controller, "Save Your Tears", None))
        self.assertEqual(controller.opened, [])

    def test_a_correct_song_is_opened(self):
        controller = FakeController(results=[_song("Ordinary")])
        match = provider._locate_song(controller, "Ordinary", None)
        self.assertEqual(match["title"], "Ordinary")
        self.assertEqual(len(controller.opened), 1)


class AddToPlaylistTests(unittest.TestCase):
    def _run(self, controller, *args, **kwargs):
        with patch.object(provider, "_ensure_ready", return_value=(controller, None)):
            return provider.music_add_to_playlist(*args, **kwargs)

    def test_a_successful_add_is_verified_against_the_playlist(self):
        controller = FakeController(results=[_song("Ordinary")], playlists=["Gym"], tracks={"Gym": []})
        result = self._run(controller, "Ordinary", "Gym")
        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])
        self.assertEqual(controller.added, [("Gym", "Ordinary")])

    def test_verification_checks_the_requested_song_not_the_matched_one(self):
        """The original bug: a wrong track was added and then "verified" by
        looking for the wrong track, which of course was there."""
        controller = FakeController(results=[_song("Ordinary")], playlists=["Gym"], tracks={"Gym": []})
        with patch.object(provider, "_playlist_contains", return_value=False) as contains:
            result = self._run(controller, "Ordinary", "Gym")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "verification_failed")
        self.assertEqual(contains.call_args.args[2], "Ordinary")

    def test_an_unknown_playlist_reports_the_ones_that_exist(self):
        controller = FakeController(results=[_song("Ordinary")], playlists=["Gym", "Focus"])
        result = self._run(controller, "Ordinary", "Party")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "playlist_not_found")
        self.assertIn("Gym", result["message"])

    def test_the_playlist_name_is_resolved_before_any_catalog_search(self):
        """A bad name used to surface as `row_menu_unavailable` after a
        pointless search, which told the user nothing."""
        controller = FakeController(results=[_song("Ordinary")], playlists=["Gym"])
        self._run(controller, "Ordinary", "Party")
        self.assertEqual(controller.opened, [])

    def test_an_unambiguous_prefix_resolves_to_the_real_name(self):
        controller = FakeController(results=[_song("Ordinary")], playlists=["JARVIS Test Two"], tracks={"JARVIS Test Two": []})
        result = self._run(controller, "Ordinary", "JARVIS Test")
        self.assertTrue(result["success"])
        self.assertEqual(result["playlist"], "JARVIS Test Two")

    def test_an_ambiguous_prefix_asks_rather_than_guessing(self):
        controller = FakeController(results=[_song("Ordinary")], playlists=["Gym A", "Gym B"])
        result = self._run(controller, "Ordinary", "Gym")
        self.assertEqual(result["error"], "playlist_ambiguous")

    def test_missing_arguments_are_refused_before_the_browser_is_touched(self):
        self.assertEqual(provider.music_add_to_playlist("", "Gym")["error"], "missing_song")
        self.assertEqual(provider.music_add_to_playlist("Ordinary", "")["error"], "missing_playlist")


class CreatePlaylistTests(unittest.TestCase):
    def _run(self, controller, *args, **kwargs):
        with patch.object(provider, "_ensure_ready", return_value=(controller, None)):
            return provider.music_create_playlist(*args, **kwargs)

    def test_a_playlist_is_created_with_its_first_song_and_verified(self):
        controller = FakeController(results=[_song("Ordinary")], playlists=[])
        result = self._run(controller, "Focus", ["Ordinary"])
        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])
        self.assertEqual(controller.created, [("Focus", "Ordinary")])

    def test_an_empty_playlist_is_refused_with_the_real_reason(self):
        """Apple Music's web player genuinely cannot create one, so this
        says so rather than failing obscurely."""
        result = provider.music_create_playlist("Focus", [])
        self.assertEqual(result["error"], "no_songs")
        self.assertIn("at least one song", result["message"])

    def test_a_duplicate_name_is_refused_rather_than_creating_a_second(self):
        controller = FakeController(results=[_song("Ordinary")], playlists=["Focus"])
        result = self._run(controller, "Focus", ["Ordinary"])
        self.assertEqual(result["error"], "playlist_exists")
        self.assertEqual(controller.created, [])

    def test_songs_that_could_not_be_added_are_reported_not_hidden(self):
        controller = FakeController(results=[_song("Ordinary")], playlists=[])

        def add(song, playlist, artist=None):
            return {"success": song == "Ordinary", "message": ""}

        with patch.object(provider, "music_add_to_playlist", side_effect=add):
            result = self._run(controller, "Focus", ["Ordinary", "Nonexistent Song"])
        self.assertTrue(result["success"])
        self.assertEqual(result["failed"], ["Nonexistent Song"])
        self.assertIn("couldn't add", result["message"])

    def test_a_creation_that_cannot_be_confirmed_is_not_reported_as_success(self):
        controller = FakeController(results=[_song("Ordinary")], playlists=[])
        with patch.object(provider, "_playlist_exists", return_value=False):
            result = self._run(controller, "Focus", ["Ordinary"])
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "verification_failed")

    def test_a_runaway_batch_is_refused(self):
        controller = FakeController(results=[_song("Ordinary")], playlists=[])
        result = self._run(controller, "Focus", ["s"] * (provider.MAX_PLAYLIST_SONGS + 1))
        self.assertEqual(result["error"], "too_many_songs")

    def test_a_missing_name_is_refused(self):
        self.assertEqual(provider.music_create_playlist("", ["x"])["error"], "missing_name")


class PlaylistExistenceTests(unittest.TestCase):
    def test_a_just_created_playlist_is_waited_for_rather_than_declared_missing(self):
        """Confirmed live: creation succeeded but the very next read of the
        library still returned the previous names -- Apple's listing is
        eventually consistent."""
        controller = FakeController(playlists=[])
        calls = {"n": 0}

        def listing():
            calls["n"] += 1
            return [{"name": "Focus"}] if calls["n"] > 2 else []

        controller.list_library_playlists = listing
        with patch.object(provider.time, "sleep"):
            self.assertTrue(provider._playlist_exists(controller, "Focus", attempts=4))

    def test_a_genuinely_absent_playlist_still_reports_absent(self):
        controller = FakeController(playlists=["Gym"])
        with patch.object(provider.time, "sleep"):
            self.assertFalse(provider._playlist_exists(controller, "Focus", attempts=2))


if __name__ == "__main__":
    unittest.main()
