import tempfile
import time
import unittest
from pathlib import Path

from tools.music.playlist_cache import PlaylistCache


class PlaylistCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = PlaylistCache(Path(self._tmp.name) / "playlists.json", ttl_seconds=1)

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_cache_is_stale_and_has_no_playlists(self):
        self.assertTrue(self.cache.is_stale())
        self.assertEqual(self.cache.playlists(), [])
        self.assertEqual(self.cache.find("gym"), [])

    def test_save_and_load_round_trip(self):
        self.cache.save([{"name": "Gym", "href": "/playlist/1"}])
        self.assertFalse(self.cache.is_stale())
        self.assertEqual(self.cache.playlists(), [{"name": "Gym", "href": "/playlist/1"}])

    def test_exact_match_scores_highest(self):
        self.cache.save([{"name": "Gym", "href": "/p/1"}, {"name": "Gym Motivation", "href": "/p/2"}])
        matches = self.cache.find("Gym")
        self.assertEqual(matches[0].name, "Gym")
        self.assertEqual(matches[0].score, 1.0)

    def test_normalized_match_ignores_the_word_playlist(self):
        self.cache.save([{"name": "Gym Playlist", "href": "/p/1"}])
        matches = self.cache.find("gym")
        self.assertTrue(matches)
        self.assertEqual(matches[0].name, "Gym Playlist")

    def test_fuzzy_match_finds_close_names(self):
        self.cache.save([{"name": "Chill Vibes", "href": "/p/1"}])
        matches = self.cache.find("chil vibs")
        self.assertTrue(matches)
        self.assertEqual(matches[0].name, "Chill Vibes")

    def test_no_match_below_threshold(self):
        self.cache.save([{"name": "Road Trip", "href": "/p/1"}])
        self.assertEqual(self.cache.find("completely unrelated playlist name"), [])

    def test_cache_becomes_stale_after_ttl(self):
        self.cache.save([{"name": "Gym", "href": "/p/1"}])
        self.assertFalse(self.cache.is_stale())
        time.sleep(1.2)
        self.assertTrue(self.cache.is_stale())

    def test_hebrew_playlist_name_matches_exactly(self):
        # An ASCII-only normalizer used to reduce every Hebrew name to an
        # empty string, making Hebrew playlists permanently unmatchable.
        self.cache.save([{"name": "ישראלי", "href": "/p/1"}])
        matches = self.cache.find("ישראלי")
        self.assertTrue(matches)
        self.assertEqual(matches[0].name, "ישראלי")
        self.assertEqual(matches[0].score, 1.0)

    def test_hebrew_fuzzy_match_finds_close_names(self):
        self.cache.save([{"name": "מוזיקה ישראלית", "href": "/p/1"}])
        matches = self.cache.find("ישראלית")
        self.assertTrue(matches)
        self.assertEqual(matches[0].name, "מוזיקה ישראלית")


if __name__ == "__main__":
    unittest.main()
