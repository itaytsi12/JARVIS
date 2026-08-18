import tempfile
import unittest
from pathlib import Path

from brain.music_state import MusicStateStore


class MusicStateStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MusicStateStore(Path(self._tmp.name) / "music_state.db")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_empty_state_is_safe_default(self):
        state = self.store.get_state()
        self.assertIsNone(state.current_song)
        self.assertFalse(state.is_playing)
        self.assertEqual(state.recent_tracks, [])

    def test_record_track_updates_state_and_history(self):
        self.store.record_track(provider="apple_music", song="Starboy", artist="The Weeknd", identifier="/song/1")
        state = self.store.get_state()
        self.assertEqual(state.current_song, "Starboy")
        self.assertEqual(state.current_artist, "The Weeknd")
        self.assertTrue(state.is_playing)
        last = self.store.last_track()
        self.assertEqual(last.song, "Starboy")
        self.assertEqual(last.identifier, "/song/1")

    def test_record_track_honors_observed_is_playing_state(self):
        # Part 7: recording from OBSERVED playback (e.g. the user just
        # paused a manually-started track) must not lie and claim it's
        # still playing.
        self.store.record_track(provider="apple_music", song="Paused Song", artist="X", is_playing=False)
        state = self.store.get_state()
        self.assertFalse(state.is_playing)
        self.assertEqual(state.current_song, "Paused Song")

    def test_last_song_resolves_from_history(self):
        self.store.record_track(provider="apple_music", song="Blinding Lights", artist="The Weeknd")
        self.store.record_track(provider="apple_music", song="Starboy", artist="The Weeknd")
        last = self.store.last_track()
        self.assertEqual(last.song, "Starboy")

    def test_empty_history_falls_back_appropriately(self):
        self.assertIsNone(self.store.last_track())
        self.assertEqual(self.store.recent_tracks(), [])

    def test_history_is_bounded(self):
        store = MusicStateStore(Path(self._tmp.name) / "bounded.db", max_history=3)
        try:
            for i in range(6):
                store.record_track(provider="apple_music", song=f"Song {i}")
            recent = store.recent_tracks(limit=10)
            self.assertEqual(len(recent), 3)
            self.assertEqual(recent[0].song, "Song 5")
        finally:
            store.close()

    def test_update_state_partial_fields(self):
        self.store.update_state(shuffle=True)
        state = self.store.get_state()
        self.assertTrue(state.shuffle)
        self.assertIsNone(state.current_song)

    def test_last_playlist_persists_across_state_updates(self):
        self.store.record_track(provider="apple_music", song="Song A", playlist="Gym")
        state = self.store.get_state()
        self.assertEqual(state.last_playlist, "Gym")
        self.store.update_state(is_playing=False)
        self.assertEqual(self.store.get_state().last_playlist, "Gym")


if __name__ == "__main__":
    unittest.main()
