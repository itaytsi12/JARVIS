"""Integration tests: brain.router.route_command dispatches music commands
to the deterministic music routes (never the local intent model / cloud
fallback for obvious phrasing), and existing non-music routing is
unaffected (regression)."""
import unittest

from brain.router import route_command


class MusicRouterIntegrationTests(unittest.TestCase):
    def test_open_music_routes_to_music_tool_not_open_application(self):
        route = route_command("open apple music")
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route["tool"], "open_music")

    def test_open_music_bare(self):
        route = route_command("open music")
        self.assertEqual(route["tool"], "open_music")

    def test_play_song_by_artist(self):
        route = route_command("play Starboy by The Weeknd")
        self.assertEqual(route["tool"], "music_play")
        self.assertEqual(route["arguments"]["song"], "Starboy")
        self.assertEqual(route["arguments"]["artist"], "The Weeknd")

    def test_playlist_request(self):
        route = route_command("play my gym playlist")
        self.assertEqual(route["arguments"]["intent"], "PLAY_PLAYLIST")
        self.assertEqual(route["arguments"]["playlist"], "gym")

    def test_pause_resume_next_previous_are_fast_tools(self):
        self.assertEqual(route_command("pause")["tool"], "music_pause")
        self.assertEqual(route_command("resume")["tool"], "music_resume")
        self.assertEqual(route_command("next")["tool"], "music_next")
        self.assertEqual(route_command("previous song")["tool"], "music_previous")

    def test_now_playing_question(self):
        route = route_command("what song is playing?")
        self.assertEqual(route["tool"], "music_now_playing")

    def test_spotify_explicit_provider_not_apple_music(self):
        route = route_command("play Starboy on Spotify")
        self.assertEqual(route["type"], "local_plan")
        self.assertIn("open.spotify.com", route["actions"][0].args["url"])

    def test_youtube_explicit_provider_not_apple_music(self):
        route = route_command("play it on YouTube")
        self.assertEqual(route["type"], "local_plan")
        self.assertIn("youtube.com", route["actions"][0].args["url"])

    # ------------------------------------------------------------------
    # Regression: pre-existing, unrelated routing must be unaffected.
    # ------------------------------------------------------------------

    def test_bare_stop_still_cancels_task(self):
        self.assertEqual(route_command("stop"), {"type": "cancel_read_only_task"})

    def test_bare_continue_still_resumes_interrupted_response(self):
        self.assertEqual(route_command("continue"), {"type": "resume_interrupted_response"})

    def test_open_notepad_still_opens_application(self):
        route = route_command("open notepad")
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route["tool"], "open_application")
        self.assertEqual(route["arguments"]["app_name"], "notepad")

    def test_volume_up_unaffected(self):
        route = route_command("volume up")
        self.assertEqual(route["tool"], "volume_up")

    def test_calculator_unaffected(self):
        route = route_command("what is 2 + 2")
        self.assertEqual(route["tool"], "calculator")


if __name__ == "__main__":
    unittest.main()
