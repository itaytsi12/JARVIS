"""Regression coverage for the spoken-response-formatting bug found while
debugging the live "Hey Jarvis, open music" path: `voice/response_formatter.py`
was flattening every failed (and some successful) music tool response into
a generic "I couldn't complete that action, sir." / "Okay, sir." -- even
though the whole point of the music feature's honest failure/status
messages (Part 20/25) is that they get SPOKEN, not swallowed.

The end-to-end pipeline itself (route_command -> tool_router -> the Apple
Music controller -> AuthenticatedBrowserSession -> CDP attach -> tab
find/open) was already proven working by the real
`logs/jarvis_background.log` trace at 2026-08-18 12:18:33 -- it correctly
reached Apple Music and reported "not signed in." The formatter is what
silently discarded that specific, correct message before it reached the
user's ears."""
import unittest

from voice.response_formatter import format_spoken_response


class MusicFailureMessagesAreSpokenTests(unittest.TestCase):
    def test_sign_in_required_message_is_spoken_specifically(self):
        route = {"type": "tool", "tool": "open_music", "arguments": {}}
        response_text = "Apple Music needs you to sign in, sir. I opened it so you can sign in manually.\nError: sign_in_required"
        spoken = format_spoken_response("open music", route, response_text, lang="en")
        self.assertIn("sign in", spoken.lower())
        self.assertNotEqual(spoken, "I couldn't complete that action, sir.")
        self.assertNotIn("error:", spoken.lower())
        self.assertNotIn("sign_in_required", spoken)

    def test_authenticated_chrome_not_running_message_is_spoken_specifically(self):
        route = {"type": "tool", "tool": "open_music", "arguments": {}}
        response_text = (
            "Authenticated Chrome is not running. Start the JARVIS browser session first.\n"
            "Error: apple_music_unavailable: Authenticated Chrome is not running. Start the JARVIS browser session first."
        )
        spoken = format_spoken_response("open music", route, response_text, lang="en")
        self.assertIn("authenticated chrome is not running", spoken.lower())
        self.assertNotEqual(spoken, "I couldn't complete that action, sir.")

    def test_playlist_not_found_message_with_natural_couldnt_wording_is_spoken(self):
        route = {"type": "tool", "tool": "music_play", "arguments": {"intent": "PLAY_PLAYLIST"}}
        response_text = "I couldn't find a playlist called Gym, sir.\nError: playlist_not_found"
        spoken = format_spoken_response("play my gym playlist", route, response_text, lang="en")
        self.assertEqual(spoken, "I couldn't find a playlist called Gym, sir.")

    def test_song_not_found_message_is_spoken(self):
        route = {"type": "tool", "tool": "music_play", "arguments": {"intent": "PLAY_SONG"}}
        response_text = "I couldn't find Starboy by The Weeknd, sir.\nError: not_found"
        spoken = format_spoken_response("play Starboy by The Weeknd", route, response_text, lang="en")
        self.assertEqual(spoken, "I couldn't find Starboy by The Weeknd, sir.")


class MusicSuccessMessagesAreSpokenTests(unittest.TestCase):
    def test_opened_apple_music_message_is_spoken_not_generic_okay(self):
        route = {"type": "tool", "tool": "open_music", "arguments": {}}
        spoken = format_spoken_response("open music", route, "Opened Apple Music, sir.", lang="en")
        self.assertEqual(spoken, "Opened Apple Music, sir.")

    def test_playing_song_message_is_spoken_verbatim(self):
        route = {"type": "tool", "tool": "music_play", "arguments": {"intent": "PLAY_SONG"}}
        spoken = format_spoken_response("play Starboy by The Weeknd", route, "Playing Starboy by The Weeknd.", lang="en")
        self.assertEqual(spoken, "Playing Starboy by The Weeknd, sir.")

    def test_paused_message_is_spoken_verbatim(self):
        route = {"type": "tool", "tool": "music_pause", "arguments": {}}
        spoken = format_spoken_response("pause", route, "Paused.", lang="en")
        self.assertEqual(spoken, "Paused, sir.")


class NonMusicToolBehaviorUnaffectedTests(unittest.TestCase):
    """The existing, deliberate generic-fallback behavior for other tools
    must be completely unchanged (regression)."""

    def test_generic_open_website_failure_still_gets_generic_fallback(self):
        route = {"type": "tool", "tool": "open_website", "arguments": {"url": "https://chrome.example"}}
        response_text = "Failed to open Google Chrome.\nError: launch failed"
        spoken = format_spoken_response("can you open youtube", route, response_text, lang="en")
        self.assertEqual(spoken, "I couldn't complete that action, sir.")

    def test_open_website_success_still_uses_its_own_dedicated_wording(self):
        route = {"type": "tool", "tool": "open_website", "arguments": {"url": "https://www.youtube.com"}}
        spoken = format_spoken_response("open YouTube", route, "Opened YouTube.", lang="en")
        self.assertEqual(spoken, "I opened YouTube, sir.")


if __name__ == "__main__":
    unittest.main()
