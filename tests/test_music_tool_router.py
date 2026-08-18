"""brain/tool_router.py dispatch wiring + brain/resource_locks.py resource
classification for the new music_* tools -- and confirmation that these
deterministic fast-path commands never touch the cloud/LLM path."""
import unittest
from unittest.mock import patch

from brain import tool_router
from brain.resource_locks import resource_for_tool


class MusicToolDispatchTests(unittest.TestCase):
    def test_no_arg_tools_dispatch_to_the_provider_module(self):
        with patch("brain.tool_router.music.music_pause", return_value={"success": True}) as fn:
            result = tool_router.execute_tool("music_pause", {})
        fn.assert_called_once_with()
        self.assertEqual(result, {"success": True})

    def test_open_music_dispatch(self):
        with patch("brain.tool_router.music.open_music", return_value={"success": True}) as fn:
            tool_router.execute_tool("open_music", {})
        fn.assert_called_once_with()

    def test_music_now_playing_passes_aspect_argument(self):
        with patch("brain.tool_router.music.music_now_playing", return_value={"success": True}) as fn:
            tool_router.execute_tool("music_now_playing", {"aspect": "artist"})
        fn.assert_called_once_with("artist")

    def test_music_now_playing_defaults_aspect_to_song(self):
        with patch("brain.tool_router.music.music_now_playing", return_value={"success": True}) as fn:
            tool_router.execute_tool("music_now_playing", {})
        fn.assert_called_once_with("song")

    def test_music_play_passes_all_arguments(self):
        with patch("brain.tool_router.music.music_play", return_value={"success": True}) as fn:
            tool_router.execute_tool("music_play", {
                "intent": "PLAY_SONG", "song": "Starboy", "artist": "The Weeknd",
                "album": None, "playlist": None, "mood": None, "scope": None,
                "contextual": False, "shuffle": False,
            })
        fn.assert_called_once_with(
            "PLAY_SONG", song="Starboy", artist="The Weeknd", album=None,
            playlist=None, mood=None, scope=None, contextual=False, shuffle=False,
        )

    def test_music_queue_next_passes_song_and_contextual(self):
        with patch("brain.tool_router.music.music_queue_next", return_value={"success": True}) as fn:
            tool_router.execute_tool("music_queue_next", {"song": "Starboy", "contextual": False})
        fn.assert_called_once_with("Starboy", False)


class MusicResourceLockTests(unittest.TestCase):
    def test_music_tools_share_a_dedicated_resource(self):
        tools = ["open_music", "music_pause", "music_resume", "music_stop", "music_next",
                 "music_previous", "music_play", "music_now_playing", "music_shuffle_on"]
        for tool in tools:
            with self.subTest(tool=tool):
                self.assertEqual(resource_for_tool(tool), "authenticated_browser")

    def test_music_resource_is_distinct_from_generic_browser_resource(self):
        self.assertEqual(resource_for_tool("open_website"), "browser_session")
        self.assertEqual(resource_for_tool("open_music"), "authenticated_browser")


class NoLLMForFastPathTests(unittest.TestCase):
    """Part 12/26: pause/next/previous/etc must never invoke the cloud
    intent router or planner -- OpenAI is never even imported by that
    path, let alone called."""

    def test_pause_route_never_touches_openai_client(self):
        from brain.router import route_command
        with patch("brain.agent.client") as fake_client:
            route = route_command("pause")
            self.assertEqual(route["type"], "tool")
            self.assertEqual(route["tool"], "music_pause")
            fake_client.responses.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
