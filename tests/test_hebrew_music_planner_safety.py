# -*- coding: utf-8 -*-
"""Regression coverage for two confirmed live bugs in the bilingual voice
work:

1. `brain/agent.py::_run_agent_impl` decided whether to invoke the task
   planner from `should_use_task_planner(command)` alone, completely
   independent of whether `route_command` had ALREADY resolved a
   deterministic music tool route. A live Hebrew music command reached the
   generic (cloud) planner this way and ended up opening
   https://music.youtube.com/ instead of using the already-correct Apple
   Music route. `_is_deterministic_music_route` now guards against this.
2. Hebrew "open <website>" phrasing ("פתח יוטיוב") never matched the
   website-alias branch (English-only "open "/"go to " prefix check), so
   it fell through to `open_application` trying to launch a nonexistent
   desktop app named after the Hebrew site word. `route_command` now
   recognizes a small set of Hebrew website names before that fallback.
"""
import unittest
from unittest.mock import patch

from brain.agent import _is_deterministic_music_route
from brain.router import route_command


class DeterministicMusicRouteGuardTests(unittest.TestCase):
    def test_recognizes_open_music_and_music_star_tools_as_deterministic(self):
        self.assertTrue(_is_deterministic_music_route({"type": "tool", "tool": "open_music", "arguments": {}}))
        self.assertTrue(_is_deterministic_music_route({"type": "tool", "tool": "music_play", "arguments": {}}))
        self.assertTrue(_is_deterministic_music_route({"type": "tool", "tool": "music_now_playing", "arguments": {}}))

    def test_rejects_non_music_tool_routes_and_other_route_types(self):
        self.assertFalse(_is_deterministic_music_route({"type": "tool", "tool": "open_website", "arguments": {}}))
        self.assertFalse(_is_deterministic_music_route({"type": "plan", "message": "x"}))
        self.assertFalse(_is_deterministic_music_route({"type": "local_plan", "actions": []}))
        self.assertFalse(_is_deterministic_music_route(None))
        self.assertFalse(_is_deterministic_music_route({}))


class MusicRouteSurvivesPlannerOverrideTests(unittest.TestCase):
    """Proves the exact live bug is fixed: even when
    `should_use_task_planner` would say True for a command's text, an
    already-resolved music route must still execute directly instead of
    being handed to the (cloud) planner."""

    def test_hebrew_music_route_bypasses_task_planner_even_if_signaled(self):
        from brain import agent as agent_module
        command = "נגן שני משוגעים"
        route = route_command(command)
        self.assertEqual(route["type"], "tool")
        self.assertTrue(route["tool"].startswith("music_"))
        with patch("brain.agent.should_use_task_planner", return_value=True), \
             patch("brain.agent.create_task_plan") as fake_task_plan, \
             patch("brain.agent.create_plan") as fake_cloud_plan:
            execution_meta = {}
            recorder = type("R", (), {"record": lambda *a, **k: None})()
            # Never actually executes the tool (no browser/session available
            # in this unit test) -- the point is only to prove neither
            # planner is even consulted once a music route is present.
            with patch("brain.agent._execute_recorded_plan", return_value=[]), \
                 patch("brain.agent.execute_tool", side_effect=Exception("stop before real execution")):
                try:
                    agent_module._run_agent_impl(command, route, recorder, "iid", execution_meta)
                except Exception:
                    pass
            fake_task_plan.assert_not_called()
            fake_cloud_plan.assert_not_called()

    def test_english_music_route_also_bypasses_task_planner(self):
        from brain import agent as agent_module
        command = "play Starboy"
        route = route_command(command)
        self.assertEqual(route["type"], "tool")
        self.assertTrue(route["tool"].startswith("music_"))
        with patch("brain.agent.should_use_task_planner", return_value=True), \
             patch("brain.agent.create_task_plan") as fake_task_plan, \
             patch("brain.agent.create_plan") as fake_cloud_plan:
            execution_meta = {}
            recorder = type("R", (), {"record": lambda *a, **k: None})()
            with patch("brain.agent._execute_recorded_plan", return_value=[]), \
                 patch("brain.agent.execute_tool", side_effect=Exception("stop before real execution")):
                try:
                    agent_module._run_agent_impl(command, route, recorder, "iid", execution_meta)
                except Exception:
                    pass
            fake_task_plan.assert_not_called()
            fake_cloud_plan.assert_not_called()


class HebrewMusicRoutingCoverageTests(unittest.TestCase):
    """Every Hebrew music phrase from the live acceptance list must reach
    a deterministic Apple-Music-backed tool route, never the generic
    planner and never music.youtube.com."""

    CASES = {
        "פתח מוזיקה": "open_music",
        "נגן מוזיקה": "music_play",
        "נגן שני משוגעים": "music_play",
        "נגן את השיר שני משוגעים": "music_play",
        "נגן את השיר האחרון ששמעתי": "music_play",
        "נגן את הפלייליסט ישראלי": "music_play",
        "נגן את הפלייליסט שלי ישראלי": "music_play",
        "תמשיך את המוזיקה": "music_play",
        "תעצור": "music_stop",
        "שיר הבא": "music_next",
        "שיר קודם": "music_previous",
        "מה מתנגן?": "music_now_playing",
        "מי שר את זה?": "music_now_playing",
    }

    def test_each_phrase_routes_to_the_expected_deterministic_music_tool(self):
        for phrase, expected_tool in self.CASES.items():
            with self.subTest(phrase=phrase):
                route = route_command(phrase)
                self.assertEqual(route["type"], "tool")
                self.assertEqual(route["tool"], expected_tool)

    def test_last_played_intent_is_not_confused_with_a_literal_play_query(self):
        route = route_command("נגן את השיר האחרון ששמעתי")
        self.assertEqual(route["arguments"].get("intent"), "PLAY_LAST_PLAYED")
        self.assertIsNone(route["arguments"].get("song"))

    def test_playlist_entity_preserved_exactly_as_hebrew_unicode(self):
        route = route_command("נגן את הפלייליסט ישראלי")
        self.assertEqual(route["arguments"].get("playlist"), "ישראלי")

    def test_song_entity_preserved_exactly_as_hebrew_unicode(self):
        route = route_command("נגן שני משוגעים")
        self.assertEqual(route["arguments"].get("song"), "שני משוגעים")


class HebrewOpenWebsiteRoutingTests(unittest.TestCase):
    def test_open_youtube_in_hebrew_opens_the_website_not_an_application(self):
        route = route_command("פתח יוטיוב")
        self.assertEqual(route, {"type": "tool", "tool": "open_website", "arguments": {"url": "https://www.youtube.com"}})

    def test_open_music_in_hebrew_still_uses_the_music_router_not_the_website_fix(self):
        # "מוזיקה" is deliberately absent from the Hebrew website-name map
        # -- must keep going through brain/music_intent.py, never treated
        # as a generic website.
        route = route_command("פתח מוזיקה")
        self.assertEqual(route, {"type": "tool", "tool": "open_music", "arguments": {}})

    def test_open_unrecognized_hebrew_target_still_falls_back_to_open_application(self):
        route = route_command("פתח נוטפד")
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route["tool"], "open_application")

    def test_english_open_youtube_unaffected(self):
        route = route_command("open YouTube")
        self.assertEqual(route, {"type": "tool", "tool": "open_website", "arguments": {"url": "https://www.youtube.com"}})


class MusicIntentDiagnosticsCliTests(unittest.TestCase):
    def test_open_music_reports_apple_music_provider(self):
        from brain.music_intent import _diagnose
        result = _diagnose("פתח מוזיקה")
        self.assertEqual(result["provider"], "apple_music")
        self.assertEqual(result["detected_language"], "he")
        self.assertEqual(result["music_intent"], "OPEN_MUSIC")

    def test_play_last_played_is_not_reported_as_play_query(self):
        from brain.music_intent import _diagnose
        result = _diagnose("נגן את השיר האחרון ששמעתי")
        self.assertEqual(result["music_intent"], "PLAY_LAST_PLAYED")
        self.assertNotEqual(result["music_intent"], "PLAY_QUERY")
        self.assertIsNone(result["entities"]["song"])

    def test_english_input_detected_as_english(self):
        from brain.music_intent import _diagnose
        result = _diagnose("play Starboy")
        self.assertEqual(result["detected_language"], "en")


if __name__ == "__main__":
    unittest.main()
