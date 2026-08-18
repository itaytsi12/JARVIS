"""Music-specific speculative partial-action behavior (Part 7): a partial
transcript may safely pre-open Apple Music, but must never guess a search
result, and the final committed transcript must reconcile against whatever
already fired instead of double-executing or being blocked incorrectly."""
import unittest

from brain.music_intent import FAST_PATH_TOOLS
from brain.router import route_command
from brain.safe_tools import CONTEXT_INDEPENDENT_TOOLS
from brain.speculative_execution import (
    PartialActionLedger,
    SAFE_PARTIAL_TOOLS,
    classify_partial_route,
    reconcile_final_route,
)


class MusicFastPathSafetyClassificationTests(unittest.TestCase):
    def test_fast_path_tools_are_context_independent_and_speculative_safe(self):
        for tool in FAST_PATH_TOOLS:
            with self.subTest(tool=tool):
                self.assertIn(tool, CONTEXT_INDEPENDENT_TOOLS)
                self.assertIn(tool, SAFE_PARTIAL_TOOLS)

    def test_entity_resolving_music_tools_are_never_speculative_safe(self):
        unsafe_tools = {
            "music_play", "music_queue_add", "music_queue_next", "music_now_playing",
            "music_add_to_library", "music_add_to_favorites", "music_artist_more",
            "music_restart_track", "music_shuffle_on", "music_shuffle_off",
            "music_repeat_on", "music_repeat_off",
        }
        for tool in unsafe_tools:
            with self.subTest(tool=tool):
                self.assertNotIn(tool, SAFE_PARTIAL_TOOLS)


class PartialMusicTranscriptTests(unittest.TestCase):
    def test_open_music_partial_is_eligible_for_speculative_firing(self):
        route = classify_partial_route("open music")
        self.assertIsNotNone(route)
        self.assertEqual(route["tool"], "open_music")

    def test_pause_partial_is_eligible(self):
        route = classify_partial_route("pause")
        self.assertEqual(route["tool"], "music_pause")

    def test_partial_song_request_never_prematurely_selects_a_result(self):
        # "play Starboy..." must never be speculatively resolved into a
        # music_play (search+play) action -- only the committed transcript
        # can safely trigger a catalog search and playback.
        for partial in ("play starboy", "play starboy by", "play starboy by the"):
            with self.subTest(partial=partial):
                self.assertIsNone(classify_partial_route(partial))

    def test_two_stable_open_music_partials_fire_exactly_once(self):
        ledger = PartialActionLedger(min_stable=2)
        self.assertIsNone(ledger.observe_partial("open music"))
        action = ledger.observe_partial("open music")
        self.assertIsNotNone(action)
        self.assertEqual(action.route["tool"], "open_music")
        self.assertIsNone(ledger.observe_partial("open music"))


class FinalTranscriptReconciliationTests(unittest.TestCase):
    def test_final_open_music_after_speculative_open_is_not_re_executed(self):
        ledger = PartialActionLedger(min_stable=1)
        fired = ledger.observe_partial("open music")
        self.assertIsNotNone(fired)
        final_route = route_command("open music")
        route_to_execute, matched = reconcile_final_route(ledger, final_route)
        self.assertIsNone(route_to_execute)
        self.assertIsNotNone(matched)

    def test_final_play_song_after_speculative_open_still_executes(self):
        # "open music" fired early is a DIFFERENT action than the eventual
        # "play Starboy by The Weeknd" -- the final search+play must still
        # run; opening early is a latency optimization, not a substitute.
        ledger = PartialActionLedger(min_stable=1)
        ledger.observe_partial("open music")
        final_route = route_command("play Starboy by The Weeknd")
        route_to_execute, matched = reconcile_final_route(ledger, final_route)
        self.assertIsNotNone(route_to_execute)
        self.assertIsNone(matched)
        self.assertEqual(route_to_execute["tool"], "music_play")

    def test_provider_qualifier_change_is_respected_in_final_command(self):
        # A partial never speculatively opens Apple Music from an
        # ambiguous "play Starboy..." (see test above), so a final
        # "...on Spotify" is free to route to Spotify with nothing to
        # reconcile against.
        ledger = PartialActionLedger(min_stable=1)
        ledger.observe_partial("play starboy")
        self.assertFalse(ledger.has_fired_anything())
        final_route = route_command("play Starboy on Spotify")
        route_to_execute, matched = reconcile_final_route(ledger, final_route)
        self.assertIsNone(matched)
        self.assertEqual(route_to_execute["type"], "local_plan")
        self.assertIn("open.spotify.com", route_to_execute["actions"][0].args["url"])


if __name__ == "__main__":
    unittest.main()
