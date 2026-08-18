import unittest

from brain.models import Action
from brain.music_intent import MusicIntentType, classify_music_intent, route_music_command


class ClassifyMusicIntentTests(unittest.TestCase):
    def test_open_music(self):
        for phrase in ("open music", "open apple music", "launch music", "launch apple music", "open my music"):
            with self.subTest(phrase=phrase):
                intent = classify_music_intent(phrase)
                self.assertIsNotNone(intent)
                self.assertEqual(intent.intent, MusicIntentType.OPEN_MUSIC)

    def test_play_generic(self):
        for phrase in ("play music", "play some music", "play some songs"):
            with self.subTest(phrase=phrase):
                intent = classify_music_intent(phrase)
                self.assertEqual(intent.intent, MusicIntentType.PLAY_GENERIC)

    def test_play_song_with_artist(self):
        intent = classify_music_intent("play Starboy by The Weeknd")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_SONG)
        self.assertEqual(intent.song, "Starboy")
        self.assertEqual(intent.artist, "The Weeknd")

    def test_play_artist_only(self):
        intent = classify_music_intent("play The Weeknd")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_QUERY)
        self.assertEqual(intent.song, "The Weeknd")

    def test_play_bare_song(self):
        intent = classify_music_intent("play Starboy")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_QUERY)
        self.assertEqual(intent.song, "Starboy")

    def test_play_album_by_artist(self):
        intent = classify_music_intent("play After Hours by The Weeknd")
        # "After Hours by The Weeknd" has no "album" keyword, so this is
        # deliberately resolved as PLAY_SONG (search-result scoring decides
        # song vs. album at execution time -- see tools/music/apple_music_provider.py).
        self.assertEqual(intent.intent, MusicIntentType.PLAY_SONG)
        self.assertEqual(intent.song, "After Hours")
        self.assertEqual(intent.artist, "The Weeknd")

    def test_play_explicit_album(self):
        intent = classify_music_intent("play the album After Hours by The Weeknd")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_ALBUM)
        self.assertEqual(intent.album, "After Hours")
        self.assertEqual(intent.artist, "The Weeknd")

    def test_play_gym_playlist(self):
        intent = classify_music_intent("play my gym playlist")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_PLAYLIST)
        self.assertEqual(intent.playlist, "gym")

    def test_play_one_of_my_playlists(self):
        intent = classify_music_intent("play one of my playlists")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_PLAYLIST)
        self.assertEqual(intent.scope, "random_user_playlist")

    def test_last_played(self):
        for phrase in (
            "play the last song I listened to",
            "play my last played song",
            "play the song I listened to before",
            "play what I was listening to earlier",
            # Confirmed live to be misclassified as a literal catalog
            # search (PLAY_QUERY) before this regex was widened:
            "play my last song",
            "play the last thing I listened to",
            "play what I listened to last",
        ):
            with self.subTest(phrase=phrase):
                intent = classify_music_intent(phrase)
                self.assertEqual(intent.intent, MusicIntentType.PLAY_LAST_PLAYED)
                # Never a literal-phrase catalog search under any name.
                self.assertIsNone(intent.song)

    def test_recently_played(self):
        intent = classify_music_intent("play my recently played music")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_RECENT)

    def test_resume_last_session(self):
        for phrase in ("continue my music", "resume what I was listening to", "continue what I was listening to"):
            with self.subTest(phrase=phrase):
                intent = classify_music_intent(phrase)
                self.assertEqual(intent.intent, MusicIntentType.RESUME_LAST_SESSION)

    def test_favorites(self):
        for phrase in ("play my favorite playlist", "play my favorites", "play my favorite songs"):
            with self.subTest(phrase=phrase):
                intent = classify_music_intent(phrase)
                self.assertEqual(intent.intent, MusicIntentType.PLAY_FAVORITES)

    def test_library(self):
        intent = classify_music_intent("play my music")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_LIBRARY)

    def test_mood(self):
        intent = classify_music_intent("play something relaxing")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_MOOD)
        self.assertEqual(intent.mood, "relaxing")
        intent2 = classify_music_intent("play some workout music")
        self.assertEqual(intent2.intent, MusicIntentType.PLAY_MOOD)

    def test_pause_resume_stop(self):
        self.assertEqual(classify_music_intent("pause").intent, MusicIntentType.PAUSE)
        self.assertEqual(classify_music_intent("resume").intent, MusicIntentType.RESUME)
        self.assertEqual(classify_music_intent("stop the music").intent, MusicIntentType.STOP)

    def test_bare_stop_is_not_music(self):
        # Bare "stop" is claimed by the existing task-cancellation route in
        # brain/router.py, checked before the music block -- this module
        # must not try to steal it.
        self.assertIsNone(classify_music_intent("stop"))

    def test_next_previous_restart(self):
        self.assertEqual(classify_music_intent("next").intent, MusicIntentType.NEXT)
        self.assertEqual(classify_music_intent("skip this song").intent, MusicIntentType.NEXT)
        self.assertEqual(classify_music_intent("previous song").intent, MusicIntentType.PREVIOUS)
        self.assertEqual(classify_music_intent("restart this song").intent, MusicIntentType.RESTART_TRACK)

    def test_shuffle_repeat(self):
        self.assertEqual(classify_music_intent("shuffle my playlist").intent, MusicIntentType.SHUFFLE_ON)
        self.assertEqual(classify_music_intent("turn shuffle on").intent, MusicIntentType.SHUFFLE_ON)
        self.assertEqual(classify_music_intent("turn shuffle off").intent, MusicIntentType.SHUFFLE_OFF)
        self.assertEqual(classify_music_intent("repeat this song").intent, MusicIntentType.REPEAT_TRACK)
        self.assertEqual(classify_music_intent("turn repeat off").intent, MusicIntentType.REPEAT_OFF)

    def test_shuffle_named_playlist(self):
        intent = classify_music_intent("shuffle my gym playlist")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_PLAYLIST)
        self.assertEqual(intent.playlist, "gym")

    def test_queue_and_play_next(self):
        self.assertEqual(classify_music_intent("queue this song").intent, MusicIntentType.QUEUE_TRACK)
        play_next = classify_music_intent("play this next")
        self.assertEqual(play_next.intent, MusicIntentType.PLAY_NEXT)
        self.assertTrue(play_next.contextual)

    def test_add_to_library_and_favorites(self):
        self.assertEqual(classify_music_intent("add this song to my library").intent, MusicIntentType.ADD_TO_LIBRARY)
        self.assertEqual(classify_music_intent("add this song to my favorites").intent, MusicIntentType.ADD_TO_FAVORITES)

    def test_now_playing_song_and_artist(self):
        song_q = classify_music_intent("what song is playing?")
        self.assertEqual(song_q.intent, MusicIntentType.NOW_PLAYING)
        self.assertEqual(song_q.aspect, "song")
        artist_q = classify_music_intent("who is this artist?")
        self.assertEqual(artist_q.intent, MusicIntentType.NOW_PLAYING)
        self.assertEqual(artist_q.aspect, "artist")

    def test_more_by_artist(self):
        intent = classify_music_intent("play more songs by this artist")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_MORE_BY_ARTIST)

    def test_explicit_provider_spotify(self):
        intent = classify_music_intent("play Starboy on Spotify")
        self.assertEqual(intent.provider, "spotify")
        self.assertEqual(intent.song, "Starboy")

    def test_explicit_provider_youtube(self):
        intent = classify_music_intent("play it on YouTube")
        self.assertEqual(intent.provider, "youtube")

    def test_unrelated_command_is_not_music(self):
        self.assertIsNone(classify_music_intent("open notepad"))
        self.assertIsNone(classify_music_intent("what time is it"))
        self.assertIsNone(classify_music_intent("send a whatsapp message to mom"))

    def test_polite_wrapper_stripped(self):
        intent = classify_music_intent("Could you please play Starboy by The Weeknd")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_SONG)
        self.assertEqual(intent.song, "Starboy")


class RouteMusicCommandTests(unittest.TestCase):
    def test_open_music_route(self):
        route = route_music_command("open music")
        self.assertEqual(route, {"type": "tool", "tool": "open_music", "arguments": {}})

    def test_simple_control_routes(self):
        self.assertEqual(route_music_command("pause")["tool"], "music_pause")
        self.assertEqual(route_music_command("next")["tool"], "music_next")
        self.assertEqual(route_music_command("shuffle my playlist")["tool"], "music_shuffle_on")

    def test_play_song_route_shape(self):
        route = route_music_command("play Starboy by The Weeknd")
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route["tool"], "music_play")
        self.assertEqual(route["arguments"]["intent"], "PLAY_SONG")
        self.assertEqual(route["arguments"]["song"], "Starboy")
        self.assertEqual(route["arguments"]["artist"], "The Weeknd")

    def test_now_playing_route(self):
        route = route_music_command("what song is playing")
        self.assertEqual(route["tool"], "music_now_playing")
        self.assertEqual(route["arguments"]["aspect"], "song")

    def test_apple_music_provider_stays_default(self):
        route = route_music_command("play Starboy")
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route["tool"], "music_play")

    def test_spotify_provider_uses_local_plan_not_apple_music(self):
        route = route_music_command("play Starboy on Spotify")
        self.assertEqual(route["type"], "local_plan")
        actions = route["actions"]
        self.assertEqual([a.tool for a in actions], ["browser_open_url", "browser_click_first_result"])
        self.assertIsInstance(actions[0], Action)
        self.assertIn("open.spotify.com", actions[0].args["url"])

    def test_youtube_provider_uses_local_plan_not_apple_music(self):
        route = route_music_command("play Blinding Lights on YouTube")
        self.assertEqual(route["type"], "local_plan")
        self.assertIn("youtube.com", route["actions"][0].args["url"])

    def test_non_music_returns_none(self):
        self.assertIsNone(route_music_command("open notepad"))


class HebrewMusicIntentTests(unittest.TestCase):
    """VOICE_LANGUAGE=he: Hebrew commands must classify correctly, and
    every extracted entity must be the EXACT original Hebrew text --
    never translated, never transliterated, never stripped."""

    def test_open_music(self):
        intent = classify_music_intent("פתח מוזיקה")
        self.assertEqual(intent.intent, MusicIntentType.OPEN_MUSIC)

    def test_play_generic(self):
        intent = classify_music_intent("נגן מוזיקה")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_GENERIC)

    def test_play_song_bare(self):
        intent = classify_music_intent("נגן שני משוגעים")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_QUERY)
        self.assertEqual(intent.song, "שני משוגעים")

    def test_play_song_with_put_on_verb(self):
        intent = classify_music_intent("שים שני משוגעים")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_QUERY)
        self.assertEqual(intent.song, "שני משוגעים")

    def test_play_the_song_x_strips_the_song_wrapper_exactly(self):
        intent = classify_music_intent("נגן את השיר שני משוגעים")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_QUERY)
        self.assertEqual(intent.song, "שני משוגעים")
        self.assertNotIn("השיר", intent.song)

    def test_play_artist_name(self):
        intent = classify_music_intent("נגן עומר אדם")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_QUERY)
        self.assertEqual(intent.song, "עומר אדם")

    def test_play_playlist(self):
        intent = classify_music_intent("נגן את הפלייליסט ישראלי")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_PLAYLIST)
        self.assertEqual(intent.playlist, "ישראלי")

    def test_play_last_played(self):
        intent = classify_music_intent("נגן את השיר האחרון ששמעתי")
        self.assertEqual(intent.intent, MusicIntentType.PLAY_LAST_PLAYED)

    def test_resume_last_session(self):
        intent = classify_music_intent("תמשיך את המוזיקה")
        self.assertEqual(intent.intent, MusicIntentType.RESUME_LAST_SESSION)

    def test_bare_resume_is_simple_resume_not_last_session(self):
        intent = classify_music_intent("תמשיך")
        self.assertEqual(intent.intent, MusicIntentType.RESUME)

    def test_stop(self):
        intent = classify_music_intent("תעצור")
        self.assertEqual(intent.intent, MusicIntentType.STOP)

    def test_next(self):
        intent = classify_music_intent("שיר הבא")
        self.assertEqual(intent.intent, MusicIntentType.NEXT)

    def test_previous(self):
        intent = classify_music_intent("שיר קודם")
        self.assertEqual(intent.intent, MusicIntentType.PREVIOUS)

    def test_now_playing_song(self):
        intent = classify_music_intent("מה מתנגן?")
        self.assertEqual(intent.intent, MusicIntentType.NOW_PLAYING)
        self.assertEqual(intent.aspect, "song")

    def test_now_playing_artist(self):
        intent = classify_music_intent("מי שר את זה?")
        self.assertEqual(intent.intent, MusicIntentType.NOW_PLAYING)
        self.assertEqual(intent.aspect, "artist")

    def test_entities_are_never_ascii_transliterated(self):
        # A translated/transliterated entity would come back as Latin
        # script; the real Hebrew Unicode must survive untouched.
        intent = classify_music_intent("נגן שני משוגעים")
        self.assertTrue(any("֐" <= ch <= "׿" for ch in intent.song))

    def test_route_music_command_produces_music_play_tool_for_hebrew(self):
        route = route_music_command("נגן שני משוגעים")
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route["tool"], "music_play")
        self.assertEqual(route["arguments"]["song"], "שני משוגעים")

    def test_route_music_command_playlist_scope_for_hebrew(self):
        route = route_music_command("נגן את הפלייליסט ישראלי")
        self.assertEqual(route["arguments"]["intent"], "PLAY_PLAYLIST")
        self.assertEqual(route["arguments"]["playlist"], "ישראלי")


if __name__ == "__main__":
    unittest.main()
