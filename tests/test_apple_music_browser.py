"""Deterministic (mocked) coverage of AppleMusicWebController. Since the
controller now delegates ALL browser lifecycle/tab-reuse to the shared
`AuthenticatedBrowserSession` (tools/browser_authenticated.py -- see its
own test file for that layer's coverage), these tests use a lightweight
fake session and focus on: correct delegation, translating
`AuthenticatedBrowserUnavailable` into this module's `AppleMusicUnavailable`,
sign-in detection, and now-playing/DOM parsing helpers."""
import unittest
from unittest.mock import Mock, patch

from tools.browser_authenticated import AuthenticatedBrowserUnavailable
from tools.music import apple_music_browser as amb
from tools.music.apple_music_browser import (
    APPLE_MUSIC_HOSTNAME,
    APPLE_MUSIC_URL,
    AppleMusicUnavailable,
    AppleMusicWebController,
    _classify_result_href,
    _parse_title_now_playing,
)


class FakeSession:
    """Minimal stand-in for AuthenticatedBrowserSession exposing exactly
    the surface AppleMusicWebController calls."""

    def __init__(self):
        self.connected = True
        self.page_to_return = Mock()
        self.ensure_page_calls: list[tuple[str, str]] = []
        self.closed = False
        self.cookie_report = {"cookie_counts": {"apple.com": 2}, "total_cookies": 2}
        self.raise_unavailable = False

    def is_connected(self):
        return self.connected

    def ensure_page(self, hostname_substring, url_if_missing, focus=True):
        if self.raise_unavailable:
            raise AuthenticatedBrowserUnavailable("Authenticated Chrome is not running. Start the JARVIS browser session first.")
        self.ensure_page_calls.append((hostname_substring, url_if_missing))
        return self.page_to_return

    def close(self):
        self.closed = True

    def cookie_counts(self, urls=None):
        if self.raise_unavailable:
            raise AuthenticatedBrowserUnavailable("Authenticated Chrome is not running. Start the JARVIS browser session first.")
        self.last_cookie_urls = urls
        return dict(self.cookie_report)


class DelegationTests(unittest.TestCase):
    """The controller owns no browser lifecycle of its own -- it must
    delegate connection/tab-reuse entirely to the shared session."""

    def test_no_dedicated_apple_music_profile_directory(self):
        controller = AppleMusicWebController(session=FakeSession())
        self.assertFalse(hasattr(controller, "profile_dir"))
        self.assertFalse(hasattr(controller, "headless"))
        self.assertFalse(hasattr(controller, "_launch"))

    def test_ensure_music_tab_delegates_to_shared_session_with_correct_target(self):
        session = FakeSession()
        controller = AppleMusicWebController(session=session)
        page = controller.ensure_music_tab()
        self.assertIs(page, session.page_to_return)
        self.assertEqual(session.ensure_page_calls, [(APPLE_MUSIC_HOSTNAME, APPLE_MUSIC_URL)])
        self.assertIs(controller.page, session.page_to_return)

    def test_is_session_live_delegates_to_shared_session(self):
        session = FakeSession()
        controller = AppleMusicWebController(session=session)
        session.connected = True
        self.assertTrue(controller.is_session_live())
        session.connected = False
        self.assertFalse(controller.is_session_live())

    def test_close_delegates_without_closing_the_users_chrome(self):
        session = FakeSession()
        controller = AppleMusicWebController(session=session)
        controller.close()
        self.assertTrue(session.closed)

    def test_unavailable_session_is_translated_to_apple_music_unavailable(self):
        session = FakeSession()
        session.raise_unavailable = True
        controller = AppleMusicWebController(session=session)
        with self.assertRaisesRegex(AppleMusicUnavailable, "Authenticated Chrome is not running"):
            controller.ensure_music_tab()

    def test_diagnose_auth_state_scopes_to_apple_domains_only(self):
        session = FakeSession()
        controller = AppleMusicWebController(session=session)
        state = controller.diagnose_auth_state()
        self.assertTrue(state["live"])
        self.assertEqual(state["cookie_counts"], {"apple.com": 2})
        self.assertTrue(all("apple" in url for url in session.last_cookie_urls))

    def test_diagnose_auth_state_reports_not_live_when_session_unavailable(self):
        session = FakeSession()
        session.raise_unavailable = True
        controller = AppleMusicWebController(session=session)
        state = controller.diagnose_auth_state()
        self.assertFalse(state["live"])
        self.assertEqual(state["cookie_counts"], {})


class SignInDetectionTests(unittest.TestCase):
    def test_signed_in_when_no_sign_in_control_visible(self):
        controller = AppleMusicWebController(session=FakeSession())
        page = Mock()
        page.get_by_role.return_value.first.is_visible.return_value = False
        self.assertTrue(controller.is_signed_in(page))

    def test_not_signed_in_when_sign_in_button_visible(self):
        controller = AppleMusicWebController(session=FakeSession())
        page = Mock()
        page.get_by_role.return_value.first.is_visible.return_value = True
        self.assertFalse(controller.is_signed_in(page))

    def test_no_page_means_not_signed_in(self):
        controller = AppleMusicWebController(session=FakeSession())
        self.assertFalse(controller.is_signed_in(None))


class CurrentTrackInfoTests(unittest.TestCase):
    def test_parses_song_and_artist_from_marquee_metadata(self):
        # Primary source, confirmed live against the real site -- the
        # player bar's [data-testid="marquee-text-item"] (song) /
        # "marquee-text-item-button" (artist). Real titles routinely carry
        # a "(feat. X)" suffix the user's request never mentions.
        controller = AppleMusicWebController(session=FakeSession())
        page = Mock()
        page.get_by_role.return_value.first.is_visible.return_value = True  # pause visible => playing
        song_el = Mock(); song_el.is_visible.return_value = True; song_el.inner_text.return_value = "Starboy (feat. Daft Punk)"
        artist_el = Mock(); artist_el.is_visible.return_value = True; artist_el.inner_text.return_value = "The Weeknd"

        def locator_side_effect(selector):
            if selector == '[data-testid="marquee-text-item"]':
                m = Mock(); m.first = song_el; return m
            if selector == '[data-testid="marquee-text-item-button"]':
                m = Mock(); m.first = artist_el; return m
            m = Mock(); m.first = Mock(is_visible=Mock(return_value=False)); return m

        page.locator.side_effect = locator_side_effect
        info = controller.current_track_info(page)
        self.assertTrue(info["observed"])
        self.assertEqual(info["song"], "Starboy (feat. Daft Punk)")
        self.assertEqual(info["artist"], "The Weeknd")
        self.assertTrue(info["is_playing"])

    def test_falls_back_to_document_title_when_marquee_unavailable(self):
        controller = AppleMusicWebController(session=FakeSession())
        page = Mock()
        page.get_by_role.return_value.first.is_visible.return_value = True  # pause visible => playing
        page.locator.return_value.first.is_visible.return_value = False  # no marquee element found
        page.title.return_value = "Starboy - The Weeknd"
        info = controller.current_track_info(page)
        self.assertTrue(info["observed"])
        self.assertEqual(info["song"], "Starboy")
        self.assertEqual(info["artist"], "The Weeknd")
        self.assertTrue(info["is_playing"])

    def test_generic_shell_title_is_never_parsed_as_a_song(self):
        controller = AppleMusicWebController(session=FakeSession())
        page = Mock()
        page.get_by_role.return_value.first.is_visible.return_value = False
        page.title.return_value = "Apple Music"
        page.locator.return_value.first.is_visible.return_value = False
        info = controller.current_track_info(page)
        self.assertFalse(info["observed"])
        self.assertIsNone(info["song"])

    def test_no_page_is_never_observed(self):
        controller = AppleMusicWebController(session=FakeSession())
        info = controller.current_track_info(None)
        self.assertFalse(info["observed"])


class ParseTitleNowPlayingTests(unittest.TestCase):
    def test_dash_separator(self):
        self.assertEqual(_parse_title_now_playing("Starboy - The Weeknd"), ("Starboy", "The Weeknd"))

    def test_en_dash_separator(self):
        self.assertEqual(_parse_title_now_playing("Starboy – The Weeknd"), ("Starboy", "The Weeknd"))

    def test_by_separator(self):
        self.assertEqual(_parse_title_now_playing("Starboy by The Weeknd"), ("Starboy", "The Weeknd"))

    def test_generic_shell_titles_return_nothing(self):
        for title in ("Apple Music", "Music", "Apple Music - Search", "Apple Music - Home"):
            with self.subTest(title=title):
                self.assertEqual(_parse_title_now_playing(title), (None, None))

    def test_real_idle_shell_title_with_invisible_chars_and_multiword_suffix(self):
        # Live-confirmed real title when nothing is playing: a leading
        # U+200E LEFT-TO-RIGHT MARK (invisible, NOT removed by str.strip()),
        # a U+00A0 NO-BREAK SPACE between "Apple" and "Music", and a
        # MULTI-WORD "Web Player" suffix an earlier `\w+`-only (single
        # word) shell-title pattern did not catch -- this used to be
        # mis-parsed as song="‎Apple Music", artist="Web Player".
        real_title = "‎Apple Music - Web Player"
        self.assertEqual(_parse_title_now_playing(real_title), (None, None))

    def test_invisible_chars_and_nbsp_do_not_break_a_real_song_title(self):
        real_title = "‎Starboy (feat. Daft Punk) - Song by The Weeknd - Apple Music"
        song, artist = _parse_title_now_playing(real_title)
        self.assertIsNotNone(song)
        self.assertNotIn("‎", song)

    def test_empty_title(self):
        self.assertEqual(_parse_title_now_playing(""), (None, None))


class ClassifyResultHrefTests(unittest.TestCase):
    def test_song_href(self):
        self.assertEqual(_classify_result_href("/us/song/starboy/123?i=456"), "song")

    def test_album_href(self):
        self.assertEqual(_classify_result_href("/us/album/after-hours/789"), "album")

    def test_artist_href(self):
        self.assertEqual(_classify_result_href("/us/artist/the-weeknd/111"), "artist")

    def test_playlist_href(self):
        self.assertEqual(_classify_result_href("/us/playlist/gym/pl.abc"), "playlist")

    def test_unrecognized_href_is_none(self):
        self.assertIsNone(_classify_result_href("/us/browse/genre/pop"))


class DiagnoseCliTests(unittest.TestCase):
    """python -m tools.music.apple_music_browser --diagnose (Part 10 of the
    live-path debug request)."""

    def tearDown(self):
        amb.reset_apple_music_controller_for_tests(None)

    def test_reports_not_reachable_honestly_without_raising(self):
        with patch("tools.browser_authenticated.is_cdp_available", return_value=False):
            report = amb.diagnose()
        self.assertFalse(report["cdp_reachable"])
        self.assertFalse(report["controller_ready"])

    def test_reports_tab_found_and_signed_in_when_reachable(self):
        session = FakeSession()
        page = Mock()
        page.url = "https://music.apple.com/listen-now"
        page.title.return_value = "Apple Music"
        page.get_by_role.return_value.first.is_visible.return_value = False  # not showing "Sign in"
        session.page_to_return = page
        fake_browser = Mock()
        fake_browser.contexts = [Mock()]
        session.ensure_connected = Mock(return_value=fake_browser)
        session.list_pages = Mock(return_value=[page])
        controller = AppleMusicWebController(session=session)
        amb.reset_apple_music_controller_for_tests(controller)
        with patch("tools.browser_authenticated.is_cdp_available", return_value=True):
            report = amb.diagnose()
        self.assertTrue(report["cdp_reachable"])
        self.assertEqual(report["contexts"], 1)
        self.assertEqual(report["pages"], 1)
        self.assertTrue(report["apple_music_tab_found"])
        self.assertTrue(report["controller_ready"])
        self.assertTrue(report["signed_in"])


class LiveConfirmedUrlTests(unittest.TestCase):
    """Confirmed live against the real site: these exact URLs are the ones
    that actually render the content (earlier guesses silently returned
    nothing)."""

    def test_list_library_playlists_uses_all_playlists_url(self):
        session = FakeSession()
        controller = AppleMusicWebController(session=session)
        page = session.page_to_return
        page.locator.return_value.all.return_value = []
        controller.list_library_playlists()
        goto_url = page.goto.call_args.args[0]
        self.assertIn("/library/all-playlists", goto_url)
        self.assertNotIn("/library/playlists", goto_url.replace("/library/all-playlists", ""))

    def test_get_recently_played_uses_listen_now_url(self):
        session = FakeSession()
        controller = AppleMusicWebController(session=session)
        page = session.page_to_return
        page.get_by_text.return_value.first.wait_for.side_effect = Exception("not found")
        controller.get_recently_played()
        goto_url = page.goto.call_args.args[0]
        self.assertIn("/listen-now", goto_url)


class PlaybackInteractionTests(unittest.TestCase):
    """Confirmed live: a track row's own "Play <title> by <artist>" button
    reliably starts that exact track; the page-level hero PLAY button can
    render `disabled` (observed live on a playlist page right after
    navigation) even while fully visible."""

    def test_play_from_current_page_prefers_row_button_over_hero(self):
        session = FakeSession()
        controller = AppleMusicWebController(session=session)
        page = session.page_to_return
        controller.ensure_music_tab()
        row_button = Mock()
        row_button.is_visible.return_value = True
        row_button.is_enabled.return_value = True

        def get_by_role(role, name=None):
            m = Mock()
            if name is not None and "Play" in getattr(name, "pattern", ""):
                m.first = row_button
            else:
                m.first = Mock(is_visible=Mock(return_value=False))
            return m

        page.get_by_role.side_effect = get_by_role
        result = controller.play_from_current_page()
        self.assertTrue(result)
        row_button.click.assert_called_once()

    def test_play_from_current_page_falls_back_to_hero_when_row_disabled(self):
        session = FakeSession()
        controller = AppleMusicWebController(session=session)
        page = session.page_to_return
        controller.ensure_music_tab()
        row_button = Mock()
        row_button.is_visible.return_value = True
        row_button.is_enabled.return_value = False  # confirmed live: hero can be enabled while row varies too
        hero_button = Mock()
        hero_button.is_visible.return_value = True

        def get_by_role(role, name=None):
            pattern = getattr(name, "pattern", "")
            m = Mock()
            if "Play" in pattern and "by" in pattern:
                m.first = row_button
            elif pattern == r"^\s*play\s*$":
                m.first = hero_button
            else:
                m.first = Mock(is_visible=Mock(return_value=False))
            return m

        page.get_by_role.side_effect = get_by_role
        result = controller.play_from_current_page()
        self.assertTrue(result)
        row_button.click.assert_not_called()
        hero_button.click.assert_called_once()

    def test_play_specific_track_targets_exact_title_and_artist(self):
        session = FakeSession()
        controller = AppleMusicWebController(session=session)
        page = session.page_to_return
        controller.ensure_music_tab()
        target_button = Mock()
        target_button.is_visible.return_value = True

        def get_by_role(role, name=None):
            pattern = getattr(name, "pattern", "")
            m = Mock()
            if pattern.startswith(r"^Play\s+Starboy") and "by" in pattern:
                m.first = target_button
            else:
                m.first = Mock(is_visible=Mock(return_value=False))
            return m

        page.get_by_role.side_effect = get_by_role
        result = controller.play_specific_track("Starboy", "The Weeknd")
        self.assertTrue(result)
        target_button.click.assert_called_once()

    def test_row_click_that_only_loads_falls_back_to_hero_play_click(self):
        # Confirmed live on a real playlist: a row-button click sometimes
        # only SELECTS/LOADS the track (player-bar metadata updates)
        # without starting playback, leaving the hero PLAY button enabled
        # but unclicked. The controller must notice (via wait_for_playing
        # never seeing Pause appear) and click the hero button too.
        session = FakeSession()
        controller = AppleMusicWebController(session=session)
        page = session.page_to_return
        controller.ensure_music_tab()
        row_button = Mock()
        row_button.is_visible.return_value = True
        hero_play_button = Mock()
        hero_play_button.is_visible.return_value = True

        def get_by_role(role, name=None):
            pattern = getattr(name, "pattern", "")
            m = Mock()
            if pattern.startswith(r"^Play\s+Starboy") and "by" in pattern:
                m.first = row_button
            elif pattern == r"^\s*pause\s*$":
                m.first = Mock(is_visible=Mock(return_value=False))  # never starts playing
            elif pattern == r"^\s*play\s*$":
                m.first = hero_play_button
            else:
                m.first = Mock(is_visible=Mock(return_value=False))
            return m

        page.get_by_role.side_effect = get_by_role
        result = controller.play_specific_track("Starboy", "The Weeknd")
        self.assertTrue(result)
        row_button.click.assert_called_once()
        hero_play_button.click.assert_called_once()

    def test_row_click_that_starts_playing_immediately_skips_hero_followup(self):
        session = FakeSession()
        controller = AppleMusicWebController(session=session)
        page = session.page_to_return
        controller.ensure_music_tab()
        row_button = Mock()
        row_button.is_visible.return_value = True
        hero_play_button = Mock()

        def get_by_role(role, name=None):
            pattern = getattr(name, "pattern", "")
            m = Mock()
            if pattern.startswith(r"^Play\s+Starboy") and "by" in pattern:
                m.first = row_button
            elif pattern == r"^\s*pause\s*$":
                m.first = Mock(is_visible=Mock(return_value=True))  # already playing
            elif pattern == r"^\s*play\s*$":
                m.first = hero_play_button
            else:
                m.first = Mock(is_visible=Mock(return_value=False))
            return m

        page.get_by_role.side_effect = get_by_role
        result = controller.play_specific_track("Starboy", "The Weeknd")
        self.assertTrue(result)
        hero_play_button.click.assert_not_called()


if __name__ == "__main__":
    unittest.main()
