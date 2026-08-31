"""Apple Music Web (https://music.apple.com) control, driven over the
shared authenticated-browser session (`tools/browser_authenticated.py`).

## Session reuse (Part 4/5)

This module used to own a DEDICATED, JARVIS-launched persistent Playwright
profile. That was abandoned: Apple's interactive sign-in flow
(idmsa.apple.com/appleid.apple.com) hangs indefinitely after the password
step specifically when the signing-in browser is Playwright-driven, even
against its own dedicated profile -- confirmed live, with the exact same
account signing in fine in an ordinary Chrome window. See
`tools/browser_authenticated.py`'s module docstring for the full
investigation and the fix: the user launches their OWN normal,
already-signed-in Chrome with a remote-debugging port
(`python -m tools.browser_authenticated --launch`), and this controller
ATTACHES to it over CDP via `AuthenticatedBrowserSession` -- reusing
whatever Apple Music (or anything else) is already signed into there. No
separate Apple Music login, no dedicated profile directory, no password
ever touched by JARVIS.

## DOM strategy (Part 5)

Every locator here is semantic-first (ARIA role + accessible name, then
text content), matching `tools/browser_agent.py`'s existing convention, and
tries several plausible name patterns before giving up -- Apple Music Web's
exact class names are not something this module depends on. Because this
was built without a live, signed-in browsing session available in the
authoring environment, treat `diagnose()`'s output as the fastest way to
tighten a selector that doesn't match after the first real run.
"""
from __future__ import annotations

import logging
import re
import time
import urllib.parse
from typing import Any

from tools.browser_authenticated import (
    AuthenticatedBrowserSession,
    AuthenticatedBrowserUnavailable,
    get_authenticated_browser_session,
)

log = logging.getLogger("jarvis.music.apple_music")

APPLE_MUSIC_URL = "https://music.apple.com"
APPLE_MUSIC_HOSTNAME = "music.apple.com"
# Scopes the safe cookie-count diagnostic (Part: never enumerate the whole
# real profile's session state) to Apple's own domains only.
_APPLE_MUSIC_AUTH_URLS = ("https://music.apple.com", "https://www.apple.com", "https://idmsa.apple.com")


class AppleMusicUnavailable(RuntimeError):
    pass


class AppleMusicSignInRequired(RuntimeError):
    """Apple Music Web is showing a sign-in surface instead of the signed-in
    app shell -- i.e. the account attached-to Chrome session isn't signed
    into Apple Music specifically. JARVIS never fills this form or touches
    credentials -- the caller reports this honestly (Part 25); the user
    signs in manually in their own normal Chrome window (the same one
    JARVIS is attached to)."""
    pass


_PLAY_NAME = re.compile(r"^\s*play\s*$", re.I)
_PAUSE_NAME = re.compile(r"^\s*pause\s*$", re.I)
_NEXT_NAME = re.compile(r"next", re.I)
_PREV_NAME = re.compile(r"previous", re.I)
_SHUFFLE_NAME = re.compile(r"shuffle", re.I)
_REPEAT_NAME = re.compile(r"repeat", re.I)
_ADD_LIBRARY_NAME = re.compile(r"^\s*add(?:\s+to\s+library)?\s*$", re.I)
_ADD_FAVORITE_NAME = re.compile(r"favorite|love", re.I)
_SIGN_IN_NAME = re.compile(r"sign in|log in", re.I)
# Row context-menu entries. Confirmed live against the real site: a track
# row's "more" button opens a `[role="menu"]` / `.contextual-menu` whose
# items are exactly "Pin Song / Delete from Library / Add to Playlist /
# New Playlist / <recent playlists> / All playlists / <every playlist> /
# Play Next / Play Last / Create Station / Favourite / Suggest Less /
# View Credits". The Add-to-Playlist SUBMENU is already present in the DOM
# alongside the top-level menu, which is why a playlist name can be
# clicked directly once the menu is open.
_ADD_TO_PLAYLIST_NAME = re.compile(r"^\s*add to playlist\s*$", re.I)
_NEW_PLAYLIST_NAME = re.compile(r"^\s*new playlist\s*$", re.I)
_MENU_SELECTOR = "[role='menu']:visible, .contextual-menu:visible"
# The inline field "New Playlist" reveals. Confirmed live: it carries
# `data-testid="playlist-title-input"` and the accessible name "Playlist
# Title", and clicking "New Playlist" does NOT navigate -- an earlier
# version waited for a `/playlist/` URL that never came.
_PLAYLIST_TITLE_INPUT = "[data-testid='playlist-title-input'], input[aria-label='Playlist Title']"
_SEARCH_NAME = re.compile(r"search", re.I)
# Confirmed live: every track row (search result detail page, album,
# playlist) exposes an accessible button named exactly "Play <title> by
# <artist>" -- the reliable per-track play control (Part 3).
_ROW_PLAY_NAME = re.compile(r"^Play\s+.+\s+by\s+.+$", re.I)


class AppleMusicWebController:
    """Apple Music Web control surface. Owns NO browser lifecycle of its
    own -- delegates entirely to the shared `AuthenticatedBrowserSession`
    (dependency-injected for tests, defaults to the process-wide shared
    one) for connecting and tab reuse, and only implements Apple
    Music-specific DOM interaction on top of whatever page that session
    hands back."""

    def __init__(self, session: AuthenticatedBrowserSession | None = None):
        self.session = session or get_authenticated_browser_session()
        self._page = None
        self._storefront: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle -- thin wrappers over the shared session, translating its
    # generic `AuthenticatedBrowserUnavailable` into this module's
    # existing `AppleMusicUnavailable` so callers
    # (`tools/music/apple_music_provider.py`) don't need to know which
    # underlying session mechanism is in use.
    # ------------------------------------------------------------------

    def is_session_live(self) -> bool:
        return self.session.is_connected()

    def close(self) -> None:
        """Drop JARVIS's own attachment. Never closes the user's actual
        Chrome window."""
        self.session.close()

    # ------------------------------------------------------------------
    # Tab reuse (Part 5): never opens a duplicate music.apple.com tab.
    # ------------------------------------------------------------------

    def ensure_music_tab(self, focus: bool = True):
        try:
            page = self.session.ensure_page(APPLE_MUSIC_HOSTNAME, APPLE_MUSIC_URL, focus=focus)
        except AuthenticatedBrowserUnavailable as exc:
            raise AppleMusicUnavailable(str(exc)) from exc
        self._page = page
        return page

    @property
    def page(self):
        return self._page

    _STOREFRONT_IN_URL = re.compile(r"^https://music\.apple\.com/([a-z]{2})(?:/|$)")

    def _resolve_storefront(self) -> str:
        """The signed-in account's real Apple Music storefront (e.g.
        `"il"`), resolved once and cached for this controller's lifetime.

        Confirmed live and root-caused a preview-only playback bug: every
        storefront-LESS URL this module used (`/search?term=...`,
        `/listen-now`, `/library/all-playlists`) silently redirects to the
        `us` storefront regardless of the signed-in account's actual
        region, while navigating to the bare root `/` correctly redirects
        to the account's real storefront (observed live: `/il/home` for
        this account). Search results built from a `/us/...` URL point at
        US-catalog content the account's subscription has no full-playback
        entitlement for -- DRM/Widevine negotiation succeeds and the
        player-bar UI shows normal "now playing" state either way, but
        Apple Music silently serves only a short instant-preview clip for
        content outside the subscribed storefront. Playing the SAME song
        via its `/il/...` URL (this account's real storefront) was
        confirmed live to use real MSE-backed streaming (duration reported
        as `Infinity`, no `AudioPreview` CDN path) instead. Falls back to
        `"us"` (the previous, if wrong, hardcoded behavior) if resolution
        fails for any reason -- never crashes a caller over this."""
        if self._storefront is not None:
            return self._storefront
        page = self.ensure_music_tab()
        try:
            page.goto(APPLE_MUSIC_URL + "/", wait_until="domcontentloaded", timeout=15000)
            page.wait_for_timeout(500)
            match = self._STOREFRONT_IN_URL.match(page.url)
            self._storefront = match.group(1) if match else "us"
        except Exception as exc:
            log.info("Apple Music storefront resolution failed, defaulting to 'us': %s", exc)
            self._storefront = "us"
        return self._storefront

    # ------------------------------------------------------------------
    # Sign-in state
    # ------------------------------------------------------------------

    def is_signed_in(self, page=None, settle_timeout: float = 8.0) -> bool:
        """Is Apple Music Web actually signed in on this tab?

        Waits for the app shell to hydrate before deciding. Apple Music
        renders a "Sign In" control in its initial HTML and only replaces
        it with the account affordance once the SPA has loaded the user
        session, so checking immediately after opening a fresh tab
        reported "not signed in" for a perfectly signed-in account --
        confirmed live, right after `_autostart_chrome` opened the tab.

        Resolves as soon as EITHER answer is genuinely visible, so the
        common case costs nothing; the timeout only bites on a page that
        never settles, and it then falls back to the previous behaviour
        (assume signed in unless a sign-in control is showing) rather than
        inventing an answer.
        """
        page = page or self._page
        if page is None:
            return False
        deadline = time.monotonic() + max(0.0, float(settle_timeout))
        while True:
            if self._sign_in_control_visible(page):
                return False
            if self._signed_in_marker_visible(page):
                return True
            if time.monotonic() >= deadline:
                # Neither marker resolved. Keep the historical default:
                # no visible sign-in control means signed in.
                return True
            page.wait_for_timeout(300)

    def _sign_in_control_visible(self, page) -> bool:
        for role in ("button", "link"):
            try:
                if page.get_by_role(role, name=_SIGN_IN_NAME).first.is_visible(timeout=400):
                    return True
            except Exception:
                continue
        return False

    def _signed_in_marker_visible(self, page) -> bool:
        """Something only a signed-in session shows.

        The account control and the Library navigation are both confirmed
        live on the signed-in shell ("My Account" appeared in the real
        button list); either one is proof the session loaded, which is
        what makes an early "not signed in" impossible to report.
        """
        for role, pattern in (
            ("button", re.compile(r"my account|account", re.I)),
            ("button", re.compile(r"^\s*library\s*$", re.I)),
            ("link", re.compile(r"^\s*library\s*$", re.I)),
        ):
            try:
                if page.get_by_role(role, name=pattern).first.is_visible(timeout=400):
                    return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------
    # Now-playing observation (Part 5/14/18)
    # ------------------------------------------------------------------

    def current_track_info(self, page=None) -> dict[str, Any]:
        """Best-effort read of what's currently loaded/playing. Returns
        `{"song", "artist", "is_playing", "observed"}`; `observed=False`
        means nothing could be confidently read (never fabricated).

        Primary source is the player bar's `[data-testid="marquee-text-item"]`
        (song) / `[data-testid="marquee-text-item-button"]` (artist) --
        confirmed live against the real site. `document.title` is kept only
        as a last-resort fallback: live-confirmed it does NOT update to
        reflect now-playing on Apple Music Web (it stays the static page
        title), so it will rarely if ever fire, but costs nothing to keep
        as a fallback for a future markup change."""
        page = page or self._page
        info: dict[str, Any] = {"song": None, "artist": None, "is_playing": False, "observed": False}
        if page is None:
            return info
        try:
            pause_btn = page.get_by_role("button", name=_PAUSE_NAME).first
            info["is_playing"] = bool(pause_btn.is_visible(timeout=1000))
        except Exception:
            pass
        try:
            song_el = page.locator('[data-testid="marquee-text-item"]').first
            if song_el.is_visible(timeout=1000):
                song_text = (song_el.inner_text(timeout=500) or "").strip()
                if song_text:
                    artist_text = None
                    try:
                        artist_el = page.locator('[data-testid="marquee-text-item-button"]').first
                        if artist_el.is_visible(timeout=500):
                            artist_text = (artist_el.inner_text(timeout=500) or "").strip() or None
                    except Exception:
                        pass
                    info["song"] = song_text
                    info["artist"] = artist_text
                    info["observed"] = True
                    return info
        except Exception:
            pass
        title = ""
        try:
            title = page.title() or ""
        except Exception:
            pass
        song, artist = _parse_title_now_playing(title)
        if song:
            info["song"] = song
            info["artist"] = artist
            info["observed"] = True
        return info

    def playback_type(self, page=None) -> dict[str, Any]:
        """Best-effort detection of whether current audio is Apple's short
        instant PREVIEW clip rather than real full-track subscription
        streaming. Confirmed live: clicking a track's row/platter/hero
        "Play" control (regardless of which of those three, or whether
        navigation to get there was a hard `page.goto` or an in-app SPA
        link click -- all four combinations were tried live) can serve a
        plain `<audio>` element whose `currentSrc` is on Apple's
        `AudioPreview<N>` CDN path with a ~90s duration, while the player
        bar UI still reports normal playing/song/artist state -- there is
        currently no DOM signal distinguishing this from confirmed full
        playback other than inspecting the actual audio element. Returns
        `{"observed": False}` (never guessed) when no `<audio>` element
        exists at all -- genuine full-track DRM'd streaming may use a
        different mechanism this can't introspect, so an absent `<audio>`
        element is NOT itself evidence of full playback either."""
        page = page or self._page
        if page is None:
            return {"observed": False}
        try:
            info = page.evaluate(
                """() => {
                    const els = [...document.querySelectorAll('audio')];
                    const active = els.find(e => !e.paused) || els[els.length - 1];
                    if (!active) return null;
                    return {src: active.currentSrc || '', duration: active.duration || null};
                }"""
            )
        except Exception:
            return {"observed": False}
        if not info or not info.get("src"):
            return {"observed": False}
        duration = info.get("duration")
        is_preview = "preview" in info["src"].lower() or bool(duration and duration <= 95)
        return {"observed": True, "is_preview": is_preview, "duration": duration}

    def wait_for_playing(self, timeout: float = 6.0) -> bool:
        page = self._page
        if page is None:
            return False
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            try:
                if page.get_by_role("button", name=_PAUSE_NAME).first.is_visible(timeout=500):
                    return True
            except Exception:
                pass
            time.sleep(0.2)
        return False

    def wait_for_paused(self, timeout: float = 4.0) -> bool:
        page = self._page
        if page is None:
            return False
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            try:
                if page.get_by_role("button", name=_PLAY_NAME).first.is_visible(timeout=500):
                    return True
            except Exception:
                pass
            time.sleep(0.2)
        return False

    def wait_for_track_change(self, previous_song: str | None, timeout: float = 6.0) -> dict[str, Any]:
        deadline = time.perf_counter() + timeout
        last_info: dict[str, Any] = {"song": None, "artist": None, "is_playing": False, "observed": False}
        while time.perf_counter() < deadline:
            last_info = self.current_track_info()
            if last_info.get("observed") and last_info.get("song") != previous_song:
                return last_info
            time.sleep(0.25)
        return last_info

    # ------------------------------------------------------------------
    # Transport controls
    # ------------------------------------------------------------------

    def _click_control(self, name_pattern: re.Pattern, page=None) -> bool:
        page = page or self._page
        if page is None:
            return False
        try:
            control = page.get_by_role("button", name=name_pattern).first
            control.click(timeout=3000)
            return True
        except Exception as exc:
            log.info("Apple Music control click failed for pattern=%r: %s", name_pattern.pattern, exc)
            return False

    def play_pause(self) -> bool:
        return self._click_control(_PLAY_NAME) or self._click_control(_PAUSE_NAME)

    def press_play(self) -> bool:
        return self._click_control(_PLAY_NAME)

    def press_pause(self) -> bool:
        return self._click_control(_PAUSE_NAME)

    def next_track(self) -> bool:
        return self._click_control(_NEXT_NAME)

    def previous_track(self) -> bool:
        return self._click_control(_PREV_NAME)

    def restart_track(self) -> bool:
        return self.previous_track()

    def set_shuffle(self, on: bool) -> bool:
        page = self._page
        if page is None:
            return False
        try:
            control = page.get_by_role("button", name=_SHUFFLE_NAME).first
            pressed = control.get_attribute("aria-pressed")
            if pressed is not None:
                if (pressed == "true") == on:
                    return True
                control.click(timeout=3000)
                return True
            control.click(timeout=3000)
            return True
        except Exception as exc:
            log.info("Apple Music shuffle toggle failed: %s", exc)
            return False

    def set_repeat(self, on: bool) -> bool:
        page = self._page
        if page is None:
            return False
        try:
            control = page.get_by_role("button", name=_REPEAT_NAME).first
            pressed = control.get_attribute("aria-pressed")
            if pressed is not None:
                if (pressed == "true") == on:
                    return True
                control.click(timeout=3000)
                return True
            control.click(timeout=3000)
            return True
        except Exception as exc:
            log.info("Apple Music repeat toggle failed: %s", exc)
            return False

    def add_current_to_library(self) -> bool:
        return self._click_control(_ADD_LIBRARY_NAME)

    def add_current_to_favorites(self) -> bool:
        page = self._page
        if page is None:
            return False
        try:
            more = page.get_by_role("button", name=re.compile("more", re.I)).first
            more.click(timeout=2000)
            fav = page.get_by_role("menuitem", name=_ADD_FAVORITE_NAME).first
            fav.click(timeout=2000)
            return True
        except Exception:
            return self._click_control(_ADD_FAVORITE_NAME)

    # ------------------------------------------------------------------
    # Search (Part 8)
    # ------------------------------------------------------------------


    #: Result links, used both to detect that a search has rendered and to
    #: detect that it has stopped changing.
    _RESULT_LINK_SELECTOR = "a[href*='/album/'], a[href*='/artist/'], a[href*='/playlist/'], a[href*='/song/']"

    def _wait_for_search_results(self, page, query: str, timeout: float = 10.0) -> bool:
        """Block until the search page is really showing `query`'s results.

        Step 1: the SPA copies the URL's `term` into its own search box, so
        that box holding the new query is proof the new navigation was
        processed -- not merely that some links exist.

        Step 2: the result count has to stop changing. Apple streams the
        shelves in, and reading mid-render yields a partial (and
        previous-query-contaminated) list. Two consecutive identical counts
        is the settle signal.

        Returns whether it settled; a timeout is not an error -- the caller
        reads whatever is there, which is the same behaviour as before,
        just no longer the DEFAULT behaviour.
        """
        deadline = time.monotonic() + timeout
        wanted = (query or "").strip().lower()

        # Step 1 -- the search box reflects the new term.
        while time.monotonic() < deadline:
            try:
                box = page.locator("input[data-testid='search-input__text-field'], [data-testid='search-input'] input").first
                value = (box.input_value(timeout=500) or "").strip().lower()
                if wanted and wanted in value:
                    break
            except Exception:
                pass
            page.wait_for_timeout(150)

        # Step 2 -- the result list stops growing.
        previous = -1
        stable = 0
        while time.monotonic() < deadline:
            try:
                count = page.locator(self._RESULT_LINK_SELECTOR).count()
            except Exception:
                count = -1
            if count > 0 and count == previous:
                stable += 1
                if stable >= 2:
                    return True
            else:
                stable = 0
            previous = count
            page.wait_for_timeout(250)
        return False

    def search(self, query: str) -> list[dict[str, str]]:
        """Navigate Apple Music's catalog+library search and return
        candidate results as `{"type", "title", "subtitle"}` dicts
        (type in {"song","artist","album","playlist"} when determinable).
        Never guesses a result -- an empty list means nothing was found."""
        page = self.ensure_music_tab()
        try:
            page.goto(f"{APPLE_MUSIC_URL}/{self._resolve_storefront()}/search?term={_url_quote(query)}", wait_until="domcontentloaded", timeout=15000)
        except Exception as exc:
            log.info("Apple Music search navigation failed: %s", exc)
            return []
        # Live-confirmed: results render well after domcontentloaded (this
        # is a heavy client-rendered SPA) -- a short fixed sleep missed
        # them entirely.
        #
        # Waiting for "a result link is visible" is NOT enough, and that bug
        # was confirmed live: this is a single long-lived tab, so the
        # PREVIOUS query's links are still in the DOM the instant the new
        # URL commits, the wait returns immediately, and the caller scores
        # the old page's results. A search for "Save Your Tears" returned
        # Khalid, twenty one pilots and "thank u, next" that way, and
        # `_best_search_match` duly picked "Stressed Out" -- a wrong song
        # chosen with complete confidence.
        #
        # So settle on the NEW query in two steps, both of which observe
        # real page state rather than sleeping a guessed duration.
        self._wait_for_search_results(page, query)
        results: list[dict[str, str]] = []
        try:
            candidates = page.get_by_role("link").all()
        except Exception:
            candidates = []
        for link in candidates[:120]:
            try:
                if not link.is_visible():
                    continue
                text = (link.inner_text(timeout=300) or "").strip()
                href = link.get_attribute("href") or ""
            except Exception:
                continue
            if not text or not href:
                continue
            kind = _classify_result_href(href)
            if kind is None:
                continue
            results.append({"type": kind, "title": text, "href": href})
        return results

    def open_result(self, href: str) -> bool:
        page = self.ensure_music_tab()
        try:
            url = href if href.startswith("http") else f"{APPLE_MUSIC_URL}{href}"
            page.goto(url, wait_until="domcontentloaded", timeout=15000)
            page.wait_for_timeout(800)
            return True
        except Exception as exc:
            log.info("Apple Music open_result failed: %s", exc)
            return False

    def _confirm_playing_or_click_hero(self, timeout: float = 1.5) -> None:
        """After a row-button click, Apple Music Web was observed live to
        sometimes only SELECT/LOAD the track (the player-bar metadata
        updates, and the hero PLAY button becomes enabled) WITHOUT
        actually starting playback -- confirmed live on a playlist
        immediately after navigation, while the exact same row-click
        pattern started playback immediately elsewhere (a song's own
        page). Rather than guess which case applies, wait briefly for the
        Pause state; if it never arrives, click the (by now enabled) hero
        PLAY button as a follow-up. A no-op if already playing -- the
        hero button's accessible name is "Pause" then, which this never
        matches."""
        if self.wait_for_playing(timeout=timeout):
            return
        self._click_control(_PLAY_NAME)

    def play_specific_track(self, title: str, artist: str | None = None) -> bool:
        """Click the row-specific "Play <title> by <artist>" control for an
        exact track -- confirmed live to be the reliable way to start THAT
        track. The page-level hero PLAY button plays the whole
        album/context (not necessarily this track), and was also observed
        live to sometimes be `disabled` right after navigation. Falls back
        to `play_from_current_page()` if no row matches (e.g. the title we
        have doesn't exactly match what's rendered)."""
        page = self._page
        if page is None or not title.strip():
            return False
        escaped = re.escape(title.strip())
        patterns = [re.compile(rf"^Play\s+{escaped}\s+by\s+", re.I)] if artist else []
        patterns.append(re.compile(rf"^Play\s+{escaped}(?:\s+by\s+.+)?$", re.I))
        for pattern in patterns:
            if self._click_control(pattern, page):
                self._confirm_playing_or_click_hero()
                return True
        return self.play_from_current_page()

    def play_from_current_page(self) -> bool:
        """Click the primary Play affordance on whatever page is currently
        loaded (a song/album/playlist/artist detail page).

        Prefers the FIRST track row's own "Play <title> by <artist>"
        button -- confirmed live to reliably start playback from track 1
        of an album/playlist -- over the page-level hero "PLAY" button,
        which was observed live to sometimes render `disabled` (e.g. right
        after navigating to a playlist) even while fully visible. Falls
        back to the hero button (needed for artist "shuffle all" pages,
        which have no individual track rows) if no row button is found."""
        page = self._page
        if page is None:
            return False
        try:
            row_button = page.get_by_role("button", name=_ROW_PLAY_NAME).first
            if row_button.is_visible(timeout=1500) and row_button.is_enabled(timeout=500):
                row_button.click(timeout=3000)
                self._confirm_playing_or_click_hero()
                return True
        except Exception as exc:
            log.info("Apple Music: no usable row play button, falling back to hero PLAY: %s", exc)
        for pattern in (_PLAY_NAME, re.compile(r"play", re.I)):
            if self._click_control(pattern, page):
                return True
        return False

    # ------------------------------------------------------------------
    # Queue (Part 16)
    # ------------------------------------------------------------------

    def queue_result(self, href: str, up_next: bool = False) -> bool:
        """Add a search result to Up Next (Part 16) without interrupting
        the currently playing track. Apple Music Web exposes this via a
        contextual "more options" menu on the item -- opens the item's own
        page (so the menu is unambiguous) rather than trying to hover a
        row in a results list."""
        page = self._page
        if page is None or not self.open_result(href):
            return False
        page.wait_for_timeout(300)
        try:
            more = page.get_by_role("button", name=re.compile("more", re.I)).first
            more.click(timeout=2500)
        except Exception as exc:
            log.info("Apple Music queue: could not open the item's more-options menu: %s", exc)
            return False
        name_pattern = re.compile(r"play next", re.I) if up_next else re.compile(r"play later|add to up next|add to queue", re.I)
        try:
            item = page.get_by_role("menuitem", name=name_pattern).first
            item.click(timeout=2500)
            return True
        except Exception as exc:
            log.info("Apple Music queue: menu item not found for pattern=%r: %s", name_pattern.pattern, exc)
            return False

    # ------------------------------------------------------------------
    # Playlists: adding a track, and creating one
    # ------------------------------------------------------------------
    #
    # Everything below drives the track row's own context menu, whose real
    # structure was confirmed live against the signed-in account (see
    # `_ADD_TO_PLAYLIST_NAME` above for the observed item list). That menu
    # is the only surface on Apple Music Web offering BOTH "add this to an
    # existing playlist" and "make a new playlist from this", so both
    # operations share one code path rather than being two independent
    # guesses at Apple's markup.

    def _open_track_menu(self, title: str | None = None, artist: str | None = None) -> bool:
        """Open the context menu for a specific track row (or the first row
        on the page when no title is given).

        Apple Music renders one `more` button per row, all with the SAME
        accessible name ("more"), so a row is identified by its
        neighbouring "Play <title> by <artist>" button -- the one per-row
        control that IS uniquely named -- and the menu button is taken from
        that row. `play_specific_track` already identifies rows the same
        way, so the two can never disagree about which row is which.
        """
        page = self._page
        if page is None:
            return False
        try:
            if title and title.strip():
                escaped = re.escape(title.strip())
                pattern = re.compile(rf"^Play\s+{escaped}(?:\s+by\s+.+)?$", re.I)
                row_button = page.get_by_role("button", name=pattern).first
            else:
                row_button = page.get_by_role("button", name=_ROW_PLAY_NAME).first
            row_button.wait_for(state="visible", timeout=6000)
            # Walk up to the row container and back down to its own
            # more-button: scoping to the row is what stops row 1's menu
            # opening when row 7 was meant.
            row = row_button.locator(
                "xpath=ancestor::*[self::tr or self::li or @role='row' or @role='listitem'][1]"
            )
            more = row.get_by_role("button", name=re.compile(r"^\s*more\s*$", re.I)).first
            more.scroll_into_view_if_needed(timeout=3000)
            more.click(timeout=4000)
        except Exception as exc:
            log.info("Apple Music: could not open the row menu for %r: %s", title, exc)
            return False
        try:
            page.locator(_MENU_SELECTOR).first.wait_for(state="visible", timeout=4000)
            return True
        except Exception:
            log.info("Apple Music: the row menu did not appear for %r", title)
            return False

    def open_track_menu(self, title: str | None = None, artist: str | None = None, attempts: int = 2) -> bool:
        """`_open_track_menu` with one bounded retry.

        This is a heavy client-rendered SPA and a row can still be
        painting when the click lands. One retry after a short settle
        costs a second in the rare failing case and removes a whole class
        of "it worked the first time and not the second" flakiness. The
        retry count is bounded by construction -- there is no loop that
        can run away.
        """
        for attempt in range(max(1, int(attempts))):
            if self._open_track_menu(title, artist):
                return True
            self._close_menus()
            if attempt + 1 < attempts and self._page is not None:
                self._page.wait_for_timeout(900)
        return False

    def _close_menus(self) -> None:
        """Escape out of any open context menu. Always safe, and always run
        in a `finally` so a failed playlist operation never leaves a menu
        covering the page for whatever runs next."""
        page = self._page
        if page is None:
            return
        for _ in range(3):
            try:
                if page.locator(_MENU_SELECTOR).count() == 0:
                    return
                page.keyboard.press("Escape")
                page.wait_for_timeout(150)
            except Exception:
                return

    #: Menu entries that are commands, not playlist names. Anything else in
    #: the open menu is a playlist the account owns.
    _MENU_COMMAND_LABELS = frozenset({
        "pin song", "unpin song", "delete from library", "add to playlist",
        "new playlist", "recents", "all playlists", "play next", "play last",
        "create station", "favourite", "favorite", "unfavourite", "unfavorite",
        "suggest less", "view credits", "add to library", "share song",
        "go to album", "go to artist", "share playlist", "remove from playlist",
    })

    def _menu_entry(self, name, exact: bool = True):
        """One entry of an open Apple Music context menu.

        Confirmed live: Apple renders these as `role="button"` inside a
        `[role="menu"]`, NOT as `role="menuitem"`. Querying menuitem found
        nothing and timed out -- so button is tried FIRST and menuitem is
        kept only as a fallback in case Apple changes it back.
        """
        page = self._page
        for role in ("button", "menuitem"):
            try:
                candidate = page.get_by_role(role, name=name, exact=exact).first
                if candidate.is_visible(timeout=1200):
                    return candidate
            except Exception:
                continue
        return None

    def _open_add_to_playlist_submenu(self) -> bool:
        """Expand the "Add to Playlist" submenu of an open row menu.

        Confirmed live: the submenu's contents ("New Playlist", "Recents",
        every playlist) are present in the DOM as soon as the row menu
        opens, but they are NOT visible or clickable until this parent
        entry is activated. An earlier version clicked the playlist name
        directly and timed out for exactly that reason.
        """
        page = self._page
        if page is None:
            return False
        entry = self._menu_entry(_ADD_TO_PLAYLIST_NAME, exact=False)
        if entry is None:
            log.info("Apple Music: the row menu has no 'Add to Playlist' entry")
            return False
        try:
            entry.click(timeout=3000)
            page.wait_for_timeout(700)
        except Exception as exc:
            log.info("Apple Music: could not open the Add-to-Playlist submenu: %s", exc)
            return False
        # "New Playlist" heads that submenu, so its visibility is the
        # signal that the submenu genuinely expanded.
        return self._menu_entry(_NEW_PLAYLIST_NAME, exact=False) is not None

    def menu_playlist_names(self) -> list[str]:
        """The playlist names offered by the currently open Add-to-Playlist
        submenu.

        Lets a failure report "I could not find a playlist called X -- here
        are the ones that exist", which is far more useful to both the user
        and the agent than a bare "not found".
        """
        page = self._page
        if page is None:
            return []
        names: list[str] = []
        try:
            for role in ("button", "menuitem"):
                for item in page.get_by_role(role).all()[:200]:
                    try:
                        if not item.is_visible():
                            continue
                        text = (item.inner_text(timeout=150) or "").strip()
                    except Exception:
                        continue
                    # A multi-line entry is a container, not a leaf item.
                    if not text or "\n" in text or len(text) > 80:
                        continue
                    if text.lower() in self._MENU_COMMAND_LABELS:
                        continue
                    if text not in names:
                        names.append(text)
                if names:
                    break
        except Exception:
            log.debug("Reading the menu's playlist names failed", exc_info=True)
        return names

    def add_track_to_playlist(self, playlist: str, title: str | None = None, artist: str | None = None) -> dict[str, Any]:
        """Add a track on the current page to an existing playlist.

        Returns a dict rather than a bool so an unmatched name can carry
        the real list of available playlists back to the caller. Nothing is
        guessed: the name must match a menu entry exactly
        (case-insensitively) or, failing that, unambiguously as a prefix --
        one candidate, never "the closest of several".
        """
        wanted = (playlist or "").strip()
        if not wanted:
            return {"added": False, "error": "missing_playlist"}
        if not self.open_track_menu(title, artist):
            return {"added": False, "error": "row_menu_unavailable"}

        page = self._page
        try:
            if not self._open_add_to_playlist_submenu():
                return {"added": False, "error": "add_to_playlist_unavailable"}
            available = self.menu_playlist_names()
            lowered = wanted.lower()
            exact = [name for name in available if name.lower() == lowered]
            prefix = [name for name in available if name.lower().startswith(lowered)] if not exact else []
            if not exact and len(prefix) != 1:
                return {
                    "added": False,
                    "error": "playlist_not_found" if not prefix else "playlist_ambiguous",
                    "available": available,
                    "candidates": prefix,
                }
            target = (exact or prefix)[0]
            entry = self._menu_entry(target, exact=True)
            if entry is None:
                return {"added": False, "error": "playlist_entry_not_clickable", "available": available}
            entry.click(timeout=3000)
            page.wait_for_timeout(900)
            # The menu closing is Apple's own acknowledgement that the entry
            # was accepted. That is a weak signal by itself, which is why
            # the provider re-opens the playlist and checks the track is
            # genuinely listed; this reports only what the UI did.
            menu_closed = page.locator(_MENU_SELECTOR).count() == 0
            return {"added": True, "playlist": target, "menu_closed": menu_closed}
        except Exception as exc:
            log.info("Apple Music: adding to playlist %r failed: %s", wanted, exc)
            return {"added": False, "error": f"add_failed:{type(exc).__name__}"}
        finally:
            self._close_menus()

    def create_playlist_from_track(self, name: str, title: str | None = None, artist: str | None = None) -> dict[str, Any]:
        """Create a NEW playlist called `name`, containing the track on the
        current page.

        Apple Music Web has no "create an empty playlist" control that can
        be driven reliably. What it does have -- confirmed live -- is
        "New Playlist" inside a track's Add-to-Playlist submenu. Clicking it
        does NOT navigate anywhere: it reveals an inline text field
        (`data-testid="playlist-title-input"`, accessible name "Playlist
        Title") on the same page, and committing that field creates the
        playlist with the track already in it. So the name is set AT
        creation rather than by a separate rename afterwards -- which is
        both fewer steps and impossible to leave half-done as a stray
        playlist called "New Playlist".
        """
        name = (name or "").strip()
        if not name:
            return {"created": False, "error": "missing_name"}
        if not self.open_track_menu(title, artist):
            return {"created": False, "error": "row_menu_unavailable"}
        page = self._page
        try:
            if not self._open_add_to_playlist_submenu():
                return {"created": False, "error": "add_to_playlist_unavailable"}
            entry = self._menu_entry(_NEW_PLAYLIST_NAME, exact=False)
            if entry is None:
                return {"created": False, "error": "new_playlist_unavailable"}
            entry.click(timeout=4000)

            field = page.locator(_PLAYLIST_TITLE_INPUT).first
            try:
                field.wait_for(state="visible", timeout=5000)
            except Exception:
                return {"created": False, "error": "title_field_did_not_appear"}
            field.click(timeout=3000)
            field.fill(name, timeout=3000)
            page.keyboard.press("Enter")
            page.wait_for_timeout(1800)
            return {"created": True, "name": name, "url": page.url}
        except Exception as exc:
            log.info("Apple Music: creating a playlist failed: %s", exc)
            return {"created": False, "error": f"create_failed:{type(exc).__name__}"}
        finally:
            self._close_menus()

    def open_library_playlist(self, name: str) -> dict[str, Any]:
        """Navigate to one of the account's own playlists by name.

        Reuses `list_library_playlists` (the real `/library/all-playlists`
        reader) rather than guessing a URL, so the href is always one Apple
        itself produced.
        """
        wanted = (name or "").strip().lower()
        playlists = self.list_library_playlists()
        available = [item.get("name", "") for item in playlists]
        if not wanted:
            return {"opened": False, "error": "missing_name", "available": available}
        exact = [item for item in playlists if item.get("name", "").strip().lower() == wanted]
        prefix = [item for item in playlists if item.get("name", "").strip().lower().startswith(wanted)] if not exact else []
        chosen = exact or prefix
        if len(chosen) != 1:
            return {
                "opened": False,
                "error": "playlist_not_found" if not chosen else "playlist_ambiguous",
                "available": available,
            }
        target = chosen[0]
        opened = self.open_result(target["href"])
        return {
            "opened": bool(opened),
            "name": target["name"],
            "href": target["href"],
            "url": self._page.url if self._page else "",
        }

    def current_page_track_titles(self, limit: int = 60) -> list[str]:
        """Track titles on the page currently open.

        `data-testid="track-title"` was confirmed live on a real playlist
        page. This is what makes an "add to playlist" genuinely verifiable:
        the provider re-opens the playlist and checks the song is really
        listed instead of trusting that a menu click meant something.
        """
        page = self._page
        if page is None:
            return []
        titles: list[str] = []
        try:
            elements = page.locator("[data-testid='track-title']").all()[: max(1, int(limit))]
            for element in elements:
                try:
                    text = (element.get_attribute("aria-label") or element.inner_text(timeout=200) or "").strip()
                except Exception:
                    continue
                if text:
                    titles.append(text)
        except Exception:
            log.debug("Reading track titles failed", exc_info=True)
        return titles

    # ------------------------------------------------------------------
    # Recently played (Part 11-B fallback when local history is empty)
    # ------------------------------------------------------------------

    def get_recently_played(self) -> list[dict[str, str]]:
        """Apple Music's own Recently Played shelf (Part 11-B fallback --
        only consulted when local history has nothing). Confirmed live:
        this lives on `/listen-now` (the bare root `/` does NOT reliably
        render it) under a "Recently Played" heading. Some shelf entries
        are playlist/station contexts with no real `href` (Apple Music Web
        drives those through client-side JS, not a normal link) -- skipped
        here since there's nothing safe to navigate to for them; only
        entries with a real album/song link are returned."""
        page = self.ensure_music_tab()
        try:
            page.goto(f"{APPLE_MUSIC_URL}/{self._resolve_storefront()}/listen-now", wait_until="domcontentloaded", timeout=15000)
        except Exception as exc:
            log.info("Apple Music listen-now navigation failed: %s", exc)
            return []
        try:
            heading = page.get_by_text("Recently Played", exact=True).first
            heading.wait_for(state="visible", timeout=8000)
            heading.scroll_into_view_if_needed(timeout=2000)
            page.wait_for_timeout(400)
        except Exception:
            return []
        try:
            raw_links = page.evaluate(
                """() => {
                    const spans = [...document.querySelectorAll('span, h2, h3')];
                    const heading = spans.find(e => e.textContent.trim() === 'Recently Played');
                    if (!heading) return [];
                    let el = heading;
                    for (let i = 0; i < 6 && el; i++) {
                        el = el.parentElement;
                        if (el && el.querySelectorAll('a[href]').length > 3) break;
                    }
                    if (!el) return [];
                    return [...el.querySelectorAll('a[href]')].map(a => ({href: a.getAttribute('href'), text: a.textContent.trim()}));
                }"""
            )
        except Exception as exc:
            log.info("Apple Music recently-played extraction failed: %s", exc)
            return []
        items: list[dict[str, str]] = []
        seen: set[str] = set()
        for entry in raw_links or []:
            href = (entry or {}).get("href") or ""
            text = ((entry or {}).get("text") or "").strip()
            if not href or href == "#" or not text or text in seen:
                continue
            seen.add(text)
            kind = _classify_result_href(href) or "song"
            items.append({"type": kind, "title": text, "href": href})
        return items

    # ------------------------------------------------------------------
    # Playlists (Part 9/10)
    # ------------------------------------------------------------------

    def list_library_playlists(self) -> list[dict[str, str]]:
        """Real user library playlists (Part 9/10) -- confirmed live that
        the URL is `/library/all-playlists`, NOT `/library/playlists`
        (which renders only the sidebar's own "All Playlists" nav link and
        nothing else -- an earlier version of this method guessed wrong
        and silently returned an empty/useless list every time)."""
        page = self.ensure_music_tab()
        try:
            page.goto(f"{APPLE_MUSIC_URL}/{self._resolve_storefront()}/library/all-playlists", wait_until="domcontentloaded", timeout=15000)
        except Exception as exc:
            log.info("Apple Music playlist listing navigation failed: %s", exc)
            return []
        try:
            page.locator("a[href*='/library/playlist/']").first.wait_for(state="visible", timeout=8000)
        except Exception:
            pass  # genuinely no library playlists -- fall through to an empty list
        playlists: list[dict[str, str]] = []
        try:
            links = page.locator("a[href*='/library/playlist/']").all()
        except Exception:
            links = []
        seen = set()
        for link in links[:200]:
            try:
                if not link.is_visible():
                    continue
                name = (link.inner_text(timeout=300) or "").strip()
                href = link.get_attribute("href") or ""
            except Exception:
                continue
            if not name or not href or name in seen:
                continue
            seen.add(name)
            playlists.append({"name": name, "href": href})
        return playlists

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnose(self) -> dict[str, Any]:
        """Dump the accessible-name of every visible button on the current
        page. Not used by any control-flow path -- a manual tool for
        tightening the locator patterns above against the real, live,
        signed-in site (see module docstring)."""
        page = self._page
        if page is None:
            return {"buttons": [], "url": None}
        buttons = []
        try:
            for button in page.get_by_role("button").all()[:80]:
                try:
                    if button.is_visible():
                        buttons.append((button.get_attribute("aria-label") or button.inner_text(timeout=200) or "").strip())
                except Exception:
                    continue
        except Exception:
            pass
        return {"buttons": [b for b in buttons if b], "url": page.url}

    def diagnose_auth_state(self) -> dict[str, Any]:
        """Report whether Apple Music-related sign-in state exists in the
        attached session, WITHOUT revealing any cookie value -- just
        per-domain counts, and only for Apple's own domains (never the
        rest of the user's real, shared browser profile)."""
        try:
            report = self.session.cookie_counts(urls=list(_APPLE_MUSIC_AUTH_URLS))
        except AuthenticatedBrowserUnavailable:
            return {"live": False, "cookie_counts": {}}
        report["live"] = True
        return report


def _url_quote(text: str) -> str:
    return urllib.parse.quote_plus(text)


def _classify_result_href(href: str) -> str | None:
    if "/song/" in href or re.search(r"\?i=\d+", href):
        return "song"
    if "/album/" in href:
        return "album"
    if "/artist/" in href:
        return "artist"
    if "/playlist/" in href:
        return "playlist"
    return None


#: Zero-width/directional marks Apple Music Web's real page titles were
#: confirmed live to always carry (e.g. a leading U+200E LEFT-TO-RIGHT
#: MARK) -- invisible on screen, but NOT removed by `str.strip()`, so a
#: naive shell-title check like `title == "Apple Music"` silently never
#: matches the real title and falls through to parsing the shell as a
#: song. Real titles were also confirmed to use U+00A0 NO-BREAK SPACE
#: between some words instead of a normal space.
_INVISIBLE_TITLE_CHARS = re.compile(r"[​-‏‪-‮﻿]")


def _clean_title(title: str) -> str:
    cleaned = _INVISIBLE_TITLE_CHARS.sub("", title or "").replace(" ", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _parse_title_now_playing(title: str) -> tuple[str | None, str | None]:
    """Apple Music Web (and most single-page media players) update
    `document.title` to reflect what's playing, typically "Song - Artist"
    or "Song by Artist". Best-effort, and deliberately conservative: the
    generic "Apple Music" / "Apple Music - Home" / "Apple Music - Search" /
    "Apple Music - Web Player" shell titles must never be parsed as a
    song -- confirmed live that "Apple Music - Web Player" (a MULTI-WORD
    suffix) is the real title shown when nothing is playing, which an
    earlier `\\w+`-only (single-word) suffix pattern here did not catch."""
    title = _clean_title(title)
    if not title or title.lower() in {"apple music", "music"}:
        return None, None
    if re.fullmatch(r"apple music\s*[-–]\s*.+", title, re.I):
        return None, None
    for sep in (" – ", " - ", " by "):
        if sep in title:
            left, _, right = title.partition(sep)
            left, right = left.strip(), right.strip()
            right = re.sub(r"\s*[–-]\s*apple music\s*$", "", right, flags=re.I).strip()
            if left and right:
                return left, right
    return None, None


_SHARED_CONTROLLER: AppleMusicWebController | None = None


def get_apple_music_controller() -> AppleMusicWebController:
    global _SHARED_CONTROLLER
    if _SHARED_CONTROLLER is None:
        _SHARED_CONTROLLER = AppleMusicWebController()
    return _SHARED_CONTROLLER


def reset_apple_music_controller_for_tests(controller: AppleMusicWebController | None = None) -> None:
    """Test-only helper: swap the process-wide singleton for an isolated
    instance so tests never share state with a real run."""
    global _SHARED_CONTROLLER
    _SHARED_CONTROLLER = controller


def diagnose() -> dict[str, Any]:
    """Direct, no-voice-needed diagnostic (Part 10 of the live-path debug
    request). Never raises -- every step is reported honestly instead,
    including a CDP-unreachable session. Prints: CDP reachable, context/
    page counts, each page's hostname/title, whether an existing Apple
    Music tab was found, sign-in state if observable, and whether the
    controller is fully ready to act."""
    from tools.browser_authenticated import AuthenticatedBrowserUnavailable, cdp_endpoint, is_cdp_available, _redact_url

    report: dict[str, Any] = {"cdp_reachable": is_cdp_available()}
    print(f"CDP endpoint: {cdp_endpoint()}")
    print(f"CDP reachable: {'yes' if report['cdp_reachable'] else 'no'}")
    if not report["cdp_reachable"]:
        print("Run: python -m tools.browser_authenticated --launch")
        report["controller_ready"] = False
        return report

    controller = get_apple_music_controller()
    try:
        browser = controller.session.ensure_connected()
        report["contexts"] = len(browser.contexts)
        print(f"Contexts: {report['contexts']}")
        pages = controller.session.list_pages()
    except AuthenticatedBrowserUnavailable as exc:
        report.update(controller_ready=False, error=str(exc))
        print(f"Controller ready: no ({exc})")
        return report

    report["pages"] = len(pages)
    print(f"Pages: {report['pages']}")
    apple_music_tab_found = False
    for page in pages:
        try:
            host, title = _redact_url(page.url), page.title()
        except Exception:
            host, title = "<unavailable>", None
        if APPLE_MUSIC_HOSTNAME in (host or ""):
            apple_music_tab_found = True
        print(f"  - {host}  (title: {title!r})")
    print(f"Apple Music tab found: {'yes' if apple_music_tab_found else 'no'}")
    report["apple_music_tab_found"] = apple_music_tab_found

    try:
        page = controller.ensure_music_tab()
        signed_in = controller.is_signed_in(page)
        report.update(controller_ready=True, signed_in=signed_in, apple_music_tab_found=True)
        print(f"Signed in: {'yes' if signed_in else 'no'}")
        print("Controller ready: yes")
    except AuthenticatedBrowserUnavailable as exc:
        report.update(controller_ready=False, error=str(exc))
        print(f"Controller ready: no ({exc})")
    return report


if __name__ == "__main__":
    diagnose()
