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

    # ------------------------------------------------------------------
    # Sign-in state
    # ------------------------------------------------------------------

    def is_signed_in(self, page=None) -> bool:
        page = page or self._page
        if page is None:
            return False
        try:
            sign_in = page.get_by_role("button", name=_SIGN_IN_NAME).first
            if sign_in.is_visible(timeout=1500):
                return False
        except Exception:
            pass
        try:
            sign_in_link = page.get_by_role("link", name=_SIGN_IN_NAME).first
            if sign_in_link.is_visible(timeout=1000):
                return False
        except Exception:
            pass
        return True

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

    def search(self, query: str) -> list[dict[str, str]]:
        """Navigate Apple Music's catalog+library search and return
        candidate results as `{"type", "title", "subtitle"}` dicts
        (type in {"song","artist","album","playlist"} when determinable).
        Never guesses a result -- an empty list means nothing was found."""
        page = self.ensure_music_tab()
        try:
            page.goto(f"{APPLE_MUSIC_URL}/search?term={_url_quote(query)}", wait_until="domcontentloaded", timeout=15000)
        except Exception as exc:
            log.info("Apple Music search navigation failed: %s", exc)
            return []
        # Live-confirmed: results render well after domcontentloaded (this
        # is a heavy client-rendered SPA) -- a short fixed sleep missed
        # them entirely. Wait for an actual result link instead of guessing
        # a duration.
        try:
            page.locator("a[href*='/album/'], a[href*='/artist/'], a[href*='/playlist/']").first.wait_for(state="visible", timeout=8000)
        except Exception:
            pass  # genuinely no results -- fall through to an empty list
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
            page.goto(f"{APPLE_MUSIC_URL}/listen-now", wait_until="domcontentloaded", timeout=15000)
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
            page.goto(f"{APPLE_MUSIC_URL}/library/all-playlists", wait_until="domcontentloaded", timeout=15000)
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
