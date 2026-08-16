from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any, Callable, TypeVar

from playwright.sync_api import Error as PlaywrightError

log = logging.getLogger("jarvis.browser")


class BrowserUnavailable(RuntimeError):
    pass


class HumanActionRequired(RuntimeError):
    pass


class BrowserSessionError(RuntimeError):
    """The Playwright session was lost (page/context/browser closed or the
    browser process disconnected) and could not be recovered in time to
    complete this specific action. A recovery attempt has already been made
    so the *next* browser action gets a fresh, coherent session."""
    pass


def _is_lifecycle_error(exc: BaseException) -> bool:
    """True when `exc` represents the page/context/browser having been
    closed out from under an in-flight Playwright call, rather than an
    ordinary operation failure (element not found, navigation timeout,
    etc). Playwright sets a stable public `.name` on these errors; the
    message substring is kept as a fallback for versions where `.name`
    isn't populated identically."""
    if not isinstance(exc, PlaywrightError):
        return False
    if getattr(exc, "name", "") == "TargetClosedError":
        return True
    return "has been closed" in str(exc)


@dataclass
class PageState:
    title: str
    url: str
    interactive_elements: list[dict[str, str]] = field(default_factory=list)
    visible_text: str = ""


T = TypeVar("T")


class BrowserAgent:
    """Small persistent Playwright session using semantic locators first.

    Lifecycle contract: every action that touches Playwright state routes
    through `ensure_live_session()`, which guarantees `self.page` is a
    connected browser's open page before any Playwright call is attempted --
    reusing the existing session when healthy, recovering it coherently when
    not. See `ensure_live_session` for the recovery rules and
    `_run_retryable` / `_run_once` for the retry policy applied around each
    action's Playwright calls.
    """

    def __init__(self, headless: bool = False):
        self.headless = headless
        self._playwright = None
        self._browser = None
        self.page = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _is_page_live(self) -> bool:
        """True only when playwright/browser/page all still reference a
        connected, open session.

        `browser.is_connected()` reflects Playwright's own last-known state
        and can lag behind an external process crash until the *next* real
        Playwright call actually fails -- so this is a fast, cheap check for
        the common cases (nothing started yet, explicit close, a closed
        page), not the only safety net. Callers still need to handle a
        lifecycle error raised from the real operation; see
        `_run_retryable`/`_run_once`.
        """
        if self._playwright is None or self._browser is None or self.page is None:
            return False
        try:
            return bool(self._browser.is_connected()) and not self.page.is_closed()
        except Exception:
            return False

    def is_session_live(self) -> bool:
        return self._is_page_live()

    def _discard_session(self) -> None:
        """Drop every Playwright reference together so a half-valid
        combination (e.g. a connected browser paired with an already-closed
        page) can never be observed by a later caller.

        Safe to call on an already-dead session: close()/stop() failures are
        swallowed here, since the objects being torn down are -- by
        definition, given this is only called during automatic recovery --
        no longer usable anyway. This is distinct from the public `close()`,
        which still surfaces a close failure to an explicit caller.
        """
        browser, playwright = self._browser, self._playwright
        self.page = self._browser = self._playwright = None
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        if playwright is not None:
            try:
                playwright.stop()
            except Exception:
                pass

    def _launch(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserUnavailable(
                "Browser automation requires the optional 'playwright' package."
            ) from exc
        self._playwright = sync_playwright().start()
        try:
            executable=os.getenv("JARVIS_BROWSER_EXECUTABLE")
            if not executable:
                from tools.browser import _resolve_chrome
                executable=_resolve_chrome()
            launch_args={"headless":self.headless}
            if executable and Path(executable).is_file():launch_args["executable_path"]=executable
            self._browser = self._playwright.chromium.launch(**launch_args)
            self.page = self._browser.new_page()
        except Exception:
            try:self._playwright.stop()
            finally:self.page=self._browser=self._playwright=None
            raise

    def ensure_live_session(self) -> None:
        """Guarantee `self.page` is a connected browser's open page: reuse
        it if healthy, recover it coherently if not.

        Recovery prefers the cheapest option that still produces a coherent
        state: if the browser process is still connected but only the page
        died (e.g. the user closed that tab), open a fresh page on the
        *same* browser instead of paying for a full relaunch. Only when the
        browser itself is disconnected -- or that cheap recovery also fails
        -- does this fall back to discarding everything and launching a new
        browser. Either path leaves `self.page` fully live; it never leaves
        a stale browser paired with a fresh page or vice versa.
        """
        if self._is_page_live():
            return
        browser_connected = False
        if self._browser is not None:
            try:
                browser_connected = bool(self._browser.is_connected())
            except Exception:
                browser_connected = False
        if browser_connected:
            try:
                self.page = self._browser.new_page()
                return
            except Exception:
                pass
        self._discard_session()
        self._launch()

    def start(self) -> None:
        self.ensure_live_session()

    def close(self) -> None:
        try:
            if self._browser:self._browser.close()
        finally:
            try:
                if self._playwright:self._playwright.stop()
            finally:self.page = self._browser = self._playwright = None

    # ------------------------------------------------------------------
    # Retry policy
    # ------------------------------------------------------------------

    def _run_retryable(self, operation: Callable[[], T]) -> T:
        """For safe/idempotent operations (navigation, read-only queries):
        ensure a live session, run once, and on a lifecycle failure recover
        and retry the *same* operation exactly once. A second lifecycle
        failure is reported truthfully as a `BrowserSessionError` -- never
        retried again, no infinite loops."""
        self.ensure_live_session()
        try:
            return operation()
        except Exception as exc:
            if not _is_lifecycle_error(exc):
                raise
            self._discard_session()
            self.ensure_live_session()
            try:
                return operation()
            except Exception as retry_exc:
                if not _is_lifecycle_error(retry_exc):
                    raise
                raise BrowserSessionError(
                    f"The browser session was lost and recovery did not hold: {retry_exc}"
                ) from retry_exc

    def _run_once(self, operation: Callable[[], T]) -> T:
        """For actions whose side effects or target-page state can't be
        trusted to carry over to a recovered session (a click, a form fill,
        a result-page click, a fullscreen toggle): ensure a live session, run
        once, and on a lifecycle failure recover state for the *next*
        command but report this action as a truthful failure instead of
        blindly repeating an action whose effect on the lost page is
        unknown -- this is what keeps a purchase/submit-style click, or a
        result click, from ever being silently double-fired."""
        self.ensure_live_session()
        try:
            return operation()
        except Exception as exc:
            if not _is_lifecycle_error(exc):
                raise
            self._discard_session()
            self.ensure_live_session()
            raise BrowserSessionError(
                f"The browser session was lost during this action, so nothing was confirmed to happen: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def open_url(self, url: str) -> PageState:
        if not url.startswith(("http://", "https://", "file://")):
            url = "https://" + url
        def _do():
            self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
            return self.get_page_state()
        return self._run_retryable(_do)

    navigate = open_url

    def _locator(self, target: str, kind: str | None = None):
        target = target.strip()
        candidates = []
        if kind:
            candidates.extend([self.page.get_by_role(kind,name=target,exact=True),self.page.get_by_role(kind,name=target,exact=False)])
        candidates.extend([
            self.page.get_by_label(target, exact=True),
            self.page.get_by_label(target, exact=False),
            self.page.get_by_placeholder(target, exact=True),
            self.page.get_by_placeholder(target, exact=False),
            self.page.get_by_role("button", name=target, exact=True),
            self.page.get_by_role("button", name=target, exact=False),
            self.page.get_by_role("link", name=target, exact=True),
            self.page.get_by_role("link", name=target, exact=False),
            self.page.get_by_text(target, exact=True),
            self.page.get_by_text(target, exact=False),
        ])
        ambiguous=False
        for locator in candidates:
            try:
                visible=[locator.nth(index) for index in range(min(locator.count(),20)) if locator.nth(index).is_visible()]
                if len(visible)==1:return visible[0]
                if len(visible)>1:ambiguous=True
            except Exception as exc:
                # A lifecycle error here means the page died mid-scan, not
                # that this particular locator strategy came up empty --
                # it must propagate so the caller's retry policy can see it,
                # not be swallowed as "try the next strategy".
                if _is_lifecycle_error(exc):raise
                continue
        if target.startswith(("#", ".", "[")):
            locator=self.page.locator(target)
            visible=[locator.nth(index) for index in range(min(locator.count(),20)) if locator.nth(index).is_visible()]
            if len(visible)==1:return visible[0]
            if len(visible)>1:raise LookupError(f"Multiple visible elements matched selector {target!r}.")
        if ambiguous:raise LookupError(f"Multiple visible elements matched {target!r}.")
        raise LookupError(f"No visible element matched {target!r}.")

    def find_element(self, target: str, kind: str | None = None) -> dict[str, str]:
        def _do():
            locator = self._locator(target, kind)
            return {"target": target, "text": (locator.inner_text(timeout=1000) or "")[:200]}
        return self._run_retryable(_do)

    def click_element(self, target: str, kind: str | None = None) -> PageState:
        def _do():
            self._locator(target, kind).click(timeout=5000)
            self.page.wait_for_timeout(100)
            self._check_handoff()
            return self.get_page_state()
        return self._run_once(_do)

    def type_into_field(self, target: str, text: str, clear: bool = True) -> bool:
        def _do():
            locator = self._locator(target, "textbox")
            locator.fill(text) if clear else locator.type(text)
            return locator.input_value(timeout=1000)==text if clear else text in locator.input_value(timeout=1000)
        return self._run_once(_do)

    def clear_field(self, target: str) -> None:
        def _do():
            self._locator(target, "textbox").fill("")
        self._run_once(_do)

    def select_option(self, target: str, option: str) -> bool:
        def _do():
            locator=self._locator(target,"combobox");selected=locator.select_option(label=option)
            return bool(selected)
        return self._run_once(_do)

    def press_key(self, key: str) -> None:
        def _do():
            self.page.keyboard.press(key)
        self._run_once(_do)

    def scroll(self, direction: str = "down", amount: int = 700) -> None:
        def _do():
            delta = abs(amount) if direction.lower() == "down" else -abs(amount)
            self.page.mouse.wheel(0, delta)
        self._run_once(_do)

    def open_link(self, target: str) -> PageState:
        return self.click_element(target, "link")

    # ------------------------------------------------------------------
    # "First search result" selection
    #
    # A plain "first visible <a>" is wrong on real result pages: logos,
    # nav bars, filter chips, and channel/account links routinely sit ahead
    # of the actual result in DOM order. Selection here always goes through
    # a provider-aware strategy first (encapsulated below, currently just
    # YouTube) and falls back to a generic, structure-aware heuristic that
    # works without any site-specific knowledge. Neither strategy is
    # allowed to just grab the first anchor tag.
    # ------------------------------------------------------------------

    _NAV_LANDMARK_SELECTOR = "nav, header, footer, [role='navigation'], [role='banner'], [role='contentinfo']"

    def _detect_search_provider(self) -> str | None:
        host = urllib.parse.urlparse(self.page.url).netloc.lower()
        if "youtube.com" in host or "youtu.be" in host:
            return "youtube"
        return None

    def _select_youtube_video_result(self):
        """YouTube-specific: the first visible link whose href is an actual
        video watch destination (/watch?v=<id>). This positively targets
        video results and, as a side effect, excludes channel links
        (/channel/, /@handle), playlists (/playlist), filter chips
        (/results?...), and the home/logo link (/) -- none of those hrefs
        ever match this pattern, so no separate exclusion list is needed."""
        links = self.page.locator('a[href*="/watch?"]')
        count = links.count()
        best = None
        for index in range(min(count, 40)):
            link = links.nth(index)
            try:
                if not link.is_visible():
                    continue
                href = link.get_attribute("href") or ""
            except Exception:
                continue
            if "/watch?" not in href:
                continue
            query = urllib.parse.parse_qs(urllib.parse.urlparse(urllib.parse.urljoin(self.page.url, href)).query)
            if not query.get("v"):
                continue
            try:
                text = (link.inner_text() or "").strip()
            except Exception:
                text = ""
            candidate = (link, href, text)
            if best is None:
                best = candidate
            # The first matching href in DOM order is often a thumbnail or
            # duration overlay ("12:34", "Now playing", or both combined on
            # separate lines) rather than the title link -- same destination
            # either way, but prefer a substantively-labeled candidate so the
            # diagnostic log below is actually useful.
            meaningful = re.sub(r"(?i)\bnow playing\b|[\d:]+", "", text).strip()
            if meaningful:
                return candidate
        return best

    def _generic_result_candidates(self, scope_selector: str):
        locator = self.page.locator(scope_selector)
        count = locator.count()
        for index in range(min(count, 60)):
            link = locator.nth(index)
            try:
                if not link.is_visible():
                    continue
                href = link.get_attribute("href") or ""
                if not href or href.startswith(("javascript:", "#")):
                    continue
                if link.evaluate(f"el => !!el.closest(\"{self._NAV_LANDMARK_SELECTOR}\")"):
                    continue
            except Exception:
                continue
            try:
                text = (link.inner_text() or "").strip()
            except Exception:
                text = ""
            yield link, href, text

    def _select_generic_first_result(self):
        """Provider-agnostic fallback: the first visible, non-navigation
        link, preferring a meaningfully-labeled one inside a main-content
        landmark (main/[role=main]/article) when the page defines one, and
        never a link that sits inside nav/header/footer/banner/contentinfo
        chrome."""
        for scope_selector in ("main a, [role='main'] a, article a", "a"):
            fallback = None
            for candidate in self._generic_result_candidates(scope_selector):
                _link, _href, text = candidate
                if text:
                    return candidate
                if fallback is None:
                    fallback = candidate
            if fallback is not None:
                return fallback
        return None

    @staticmethod
    def _destination_matches(expected_href: str, final_url: str) -> bool:
        """Generic post-click sanity check: a click on a real content link
        must not have silently landed on the site's bare homepage, and must
        not have wandered to a different domain. This is what turns "the URL
        changed" (true even when a result click regresses to the homepage)
        into "the URL changed to where we actually meant to go"."""
        expected = urllib.parse.urlparse(expected_href)
        final = urllib.parse.urlparse(final_url)
        if expected.netloc and final.netloc and expected.netloc.split(":")[0] != final.netloc.split(":")[0]:
            return False
        expected_root = expected.path in ("", "/")
        final_root = final.path in ("", "/")
        if not expected_root and final_root:
            return False
        return True

    def click_first_result(self) -> PageState:
        def _do():
            provider = self._detect_search_provider()
            strategy = None
            selection = None
            if provider == "youtube":
                selection = self._select_youtube_video_result()
                if selection is not None:
                    strategy = "youtube_video_result"
            if selection is None:
                selection = self._select_generic_first_result()
                strategy = "generic_content_link"
            if selection is None:
                raise LookupError("No unambiguous first search result was found on this page.")
            link, href, text = selection
            resolved_href = urllib.parse.urljoin(self.page.url, href)
            log.info(
                "click_first_result selecting strategy=%s href=%r text=%r",
                strategy, resolved_href, text[:120],
            )
            link.click(timeout=5000)
            self.page.wait_for_load_state("domcontentloaded", timeout=10000)
            final_url = self.page.url
            if not self._destination_matches(resolved_href, final_url):
                raise LookupError(
                    f"Clicking the first result (strategy={strategy}, expected {resolved_href!r}) "
                    f"did not navigate to the expected destination; landed on {final_url!r} instead."
                )
            return self.get_page_state()
        return self._run_once(_do)

    def wait_for_element(self, target: str, timeout_ms: int = 5000) -> None:
        def _do():
            self._locator(target).wait_for(state="visible", timeout=timeout_ms)
        self._run_retryable(_do)

    def read_visible_text(self, limit: int = 2000) -> str:
        return self.page.locator("body").inner_text(timeout=3000)[:limit]

    def get_current_url(self) -> str:
        return self.page.url

    def get_page_title(self) -> str:
        return self.page.title()

    def go_back(self) -> PageState:
        def _do():
            self.page.go_back(wait_until="domcontentloaded")
            return self.get_page_state()
        return self._run_once(_do)

    def go_forward(self) -> PageState:
        def _do():
            self.page.go_forward(wait_until="domcontentloaded")
            return self.get_page_state()
        return self._run_once(_do)

    def get_page_state(self) -> PageState:
        elements: list[dict[str, str]] = []
        for role, selector in (("button", "button"), ("link", "a"), ("input", "input"), ("select", "select")):
            locator = self.page.locator(selector)
            for index in range(min(locator.count(), 12)):
                item = locator.nth(index)
                try:
                    if not item.is_visible():
                        continue
                    name = item.get_attribute("aria-label") or item.get_attribute("placeholder") or item.inner_text() or item.get_attribute("name") or ""
                    if name.strip():
                        elements.append({"role": role, "name": name.strip()[:120]})
                except Exception:
                    continue
        return PageState(self.page.title(), self.page.url, elements[:30], self.read_visible_text(1000))

    def _check_handoff(self) -> None:
        text = self.read_visible_text(3000).lower()
        markers = ("captcha", "sms verification", "email verification", "two-factor", "2fa", "passkey", "manual verification")
        found = next((marker for marker in markers if marker in text), None)
        if found:
            raise HumanActionRequired(f"The page requires {found}.")
