from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any


class BrowserUnavailable(RuntimeError):
    pass


class HumanActionRequired(RuntimeError):
    pass


@dataclass
class PageState:
    title: str
    url: str
    interactive_elements: list[dict[str, str]] = field(default_factory=list)
    visible_text: str = ""


class BrowserAgent:
    """Small persistent Playwright session using semantic locators first."""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self._playwright = None
        self._browser = None
        self.page = None

    def start(self) -> None:
        if self.page is not None:
            return
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

    def close(self) -> None:
        try:
            if self._browser:self._browser.close()
        finally:
            try:
                if self._playwright:self._playwright.stop()
            finally:self.page = self._browser = self._playwright = None

    def open_url(self, url: str) -> PageState:
        self.start()
        if not url.startswith(("http://", "https://", "file://")):
            url = "https://" + url
        self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
        return self.get_page_state()

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
            except Exception:
                continue
        if target.startswith(("#", ".", "[")):
            locator=self.page.locator(target)
            visible=[locator.nth(index) for index in range(min(locator.count(),20)) if locator.nth(index).is_visible()]
            if len(visible)==1:return visible[0]
            if len(visible)>1:raise LookupError(f"Multiple visible elements matched selector {target!r}.")
        if ambiguous:raise LookupError(f"Multiple visible elements matched {target!r}.")
        raise LookupError(f"No visible element matched {target!r}.")

    def find_element(self, target: str, kind: str | None = None) -> dict[str, str]:
        locator = self._locator(target, kind)
        return {"target": target, "text": (locator.inner_text(timeout=1000) or "")[:200]}

    def click_element(self, target: str, kind: str | None = None) -> PageState:
        self._locator(target, kind).click(timeout=5000)
        self.page.wait_for_timeout(100)
        self._check_handoff()
        return self.get_page_state()

    def type_into_field(self, target: str, text: str, clear: bool = True) -> bool:
        locator = self._locator(target, "textbox")
        locator.fill(text) if clear else locator.type(text)
        return locator.input_value(timeout=1000)==text if clear else text in locator.input_value(timeout=1000)

    def clear_field(self, target: str) -> None:
        self._locator(target, "textbox").fill("")

    def select_option(self, target: str, option: str) -> bool:
        locator=self._locator(target,"combobox");selected=locator.select_option(label=option)
        return bool(selected)

    def press_key(self, key: str) -> None:
        self.page.keyboard.press(key)

    def scroll(self, direction: str = "down", amount: int = 700) -> None:
        delta = abs(amount) if direction.lower() == "down" else -abs(amount)
        self.page.mouse.wheel(0, delta)

    def open_link(self, target: str) -> PageState:
        return self.click_element(target, "link")

    def click_first_result(self) -> PageState:
        selectors = ["main a", "#search a", "a:visible"]
        for selector in selectors:
            links = self.page.locator(selector)
            for index in range(min(links.count(), 20)):
                link = links.nth(index)
                href = link.get_attribute("href") or ""
                if href and not href.startswith(("javascript:", "#")) and link.is_visible():
                    link.click(timeout=5000)
                    self.page.wait_for_load_state("domcontentloaded", timeout=10000)
                    return self.get_page_state()
        raise LookupError("No usable search result link was found.")

    def wait_for_element(self, target: str, timeout_ms: int = 5000) -> None:
        self._locator(target).wait_for(state="visible", timeout=timeout_ms)

    def read_visible_text(self, limit: int = 2000) -> str:
        return self.page.locator("body").inner_text(timeout=3000)[:limit]

    def get_current_url(self) -> str:
        return self.page.url

    def get_page_title(self) -> str:
        return self.page.title()

    def go_back(self) -> PageState:
        self.page.go_back(wait_until="domcontentloaded")
        return self.get_page_state()

    def go_forward(self) -> PageState:
        self.page.go_forward(wait_until="domcontentloaded")
        return self.get_page_state()

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
