"""Deterministic spoken response formatter.

Provides `format_spoken_response(command, route, response_text)` which returns
a short, user-friendly sentence suitable for TTS while leaving the full
response_text printed to the terminal.

The formatter prefers structured `route` information from `brain.router.route_command()`
and avoids reading raw URLs, PIDs, file paths, JSON, or stack traces aloud.
"""
from __future__ import annotations

import urllib.parse
import re
from typing import Optional

from tools.registry import WEBSITE_ALIASES


def _pretty_site_name(url: str) -> str:
    try:
        p = urllib.parse.urlparse(url)
        host = (p.hostname or url).lower()
        # strip common prefixes
        host = re.sub(r'^www\.', '', host)
        # pick first part before dot
        name = host.split('.')[0]
        return name.capitalize()
    except Exception:
        return url


def _extract_search_query(url: str) -> Optional[str]:
    try:
        p = urllib.parse.urlparse(url)
        q = urllib.parse.parse_qs(p.query)
        # common param names
        for k in ("q", "search_query", "query"):
            if k in q and q[k]:
                return q[k][0]
        return None
    except Exception:
        return None


def _sanitize_for_speech(text: str) -> str:
    # remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # remove file paths like C:\... or /home/... (basic)
    text = re.sub(r'[A-Za-z]:\\\\[^\s]+', '', text)
    text = re.sub(r'/[\w\-_/\.]+', '', text)
    # remove JSON-like braces
    text = re.sub(r'[{}\[\]]', '', text)
    # collapse extra whitespace
    return re.sub(r'\s+', ' ', text).strip()


def format_spoken_response(command: str, route: dict, response_text: str) -> str:
    """Return a short spoken response for the given command/route/result.

    - `command`: original user text
    - `route`: route dict from `route_command(command)`
    - `response_text`: full textual result (printed to terminal)
    """
    # Default fallbacks
    default_done = "Done."
    default_ok = "Okay."

    try:
        rtype = route.get("type") if route else None
    except Exception:
        rtype = None

    # Tool-level routes
    if rtype == "tool":
        tool = route.get("tool")
        args = route.get("arguments", {}) or {}

        if tool == "open_website":
            url = args.get("url") or response_text
            site = _pretty_site_name(url)
            return f"Okay, opening {site}."

        if tool == "open_application":
            app = args.get("app_name") or "application"
            return f"Okay, opening {app.capitalize()}."

        if tool in ("volume_up",):
            return "Turning it up."

        if tool in ("volume_down",):
            return "Turning it down."

        if tool == "mute_volume":
            return "Muted."

        if tool == "take_screenshot":
            return "Screenshot taken."

        if tool == "type_text":
            return default_done

        if tool == "press_key":
            key = args.get("key")
            if key:
                return f"Pressed {key}."
            return default_ok

        # Generic open/close responses
        if response_text and response_text.lower().startswith("opened"):
            # try to infer resource
            m = re.match(r'Opened\s+(https?://\S+|\S+)\s+in', response_text)
            if m:
                site = _pretty_site_name(m.group(1))
                return f"Okay, opening {site}."
            return default_ok

        # Fallback: short sanitized version
        s = _sanitize_for_speech(response_text)
        if s:
            return s if len(s) < 200 else s[:200].rsplit(' ', 1)[0] + '...'

        return default_ok

    # Local multi-step plans: speak once at the end
    if rtype in ("local_plan", "plan", "tools"):
        # If the response_text indicates failure, speak a short failure
        if "error" in (response_text or "").lower():
            return "Something failed while performing the actions."
        return default_done

    # AI responses: speak a short summary (first sentence)
    if rtype == "ai" or (not rtype and response_text):
        # Prefer first sentence
        s = response_text.strip()
        # Don't speak very long answers fully
        if len(s) > 400:
            # speak first ~200 chars but trim cleanly
            short = s[:200].rsplit(' ', 1)[0]
            return short + '...'

        # Return up to the first sentence
        m = re.match(r'(.+?[\.\!?])(\s|$)', s)
        if m:
            return m.group(1).strip()
        return s

    # Default
    return default_ok
