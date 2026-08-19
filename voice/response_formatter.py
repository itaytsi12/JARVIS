"""Deterministic spoken response formatter.

Provides `format_spoken_response(command, route, response_text)` which returns
a short, user-friendly sentence suitable for TTS while leaving the full
response_text printed to the terminal.

The formatter prefers structured `route` information from `brain.router.route_command()`
and avoids reading raw URLs, PIDs, file paths, JSON, or stack traces aloud.
"""
from __future__ import annotations

import random
import urllib.parse
import re
from typing import Optional

from tools.registry import WEBSITE_ALIASES

# Hebrew-mode TTS (Part: language UX rule) -- JARVIS speaks English only,
# always. For an English command that's already guaranteed by
# `format_spoken_response` below (it only ever composes English phrasing
# from English route/tool data). For a HEBREW command, the response would
# otherwise need to speak the recognized Hebrew entity/song/playlist text
# verbatim (or, worse, someone might be tempted to translate/transliterate
# it) -- neither is acceptable, so Hebrew-mode interactions use ONLY one of
# these short, generic, entity-free acknowledgements instead of a
# contextual one. Never derived from the command or response text, so it
# can never leak foreign-language content into TTS.
_GENERIC_ACKNOWLEDGEMENTS = (
    "Okay, on it, sir.",
    "Certainly, sir.",
    "Right away, sir.",
    "On it, sir.",
)

#: Same rule for a failed Hebrew-mode command: one short, generic, English
#: failure message -- never the contextual message a failed English
#: command gets, since that could contain the recognized Hebrew text
#: (e.g. "I couldn't find <Hebrew playlist name>").
_GENERIC_FAILURE_MESSAGE = "I couldn't complete that action, sir."


def generic_acknowledgement() -> str:
    """A short, generic, ALWAYS-English acknowledgement for Hebrew-mode
    commands -- see module note above."""
    return random.choice(_GENERIC_ACKNOWLEDGEMENTS)


def generic_failure_message() -> str:
    """The one short, generic, ALWAYS-English message spoken after a
    Hebrew-mode command fails -- see module note above."""
    return _GENERIC_FAILURE_MESSAGE


_FALLBACK_ACK = "On it, sir."


def _ack_for_tool(tool: str | None, args: dict) -> str | None:
    """Deterministic (no LLM), present-progressive acknowledgement text for
    a route BEFORE it has executed -- reuses the same friendly-name helpers
    `_describe_action`/`format_spoken_response` use for the (unrelated,
    post-execution) completion phrasing, so the two never invent
    conflicting names for the same site/app. Returns None for a tool this
    composer has no specific phrasing for -- callers fall back to
    `_FALLBACK_ACK`."""
    if tool == "open_website":
        url = args.get("url", "")
        query = _extract_search_query(url)
        site = _pretty_site_name(url)
        if query:
            return f"Okay, I'm searching for {urllib.parse.unquote_plus(query)}, sir."
        return f"Opening {site}, sir."
    if tool == "youtube_search":
        query = args.get("query", "")
        return f"Okay, I'm searching for {query}, sir."
    if tool == "open_application":
        return f"Opening {_natural_name(str(args.get('app_name') or 'that'))}, sir."
    if tool in ("close_application", "close_window"):
        return f"Closing {_natural_name(str(args.get('app_name') or 'that'))}, sir."
    if tool == "music_play":
        song, artist, playlist = args.get("song"), args.get("artist"), args.get("playlist")
        intent = args.get("intent")
        if song:
            return f"Playing {song}, sir."
        if artist:
            return f"Playing {artist}, sir."
        if playlist:
            return f"Okay, playing your {playlist} playlist, sir."
        if intent == "PLAY_LAST_PLAYED":
            return "Okay, playing the last song you listened to, sir."
        if intent == "RESUME_LAST_SESSION":
            return "Okay, resuming your music, sir."
        return "Okay, playing music, sir."
    if tool == "open_music":
        return "Opening Apple Music, sir."
    if tool == "music_next":
        return "Okay, next song, sir."
    if tool == "music_previous":
        return "Okay, previous song, sir."
    if tool in ("music_pause",):
        return "Pausing, sir."
    if tool in ("music_resume",):
        return "Resuming, sir."
    if tool in ("music_stop",):
        return "Stopping, sir."
    if tool == "music_now_playing":
        return "Okay, checking, sir."
    if tool == "calculator":
        return "Okay, calculating, sir."
    if tool == "take_screenshot":
        return "Okay, taking a screenshot, sir."
    if tool == "analyze_screen":
        return "Okay, I'll take a look, sir."
    if tool == "volume_up":
        return "Turning the volume up, sir."
    if tool == "volume_down":
        return "Turning the volume down, sir."
    if tool == "mute_volume":
        return "Muting, sir."
    return None


def compose_contextual_ack(route: dict | None) -> str:
    """The immediate, pre-execution English acknowledgement for an ENGLISH
    command (Part 7/9/10 of the bilingual voice UX rule): deterministic,
    derived only from the resolved route/tool -- never an LLM call, never
    claims completion ("Opening YouTube, sir." not "YouTube is open,
    sir."). Hebrew commands never call this -- see `generic_acknowledgement`
    instead, which is entity-free by construction."""
    if not route:
        return _FALLBACK_ACK
    rtype = route.get("type")
    if rtype == "tool":
        return _ack_for_tool(route.get("tool"), route.get("arguments") or {}) or _FALLBACK_ACK
    if rtype in ("plan", "ai"):
        return "I'll check that, sir."
    if rtype == "remember":
        return "I'll remember that, sir."
    if rtype == "agent_task":
        return "I'll work on that, sir."
    if rtype == "local_plan":
        actions = route.get("actions") or []
        if actions:
            first_tool, first_args = _action_parts(actions[0])
            ack = _ack_for_tool(first_tool, first_args)
            if ack:
                return ack
        return "Okay, I'll get started, sir."
    return _FALLBACK_ACK


def _pretty_site_name(url: str) -> str:
    try:
        p = urllib.parse.urlparse(url)
        host = (p.hostname or url).lower()
        # strip common prefixes
        host = re.sub(r'^www\.', '', host)
        # pick first part before dot
        name = host.split('.')[0]
        return _natural_name(name)
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


def _natural_name(value: str) -> str:
    names = {
        "youtube": "YouTube",
        "google": "Google",
        "github": "GitHub",
        "reddit": "Reddit",
        "notepad": "Notepad",
        "calculator": "Calculator",
    }
    return names.get(value.strip().lower(), value.strip())


def _strip_error_suffix(text: str) -> str:
    """Strip a trailing "\\nError: <code>" suffix (appended by
    `brain/agent.py`'s tool-route failure path for logs/terminal output,
    never meant for TTS) before a message is spoken."""
    return re.split(r"\n\s*Error:\s*", text or "", maxsplit=1)[0].strip()


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


def _action_parts(action) -> tuple[Optional[str], dict]:
    if isinstance(action, dict):
        return action.get("tool"), action.get("arguments") or action.get("args") or {}
    return getattr(action, "tool", None), getattr(action, "args", {}) or {}


def _describe_action(action, lang: str) -> Optional[str]:
    tool, args = _action_parts(action)
    if tool == "open_website":
        url = args.get("url", "")
        query = _extract_search_query(url)
        site = _pretty_site_name(url)
        if query:
            query = urllib.parse.unquote_plus(query)
            return f"חיפשתי ב{site} את {query}" if lang == "he" else f"searched {site} for {query}"
        return f"פתחתי את {site}" if lang == "he" else f"opened {site}"
    if tool == "youtube_search":
        query = args.get("query", "")
        return f"חיפשתי ביוטיוב את {query}" if lang == "he" else f"searched YouTube for {query}"
    if tool == "open_application":
        app = _natural_name(str(args.get("app_name") or "the application"))
        return f"פתחתי את {app}" if lang == "he" else f"opened {app}"
    if tool == "close_application":
        app = _natural_name(str(args.get("app_name") or "the application"))
        return f"סגרתי את {app}" if lang == "he" else f"closed {app}"
    if tool == "type_text":
        typed = str(args.get("text") or "the text")
        return f"הקלדתי {typed}" if lang == "he" else f"typed {typed}"
    if tool == "press_key":
        key = str(args.get("key") or "the key")
        return f"לחצתי על {key}" if lang == "he" else f"pressed {key}"
    if tool == "volume_up":
        return "הגברתי את עוצמת הקול" if lang == "he" else "turned the volume up"
    if tool == "volume_down":
        return "הנמכתי את עוצמת הקול" if lang == "he" else "turned the volume down"
    if tool == "mute_volume":
        return "השתקתי את הקול" if lang == "he" else "muted the volume"
    if tool == "send_whatsapp_message":
        recipient=_natural_name(str(args.get("recipient") or "the contact"))
        return f"sent the WhatsApp message to {recipient}"
    return None


EXECUTABLE_ROUTE_TYPES = {"tool", "local_plan", "plan", "tools"}


def _join_descriptions(descriptions: list[str], lang: str) -> str:
    if len(descriptions) == 1:
        joined = descriptions[0]
    elif len(descriptions) == 2:
        joined = f"{descriptions[0]} {'ו' if lang == 'he' else 'and '}{descriptions[1]}"
    else:
        conjunction = "ו" if lang == "he" else "and "
        joined = ", ".join(descriptions[:-1]) + f", {conjunction}{descriptions[-1]}"
    return f"{joined}." if lang == "he" else f"I {joined}."


def _format_spoken_response_base(command: str, route: dict, response_text: str, lang: str | None = None, execution: dict | None = None) -> str:
    """Return a short spoken response for the given command/route/result.

    - `command`: original user text
    - `route`: route dict from `route_command(command)`
    - `response_text`: full textual result (printed to terminal)
    - `execution`: real execution outcome from `run_agent(..., execution_outcome=...)`,
      describing what actually ran rather than what `route` merely planned.
    """
    # Default fallbacks
    default_done = "Done."
    default_ok = "Okay."

    try:
        rtype = route.get("type") if route else None
    except Exception:
        rtype = None

    if rtype == "question":
        return _sanitize_for_speech(response_text)

    if rtype in {"cancel_read_only_task","task_status"}:
        return response_text
    if rtype in {"resume_interrupted_response","correct_interrupted_response"}:
        return _sanitize_for_speech(response_text)
    if rtype in {"agent_task", "remember"}:
        # The agent runtime already produces one short, user-facing English
        # sentence (brain/agent_loop.py's system prompt requires exactly
        # that), and it is the only description of what actually happened
        # -- flattening it to "Okay." below would throw away the entire
        # answer. It is still sanitized, so URLs/JSON never reach TTS.
        return _sanitize_for_speech(response_text) or "Done."

    # If language not provided, attempt to detect from command
    if lang is None:
        try:
            from .language_utils import detect_dominant_language

            lang = detect_dominant_language(command)
        except Exception:
            lang = 'en'

    # No structured execution evidence = no success response, regardless of
    # whether response_text happens to contain a recognized failure keyword.
    no_evidence = rtype in EXECUTABLE_ROUTE_TYPES and execution is not None and execution.get("executed") is False
    if no_evidence:
        sanitized = _sanitize_for_speech(response_text or "")
        if sanitized:
            return sanitized
        return "I didn't perform any actions." if lang != 'he' else "לא ביצעתי שום פעולה."

    failure_text = (response_text or "").lower()
    # Music tools (brain/music_intent.py's `open_music`/`music_*` routes)
    # deliberately craft short, specific, TTS-safe failure/status sentences
    # ("Apple Music needs you to sign in, sir.", "I couldn't find that
    # playlist, sir.") -- speaking THOSE is the whole point (Part 20/25 of
    # the music feature). The generic keyword-based flattening below exists
    # for tools whose raw message/error text is not meant for TTS; it must
    # not swallow a message a tool specifically wrote to be spoken.
    tool_name = route.get("tool") if rtype == "tool" else None
    speaks_for_itself = isinstance(tool_name, str) and (tool_name == "open_music" or tool_name.startswith("music_"))
    if rtype in EXECUTABLE_ROUTE_TYPES and not speaks_for_itself and any(marker in failure_text for marker in ("failed", "couldn't", "could not", "error:")):
        return "I couldn't complete that action."

    # Tool-level routes
    if rtype == "tool":
        tool = route.get("tool")
        args = route.get("arguments", {}) or {}

        if tool == "open_website":
            url = args.get("url") or response_text
            site = _pretty_site_name(url)
            # If this is a search URL with a query, make the spoken reply a search phrase
            q = _extract_search_query(url)
            # Prefer extracting query from the original command (preserve case)
            query_text = None
            if q:
                # try to extract original-case query from `command` using simple patterns
                try:
                    patterns = [r'(?i)search youtube for (.+)', r'(?i)youtube search (.+)', r'(?i)youtube (.+)',
                                r'(?i)search google for (.+)', r'(?i)google (.+)']
                    for pat in patterns:
                        m = re.search(pat, command)
                        if m:
                            query_text = m.group(1).strip()
                            break
                except Exception:
                    query_text = None

                if not query_text:
                    # fallback to URL-decoded q
                    try:
                        query_text = urllib.parse.unquote_plus(q)
                    except Exception:
                        query_text = q

            if query_text:
                if 'youtube' in url:
                    if lang == 'he':
                        return f"פתחתי את יוטיוב וחיפשתי את {query_text}."
                    return f"I opened YouTube and searched for {query_text}."
                if 'google' in url:
                    if lang == 'he':
                        return f"פתחתי את גוגל וחיפשתי את {query_text}."
                    return f"I opened Google and searched for {query_text}."

            if lang == 'he':
                return f"פתחתי את {site}."
            return f"I opened {site}."

        if tool == "open_application":
            app = args.get("app_name") or "application"
            if lang == 'he':
                return f"פתחתי את {_natural_name(app)}."
            return f"I opened {_natural_name(app)}."

        if tool in ("volume_up",):
            return "Turning it up." if lang != 'he' else "מגביר."

        if tool in ("volume_down",):
            return "Turning it down." if lang != 'he' else "מנמיך."

        if tool == "mute_volume":
            return "Muted." if lang != 'he' else "השתקתי."

        if tool == "take_screenshot":
            return "Screenshot taken." if lang != 'he' else "תצלום מסך נלקח."

        if tool == "type_text":
            return default_done if lang != 'he' else "בוצע."

        if tool == "press_key":
            key = args.get("key")
            if key:
                if lang == 'he':
                    return f"הוקש {key}."
                return f"Pressed {key}."
            return default_ok if lang != 'he' else "בסדר."

        # NOTE: a former "Generic open/close responses" block used to live
        # here, matching any message starting with "opened" against the
        # `open_website` tool's specific "Opened <url> in <browser>."
        # format. Both `open_website` and `open_application` already return
        # from their own dedicated branches above, so that block was
        # unreachable for its intended purpose -- it only ever fired for
        # OTHER tools whose message happened to start with "Opened" (e.g.
        # music's "Opened Apple Music, sir."), silently flattening them to
        # "Okay." Removed rather than reused, since the fallback below
        # already speaks any such message correctly.

        # Fallback: short sanitized version. Tool-route failures append a
        # technical "\nError: <code>" suffix meant for logs/terminal output
        # (see brain/agent.py's `route["type"]=="tool"` failure branch) --
        # strip it before speaking so a clean message like "Apple Music
        # needs you to sign in, sir." doesn't have a raw error code read
        # aloud right after it.
        s = _sanitize_for_speech(_strip_error_suffix(response_text))
        if s:
            return s if len(s) < 200 else s[:200].rsplit(' ', 1)[0] + '...'

        return default_ok

    # Local multi-step plans: speak once at the end
    if rtype in ("local_plan", "plan", "tools"):
        # If the response_text indicates failure, speak a short failure
        if "error" in (response_text or "").lower():
            return "Something failed while performing the actions." if lang != 'he' else "שגיאה בביצוע הפעולות."
        # If there is no structured execution evidence (empty result text),
        # do not speak a success message based solely on the planned actions.
        # This enforces: No structured execution evidence = no success response.
        if not (response_text or "").strip():
            return "I didn't perform any actions." if lang != 'he' else "לא ביצעתי שום פעולה."
        if execution is not None:
            successful = [item for item in execution.get("actions", []) if item.get("success")]
            descriptions = [d for item in successful if (d := _describe_action(item, lang))]
            if descriptions:
                joined = _join_descriptions(descriptions, lang)
                if execution.get("partial"):
                    if lang == 'he':
                        return f"רק חלק מהפעולות הושלמו: {joined}"
                    return joined[:-1] + " -- the rest of that request did not complete."
                return joined
            sanitized = _sanitize_for_speech(response_text or "")
            if sanitized:
                return sanitized[:240]
            return default_done if lang != 'he' else "בוצע."
        # No execution evidence supplied: never fall back to route["actions"] --
        # that is exactly the stale-route pattern that caused the bug. Speak
        # the real response text instead.
        sanitized = _sanitize_for_speech(response_text or "")
        if sanitized:
            return sanitized[:240]
        return default_done if lang != 'he' else "בוצע."

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


def _address_as_sir(text:str) -> str:
    """Add one natural sentence-final form of address to final spoken text only."""
    spoken=str(text or "").strip()
    if not spoken or re.search(r"(?i)\bsir\s*[.!?…]*$",spoken):return spoken
    if spoken.endswith("...") or spoken.endswith("…"):return f"{spoken} Sir."
    punctuation=spoken[-1] if spoken[-1] in ".!?" else "."
    stem=spoken[:-1].rstrip() if spoken[-1] in ".!?" else spoken.rstrip(" ,")
    return f"{stem}, sir{punctuation}"


def format_spoken_response(command: str, route: dict, response_text: str, lang: str | None = None, execution: dict | None = None) -> str:
    return _address_as_sir(_format_spoken_response_base(command,route,response_text,lang,execution))
