import json
import os
import re
from typing import Optional
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError

from config import events


INTENT_SERVICE_URL = "http://127.0.0.1:5050/predict"
INTENT_SERVICE_TIMEOUT = float(os.getenv("JARVIS_INTENT_SERVICE_TIMEOUT", "0.35"))

MIN_CONFIDENCE = 0.70


# ============================================================
# Known applications
# ============================================================

APP_ALIASES = {
    "notepad": "notepad",
    "פנקס רשימות": "notepad",

    "calculator": "calculator",
    "calc": "calculator",
    "מחשבון": "calculator",

    "vscode": "vscode",
    "vs code": "vscode",
    "visual studio code": "vscode",

    "chrome": "chrome",
    "google chrome": "chrome",
    "כרום": "chrome",

    "spotify": "spotify",
    "ספוטיפיי": "spotify",

    "discord": "discord",
    "דיסקורד": "discord",

    "paint": "paint",
    "צייר": "paint",

    "settings": "settings",
    "הגדרות": "settings",

    "file explorer": "explorer",
    "explorer": "explorer",
    "סייר הקבצים": "explorer",
}


# ============================================================
# Known websites
# ============================================================

WEBSITE_ALIASES = {
    "youtube": "https://www.youtube.com",
    "יוטיוב": "https://www.youtube.com",

    "google": "https://www.google.com",
    "גוגל": "https://www.google.com",

    "github": "https://github.com",
    "גיטהאב": "https://github.com",

    "reddit": "https://www.reddit.com",

    "wikipedia": "https://www.wikipedia.org",
    "ויקיפדיה": "https://www.wikipedia.org",

    "netflix": "https://www.netflix.com",
    "נטפליקס": "https://www.netflix.com",

    "twitch": "https://www.twitch.tv",

    "instagram": "https://www.instagram.com",
    "אינסטגרם": "https://www.instagram.com",
}


# ============================================================
# Intent service
# ============================================================

def predict_local_intent(text: str) -> Optional[dict]:
    payload = json.dumps(
        {
            "text": text,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    request = Request(
        INTENT_SERVICE_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    # UI/status hook (config/events.py). The service being absent is the
    # NORMAL state, not a failure -- a refused connection publishes nothing
    # at all, so the local node stays honestly `offline` instead of
    # flashing an error every time a command is routed without it.
    events.publish(events.MODEL_REQUEST_STARTED, model="local")

    try:
        with urlopen(
            request,
            timeout=INTENT_SERVICE_TIMEOUT,
        ) as response:

            data = json.loads(
                response.read().decode("utf-8")
            )

    except URLError:
        # The service simply is not running -- the documented, supported
        # state (`brain/router.py` degrades gracefully). Reported as a
        # completion, not an error: what matters is that the "started"
        # above is always balanced, so nothing is left showing as in
        # flight forever. The bridge renders an unavailable module as
        # offline regardless, so this never flashes a red node.
        events.publish(events.MODEL_REQUEST_FAILED, model="local", error="service_unavailable")
        return None

    except Exception as exc:
        events.publish(events.MODEL_REQUEST_FAILED, model="local", error=type(exc).__name__)
        print(
            f"[Intent Model] Error: {exc}"
        )
        return None

    if not data.get("success"):
        return None

    events.publish(events.MODEL_REQUEST_SUCCEEDED, model="local")
    return {
        "intent": data.get("intent"),
        "confidence": data.get("confidence"),
    }


# ============================================================
# Utility
# ============================================================

def normalize_text(text: str) -> str:
    return " ".join(
        text.strip().split()
    )


def remove_wake_words(text: str) -> str:
    result = text.strip()

    prefixes = [
        "hey jarvis ",
        "jarvis ",
        "היי jarvis ",
        "היי ג'רוויס ",
        "ג'רוויס ",
    ]

    changed = True

    while changed:
        changed = False

        lowered = result.lower()

        for prefix in prefixes:
            if lowered.startswith(
                prefix.lower()
            ):
                result = result[
                    len(prefix):
                ].strip()

                changed = True
                break

    return result


# ============================================================
# Application extraction
# ============================================================

def extract_app_name(
    text: str,
) -> Optional[str]:

    lowered = text.lower()

    # Prefer explicit known app names.
    matches = []

    for alias, app_name in APP_ALIASES.items():
        position = lowered.find(
            alias.lower()
        )

        if position != -1:
            matches.append(
                (
                    len(alias),
                    app_name,
                )
            )

    if matches:
        # Longer aliases win.
        matches.sort(
            reverse=True,
        )

        return matches[0][1]

    # Generic extraction fallback.
    cleaned = remove_wake_words(text)

    patterns = [
        r"^(?:open|launch|start|run)\s+(.+)$",

        r"^(?:תפתח|פתח|תפעיל|תריץ)\s+(?:לי\s+)?(?:את\s+)?(.+)$",

        r"^תעלה\s+(?:לי\s+)?(?:את\s+)?(.+)$",

        r"^אני רוצה להשתמש ב\s*(.+)$",

        r"^אני צריך(?: את)?\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.match(
            pattern,
            cleaned,
            re.IGNORECASE,
        )

        if match:
            return match.group(1).strip()

    return None


# ============================================================
# Website extraction
# ============================================================

def extract_website(
    text: str,
) -> Optional[str]:

    lowered = text.lower()

    matches = []

    for alias, url in WEBSITE_ALIASES.items():
        position = lowered.find(
            alias.lower()
        )

        if position != -1:
            matches.append(
                (
                    len(alias),
                    url,
                )
            )

    if matches:
        matches.sort(
            reverse=True,
        )

        return matches[0][1]

    # Full URL.
    url_match = re.search(
        r"https?://[^\s]+",
        text,
        re.IGNORECASE,
    )

    if url_match:
        return url_match.group(0)

    # Domain without protocol.
    domain_match = re.search(
        r"\b[\w\-]+\."
        r"(?:com|org|net|io|co|ai|tv)\b",
        text,
        re.IGNORECASE,
    )

    if domain_match:
        return (
            "https://"
            + domain_match.group(0)
        )

    return None


# ============================================================
# YouTube query extraction
# ============================================================

def extract_youtube_query(
    text: str,
) -> Optional[str]:

    cleaned = remove_wake_words(text)
    # Prioritize common English phrasings that place the query before
    # the trailing 'on YouTube' phrase. Use non-greedy capture so we
    # don't swallow extraneous trailing words.
    patterns = [
        # Direct search phrasing
        r"^(?:search youtube for|search for|search)\s+(.+?)\s*(?:on\s+youtube)?\W*$",
        r"^(?:youtube search)\s+(.+?)\W*$",

        # Watch / play / show phrasing
        r"^(?:i want to watch|i wanna watch|i'd like to watch|i would like to watch|i'd love to watch|i want to see|i wanna see)\s+(.+?)\s*(?:on\s+youtube)?\W*$",
        r"^(?:show me|find|play|look up|look for)\s+(.+?)\s*(?:on\s+youtube)?\W*$",

        # Verb-first variants
        r"^(?:play)\s+(.+?)\s*(?:on\s+youtube)?\W*$",

        # Mixed / language-agnostic fallbacks that mention youtube later
        r"^(.+?)\s+on\s+youtube\W*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            query = match.group(1).strip()
            # Remove trailing prepositions or stray words like 'on'
            query = re.sub(r"\b(on|in)\s*$", "", query, flags=re.IGNORECASE).strip()
            # Strip trailing punctuation
            query = query.rstrip('.?!,;:')
            if query:
                return query

    # Preserve existing Hebrew/mixed patterns as a secondary pass
    legacy_patterns = [
        r"תחפש(?: לי)? ביוטיוב\s+(.+)$",
        r"חפש(?: לי)? ביוטיוב\s+(.+)$",
        r"תחפש(?: לי)?\s+(.+?)\s+ביוטיוב$",
        r"חפש(?: לי)?\s+(.+?)\s+ביוטיוב$",
        r"תמצא(?: לי)?\s+(.+?)\s+ביוטיוב$",
        r"שים(?: לי)?\s+(.+?)\s+ביוטיוב$",
        r"תראה(?: לי)?\s+(.+?)\s+ביוטיוב$",
        r"אני רוצה לראות(?: ביוטיוב)?\s+(.+)$",
        r"בא לי לראות\s+(.+?)\s+ביוטיוב$",
        r"בא לי\s+(.+?)\s+ביוטיוב$",
        r"תבדוק ביוטיוב\s+(.+)$",
        r"תמצא סרטונים על\s+(.+)$",
        r"search ביוטיוב\s+(?:for\s+)?(.+)$",
        r"youtube תחפש\s+(.+)$",
        r"תחפש youtube\s+(.+)$",
    ]

    for pattern in legacy_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            query = match.group(1).strip()
            query = query.rstrip('.?!,;:')
            return query or None

    # Final fallback: try removing common intent phrases and youtube mentions
    result = cleaned
    removable_phrases = [
        "search youtube for",
        "search youtube",
        "youtube search",
        "search for",
        "look up",
        "show me",
        "find me",
        "find",
        "search",
        "play",
        "i want to watch",
        "i wanna watch",
        "i'd like to watch",
        "i want to see",
        "i'd love to watch",
        "i would like to watch",
        "look for",
        "show",
        "youtube",
    ]

    for phrase in removable_phrases:
        result = re.sub(re.escape(phrase), " ", result, flags=re.IGNORECASE)

    result = " ".join(result.split()).strip()
    result = result.rstrip('.?!,;:')
    # If result ends with 'on' (leftover from 'on YouTube'), drop it
    result = re.sub(r"\b(on|in)\s*$", "", result, flags=re.IGNORECASE).strip()

    return result or None


# ============================================================
# Intent -> JARVIS action
# ============================================================

def intent_to_action(
    intent: str,
    text: str,
) -> Optional[dict]:

    if intent == "volume_up":
        return {
            "type": "tool",
            "tool": "volume_up",
            "arguments": {
                "amount": 1,
            },
        }

    if intent == "volume_down":
        return {
            "type": "tool",
            "tool": "volume_down",
            "arguments": {
                "amount": 1,
            },
        }

    if intent == "open_application":
        app_name = extract_app_name(
            text
        )

        if not app_name:
            return None

        return {
            "type": "tool",
            "tool": "open_application",
            "arguments": {
                "app_name": app_name,
            },
        }

    if intent == "open_website":
        url = extract_website(
            text
        )

        if not url:
            return None

        return {
            "type": "tool",
            "tool": "open_website",
            "arguments": {
                "url": url,
            },
        }

    if intent == "youtube_search":
        query = extract_youtube_query(
            text
        )

        if not query:
            return None

        url = (
            "https://www.youtube.com/results"
            "?search_query="
            + quote_plus(query)
        )

        return {
            "type": "tool",
            "tool": "open_website",
            "arguments": {
                "url": url,
            },
        }

    return None


# ============================================================
# Main entry point for router
# ============================================================

def route_with_local_model(
    command: str,
) -> Optional[dict]:

    command = normalize_text(
        command
    )

    if not command:
        return None

    prediction = predict_local_intent(
        command
    )

    if prediction is None:
        return None

    intent = prediction.get(
        "intent"
    )

    confidence = prediction.get(
        "confidence"
    )

    confidence_text = (
        f"{confidence:.3f}"
        if confidence is not None
        else "unknown"
    )

    print(
        f"[Intent Model] "
        f"intent={intent} "
        f"confidence={confidence_text}"
    )

    if not intent:
        return None

    if (
        confidence is not None
        and confidence < MIN_CONFIDENCE
    ):
        print(
            "[Intent Model] "
            "Confidence too low."
        )

        return None

    action = intent_to_action(
        intent,
        command,
    )

    if action is None:
        print(
            "[Intent Model] "
            "Intent understood but "
            "arguments could not be extracted."
        )

        return None

    action["route_source"] = "local_learned_classifier"
    return action
