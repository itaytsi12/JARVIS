import re

from brain.models import Action
from tools.registry import WEBSITE_ALIASES, APP_ALIASES
import urllib.parse


def create_local_plan(command: str) -> list[Action] | None:
    text = command.strip()

    if not text:
        return None

    parts = re.split(
        r"\s+(?:and then|then|and|ואז|ואחר כך)\s+",
        text,
        flags=re.IGNORECASE
    )

    if len(parts) < 2:
        return None

    actions = []

    for part in parts:
        action = parse_local_action(part.strip())

        if action is None:
            return None

        actions.append(action)

    return actions


def parse_local_action(text: str) -> Action | None:
    lowered = text.lower().strip()

    # -------------------------
    # YouTube search
    # -------------------------

    match = re.match(
        r"^(?:search youtube for|youtube search)\s+(.+)$",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return Action(
            tool="youtube_search",
            args={
                "query": match.group(1).strip()
            }
        )

    # -------------------------
    # Open website
    # -------------------------

    match = re.match(
        r"^(?:open|go to)\s+(https?://\S+)$",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return Action(
            tool="open_website",
            args={
                "url": match.group(1).strip()
            }
        )

    # -------------------------
    # Open application
    # -------------------------

    match = re.match(
        r"^(?:open|launch|start)\s+(.+)$",
        lowered
    )

    if match:
        return Action(
            tool="open_application",
            args={
                "app_name": match.group(1).strip()
            }
        )

    match = re.match(
        r"^(?:תפתח|פתח)\s+(.+)$",
        lowered
    )

    if match:
        return Action(
            tool="open_application",
            args={
                "app_name": match.group(1).strip()
            }
        )

    # -------------------------
    # Close application
    # -------------------------

    match = re.match(
        r"^(?:close|quit)\s+(.+)$",
        lowered
    )

    if match:
        return Action(
            tool="close_application",
            args={
                "app_name": match.group(1).strip()
            }
        )

    match = re.match(
        r"^(?:סגור|תסגור)\s+(.+)$",
        lowered
    )

    if match:
        return Action(
            tool="close_application",
            args={
                "app_name": match.group(1).strip()
            }
        )

    # -------------------------
    # Type text
    # -------------------------

    match = re.match(
        r"^(?:type|write)\s+(.+)$",
        text,
        flags=re.IGNORECASE
    )

    if match:
        typed_text = match.group(1).strip()

        typed_text = remove_optional_quotes(
            typed_text
        )

        return Action(
            tool="type_text",
            args={
                "text": typed_text,
                "delay": 0.02
            }
        )

    # -------------------------
    # Open website aliases (in multi-step commands)
    # -------------------------
    m = re.match(r"^(?:open|go to)\s+(.+)$", text, flags=re.IGNORECASE)
    if m:
        target = m.group(1).strip().lower()

        if target in WEBSITE_ALIASES:
            return Action(
                tool="open_website",
                args={"url": WEBSITE_ALIASES[target]}
            )

    # -------------------------
    # Generic searches in multi-step commands
    # -------------------------
    search_map = {
        r"^(?:search google for|google search|google)\s+(.+)$": "https://www.google.com/search?q={}",
        r"^(?:search youtube for|youtube search|youtube)\s+(.+)$": "https://www.youtube.com/results?search_query={}",
        r"^(?:search reddit for|reddit search|reddit)\s+(.+)$": "https://www.reddit.com/search/?q={}",
        r"^(?:search github for|github search|github)\s+(.+)$": "https://github.com/search?q={}",
    }

    for pattern, template in search_map.items():
        mm = re.match(pattern, text, flags=re.IGNORECASE)
        if mm:
            q = urllib.parse.quote_plus(mm.group(1).strip())
            return Action(
                tool="open_website",
                args={"url": template.format(q)}
            )

    match = re.match(
        r"^(?:תרשום|תכתוב|כתוב)\s+(.+)$",
        text,
        flags=re.IGNORECASE
    )

    if match:
        typed_text = match.group(1).strip()

        typed_text = remove_optional_quotes(
            typed_text
        )

        return Action(
            tool="type_text",
            args={
                "text": typed_text,
                "delay": 0.02
            }
        )

    # -------------------------
    # Press key
    # -------------------------

    match = re.match(
        r"^(?:press|hit)\s+(.+)$",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return Action(
            tool="press_key",
            args={
                "key": match.group(1).strip()
            }
        )

    # -------------------------
    # Volume
    # -------------------------

    if lowered in [
        "volume down",
        "turn volume down",
        "decrease volume",
        "תנמיך",
        "תנמיך ווליום",
    ]:
        return Action(
            tool="volume_down",
            args={
                "amount": 1
            }
        )

    if lowered in [
        "volume up",
        "turn volume up",
        "increase volume",
        "תגביר",
        "תגביר ווליום",
    ]:
        return Action(
            tool="volume_up",
            args={
                "amount": 1
            }
        )

    if lowered in [
        "mute",
        "mute volume",
        "תשתיק",
        "השתק",
    ]:
        return Action(
            tool="mute_volume",
            args={}
        )

    return None


def remove_optional_quotes(text: str) -> str:
    if len(text) >= 2:
        if (
            text.startswith('"')
            and text.endswith('"')
        ):
            return text[1:-1]

        if (
            text.startswith("'")
            and text.endswith("'")
        ):
            return text[1:-1]

    return text