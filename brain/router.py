import re

from brain.intent_router import classify_intent
from brain.local_planner import create_local_plan
from tools.registry import WEBSITE_ALIASES, APP_ALIASES
import urllib.parse
from brain.local_intent_model import route_with_local_model

def looks_like_math(text: str) -> bool:
    math_pattern = r"^[\d\s\.\+\-\*\/\%\(\)]+$"

    return bool(
        re.fullmatch(
            math_pattern,
            text
        )
    )


def clean_math_command(text: str) -> str:
    text = text.lower().strip()

    prefixes = [
        "what is ",
        "what's ",
        "calculate ",
        "compute ",
        "כמה זה ",
        "חשב ",
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]

    text = text.replace(
        "?",
        ""
    )

    return text.strip()


def route_command(command: str) -> dict:
    text = command.lower().strip()

    # -------------------------
    # Calculator
    # -------------------------

    math_expression = clean_math_command(text)

    if looks_like_math(math_expression):
        return {
            "type": "tool",
            "tool": "calculator",
            "arguments": {
                "expression": math_expression
            }
        }

    # -------------------------
    # Volume
    # -------------------------

    if text in [
        "mute",
        "mute volume",
        "volume mute",
        "השתק",
        "תשתיק"
    ]:
        return {
            "type": "tool",
            "tool": "mute_volume",
            "arguments": {}
        }

    if text in [
        "volume up",
        "turn volume up",
        "increase volume",
        "תגביר",
        "תגביר ווליום"
    ]:
        return {
            "type": "tool",
            "tool": "volume_up",
            "arguments": {
                "amount": 1
            }
        }

    if text in [
        "volume down",
        "turn volume down",
        "decrease volume",
        "תנמיך",
        "תנמיך ווליום"
    ]:
        return {
            "type": "tool",
            "tool": "volume_down",
            "arguments": {
                "amount": 1
            }
        }

    # -------------------------
    # Screenshot
    # -------------------------

    if text in [
        "take a screenshot",
        "screenshot",
        "take screenshot",
        "צלם מסך",
        "תצלם מסך"
    ]:
        return {
            "type": "tool",
            "tool": "take_screenshot",
            "arguments": {}
        }

    # -------------------------
    # Analyze screen
    # -------------------------

    screen_phrases = [
        "what is on my screen",
        "what's on my screen",
        "look at my screen",
        "analyze my screen",
        "מה יש לי על המסך",
        "תסתכל על המסך",
    ]

    if text in screen_phrases:
        return {
            "type": "tool",
            "tool": "analyze_screen",
            "arguments": {
                "question": command
            }
        }

    # -------------------------
    # Active window
    # -------------------------

    if text in [
        "what app am i using",
        "what window am i in",
        "what is my active window",
        "where am i",
        "באיזה תוכנה אני",
        "איזה חלון פתוח"
    ]:
        return {
            "type": "tool",
            "tool": "active_window",
            "arguments": {}
        }

    # -------------------------
    # Open websites
    # -------------------------

    website_patterns = [
        r"^(open|go to)\s+(https?://\S+)$",
        r"^(open|go to)\s+([\w\-]+\.(?:com|org|net|io|co|ai))$",
    ]

    for pattern in website_patterns:
        match = re.match(
            pattern,
            text
        )

        if match:
            return {
                "type": "tool",
                "tool": "open_website",
                "arguments": {
                    "url": match.group(2).strip()
                }
            }

        # -------------------------
        # Website aliases (immediate local open)
        # -------------------------
        if text.startswith("open ") or text.startswith("go to "):
            # extract target
            parts = text.split(None, 1)
            if len(parts) > 1:
                target = parts[1].strip()
                # normalize: remove trailing punctuation that may come from STT
                target = target.rstrip(".?!,;:")

                # normalize key for alias lookup
                key = target.rstrip('/').lower()
                if key in WEBSITE_ALIASES:
                    return {
                        "type": "tool",
                        "tool": "open_website",
                        "arguments": {"url": WEBSITE_ALIASES[key]}
                    }

                # Open known folders (downloads, documents, desktop)
                known_folders = {"downloads", "documents", "desktop", "pictures", "music", "videos"}
                if key in known_folders:
                    return {"type": "tool", "tool": "open_folder", "arguments": {"name": target}}

                # Application aliases
                if key in APP_ALIASES:
                    return {"type": "tool", "tool": "open_application", "arguments": {"app_name": APP_ALIASES[key]}}

                # Browser navigation shortcuts (single-word commands)
                nav_map = {
                    "refresh": {"tool": "press_key", "args": {"key": "f5"}},
                    "reload": {"tool": "press_key", "args": {"key": "f5"}},
                    "new tab": {"tool": "press_key", "args": {"key": "ctrl+t"}},
                    "close tab": {"tool": "press_key", "args": {"key": "ctrl+w"}},
                    "reopen closed tab": {"tool": "press_key", "args": {"key": "ctrl+shift+t"}},
                    "next tab": {"tool": "press_key", "args": {"key": "ctrl+tab"}},
                    "previous tab": {"tool": "press_key", "args": {"key": "ctrl+shift+tab"}},
                    "go back": {"tool": "press_key", "args": {"key": "alt+left"}},
                    "go forward": {"tool": "press_key", "args": {"key": "alt+right"}},
                }

                if key in nav_map:
                    mapping = nav_map[key]
                    return {"type": "tool", "tool": mapping["tool"], "arguments": mapping.get("args", {})}

        # -------------------------
        # YouTube / Google / Reddit / GitHub searches (single command)
        # -------------------------
        search_patterns = [
            (r"^(?:search google for|google search|google)\s+(.+)$", "https://www.google.com/search?q={}"),
            (r"^(?:search youtube for|youtube search|youtube)\s+(.+)$", "https://www.youtube.com/results?search_query={}"),
            (r"^(?:search reddit for|reddit search|reddit)\s+(.+)$", "https://www.reddit.com/search/?q={}"),
            (r"^(?:search github for|github search|github)\s+(.+)$", "https://github.com/search?q={}"),
        ]

        for pattern, url_template in search_patterns:
            m = re.match(pattern, text, flags=re.IGNORECASE)
            if m:
                q = urllib.parse.quote_plus(m.group(1).strip())
                return {
                    "type": "tool",
                    "tool": "open_website",
                    "arguments": {"url": url_template.format(q)}
                }

        # (redundant YouTube direct-match removed — handled by search_patterns above)

    # -------------------------
    # Single-keyboard / system commands
    # -------------------------
    single_map = {
        "refresh": {"tool": "press_key", "arguments": {"key": "f5"}},
        "reload": {"tool": "press_key", "arguments": {"key": "f5"}},
        "new tab": {"tool": "press_key", "arguments": {"key": "ctrl+t"}},
        "close tab": {"tool": "press_key", "arguments": {"key": "ctrl+w"}},
        "reopen closed tab": {"tool": "press_key", "arguments": {"key": "ctrl+shift+t"}},
        "next tab": {"tool": "press_key", "arguments": {"key": "ctrl+tab"}},
        "previous tab": {"tool": "press_key", "arguments": {"key": "ctrl+shift+tab"}},
        "go back": {"tool": "press_key", "arguments": {"key": "alt+left"}},
        "go forward": {"tool": "press_key", "arguments": {"key": "alt+right"}},
        "show desktop": {"tool": "show_desktop", "arguments": {}},
        "open task manager": {"tool": "open_task_manager", "arguments": {}},
        "minimize window": {"tool": "minimize_window", "arguments": {}},
        "maximize window": {"tool": "maximize_window", "arguments": {}},
        "restore window": {"tool": "restore_window", "arguments": {}},
        "close window": {"tool": "close_window", "arguments": {}},
    }

    if text in single_map:
        m = single_map[text]
        return {"type": "tool", "tool": m["tool"], "arguments": m.get("arguments", {})}

    # Single press command pattern e.g. "press enter" or "press ctrl s"
    m_press = re.match(r"^(?:press|hit)\s+(.+)$", text, flags=re.IGNORECASE)
    if m_press:
        key = m_press.group(1).strip()
        return {"type": "tool", "tool": "press_key", "arguments": {"key": key}}

    # Switch/focus to application
    m_switch = re.match(r"^(?:switch to|focus)\s+(.+)$", text, flags=re.IGNORECASE)
    if m_switch:
        app = m_switch.group(1).strip().lower()
        # If an alias exists, resolve it
        if app in APP_ALIASES:
            app = APP_ALIASES[app]
        return {"type": "tool", "tool": "focus_application", "arguments": {"app_name": app}}

    # -------------------------
    # Local multi-step plan
    # -------------------------

    local_plan = create_local_plan(
        command
    )

    if local_plan:
        return {
            "type": "local_plan",
            "actions": local_plan
        }

    # -------------------------
    # Multi-step AI fallback
    # -------------------------

    multi_step_words = [
        " and ",
        " then ",
        " and then ",
        " ואז ",
        " ואחר כך ",
    ]

    if any(
        word in text
        for word in multi_step_words
    ):
        return {
            "type": "plan",
            "message": command
        }

    # -------------------------
    # Open applications
    # -------------------------

    open_patterns = [
        r"^(open|launch|start)\s+(.+)$",
        r"^(תפתח|פתח)\s+(.+)$",
    ]

    for pattern in open_patterns:
        match = re.match(
            pattern,
            text
        )

        if match:
            return {
                "type": "tool",
                "tool": "open_application",
                "arguments": {
                    "app_name": match.group(2).strip()
                }
            }

    # -------------------------
    # Close applications
    # -------------------------

    close_patterns = [
        r"^(close|quit)\s+(.+)$",
        r"^(סגור|תסגור)\s+(.+)$",
    ]

    for pattern in close_patterns:
        match = re.match(
            pattern,
            text
        )

        if match:
            return {
                "type": "tool",
                "tool": "close_application",
                "arguments": {
                    "app_name": match.group(2).strip()
                }
            }

        # -------------------------
    # Local trained intent model
    # -------------------------

    local_model_action = route_with_local_model(command)

    if local_model_action is not None:
        return local_model_action
        

    # -------------------------
    # AI / Intent fallback
    # -------------------------

    # -------------------------
    # Time / Date local queries
    # -------------------------
    if text in [
        "what time is it",
        "what is the time",
        "time",
        "what time",
    ]:
        return {"type": "tool", "tool": "get_time", "arguments": {}}

    if text in [
        "what is today's date",
        "what is the date",
        "what day is it",
        "date",
    ]:
        return {"type": "tool", "tool": "get_date", "arguments": {}}


    return classify_intent(
        command
    )