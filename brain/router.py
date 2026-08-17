import re
import time

from brain.intent_router import classify_intent
from brain.local_planner import create_local_plan
from tools.registry import WEBSITE_ALIASES, APP_ALIASES
import urllib.parse
from brain.local_intent_model import route_with_local_model
from brain.request_intent import RequestKind, classify_request_kind

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

    if text.rstrip(".?!,;:") in {
        "what are you doing",
        "what tasks are running",
        "are any tasks running",
        "what is running",
    }:
        return {"type":"task_status"}

    # Highest-priority deterministic control command. This must never reach an
    # intent model because it exists specifically to interrupt active work.
    if text.rstrip(".?!,;:") in {
        "cancel",
        "cancel that",
        "stop that",
        "stop the current task",
        "stop",
        "never mind",
    }:
        return {"type": "cancel_read_only_task"}
    if text.rstrip(".?!,;:") == "continue":
        return {"type":"resume_interrupted_response"}

    # Deterministic voice-approved continual learning commands (never routed
    # through the local intent model or the cloud planner -- see
    # brain/learning_orchestrator.py::start_learning and
    # voice/background_assistant.py's dispatch of these two route types).
    if text.rstrip(".?!,;:") in {"start learning", "start the learning", "begin learning"}:
        return {"type": "start_learning"}
    if text.rstrip(".?!,;:") in {"stop learning", "stop the learning", "cancel learning", "cancel the learning"}:
        return {"type": "stop_learning"}
    if text.rstrip(".?!,;:") in {"learning status", "what is the learning status", "what's the learning status", "status of learning"}:
        return {"type": "learning_status"}
    if re.fullmatch(r"(?:make it (?:much )?shorter|shorten that|only (?:tell|give) me (?:the )?(?:top )?\d+)",text.rstrip(".?!,;:")):
        return {"type":"correct_interrupted_response","instruction":text.rstrip(".?!,;:")}
    recipient_correction=re.fullmatch(r"(?:don't send it to \S+,\s*)?send it to (.+?) instead",text.rstrip(".?!;:"))
    if recipient_correction:return {"type":"revise_whatsapp_recipient","recipient":recipient_correction.group(1).strip()}

    # Polite wrappers do not change a concrete computer-control request into
    # a question. Strip them only when followed by a known action verb so the
    # existing deterministic routes remain in control.
    text = re.sub(
        r"^(?:can|could|would) you(?: please)?\s+(?=(?:open|launch|start|close|mute|turn|increase|decrease)\b)",
        "",
        text,
    )
    text = re.sub(r"^please\s+(?=(?:open|launch|start|close|mute|turn|increase|decrease)\b)","",text)
    if re.match(r"^(?:open|launch|start|close|mute|turn|increase|decrease)\b",text):
        text = re.sub(r"\s+(?:please|for me|if possible)[.?!,;:]*$","",text).strip()

    # Natural browser phrasing sometimes produced by voice transcription.
    website_on_browser = re.fullmatch(
        r"open\s+(youtube|tiktok)(?:\s+(?:on|in)\s+(?:google|google chrome|chrome))?[.?!,;:]*",
        text,
    )
    if website_on_browser:
        website = website_on_browser.group(1)
        return {
            "type": "tool",
            "tool": "open_website",
            "arguments": {"url": WEBSITE_ALIASES[website]},
        }

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
        "turn the volume up",
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
        "turn the volume down",
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

    inspect_match=re.fullmatch(r"(?:inspect|list (?:the )?controls in|what controls are in)\s+(.+?)[.?!]?",text,re.I)
    if inspect_match:
        app=inspect_match.group(1).strip().lower();app=APP_ALIASES.get(app,app)
        return {"type":"tool","tool":"inspect_window","arguments":{"app_name":app,"limit":50}}

    click_match=re.fullmatch(r"click\s+(?:the\s+)?(.+?)(?:\s+(button|link|menu item))?\s+(?:in|on)\s+(.+?)[.?!]?",command.strip(),re.I)
    if click_match:
        name,control_type,app=click_match.groups();app=APP_ALIASES.get(app.strip().lower(),app.strip().lower())
        arguments={"app_name":app,"name":name.strip()}
        if control_type:arguments["control_type"]={"button":"Button","link":"Hyperlink","menu item":"MenuItem"}[control_type.lower()]
        return {"type":"tool","tool":"click_ui_element","arguments":arguments}

    # Literal typing is data, not a recursively routable command. References
    # to prior assistant output are handled earlier by the task planner.
    m_type = re.match(r"^(?:type|write)\s+(.+)$", command.strip(), flags=re.IGNORECASE)
    if m_type:
        payload=m_type.group(1).strip().rstrip(".?!")
        if len(payload)>=2 and payload[0]==payload[-1] and payload[0] in {'"',"'"}:payload=payload[1:-1]
        return {"type":"tool","tool":"type_text","arguments":{"text":payload,"delay":.02}}

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

    local_plan = create_local_plan(text)

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
    # Existing local time/date queries remain local and never need web search.
    if text in ["what time is it", "what is the time", "time", "what time"]:
        return {"type": "tool", "tool": "get_time", "arguments": {}}
    if text in ["what is today's date", "what is the date", "what day is it", "date"]:
        return {"type": "tool", "tool": "get_date", "arguments": {}}

    # Read-only informational question path. Action-like polite requests stay
    # with the existing local intent/action router below.
    classification_started = time.perf_counter()
    request_kind = classify_request_kind(command)
    classification_ms = (time.perf_counter() - classification_started) * 1000
    if request_kind.kind is RequestKind.QUESTION:
        return {
            "type": "question",
            "message": command,
            "confidence": request_kind.confidence,
            "intent_classification_ms": classification_ms,
        }

    # Local trained intent model
    # -------------------------

    local_model_action = route_with_local_model(command)

    if local_model_action is not None:
        return local_model_action
        

    # -------------------------
    # AI / Intent fallback
    # -------------------------

    fallback=classify_intent(command)
    fallback.setdefault("route_source","cloud_intent_router")
    fallback.setdefault("model","gpt-5-mini")
    fallback.setdefault("model_calls",1)
    fallback.setdefault("fallback_from",["local_learned_classifier"])
    fallback.setdefault("fallback_reason","no_confident_local_route")
    return fallback
