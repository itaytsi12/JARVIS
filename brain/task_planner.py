from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote_plus

from brain.models import Action, ActionRisk, Plan
from brain.session_context import SessionContext
from tools.files import get_desktop_path
from tools.registry import APP_ALIASES, WEBSITE_ALIASES


SEARCH_URLS = {
    "google": "https://www.google.com/search?q={}",
    "youtube": "https://www.youtube.com/results?search_query={}",
}
def _desktop_path(filename: str) -> str:
    return str(get_desktop_path() / filename)


def _clean(value: str) -> str:
    return value.strip(" \t\r\n,.;!?'\"")


def _search_parts(text: str) -> tuple[str, str] | None:
    lowered = text.lower()
    provider = "youtube" if "youtube" in lowered else "google" if "google" in lowered else ""
    if not provider:
        return None
    patterns = [
        rf"(?:search(?:\s+{provider})?\s+for|look\s+up|look\s+for|find|play)\s+(.+?)(?:\s+(?:on|in)\s+{provider})?(?:\s+for\s+me)?$",
        rf"(?:{provider}\s+search)\s+(.+)$",
        rf"search\s+{provider}\s+(.+)$",
        r"search\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            query = _clean(match.group(1))
            query = re.split(
                r"\s*,?\s*(?:and\s+then\s+|then\s+|and\s+)?(?:open|choose)\s+the\s+first\s+(?:result|video)\b",
                query,
                maxsplit=1,
                flags=re.I,
            )[0]
            query = re.sub(r"\s+(?:on|in)\s+(?:youtube|google)$", "", query, flags=re.I)
            if query:
                if query.lower() in {"him", "her", "it", "that"}:
                    earlier = re.search(r"(?:see|watch)\s+(?:some\s+)?(.+?)\s+videos?", text, re.I)
                    if earlier:
                        query = _clean(earlier.group(1))
                return provider, query
    if provider == "youtube":
        match = re.search(r"(?:videos?(?:\s+about)?|watch)\s+(.+?)(?:\s+on\s+youtube)?$", text, re.I)
        if match:
            return provider, _clean(match.group(1))
    return None


def _extract_type_text(text: str) -> str | None:
    match = re.search(r"(?:type|write)\s+(.+?)(?=\s*,?\s*(?:(?:then|and then|and|finally)\s+)?(?:save|close)\b|$)", text, re.I)
    return _clean(match.group(1)) if match else None


def _app_mentions(text: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    lowered = text.lower()
    for alias, app in APP_ALIASES.items():
        pattern = rf"\b(?:open|launch|start|bring\s+up)\s+(?:my\s+)?{re.escape(alias)}\b"
        for match in re.finditer(pattern, lowered):
            found.append((match.start(), app))
        implied_pattern = rf"\b(?:then|and\s+then)\s+{re.escape(alias)}\b"
        for match in re.finditer(implied_pattern, lowered):
            found.append((match.start(), app))
    unique = []
    for item in sorted(found):
        if not unique or item[1] != unique[-1][1]:
            unique.append(item)
    return unique


def should_use_task_planner(command: str) -> bool:
    lowered = command.lower()
    signals = (",", " then ", " and then ", "after that", "afterwards", "finally", "once that opens")
    goal_patterns = ("create a text file", "containing ", "first result", "first video", "switch back", "maximize it", "open the screenshots folder")
    continuation = ("open the first", "go back", "go forward", "scroll down", "scroll up", "select option", "continue until verification")
    action_connector = re.search(r"\band\s+(?:type|write|search|look|open|close|maximize|minimize|save|switch)\b", lowered)
    return (
        any(s in lowered for s in signals + goal_patterns)
        or (" and " in lowered and len(_app_mentions(command)) > 1)
        or (_search_parts(command) is not None and " and " in lowered)
        or any(lowered.startswith(item) for item in continuation)
        or action_connector is not None
    )


def create_task_plan(command: str, context: SessionContext | None = None) -> Plan | None:
    context = context or SessionContext()
    text = " ".join(command.strip().split())
    lowered = text.lower()
    if not text:
        return None
    actions: list[Action] = []

    # Safe browser continuation commands resolve against the persistent session.
    if re.match(r"^(?:open|choose)\s+the\s+first\s+(?:one|result|video)", lowered):
        return Plan(text, [Action("browser_click_first_result", {}, verify="url_changed")])
    if lowered.startswith("go back"):
        return Plan(text, [Action("browser_go_back", {}, verify="url_changed")])
    if lowered.startswith("go forward"):
        return Plan(text, [Action("browser_go_forward", {}, verify="url_changed")])
    scroll_match = re.match(r"^scroll\s+(down|up)", lowered)
    if scroll_match:
        return Plan(text, [Action("browser_scroll", {"direction": scroll_match.group(1)})])
    option_match = re.match(r"^select\s+option\s+(.+?)[.!]?$", text, re.I)
    if option_match:
        return Plan(text, [Action("browser_select", {"target": "Options", "option": _clean(option_match.group(1))})])
    if lowered.startswith("continue until verification"):
        return Plan(text, [Action("browser_click", {"target": "Continue", "kind": "button"}, stop_condition="human_verification")])

    # Generic URL + legitimate login form flow. Values remain ephemeral and redacted.
    url_match = re.search(r"(?:https?://|file://)\S+", text, re.I)
    login_match = re.search(r"log\s*in\s+with\s+username\s+(.+?)\s+and\s+password\s+(.+?)(?:[.!]|$)", text, re.I)
    if url_match and login_match:
        url = url_match.group(0).rstrip(",.;")
        username, password = _clean(login_match.group(1)), _clean(login_match.group(2))
        return Plan(text, [
            Action("browser_open_url", {"url": url}, verify="url_loaded"),
            Action("browser_click", {"target": "Login", "kind": "link"}, depends_on=[0]),
            Action("browser_type", {"target": "Username", "text": username}, depends_on=[1]),
            Action("browser_type", {"target": "Password", "text": password}, depends_on=[1], sensitive_fields={"text"}),
            Action("browser_click", {"target": "Log in", "kind": "button"}, depends_on=[2, 3], risk=ActionRisk.CAUTION),
        ])

    # Goal-level direct file creation.
    file_goal = re.search(r"create\s+(?:a\s+)?(?:new\s+)?text file\s+(?:on\s+my\s+desktop\s+)?(?:called|named)\s+([^,\s]+)\s+containing\s+(.+)$", text, re.I)
    if file_goal:
        path = _desktop_path(_clean(file_goal.group(1)))
        contents = _clean(file_goal.group(2)).replace(", ", "\n").replace(" and ", "\n")
        return Plan(text, [
            Action("create_text_file", {"path": path, "contents": contents}, verify="file_exists"),
            Action("verify_file", {"path": path}, depends_on=[0]),
        ])

    # Screenshot + folder is a precise goal and does not need language splitting.
    if "screenshot" in lowered:
        actions.append(Action("take_screenshot", {}, verify="file_exists_from_result"))
        if "open" in lowered and "screenshots folder" in lowered:
            actions.append(Action("open_file_explorer", {"path": str(Path.cwd() / "screenshots")}, depends_on=[0]))
        return Plan(text, actions)

    search = _search_parts(text)
    app_mentions = _app_mentions(text)

    # Open apps in their textual order, including readiness dependencies.
    for _, app in app_mentions:
        open_index = len(actions)
        actions.append(Action("open_application", {"app_name": app}, verify="process_started"))
        actions.append(Action("wait_for_window", {"app_name": app}, depends_on=[open_index], verify="window_exists"))

    # Browser goals use a persistent semantic browser session.
    if search:
        provider, query = search
        actions = [a for a in actions if a.args.get("app_name") not in {"chrome", "edge"}]
        actions.extend([
            Action("browser_open_url", {"url": SEARCH_URLS[provider].format(quote_plus(query))}, verify="url_loaded"),
        ])
        if re.search(r"(?:open|choose)\s+the\s+first\s+(?:result|video)", lowered):
            actions.append(Action("browser_click_first_result", {}, depends_on=[len(actions)-1], verify="url_changed"))
        if "fullscreen" in lowered or "full screen" in lowered:
            actions.append(Action("browser_fullscreen", {}, depends_on=[len(actions)-1]))
        return Plan(text, actions)

    typed = _extract_type_text(text)
    if typed:
        actions.append(Action("type_text", {"text": typed, "delay": 0.02}, depends_on=[len(actions)-1] if actions else []))

    save_match = re.search(r"save\s+(?:it\s+)?(?:(?:to\s+(?:(?:my|the)\s+)?desktop\s+)?as|to\s+(?:(?:my|the)\s+)?desktop\s+as)\s+([^,\s]+)", text, re.I)
    if save_match:
        path = _desktop_path(_clean(save_match.group(1)))
        actions.append(Action("write_text_file", {"path": path, "contents": typed or ""}, depends_on=[len(actions)-1] if actions else [], verify="file_exists"))
        actions.append(Action("verify_file", {"path": path, "expected_content": typed or ""}, depends_on=[len(actions)-1]))

    if re.search(r"\bmaximize\s+(?:it|the app|chrome|notepad|vscode)\b", lowered):
        target = app_mentions[-1][1] if app_mentions else context.resolve_target("it")
        if target:
            actions.append(Action("focus_application", {"app_name": target}))
            actions.append(Action("maximize_window", {}, depends_on=[len(actions)-1]))

    switch = re.search(r"(?:switch|go)\s+back\s+to\s+(.+?)(?:[.!]|$)", text, re.I)
    if switch:
        target_text = _clean(switch.group(1)).lower()
        target = APP_ALIASES.get(target_text, target_text)
        actions.append(Action("focus_application", {"app_name": target}))

    if re.search(r"(?:then\s+)?close\s+(?:it|notepad|the app)", lowered):
        target = "notepad" if "notepad" in lowered else (app_mentions[-1][1] if app_mentions else context.resolve_target("it"))
        if target:
            actions.append(Action("close_application", {"app_name": target}, risk=ActionRisk.CAUTION, verify="window_closed"))

    if "downloads folder" in lowered:
        actions.append(Action("open_folder", {"name": "downloads"}))

    return Plan(text, actions) if actions else None


def format_plan(plan: Plan) -> str:
    lines = ["PLAN", "", f"Goal: {plan.safe_goal()}", ""]
    for index, action in enumerate(plan.actions, 1):
        args = ", ".join(f"{key}={value!r}" for key, value in action.safe_args().items())
        lines.append(f"{index}. {action.tool}({args})")
    return "\n".join(lines)
