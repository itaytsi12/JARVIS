import re

from brain.models import Action, ActionRisk


HIGH_IMPACT_TOOLS = {"delete_file", "shutdown", "restart", "modify_registry"}
DESTRUCTIVE_COMMAND = re.compile(
    r"(?:\bshutdown\b|\brestart\b|\breboot\b|\bformat\b|\bdiskpart\b|"
    r"\brm\s+-rf\b|\bgit\s+reset\s+--hard\b|\bgit\s+clean\b|"
    r"\breg(?:\.exe)?\s+(?:add|delete)\b|\btaskkill\b|\b(?:un)?install\b)",
    re.IGNORECASE,
)


def classify_action(action: Action) -> ActionRisk:
    if action.tool in HIGH_IMPACT_TOOLS:
        return ActionRisk.HIGH_IMPACT
    if action.tool in {"submit_form", "overwrite_file", "send_message", "upload_file"}:
        return ActionRisk.CAUTION
    if action.tool == "run_command" and DESTRUCTIVE_COMMAND.search(str(action.args.get("command", ""))):
        return ActionRisk.HIGH_IMPACT
    return action.risk


def may_auto_execute(action: Action) -> bool:
    return classify_action(action) is not ActionRisk.HIGH_IMPACT
