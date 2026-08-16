from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import quote_plus

from brain.models import Action, ActionRisk, Plan
from brain.session_context import SessionContext
from tools.files import get_desktop_path,get_documents_path
from tools.registry import APP_ALIASES, WEBSITE_ALIASES


SEARCH_URLS = {
    "google": "https://www.google.com/search?q={}",
    "youtube": "https://www.youtube.com/results?search_query={}",
}
def _desktop_path(filename: str) -> str:
    return str(get_desktop_path() / filename)


def _clean(value: str) -> str:
    return value.strip(" \t\r\n,.;!?'\"")


_COMMAND_START = r"(?:open|launch|start|play|type|write|save|close|quit|go\s+to|search|press|hit|click|maximize|minimize|switch|focus|send|message|tell|create|rename|move|read)"
_TEXT_REFERENCES = re.compile(
    r"^(?:exactly\s+)?(?:what you(?: just| last)? said|what you(?: just| last)? told me|your last (?:answer|response)|the previous (?:answer|response)|your previous (?:answer|response)|the last thing you said|that|the last thing)$",
    re.I,
)


def segment_sequential_commands(text: str) -> list[str]:
    """Split clear imperative sequences while protecting quoted payloads."""
    mask=list(text);quote=None;escaped=False
    for index,char in enumerate(text):
        if escaped:mask[index]=" ";escaped=False;continue
        if char=="\\" and quote:mask[index]=" ";escaped=True;continue
        if char in {'"',"'"}:
            quote=None if quote==char else char if quote is None else quote
            mask[index]=" ";continue
        if quote:mask[index]=" "
    visible="".join(mask)
    connector=re.compile(
        rf"(?:\s+and\s+then\s+|\s+then\s+|\s+after\s+that\s+|\s+next\s+|\s+and\s+|,\s*(?:then\s+|next\s+)?)(?={_COMMAND_START}\b)",
        re.I,
    )
    parts=[];start=0
    for match in connector.finditer(visible):
        part=text[start:match.start()].strip(" ,")
        if part:parts.append(part)
        start=match.end()
    tail=text[start:].strip(" ,")
    if tail:parts.append(tail)
    return parts


def _literal_payload(value: str) -> str:
    value=value.strip()
    if len(value)>=2 and value[0]==value[-1] and value[0] in {'"',"'"}:return value[1:-1]
    return value.rstrip(".?!")


def _type_payload(value: str, context: SessionContext):
    """Parse type/write data, preserving quoted references as literal text."""
    raw = value.strip()
    quoted_match = re.fullmatch(r'''(["'])(.*)\1[.?!]?''', raw, re.S)
    quoted = quoted_match is not None
    payload = quoted_match.group(2) if quoted_match else _literal_payload(raw)
    detected_reference = payload if not quoted and _TEXT_REFERENCES.fullmatch(payload) else None
    trace = {"type_payload_before_reference_resolution": payload, "quoted_literal": quoted, "detected_reference_phrase": detected_reference}
    if detected_reference:
        resolved = context.resolve_text_reference(payload)
        trace["reference_resolution_result"] = resolved
        trace["reference_source"] = "last_spoken_response" if resolved is not None and resolved==context.last_spoken_response and payload.lower().removeprefix("exactly ") in {"what you said","what you just said","what you last said","what you told me","what you just told me","what you last told me","the last thing you said"} else "last_assistant_response" if resolved is not None else None
        if resolved is None:
            return None, trace
        return resolved, trace
    trace["reference_resolution_result"] = None
    return payload, trace


def _whatsapp_parts(segment:str,context:SessionContext):
    patterns=[r"^(?:send|message)\s+(.+?)\s+on\s+whatsapp\s*(?::|that\s+)?\s*(.+)$",r"^(?:send|message)\s+(.+?)\s*:\s*(.+)$",r"^(?:send|message|tell)\s+(\S+)\s+(.+)$"]
    for pattern in patterns:
        match=re.match(pattern,segment,re.I)
        if not match:continue
        recipient=_clean(match.group(1));raw=match.group(2).strip();literal=(len(raw)>=2 and raw[0]==raw[-1] and raw[0] in {'"',"'"}) or raw.lower().startswith("exactly:")
        if recipient.lower()=="me":return None
        if raw.lower().startswith("exactly:"):raw=raw.split(":",1)[1].strip()
        payload=raw[1:-1] if len(raw)>=2 and raw[0]==raw[-1] and raw[0] in {'"',"'"} else raw
        if recipient.lower() in {"him","her","them","that person","the same person"}:
            if not context.last_messaging_recipient:return recipient,None,{"clarification":"I don't have a recent WhatsApp recipient to use."},False
            recipient=context.last_messaging_recipient
        reference_candidate=payload.rstrip(".?!")
        if _TEXT_REFERENCES.fullmatch(reference_candidate):
            resolved=context.resolve_text_reference(reference_candidate)
            if resolved is None:return recipient,None,{"clarification":"I don't have a previous answer to send."},True
            return recipient,resolved,{"reference":reference_candidate,"resolved_value":"<REDACTED_MESSAGE>","literal":True},True
        return recipient,payload,{},literal
    return None


def _segmented_plan(text: str,context:SessionContext) -> Plan | None:
    segmentation_started=time.perf_counter();segments=segment_sequential_commands(text);segmentation_ms=(time.perf_counter()-segmentation_started)*1000
    if len(segments)<2:return None
    actions=[];resolved=[];type_traces=[];reference_started=time.perf_counter()
    for segment in segments:
        lowered=segment.lower().strip()
        whatsapp=_whatsapp_parts(segment,context)
        if whatsapp:
            recipient,payload,metadata,literal=whatsapp
            if payload is None:return Plan(text,[],context=metadata)
            if metadata:resolved.append(metadata)
            actions.append(Action("send_whatsapp_message",{"recipient":recipient,"message":payload,"literal":literal},depends_on=[len(actions)-1] if actions else [],risk=ActionRisk.CAUTION,max_attempts=1,sensitive_fields={"message"}))
            continue
        open_match=re.match(r"^(?:open|launch|start|go\s+to)\s+(.+)$",segment,re.I)
        if open_match:
            target=_clean(open_match.group(1)).lower()
            if target in WEBSITE_ALIASES:
                actions.append(Action("open_website",{"url":WEBSITE_ALIASES[target]},depends_on=[len(actions)-1] if actions else [],max_attempts=1))
            elif target in APP_ALIASES:
                app=APP_ALIASES[target];open_index=len(actions);actions.append(Action("open_application",{"app_name":app},depends_on=[open_index-1] if open_index else [],verify="process_started",max_attempts=1));actions.append(Action("wait_for_window",{"app_name":app},depends_on=[open_index],verify="window_exists",max_attempts=1))
            else:return None
            continue
        type_match=re.match(r"^(?:type|write)\s+(.+)$",segment,re.I)
        if type_match:
            payload,trace=_type_payload(type_match.group(1),context);type_traces.append(trace)
            if payload is None:
                return Plan(text,[],context={"clarification":"I don't have a previous answer to type.","planning_trace":{"segments":segments,"type_text":type_traces}})
            if trace["reference_resolution_result"] is not None:
                resolved.append({"reference":trace["type_payload_before_reference_resolution"],"resolved_value":payload})
            actions.append(Action("type_text",{"text":payload,"delay":.02},depends_on=[len(actions)-1] if actions else [],max_attempts=1,sensitive_fields={"text"}))
            continue
        search_match=re.match(r"^search(?:\s+(?:google|chrome))?\s+(?:for\s+)?(.+)$",segment,re.I)
        if search_match:
            app=next((a.args.get("app_name") for a in reversed(actions) if a.tool=="open_application"),None) or context.active_app
            if app!="chrome":return None
            actions.append(Action("open_website",{"url":SEARCH_URLS["google"].format(quote_plus(_clean(search_match.group(1))))},depends_on=[len(actions)-1] if actions else [],max_attempts=1))
            continue
        if re.fullmatch(r"save(?:\s+(?:it|the document|the file))?",lowered.rstrip(".?!")):
            actions.append(Action("save_current_document",{},depends_on=[len(actions)-1] if actions else [],max_attempts=1))
            continue
        click_match=re.match(r"^click\s+(?:the\s+)?(.+?)(?:\s+(button|link|menu item))?(?:\s+(?:in|on)\s+(.+?))?[.?!]?$",segment,re.I)
        if click_match:
            name,control_type,explicit_app=click_match.groups()
            app=(explicit_app or next((a.args.get("app_name") for a in reversed(actions) if a.tool=="open_application"),None) or context.active_app)
            if not app:return None
            app=APP_ALIASES.get(_clean(app).lower(),_clean(app).lower());arguments={"app_name":app,"name":_clean(name)}
            if control_type:arguments["control_type"]={"button":"Button","link":"Hyperlink","menu item":"MenuItem"}[control_type.lower()]
            actions.append(Action("click_ui_element",arguments,depends_on=[len(actions)-1] if actions else [],max_attempts=1))
            continue
        close_match=re.match(r"^(?:close|quit)\s+(.+)$",segment,re.I)
        if close_match:
            target=_clean(close_match.group(1)).lower()
            if target in {"it","the app"}:target=context.last_opened_app or next((a.args.get("app_name") for a in reversed(actions) if a.tool=="open_application"),None)
            target=APP_ALIASES.get(target,target)
            if not target:return None
            actions.append(Action("close_application",{"app_name":target},depends_on=[len(actions)-1] if actions else [],risk=ActionRisk.CAUTION,verify="window_closed",max_attempts=1))
            continue
        return None
    reference_ms=(time.perf_counter()-reference_started)*1000
    return Plan(text,actions,context={"segments":segments,"represented_clause_count":len(segments),"resolved_references":{"items":resolved} if resolved else {},"planning_trace":{"segments":segments,"type_text":type_traces,"had_last_assistant_response":bool(context.last_assistant_response),"had_last_spoken_response":bool(context.last_spoken_response)},"segmentation_ms":segmentation_ms,"reference_resolution_ms":reference_ms,"model_calls":0,"planning_ms":segmentation_ms+reference_ms})


def assess_plan_completeness(command:str,plan:Plan | None) -> dict:
    clauses=segment_sequential_commands(command)
    if not plan or not plan.actions:return {"complete":False,"clauses":clauses,"reason":"missing_local_plan"}
    represented=plan.context.get("represented_clause_count") if isinstance(plan.context,dict) else None
    complete=len(clauses)<=1 or represented==len(clauses)
    return {"complete":complete,"clauses":clauses,"represented_clause_count":represented,"reason":None if complete else "incomplete_local_plan"}


def validate_goal_coverage(goal:str,actions:list[Action]) -> list[str]:
    """Check semantic clause and explicit app identity fidelity before execution."""
    clauses=segment_sequential_commands(goal);errors=[];tools=[action.tool for action in actions]
    open_clause=next((clause for clause in clauses if re.match(r"^(?:open|launch|start)\b",clause,re.I)),None)
    if open_clause:
        match=re.match(r"^(?:open|launch|start)(?:\s+up)?\s+(.+)$",open_clause,re.I);requested=_clean(match.group(1)).lower() if match else ""
        expected=APP_ALIASES.get(requested,requested)
        planned=[str(action.args.get("app_name","")).strip().lower() for action in actions if action.tool=="open_application"]
        if requested in {"a music","music","the music"}:errors.append("app_identification_uncertain")
        elif not planned:errors.append("missing_open_application_clause")
        elif expected not in planned:errors.append("app_identity_mismatch")
    for clause in clauses:
        lowered=clause.lower()
        if re.match(r"^play\b",lowered) and not ("type_text" in tools and any(tool in tools for tool in {"press_key","click_ui_element"})):errors.append("missing_playback_clause")
        if re.match(r"^search\b",lowered) and not any(tool in tools for tool in {"open_website","browser_open_url","browser_type","type_text"}):errors.append("missing_search_clause")
    return errors


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


_BROWSER_CAPABLE_APPS = {"chrome", "edge"}
_SEARCH_CLAUSE_START = re.compile(
    r"^(?:search(?:\s+(?:google|youtube|chrome))?\s+(?:for\s+)?|look\s+up\s+|look\s+for\s+|find\s+|play\s+|watch\s+)",
    re.I,
)
# fullmatch, not match: the sequential-command segmenter only splits before a
# recognized command-start verb (see _COMMAND_START), so an unhandled clause
# like "mute the volume" that follows a comma can end up merged onto the tail
# of "open the first result" as one clause. Anchoring both ends keeps that
# merged clause from being falsely counted as covered by the click action.
_FIRST_RESULT_CLAUSE = re.compile(r"(?:open|choose)\s+the\s+first\s+(?:result|video|one)[.,!?]*", re.I)
_FULLSCREEN_CLAUSE = re.compile(r"(?:go\s+)?(?:fullscreen|full\s+screen)[.,!?]*", re.I)


def _browser_goal_represented_clauses(clauses: list[str], provider: str, has_first_result: bool, has_fullscreen: bool) -> int:
    """Count how many of the original command's clauses a browser-goal plan
    (browser_open_url [+ browser_click_first_result] [+ browser_fullscreen])
    genuinely accounts for.

    That branch folds "open <browser>", "go to <site>", and "search ... for
    ..." into a single semantic browser_open_url action, so it never produces
    the 1:1 clause-to-action mapping assess_plan_completeness normally
    expects from a plan. This walks the same clauses assess_plan_completeness
    will independently re-derive and classifies each one against what the
    browser plan actually does -- opens a browser-capable app, navigates to
    the site the search itself targets, performs the search, optionally
    clicks the first result, optionally goes fullscreen -- so a genuinely
    complete compound browser command isn't discarded as incomplete merely
    because its representation isn't one-action-per-clause. A clause that
    doesn't fit any of those categories is left uncovered on purpose, so an
    unrelated trailing instruction (e.g. "...and take a screenshot") still
    correctly falls back to the cloud planner instead of being silently
    dropped.
    """
    covered = 0
    search_clause_claimed = False
    for clause in clauses:
        stripped = clause.lower().strip()
        if has_first_result and _FIRST_RESULT_CLAUSE.fullmatch(stripped):
            covered += 1
            continue
        open_match = re.match(r"^(?:open|launch|start|go\s+to)\s+(.+)$", clause, re.I)
        if open_match:
            target = _clean(open_match.group(1)).lower()
            resolved_app = APP_ALIASES.get(target, target)
            resolved_site = WEBSITE_ALIASES.get(target, "")
            if resolved_app in _BROWSER_CAPABLE_APPS or (provider and (provider in target or provider in resolved_site)):
                covered += 1
            continue
        if has_fullscreen and _FULLSCREEN_CLAUSE.fullmatch(stripped):
            covered += 1
            continue
        if not search_clause_claimed and _SEARCH_CLAUSE_START.match(stripped):
            covered += 1
            search_clause_claimed = True
            continue
    return covered


def should_use_task_planner(command: str) -> bool:
    lowered = command.lower()
    signals = (",", " then ", " and then ", "after that", "afterwards", "finally", "once that opens")
    goal_patterns = ("create a text file", "create a file", "containing ", "first result", "first video", "switch back", "maximize it", "open the screenshots folder","file i just created","rename it","move it to documents","read that file","tell me what is inside")
    continuation = ("open the first", "go back", "go forward", "scroll down", "scroll up", "select option", "continue until verification")
    action_connector = re.search(r"\band\s+(?:play|type|write|search|look|open|close|maximize|minimize|save|switch)\b", lowered)
    local_reference = any(phrase in lowered for phrase in ("what you said","what you just said","what you last said","what you told me","what you just told me","what you last told me","your last answer","your last response","the previous answer","the previous response","your previous answer","your previous response","the last thing you said")) or re.fullmatch(r"(?:type|write)\s+(?:exactly\s+)?(?:that|the last thing)[.?!]?",lowered) is not None
    return (
        any(s in lowered for s in signals + goal_patterns)
        or (" and " in lowered and len(_app_mentions(command)) > 1)
        or (_search_parts(command) is not None and " and " in lowered)
        or any(lowered.startswith(item) for item in continuation)
        or action_connector is not None
        or local_reference
        or re.match(r"^(?:type|write)\s+",lowered) is not None
        or re.match(r"^(?:send|message|tell)\s+",lowered) is not None
    )


def create_task_plan(command: str, context: SessionContext | None = None) -> Plan | None:
    context = context or SessionContext()
    text = " ".join(command.strip().split())
    lowered = text.lower()
    if not text:
        return None
    actions: list[Action] = []

    if re.match(r"^(?:ask|find out|tell me|who|what|when|where|why|how|which)\b",text,re.I) and re.search(r"\b(?:write|type)\s+(?:the\s+)?answer\b",text,re.I):
        return Plan(text,[],context={"clarification":"Ask me the question first, then tell me to write my last answer. I won't type an unresolved answer reference."})

    empty_file=re.fullmatch(r"create\s+(?:a\s+)?(?:new\s+)?(?:text\s+)?file\s+(?:on\s+(?:my|the)\s+desktop\s+)?(?:called|named)\s+([^,\s]+)[.?!]?",text,re.I)
    if empty_file:
        path=_desktop_path(_clean(empty_file.group(1)))
        return Plan(text,[Action("create_text_file",{"path":path,"contents":""},verify="file_exists",sensitive_fields={"contents"}),Action("verify_file",{"path":path,"expected_content":""},depends_on=[0],verify="file_exists")],context={"model_calls":0})

    file_reference=context.last_opened_file
    if re.fullmatch(r"open\s+(?:the\s+)?file\s+(?:i\s+)?just\s+created[.?!]?",text,re.I):
        return Plan(text,[Action("open_path",{"path":file_reference},verify="path_opened")],context={"model_calls":0}) if file_reference else Plan(text,[],context={"clarification":"I don't have a recently created file to open."})
    rename_match=re.fullmatch(r"rename\s+(?:it|that file|the file)\s+to\s+([^,\s]+)[.?!]?",text,re.I)
    if rename_match:
        return Plan(text,[Action("rename_path",{"path":file_reference,"new_name":_clean(rename_match.group(1))},risk=ActionRisk.CAUTION,max_attempts=1,verify="destination_exists")],context={"model_calls":0}) if file_reference else Plan(text,[],context={"clarification":"I don't have a recent file to rename."})
    move_match=re.fullmatch(r"move\s+(?:it|that file|the file)\s+to\s+(?:my\s+)?documents[.?!]?",text,re.I)
    if move_match:
        destination=str(get_documents_path()/Path(file_reference).name) if file_reference else None
        return Plan(text,[Action("move_path",{"source":file_reference,"destination":destination},risk=ActionRisk.CAUTION,max_attempts=1,verify="destination_exists")],context={"model_calls":0}) if file_reference else Plan(text,[],context={"clarification":"I don't have a recent file to move."})
    if re.fullmatch(r"(?:read\s+(?:it|that file|the file)|tell me what(?:'s| is) inside (?:it|that file|the file))[.?!]?",text,re.I):
        return Plan(text,[Action("read_text_file",{"path":file_reference},verify="file_read")],context={"model_calls":0}) if file_reference else Plan(text,[],context={"clarification":"I don't have a recent file to read."})

    segmented=_segmented_plan(text,context)
    if segmented is not None:return segmented

    whatsapp=_whatsapp_parts(text,context)
    if whatsapp:
        recipient,payload,metadata,literal=whatsapp
        if payload is None:return Plan(text,[],context=metadata)
        return Plan(text,[Action("send_whatsapp_message",{"recipient":recipient,"message":payload,"literal":literal},risk=ActionRisk.CAUTION,max_attempts=1,sensitive_fields={"message"})],context={"resolved_references":{"items":[metadata]} if metadata else {},"model_calls":0})

    single_type=re.match(r"^(?:type|write)\s+(.+)$",text,re.I)
    if single_type:
        payload,trace=_type_payload(single_type.group(1),context)
        if payload is None:return Plan(text,[],context={"clarification":"I don't have a previous answer to type.","planning_trace":{"segments":[text],"type_text":[trace]}})
        resolved={"items":[{"reference":trace["type_payload_before_reference_resolution"],"resolved_value":payload}]} if trace["reference_resolution_result"] is not None else {}
        return Plan(text,[Action("type_text",{"text":payload,"delay":.02},max_attempts=1,sensitive_fields={"text"})],context={"resolved_references":resolved,"planning_trace":{"segments":[text],"type_text":[trace],"had_last_assistant_response":bool(context.last_assistant_response),"had_last_spoken_response":bool(context.last_spoken_response)},"model_calls":0})

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
            Action("browser_type", {"target": "Username", "text": username}, depends_on=[1], sensitive_fields={"text"}),
            Action("browser_type", {"target": "Password", "text": password}, depends_on=[1], sensitive_fields={"text"}),
            Action("browser_click", {"target": "Log in", "kind": "button"}, depends_on=[2, 3], risk=ActionRisk.CAUTION),
        ])

    # Goal-level direct file creation.
    file_goal = re.search(r"create\s+(?:a\s+)?(?:new\s+)?text file\s+(?:on\s+my\s+desktop\s+)?(?:called|named)\s+([^,\s]+)\s+containing\s+(.+)$", text, re.I)
    if file_goal:
        path = _desktop_path(_clean(file_goal.group(1)))
        contents = _clean(file_goal.group(2)).replace(", ", "\n").replace(" and ", "\n")
        return Plan(text, [
            Action("create_text_file", {"path": path, "contents": contents}, verify="file_exists", sensitive_fields={"contents"}),
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
        actions.append(Action("wait_for_window", {"app_name": app}, depends_on=[open_index], verify="window_exists", max_attempts=1))

    # Browser goals use a persistent semantic browser session.
    if search:
        provider, query = search
        actions = [a for a in actions if a.args.get("app_name") not in {"chrome", "edge"}]
        actions.extend([
            Action("browser_open_url", {"url": SEARCH_URLS[provider].format(quote_plus(query))}, verify="url_loaded"),
        ])
        has_first_result = bool(re.search(r"(?:open|choose)\s+the\s+first\s+(?:result|video)", lowered))
        if has_first_result:
            actions.append(Action("browser_click_first_result", {}, depends_on=[len(actions)-1], verify="url_changed"))
        has_fullscreen = "fullscreen" in lowered or "full screen" in lowered
        if has_fullscreen:
            actions.append(Action("browser_fullscreen", {}, depends_on=[len(actions)-1]))
        clauses = segment_sequential_commands(command)
        represented = _browser_goal_represented_clauses(clauses, provider, has_first_result, has_fullscreen)
        return Plan(text, actions, context={
            "represented_clause_count": represented,
            "planning_trace": {"segments": clauses},
            "model_calls": 0,
        })

    typed = _extract_type_text(text)
    if typed:
        actions.append(Action("type_text", {"text": typed, "delay": 0.02}, depends_on=[len(actions)-1] if actions else [], sensitive_fields={"text"}))

    save_match = re.search(r"save\s+(?:it\s+)?(?:(?:to\s+(?:(?:my|the)\s+)?desktop\s+)?as|to\s+(?:(?:my|the)\s+)?desktop\s+as)\s+([^,\s]+)", text, re.I)
    if save_match:
        path = _desktop_path(_clean(save_match.group(1)))
        actions.append(Action("write_text_file", {"path": path, "contents": typed or ""}, depends_on=[len(actions)-1] if actions else [], verify="file_exists", sensitive_fields={"contents"}))
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
