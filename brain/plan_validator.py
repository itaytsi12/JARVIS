"""Validation boundary for model-generated desktop actions."""
from __future__ import annotations

from pathlib import Path

from brain.intent_router import TOOLS
from brain.models import Action,Plan
from brain.session_context import SessionContext
from security.safety import classify_action


SCHEMAS={tool["name"]:tool["parameters"] for tool in TOOLS}
TYPE_MAP={"string":str,"integer":int,"number":(int,float),"boolean":bool,"object":dict,"array":list}

RUNTIME_TOOL_ARGUMENTS={
    "open_application":{"app_name"},"close_application":{"app_name"},"open_website":{"url"},"calculator":{"expression"},
    "wait_for_window":{"app_name"},"focus_application":{"app_name"},"inspect_window":{"app_name"},
    "click_ui_element":{"app_name","name"},"create_text_file":{"path","contents"},"open_path":{"path"},
    "read_text_file":{"path"},"rename_path":{"path","new_name"},"move_path":{"source","destination"},
    "verify_file":{"path"},"write_text_file":{"path","contents"},"type_text":{"text"},"press_key":{"key"},
    "send_whatsapp_message":{"recipient","message"},"browser_open_url":{"url"},"browser_click":{"target"},
    "browser_type":{"target","text"},"browser_select":{"target","option"},"append_text_file":{"path","contents"},
    "copy_path":{"source","destination"},"find_file":{"path","name"},"search_text":{"path","query"},
    "analyze_screen":{"question"},"click_at":{"x","y"},
}
RUNTIME_NO_ARGUMENT_TOOLS={
    "save_current_document","save_active_document","browser_click_first_result","browser_scroll","browser_go_back",
    "browser_go_forward","browser_fullscreen","minimize_window","maximize_window","restore_window","close_window",
    "take_screenshot","show_desktop","open_task_manager","mute_volume","get_time","get_date","get_day","active_window",
}
RUNTIME_TOOLS=set(SCHEMAS)|set(RUNTIME_TOOL_ARGUMENTS)|RUNTIME_NO_ARGUMENT_TOOLS|{
    "open_folder","open_file_explorer","list_files","exists","append_text_file","copy_path","find_file","search_text",
    "lock_computer","analyze_screen","click_at","calculator","volume_up","volume_down","open_website","open_application","close_application",
}


def validate_plan_preflight(plan:Plan,context:SessionContext) -> list[str]:
    """Validate a complete local plan without causing any external side effect."""
    errors=[]
    if not isinstance(plan,Plan) or not plan.actions:return ["empty_plan"]
    completed=set();available_apps=set()
    if context.active_app and context.last_hwnd:available_apps.add(context.active_app)
    for index,action in enumerate(plan.actions):
        if not isinstance(action,Action):errors.append(f"action_{index}:not_action");continue
        if action.tool not in RUNTIME_TOOLS:errors.append(f"action_{index}:unknown_tool:{action.tool}");continue
        if not isinstance(action.args,dict):errors.append(f"action_{index}:arguments_not_object");continue
        required=RUNTIME_TOOL_ARGUMENTS.get(action.tool,set())
        missing=required-set(action.args)
        if missing:errors.append(f"action_{index}:missing:{','.join(sorted(missing))}")
        if any(not isinstance(dep,int) or dep<0 or dep>=index for dep in action.depends_on):errors.append(f"action_{index}:invalid_dependency")
        if any(dep not in completed for dep in action.depends_on):errors.append(f"action_{index}:unavailable_dependency")
        if action.tool in {"open_application","close_application","wait_for_window","focus_application","inspect_window","click_ui_element"} and not str(action.args.get("app_name","")).strip():errors.append(f"action_{index}:empty_app_name")
        if action.tool=="type_text" and (not isinstance(action.args.get("text"),str) or not action.args.get("text") or not 0<=action.args.get("delay",.02)<=1):errors.append(f"action_{index}:invalid_typing_arguments")
        if action.tool in {"open_path","read_text_file","verify_file","rename_path","create_text_file","write_text_file"} and not str(action.args.get("path","")).strip():errors.append(f"action_{index}:empty_path")
        if action.tool in {"open_path","read_text_file","rename_path"} and str(action.args.get("path","")).strip() and not Path(action.args["path"]).exists():errors.append(f"action_{index}:path_not_found")
        if action.tool in {"create_text_file","write_text_file"} and str(action.args.get("path","")).strip() and not Path(action.args["path"]).expanduser().parent.exists():errors.append(f"action_{index}:parent_not_found")
        if action.tool=="wait_for_window":
            available_apps.add(str(action.args.get("app_name")))
        if action.tool in {"type_text","save_current_document"} and not available_apps:
            errors.append(f"action_{index}:missing_target_window_context")
        if action.tool=="click_ui_element" and str(action.args.get("app_name")) not in available_apps:
            errors.append(f"action_{index}:missing_app_window_context")
        completed.add(index)
    trace=plan.context.get("planning_trace",{}) if isinstance(plan.context,dict) else {}
    segments=trace.get("segments")
    represented=plan.context.get("represented_clause_count") if isinstance(plan.context,dict) else None
    if segments and represented is not None and represented!=len(segments):errors.append("unrepresented_clause")
    return errors


def validate_generated_actions(raw_actions) -> tuple[list[Action],list[str]]:
    if not isinstance(raw_actions,list):return [],["actions_not_list"]
    validated=[];errors=[]
    for index,raw in enumerate(raw_actions):
        if not isinstance(raw,dict):errors.append(f"action_{index}:not_object");continue
        tool=raw.get("tool");arguments=raw.get("arguments")
        schema=SCHEMAS.get(tool)
        if schema is None:errors.append(f"action_{index}:unknown_tool:{tool}");continue
        if not isinstance(arguments,dict):errors.append(f"action_{index}:arguments_not_object");continue
        properties=schema.get("properties",{});required=set(schema.get("required",[]));keys=set(arguments)
        missing=required-keys;extra=keys-set(properties)
        if missing:errors.append(f"action_{index}:missing:{','.join(sorted(missing))}");continue
        if extra:errors.append(f"action_{index}:extra:{','.join(sorted(extra))}");continue
        invalid=None
        for key,value in arguments.items():
            property_schema=properties[key];expected=TYPE_MAP.get(property_schema.get("type"))
            if expected and (isinstance(value,bool) and property_schema.get("type") in {"integer","number"} or not isinstance(value,expected)):
                invalid=key;break
            if isinstance(value,(int,float)) and not isinstance(value,bool) and (value<property_schema.get("minimum",value) or value>property_schema.get("maximum",value)):
                invalid=key;break
        if invalid:errors.append(f"action_{index}:invalid_type:{invalid}");continue
        if tool=="open_website" and not str(arguments["url"]).lower().startswith(("http://","https://")):errors.append(f"action_{index}:invalid_url_scheme");continue
        if tool in {"volume_up","volume_down"} and not 1<=arguments["amount"]<=20:errors.append(f"action_{index}:amount_out_of_range");continue
        if tool=="type_text" and (not arguments["text"] or len(arguments["text"])>10000 or not 0<=arguments["delay"]<=1):errors.append(f"action_{index}:unsafe_typing_arguments");continue
        if tool in {"open_application","close_application"} and not arguments["app_name"].strip():errors.append(f"action_{index}:empty_app_name");continue
        sensitive={"text"} if tool in {"type_text","browser_type"} else {"message"} if tool=="send_whatsapp_message" else {"contents"} if tool in {"create_text_file","write_text_file","append_text_file"} else set()
        action=Action(tool,arguments,sensitive_fields=sensitive);action.risk=classify_action(action);validated.append(action)
    return (validated,errors) if not errors else ([],errors)
