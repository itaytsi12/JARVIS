from __future__ import annotations
import re
import hashlib
from pathlib import Path
from memory import redact

PATTERNS=[
 re.compile(r"(?i)\b(?:password|passcode|api[ _-]?key|token|2fa|verification code)\b\s*(?:is|=|:)?\s*.+?(?=\s+(?:and then|then)\b|$)"),
 re.compile(r"(?i)\b(?:sk-[a-z0-9_-]{12,}|gh[pousr]_[a-z0-9]{12,})\b"),
 re.compile(r"(?i)\bBearer\s+[a-z0-9._~+/-]+=*"),
 re.compile(r"(?i)\b(?:Cookie|Set-Cookie|Authorization)\s*:\s*[^\r\n]+"),
 re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----[\s\S]*?-----END (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----"),
 re.compile(r"(?im)^\s*[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|COOKIE)\s*=\s*.+$")]
SENSITIVE_KEYS=re.compile(r"(?i)(password|passwd|secret|token|api.?key|authorization|cookie|2fa|verification.?code)")
SAFE_NUMERIC_METRIC_KEYS={"input_tokens","output_tokens","translation_input_tokens","translation_output_tokens"}
CONTENT_BEARING_COMMAND=re.compile(r"(?i)\b(?P<verb>send|message|type|write)\b|\b(?P<tell>tell)\s+(?!me\b)")
def sanitize_text(text):
    value=redact(str(text))
    for pattern in PATTERNS:value=pattern.sub("<REDACTED>",value)
    return value
def sanitize_user_request(text):
    """Keep action intent while excluding private typed/message bodies."""
    value=sanitize_text(text)
    match=CONTENT_BEARING_COMMAND.search(value)
    if not match:return value
    verb=match.group("verb") or match.group("tell")
    return f"{verb.lower()} request <CONTENT_REDACTED; original_length={len(value)}>"
def sanitize(value,key=None):
    if key and str(key).lower() in SAFE_NUMERIC_METRIC_KEYS and isinstance(value,(int,float)) and not isinstance(value,bool):return value
    if key and SENSITIVE_KEYS.search(str(key)):return "<REDACTED>"
    if isinstance(value,str):return sanitize_text(value)
    if isinstance(value,dict):return {str(k):sanitize(v,str(k)) for k,v in value.items() if str(k).lower() not in {"env","environment"}}
    if isinstance(value,(list,tuple)):return [sanitize(v) for v in value]
    if value is None or isinstance(value,(bool,int,float)):return value
    return sanitize_text(repr(value))
def sanitize_action_arguments(tool,arguments):
    safe=sanitize(dict(arguments or {}))
    private_field={"type_text":"text","browser_type":"text","send_whatsapp_message":"message"}.get(str(tool))
    if private_field and private_field in safe:
        raw=str((arguments or {}).get(private_field,""));safe[private_field]="<REDACTED>";safe[f"{private_field}_length"]=len(raw)
    if "contents" in safe and safe["contents"]!="<REDACTED>":
        raw=str((arguments or {}).get("contents",""));safe.pop("contents",None);safe["content_metadata"]={"length":len(raw),"sha256":hashlib.sha256(raw.encode()).hexdigest()}
    return safe
def privacy_safe_event(event_type,payload):
    """Sanitize both current and legacy event payloads at read/export time."""
    safe=sanitize(payload)
    if not isinstance(safe,dict):return safe
    request_fields={"USER_REQUEST":["original_user_text"],"STT_TRANSCRIPT":["raw_stt_transcript"],"NORMALIZED_REQUEST":["normalized_text","final_interpreted_command"],"RESOLVED_REFERENCES":["request"],"REASONING_REQUEST":["request","query","input"]}
    for key in request_fields.get(str(event_type),[]):
        if isinstance(safe.get(key),str):safe[key]=sanitize_user_request(safe[key])
    if str(event_type) in {"TOOL_CALL","PLAN_CREATED"}:
        if isinstance(safe.get("arguments"),dict):safe["arguments"]=sanitize_action_arguments(safe.get("name") or safe.get("tool"),safe["arguments"])
        if isinstance(safe.get("actions"),list):
            for action in safe["actions"]:
                if isinstance(action,dict) and isinstance(action.get("arguments"),dict):action["arguments"]=sanitize_action_arguments(action.get("tool"),action["arguments"])
    return safe
def code_reference(path,content=None):
    import hashlib
    p=Path(path); raw=(content if content is not None else p.read_text(encoding="utf-8",errors="replace")); safe=sanitize_text(raw)
    return {"source_file":str(p),"content_hash":hashlib.sha256(safe.encode()).hexdigest(),"content":safe}
