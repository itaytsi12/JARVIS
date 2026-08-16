from __future__ import annotations
import json,os,re,subprocess,threading,time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import quote
from tools.window import find_application_window

class RecipientStatus(str,Enum):UNIQUE="UNIQUE";AMBIGUOUS="AMBIGUOUS";NOT_FOUND="NOT_FOUND";INVALID="INVALID"
@dataclass
class RecipientResolution:
    status:RecipientStatus
    contact:dict|None=None
    matches:list[str]|None=None

_SEND_LOCK=threading.Lock()
def _active_chat_verified(text_controls,display,box):
    exact=[control for control in text_controls if str(control.element_info.name).strip().casefold()==display.casefold()]
    if len(exact)!=1:return False
    try:
        header_rect=exact[0].rectangle();input_rect=box.rectangle()
        return header_rect.bottom<input_rect.top and header_rect.left>=input_rect.left-50
    except Exception:
        # Some UIA wrappers do not expose geometry. A unique exact semantic
        # match is the conservative fallback; multiple chat-list/header hits fail.
        return True
def _contacts_path():
    configured=os.getenv("JARVIS_WHATSAPP_CONTACTS_PATH")
    if configured:return Path(configured)
    return Path(os.getenv("LOCALAPPDATA",Path.home()))/"Jarvis"/"whatsapp_contacts.json"
def _load_contacts():
    path=_contacts_path()
    if not path.is_file():return []
    data=json.loads(path.read_text(encoding="utf-8"));items=data.get("contacts",[]) if isinstance(data,dict) else data
    return [item for item in items if isinstance(item,dict)]
def resolve_recipient(name:str)->RecipientResolution:
    query=" ".join(name.lower().split())
    if not query:return RecipientResolution(RecipientStatus.INVALID)
    exact=[];partial=[]
    for item in _load_contacts():
        names=[str(item.get("name","")).lower(),*[str(v).lower() for v in item.get("aliases",[])]]
        if query in names:exact.append(item)
        elif any(query in value for value in names):partial.append(item)
    matches=exact or partial
    if len(matches)==1 and re.fullmatch(r"\+?\d{7,15}",str(matches[0].get("phone","")).replace(" ","")):return RecipientResolution(RecipientStatus.UNIQUE,matches[0])
    if len(matches)>1:return RecipientResolution(RecipientStatus.AMBIGUOUS,matches=[str(x.get("name")) for x in matches])
    return RecipientResolution(RecipientStatus.NOT_FOUND if not matches else RecipientStatus.INVALID)
def _language(text):
    hebrew=sum('\u0590'<=c<='\u05ff' for c in text);latin=sum(c.isascii() and c.isalpha() for c in text)
    return "he" if hebrew>latin else "en"
def _translate(message,target_language):
    if not target_language or _language(message)==target_language:return message,{"translated":False,"translation_ms":0.0,"translation_model":None,"translation_input_tokens":0,"translation_output_tokens":0}
    from openai import OpenAI
    began=time.perf_counter();model=os.getenv("JARVIS_TRANSLATION_MODEL","gpt-5-mini")
    response=OpenAI(api_key=os.getenv("OPENAI_API_KEY")).responses.create(model=model,input=[{"role":"system","content":f"Translate only the message into {target_language}. Preserve meaning, casual tone, names, numbers, URLs, emojis, and punctuation. Return only the translation."},{"role":"user","content":message}],max_output_tokens=160,store=False,timeout=float(os.getenv("JARVIS_TRANSLATION_TIMEOUT","10")))
    usage=getattr(response,"usage",None);translated=(getattr(response,"output_text","") or "").strip()
    if not translated:raise RuntimeError("empty_translation")
    return translated,{"translated":True,"translation_ms":(time.perf_counter()-began)*1000,"translation_model":model,"translation_input_tokens":getattr(usage,"input_tokens",0) or 0,"translation_output_tokens":getattr(usage,"output_tokens",0) or 0}
def send_whatsapp_message(recipient:str,message:str,literal:bool=False,cancellation_token=None)->dict:
    started=time.perf_counter();resolution=resolve_recipient(recipient);resolved_ms=(time.perf_counter()-started)*1000
    if resolution.status is not RecipientStatus.UNIQUE:
        prompt=f"Which {recipient}?" if resolution.status is RecipientStatus.AMBIGUOUS else f"I couldn't safely resolve WhatsApp contact {recipient}."
        return {"success":False,"recipient":recipient,"message":prompt,"error":resolution.status.value.lower(),"recipient_status":resolution.status.value,"matches":resolution.matches or [],"recipient_resolution_ms":resolved_ms}
    contact=resolution.contact;phone=re.sub(r"\D","",str(contact["phone"]));display=str(contact.get("name") or recipient)
    try:final_message,translation= (message,{"translated":False,"translation_ms":0.0,"translation_model":None,"translation_input_tokens":0,"translation_output_tokens":0}) if literal else _translate(message,contact.get("preferred_language"))
    except Exception as exc:return {"success":False,"recipient":display,"message":"The message was not sent because translation failed.","error":f"translation_failed: {exc}","recipient_status":"UNIQUE","recipient_resolution_ms":resolved_ms}
    if cancellation_token is not None and cancellation_token.cancelled:return {"success":False,"recipient":display,"message":"The WhatsApp message was cancelled.","error":"cancelled","recipient_status":"UNIQUE",**translation}
    with _SEND_LOCK:
        launch=time.perf_counter();subprocess.Popen(["explorer.exe",f"whatsapp://send?phone={phone}&text={quote(final_message)}"]);deadline=time.perf_counter()+8;hwnd=None
        while time.perf_counter()<deadline:
            if cancellation_token is not None and cancellation_token.cancelled:return {"success":False,"recipient":display,"message":"The WhatsApp message was cancelled.","error":"cancelled","recipient_status":"UNIQUE",**translation}
            hwnd=find_application_window("whatsapp")
            if hwnd:break
            time.sleep(.1)
        app_ms=(time.perf_counter()-launch)*1000
        if not hwnd:return {"success":False,"recipient":display,"message":"WhatsApp did not become ready.","error":"window_unavailable","recipient_status":"UNIQUE","recipient_resolution_ms":resolved_ms,"app_ready_ms":app_ms}
        try:
            from pywinauto import Desktop
            window=Desktop(backend="uia").window(handle=hwnd);window.wait("visible enabled ready",timeout=3)
            selection=time.perf_counter();text_controls=window.descendants(control_type="Text")
            edits=window.descendants(control_type="Edit");box=next((c for c in edits if "message" in str(c.element_info.name).lower()),edits[-1] if edits else None)
            if box is None:return {"success":False,"recipient":display,"message":"WhatsApp message input was unavailable.","error":"message_input_unavailable"}
            if not _active_chat_verified(text_controls,display,box):return {"success":False,"recipient":display,"message":f"Could not verify the active WhatsApp chat for {display}.","error":"target_verification_failed","recipient_status":"UNIQUE","app_ready_ms":app_ms,"chat_selection_ms":(time.perf_counter()-selection)*1000}
            if cancellation_token is not None and cancellation_token.cancelled:return {"success":False,"recipient":display,"message":"The WhatsApp message was cancelled.","error":"cancelled","recipient_status":"UNIQUE",**translation}
            insert=time.perf_counter();box.set_edit_text(final_message);value=box.get_value() if hasattr(box,"get_value") else box.window_text()
            if final_message not in str(value):return {"success":False,"recipient":display,"message":"WhatsApp message verification failed.","error":"message_verification_failed"}
            insert_ms=(time.perf_counter()-insert)*1000
            if cancellation_token is not None and cancellation_token.cancelled:return {"success":False,"recipient":display,"message":"The WhatsApp message was cancelled before sending.","error":"cancelled","recipient_status":"UNIQUE",**translation}
            send_started=time.perf_counter();box.type_keys("{ENTER}");send_ms=(time.perf_counter()-send_started)*1000
            verify_deadline=time.perf_counter()+2;send_verified=False
            while time.perf_counter()<verify_deadline:
                visible=[str(c.element_info.name).strip() for c in window.descendants(control_type="Text")]
                if any(text==final_message for text in visible):send_verified=True;break
                time.sleep(.1)
            common={"recipient":display,"outgoing_message":final_message,"recipient_status":"UNIQUE","recipient_resolution_ms":resolved_ms,"app_ready_ms":app_ms,"chat_selection_ms":(time.perf_counter()-selection)*1000,"message_insert_ms":insert_ms,"send_ms":send_ms,"total_ms":(time.perf_counter()-started)*1000,"committed":True,"verified":send_verified,**translation}
            if not send_verified:return {"success":False,"message":f"The message to {display} was submitted, but I could not verify it appeared. I will not retry it.","error":"send_unverified",**common}
            return {"success":True,"message":f"Sent the WhatsApp message to {display}.",**common}
        except Exception as exc:return {"success":False,"recipient":display,"message":"The WhatsApp message was not sent.","error":f"ui_automation_failed: {exc}","recipient_status":"UNIQUE","app_ready_ms":app_ms}
