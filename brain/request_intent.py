from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum

class RequestKind(str,Enum):ACTION="ACTION"; QUESTION="QUESTION"; OTHER="OTHER"
@dataclass(frozen=True)
class IntentDecision:
    kind:RequestKind; confidence:float; reason:str

ACTION_TARGETS=("open","launch","start","close","quit","mute","volume","turn up","turn down","click","type","write","save","download","run","press","maximize","minimize","switch","focus")
INFORMATION_PREDICATES=("who","what","when","where","why","how","which","how many","how much","tell me who","tell me what","latest","recently","today","yesterday","weather","age","distance","population")
CONVERSATIONAL=("how are you","what can you do","who are you")
LOCAL_QUERIES=("what time is it","what is the time","what time","what is today's date","what is the date","what day is it")

def classify_request_kind(text:str)->IntentDecision:
    normalized=" ".join(text.lower().strip().rstrip("?.!,").split())
    if not normalized:return IntentDecision(RequestKind.OTHER,1.0,"empty")
    if normalized in LOCAL_QUERIES:return IntentDecision(RequestKind.OTHER,.99,"existing_local_query")
    if any(phrase in normalized for phrase in CONVERSATIONAL):return IntentDecision(RequestKind.OTHER,.95,"conversation")
    asks_to_tell=bool(re.search(r"\b(?:can|could|would) you (?:tell|explain)\b",normalized)) or normalized.startswith(("tell me who","tell me what"))
    action_hits=sum(1 for phrase in ACTION_TARGETS if re.search(rf"(?:^|\s){re.escape(phrase)}(?:\s|$)",normalized))
    info_hits=sum(1 for phrase in INFORMATION_PREDICATES if phrase in normalized)
    # Polite grammar does not change a concrete computer-control request into a question.
    if action_hits and not asks_to_tell:return IntentDecision(RequestKind.ACTION,.98,"computer_action_predicate")
    if asks_to_tell and info_hits:return IntentDecision(RequestKind.QUESTION,.97,"information_request_under_polite_wrapper")
    if info_hits>=2:return IntentDecision(RequestKind.QUESTION,.96,"multiple_information_signals")
    if info_hits==1 and (normalized.startswith(INFORMATION_PREDICATES) or len(normalized.split())>=3):return IntentDecision(RequestKind.QUESTION,.9,"information_predicate")
    return IntentDecision(RequestKind.OTHER,.55,"insufficient_information_evidence")
