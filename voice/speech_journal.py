"""Privacy-bounded action journaling for asynchronous speech output."""
from __future__ import annotations

import time,uuid
from datetime import datetime,timezone
from training_data import EventType,get_recorder


def begin_speech(language,length,parent_interaction_id=None):
    recorder=get_recorder();iid=recorder.begin("speak assistant response",task_id=parent_interaction_id,metadata={"parent_interaction_id":parent_interaction_id,"language":language,"text_length":length})
    action_id=uuid.uuid4().hex;started=time.perf_counter();started_at=datetime.now(timezone.utc).isoformat()
    recorder.record(EventType.TASK_STARTED,{"operation":"speech_output"},iid,parent_interaction_id)
    recorder.record(EventType.TOOL_CALL,{"action_id":action_id,"status":"prepared","name":"speak_response","arguments":{"language":language,"text_length":length},"execution":{"started_at":started_at}},iid,parent_interaction_id)
    return recorder,iid,action_id,started,started_at,parent_interaction_id


def finish_speech(journal,result=None,error=None):
    recorder,iid,action_id,started,started_at,parent=journal;data=result if isinstance(result,dict) else {}
    success=bool(data.get("success")) if isinstance(result,dict) else error is None
    payload={"action_id":action_id,"status":"committed" if success else "failed","name":"speak_response","success":success,"error":str(error) if error else data.get("error"),"metadata":{"provider":data.get("provider"),"attempted_providers":data.get("attempted_providers",[]),"fallback_from":data.get("fallback_from",[]),"spoken_chars":data.get("spoken_chars",0),"resource":data.get("resource"),"resource_wait_ms":data.get("resource_wait_ms",0),"verified":success},"execution":{"started_at":started_at,"finished_at":datetime.now(timezone.utc).isoformat(),"latency_ms":round((time.perf_counter()-started)*1000,3)},"automatically_verified":success}
    recorder.record(EventType.TOOL_RESULT,payload,iid,parent)
    recorder.record(EventType.TASK_COMPLETED if success else EventType.TASK_FAILED,{"success":success,"verified":success},iid,parent)
    recorder.finalize(iid,success,success,parent,"Speech completed." if success else "Speech failed.")
    return success
