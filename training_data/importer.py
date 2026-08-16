"""Optional conservative importer for reconstructable legacy task outcomes."""
import hashlib,json,sqlite3
from .recorder import DatasetRecorder
from .schema import EventType
def _legacy_result_metadata(value):
    raw=str(value or "")
    return {"available":bool(raw),"length":len(raw),"sha256":hashlib.sha256(raw.encode()).hexdigest() if raw else None}
def import_legacy_tasks(memory_database,recorder):
    source=sqlite3.connect(memory_database);source.row_factory=sqlite3.Row;count=0
    for task in source.execute("SELECT * FROM tasks WHERE status IN ('COMPLETED','FAILED')"):
        iid=recorder.begin(task["goal"],task_id=task["id"],metadata={"source":"legacy_task_history","confidence":"limited"})
        success=task["status"]=="COMPLETED";recorder.record(EventType.TASK_COMPLETED if success else EventType.TASK_FAILED,{"message":"Imported completed legacy task." if success else "Imported failed legacy task.","success":success,"verified":False,"imported":True,"legacy_result_metadata":_legacy_result_metadata(task["last_result"])},iid,task["id"],source="legacy_task_history");recorder.finalize(iid,success=False if not success else True,verified=False,task_id=task["id"],response="Imported legacy task outcome.");count+=1
    source.close();recorder.flush();return count
