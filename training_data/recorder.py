from __future__ import annotations
import atexit,hashlib,json,logging,os,queue,shutil,sqlite3,sys,tempfile,threading,time,uuid
from collections import deque
from datetime import datetime,timedelta,timezone
from pathlib import Path
from .sanitizer import sanitize,sanitize_user_request
from .sanitizer import code_reference
from .schema import CAPTURE_VERSION,DATASET_VERSION,SCHEMA_VERSION,EventType,QualityLabel

def now():return datetime.now(timezone.utc).isoformat()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False)
def signature(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def numeric_metric(value):return int(value) if isinstance(value,(int,float)) and not isinstance(value,bool) else 0
class DatasetRecorder:
    def __init__(self,path=None,enabled=None,async_writes=True):
        self.enabled=(os.getenv("TRAINING_DATA_ENABLED","true").lower() in {"1","true","yes"}) if enabled is None else enabled
        self.path=Path(path or os.getenv("TRAINING_DATA_DB_PATH") or Path.cwd()/"data"/"training_dataset.sqlite3")
        self.dedup=os.getenv("TRAINING_DEDUP_ENABLED","true").lower() in {"1","true","yes"}; self.max_mb=int(os.getenv("TRAINING_MAX_LOCAL_MB","512")); self.raw_days=int(os.getenv("TRAINING_RAW_RETENTION_DAYS","30")); self.capture_audio=os.getenv("TRAINING_CAPTURE_AUDIO","false").lower() in {"1","true","yes"}
        self.log=logging.getLogger("jarvis.training_data"); self._seq={}; self._lock=threading.Lock(); self._db_lock=threading.RLock(); self._queue=queue.Queue(maxsize=10000);self._overflow=deque(maxlen=1000);self._overflow_lock=threading.Lock();self._dropped_events=0; self._disabled_reason=None; self._async=async_writes; self._connection=None;self._closed=False
        if self.enabled:
            try:self._open()
            except Exception as e:self._degrade(e)
        self._thread=None
        if self.enabled and async_writes:
            self._thread=threading.Thread(target=self._worker,name="jarvis-dataset",daemon=True); self._thread.start()
        if self.enabled:atexit.register(self.close)
    def _open(self):
        self.path.parent.mkdir(parents=True,exist_ok=True); self._connection=sqlite3.connect(self.path,check_same_thread=False); self._connection.row_factory=sqlite3.Row; self._connection.execute("PRAGMA journal_mode=WAL"); self._connection.executescript("""
        CREATE TABLE IF NOT EXISTS dataset_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS raw_events(event_id TEXT PRIMARY KEY,interaction_id TEXT NOT NULL,task_id TEXT,session_id TEXT,parent_event_id TEXT,sequence_number INTEGER NOT NULL,timestamp TEXT NOT NULL,event_type TEXT NOT NULL,payload_json TEXT NOT NULL,signature TEXT NOT NULL,duplicate_count INTEGER NOT NULL DEFAULT 0,source TEXT NOT NULL DEFAULT 'runtime');
        CREATE UNIQUE INDEX IF NOT EXISTS raw_order ON raw_events(interaction_id,sequence_number); CREATE INDEX IF NOT EXISTS raw_task ON raw_events(task_id,sequence_number);
        CREATE TABLE IF NOT EXISTS training_examples(example_id TEXT PRIMARY KEY,interaction_id TEXT NOT NULL,task_id TEXT,category TEXT NOT NULL,quality_label TEXT NOT NULL,training_eligible INTEGER NOT NULL,quality_score REAL NOT NULL,quality_evidence_json TEXT NOT NULL,input_json TEXT NOT NULL,context_json TEXT NOT NULL,output_json TEXT NOT NULL,event_ids_json TEXT NOT NULL,signature TEXT NOT NULL,duplicate_count INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL,schema_version INTEGER NOT NULL,dataset_version TEXT NOT NULL,capture_version TEXT NOT NULL,source TEXT NOT NULL DEFAULT 'runtime');
        CREATE INDEX IF NOT EXISTS examples_quality ON training_examples(training_eligible,quality_label,category);
        CREATE TABLE IF NOT EXISTS content_blobs(content_hash TEXT PRIMARY KEY,content TEXT NOT NULL,source_project TEXT,source_file TEXT,created_at TEXT NOT NULL);
        """);
        for k,v in {"schema_version":SCHEMA_VERSION,"dataset_version":DATASET_VERSION,"capture_version":CAPTURE_VERSION}.items():self._connection.execute("INSERT OR REPLACE INTO dataset_meta VALUES(?,?)",(k,str(v)))
        self._repair_sequences()
        self._connection.commit()
    def _repair_sequences(self):
        for iid, in self._connection.execute("SELECT DISTINCT interaction_id FROM raw_events"):
            rows=list(self._connection.execute("SELECT event_id FROM raw_events WHERE interaction_id=? ORDER BY sequence_number,timestamp,event_id",(iid,)))
            for number,row in enumerate(rows,1):self._connection.execute("UPDATE raw_events SET sequence_number=? WHERE event_id=?",(-number,row[0]))
            for number,row in enumerate(rows,1):self._connection.execute("UPDATE raw_events SET sequence_number=? WHERE event_id=?",(number,row[0]))
    def _degrade(self,error):self.enabled=False; self._disabled_reason=str(error); self.log.exception("Dataset capture disabled after recorder failure")
    def begin(self,original_user_text,session_id=None,task_id=None,metadata=None):
        if not self.enabled:return None
        iid=uuid.uuid4().hex; self.record(EventType.USER_REQUEST,{"original_user_text":sanitize_user_request(original_user_text),"metadata":metadata or {}},iid,task_id,session_id); return iid
    def record(self,event_type,payload,interaction_id,task_id=None,session_id=None,parent_event_id=None,source="runtime"):
        try:return self._record(event_type,payload,interaction_id,task_id,session_id,parent_event_id,source)
        except Exception as error:self._degrade(error); return None
    def _record(self,event_type,payload,interaction_id,task_id=None,session_id=None,parent_event_id=None,source="runtime"):
        if not self.enabled:return None
        kind=event_type.value if hasattr(event_type,"value") else str(event_type); safe=sanitize(payload)
        with self._lock:
            if interaction_id not in self._seq:
                with self._db_lock:self._seq[interaction_id]=self._connection.execute("SELECT COALESCE(MAX(sequence_number),0) FROM raw_events WHERE interaction_id=?",(interaction_id,)).fetchone()[0]
            seq=self._seq[interaction_id]+1; self._seq[interaction_id]=seq
        row={"event_id":uuid.uuid4().hex,"interaction_id":interaction_id,"task_id":task_id,"session_id":session_id,"parent_event_id":parent_event_id,"sequence_number":seq,"timestamp":now(),"event_type":kind,"payload":safe,"signature":signature({"type":kind,"payload":safe}),"source":source}
        self._submit("event",row); return row["event_id"]
    def finalize(self,interaction_id,success=None,verified=False,task_id=None,response=None):
        if response is not None:self.record(EventType.FINAL_RESPONSE,{"response":response,"success":success,"verified":verified},interaction_id,task_id)
        self._submit("curate",{"interaction_id":interaction_id,"task_id":task_id})
    def correction(self,interaction_id,original,correction,successful=False):
        self.record(EventType.USER_CORRECTION,{"original":original,"correction":correction,"successful":successful},interaction_id)
    def approval(self,interaction_id,approved,details=None):self.record(EventType.USER_APPROVAL if approved else EventType.USER_REJECTION,{"details":details or {}},interaction_id)
    def store_code_context(self,source_project,source_file,content=None):
        if not self.enabled or os.getenv("TRAINING_CAPTURE_CODE_CONTEXT","true").lower() not in {"1","true","yes"}:return None
        if Path(source_file).name.lower()==".env" or Path(source_file).suffix.lower() in {".pem",".key"}:return None
        try:
            ref=code_reference(source_file,content)
            with self._db_lock,self._connection:self._connection.execute("INSERT OR IGNORE INTO content_blobs VALUES(?,?,?,?,?)",(ref["content_hash"],ref.pop("content"),sanitize(source_project),sanitize(source_file),now()))
            return ref
        except Exception as error:self._degrade(error); return None
    def _submit(self,kind,data):
        if not self.enabled:return
        if self._async:
            try:self._queue.put_nowait((kind,data))
            except queue.Full:
                with self._overflow_lock:
                    if len(self._overflow)==self._overflow.maxlen:self._dropped_events+=1
                    self._overflow.append((kind,data))
                self.log.warning("Dataset queue full; event moved to bounded overflow buffer")
        else:self._handle(kind,data)
    def _worker(self):
        while True:
            item=self._queue.get()
            try:
                if item is None:return
                self._handle(*item)
                while True:
                    with self._overflow_lock:
                        overflow_item=self._overflow.popleft() if self._overflow else None
                    if overflow_item is None:break
                    self._handle(*overflow_item)
            except Exception as e:self._degrade(e)
            finally:self._queue.task_done()
    def _handle(self,kind,data):
        with self._db_lock:
            if kind=="event":self._write_event(data)
            elif kind=="curate":self._curate(data["interaction_id"],data.get("task_id"))
    def _write_event(self,r):
        with self._connection:
            duplicate_count=self._connection.execute("SELECT COUNT(*) FROM raw_events WHERE signature=?",(r["signature"],)).fetchone()[0] if self.dedup else 0
            self._connection.execute("INSERT INTO raw_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(r["event_id"],r["interaction_id"],r["task_id"],r["session_id"],r["parent_event_id"],r["sequence_number"],r["timestamp"],r["event_type"],canonical(r["payload"]),r["signature"],duplicate_count,r["source"]))
    def _curate(self,iid,task_id=None):
        rows=list(self._connection.execute("SELECT * FROM raw_events WHERE interaction_id=? ORDER BY sequence_number",(iid,)))
        if not rows:return
        events=[{"id":r["event_id"],"type":r["event_type"],"payload":json.loads(r["payload_json"])} for r in rows]; types={e["type"] for e in events}; user=next((e["payload"] for e in events if e["type"]=="USER_REQUEST"),{})
        final=next((e["payload"] for e in reversed(events) if e["type"] in {"FINAL_RESPONSE","TASK_COMPLETED","TASK_FAILED"}),{})
        passed=any(e["type"]=="TEST_RESULT" and e["payload"].get("exit_code")==0 for e in events); regressed=any(e["payload"].get("regression_rolled_back") for e in events); approved="USER_APPROVAL" in types; rejected="USER_REJECTION" in types
        success=final.get("success") is True if isinstance(final.get("success"),bool) else "TASK_COMPLETED" in types; verified=bool(final.get("verified")) or passed or approved
        if rejected:label=QualityLabel.USER_REJECTED
        elif approved:label=QualityLabel.USER_APPROVED
        elif "USER_CORRECTION" in types:label=QualityLabel.CORRECTED
        elif regressed and not success:label=QualityLabel.ROLLED_BACK
        elif verified and success:label=QualityLabel.VERIFIED_SUCCESS
        elif success:label=QualityLabel.SUCCESS
        elif "TASK_BLOCKED" in types:label=QualityLabel.BLOCKED
        else:label=QualityLabel.FAILED
        eligible=label in {QualityLabel.VERIFIED_SUCCESS,QualityLabel.USER_APPROVED} or (label==QualityLabel.CORRECTED and success)
        evidence=[]
        if passed:evidence.append("tests_passed")
        if approved:evidence.append("user_approved")
        if success:evidence.append("task_or_tool_succeeded")
        if regressed:evidence.append("regression_rolled_back")
        category="coding" if types & {"CODE_PATCH","TEST_RUN","TEST_RESULT"} else "tool" if "TOOL_CALL" in types else "correction" if "USER_CORRECTION" in types else "personalization" if "RESOLVED_REFERENCES" in types and success else "interaction"
        output={"final":final,"events":[e for e in events if e["type"] in {"PLAN_CREATED","TOOL_CALL","TOOL_RESULT","CODE_PATCH","TEST_RESULT","FINAL_RESPONSE"}]}; semantic_output={"final":final,"events":[{"type":e["type"],"payload":e["payload"]} for e in output["events"]]}; sig=signature({"input":user,"output":semantic_output,"category":category})
        with self._connection:
            duplicate_count=self._connection.execute("SELECT COUNT(*) FROM training_examples WHERE signature=?",(sig,)).fetchone()[0] if self.dedup else 0
            score=1.0 if eligible and passed else .9 if eligible else .1 if regressed or rejected else .4 if success else 0.0
            self._connection.execute("INSERT INTO training_examples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(uuid.uuid4().hex,iid,task_id,category,label.value,int(eligible),score,canonical(evidence),canonical(user),canonical(user.get("metadata",{})),canonical(output),canonical([e["id"] for e in events]),sig,duplicate_count,now(),SCHEMA_VERSION,DATASET_VERSION,CAPTURE_VERSION,"runtime"))
    def flush(self):
        if self._async:self._queue.join()
    def stats(self):
        self.flush(); c=self._connection
        with self._db_lock:
            labels={row[0]:row[1] for row in c.execute("SELECT quality_label,COUNT(*) FROM training_examples GROUP BY quality_label")}
            action_types={}
            routes={}
            interaction_features={}
            model_operations={}
            model_usage={}
            ui_tools={"focus_application","wait_for_window","type_text","press_key","click_at","click_ui_element","inspect_window","browser_click","browser_type","browser_select","browser_scroll"}
            for interaction_id,event_type,payload_json in c.execute("SELECT interaction_id,event_type,payload_json FROM raw_events"):
                payload=json.loads(payload_json)
                feature=interaction_features.setdefault(interaction_id,{"actions":set(),"multi_step":False,"correction":False,"cancellation":False,"action_plan":False})
                if event_type=="TOOL_CALL" and payload.get("name"):
                    name=payload["name"];action_types[name]=action_types.get(name,0)+1;feature["actions"].add(name)
                if event_type=="PLAN_CREATED" and (payload.get("route_source") or payload.get("route_type")):
                    route=payload.get("route_source") or payload["route_type"];routes[route]=routes.get(route,0)+1
                if event_type=="PLAN_CREATED" and isinstance(payload.get("actions"),list):feature["action_plan"]=True;feature["multi_step"]|=len(payload["actions"])>1
                if event_type=="USER_CORRECTION":feature["correction"]=True
                if event_type=="TASK_CANCELLED" or event_type=="TOOL_CALL" and payload.get("name")=="cancel_active_task":feature["cancellation"]=True
                if event_type=="REASONING_REQUEST":
                    operation=payload.get("operation") or payload.get("provider") or "unspecified";model_operations[operation]=model_operations.get(operation,0)+1
                    usage=model_usage.setdefault(operation,{"calls":0,"input_tokens":0,"output_tokens":0});usage["calls"]+=numeric_metric(payload.get("model_calls",1));usage["input_tokens"]+=numeric_metric(payload.get("input_tokens",0));usage["output_tokens"]+=numeric_metric(payload.get("output_tokens",0))
                if event_type=="REASONING_RESPONSE" and payload.get("operation"):
                    usage=model_usage.setdefault(payload["operation"],{"calls":0,"input_tokens":0,"output_tokens":0});usage["input_tokens"]+=numeric_metric(payload.get("input_tokens",0));usage["output_tokens"]+=numeric_metric(payload.get("output_tokens",0))
                if event_type=="TOOL_RESULT" and isinstance(payload.get("metadata"),dict):
                    metadata=payload["metadata"]
                    if metadata.get("model_calls") and payload.get("name")!="web_search":
                        operation=payload.get("name") or "tool_model";usage=model_usage.setdefault(operation,{"calls":0,"input_tokens":0,"output_tokens":0});usage["calls"]+=numeric_metric(metadata.get("model_calls",0));usage["input_tokens"]+=numeric_metric(metadata.get("input_tokens",0));usage["output_tokens"]+=numeric_metric(metadata.get("output_tokens",0))
                    if metadata.get("translated") and metadata.get("translation_model"):
                        usage=model_usage.setdefault("message_translation",{"calls":0,"input_tokens":0,"output_tokens":0});usage["calls"]+=1;usage["input_tokens"]+=numeric_metric(metadata.get("translation_input_tokens",0));usage["output_tokens"]+=numeric_metric(metadata.get("translation_output_tokens",0))
            total=c.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0];verified=labels.get("VERIFIED_SUCCESS",0)+labels.get("USER_APPROVED",0)
            outcomes=[]
            for output_json, in c.execute("SELECT output_json FROM training_examples"):
                final=json.loads(output_json).get("final",{});outcomes.append(final.get("success"))
            successful_records=sum(value is True for value in outcomes);failure_records=sum(value is False for value in outcomes)
            with self._overflow_lock:overflow_pending=len(self._overflow)
            raw_events=c.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0];oldest,newest=c.execute("SELECT MIN(timestamp),MAX(timestamp) FROM raw_events").fetchone();size_bytes=sum(p.stat().st_size for p in (self.path,Path(str(self.path)+'-wal'),Path(str(self.path)+'-shm')) if p.exists());span_seconds=max(0.0,(datetime.fromisoformat(newest)-datetime.fromisoformat(oldest)).total_seconds()) if oldest and newest else 0.0
            stored_versions=[{"schema_version":row[0],"dataset_version":row[1],"capture_version":row[2],"count":row[3]} for row in c.execute("SELECT schema_version,dataset_version,capture_version,COUNT(*) FROM training_examples GROUP BY schema_version,dataset_version,capture_version ORDER BY schema_version,dataset_version,capture_version")]
            return {"schema_version":SCHEMA_VERSION,"dataset_version":DATASET_VERSION,"capture_version":CAPTURE_VERSION,"stored_example_versions":stored_versions,"raw_events":raw_events,"raw_interactions":c.execute("SELECT COUNT(DISTINCT interaction_id) FROM raw_events").fetchone()[0],"curated_examples":total,"eligible_examples":c.execute("SELECT COUNT(*) FROM training_examples WHERE training_eligible=1").fetchone()[0],"successful_records":successful_records,"verified_records":verified,"unverified_records":total-verified,"failures":failure_records,"corrections":sum(f["correction"] for f in interaction_features.values()),"cancellations":sum(f["cancellation"] for f in interaction_features.values()),"multi_step_examples":sum(f["multi_step"] for f in interaction_features.values()),"ui_examples":sum(bool(f["actions"]&ui_tools) for f in interaction_features.values()),"action_plan_examples":sum(f["action_plan"] for f in interaction_features.values()),"coding_examples":c.execute("SELECT COUNT(*) FROM training_examples WHERE category='coding'").fetchone()[0],"eligible_coding_examples":c.execute("SELECT COUNT(*) FROM training_examples WHERE category='coding' AND training_eligible=1").fetchone()[0],"quality_labels":labels,"action_types":action_types,"planner_routes":routes,"model_operations":model_operations,"model_usage":model_usage,"oldest_event_at":oldest,"newest_event_at":newest,"observed_event_span_seconds":round(span_seconds,3),"average_bytes_per_raw_event":round(size_bytes/raw_events,1) if raw_events else 0.0,"overflow_pending":overflow_pending,"dropped_events":self._dropped_events,"size_bytes":size_bytes}
    def cleanup(self):
        self.flush(); cutoff=(datetime.now(timezone.utc)-timedelta(days=self.raw_days)).isoformat()
        with self._connection:self._connection.execute("DELETE FROM raw_events WHERE timestamp<? AND interaction_id IN (SELECT interaction_id FROM training_examples)",(cutoff,))
        if self.stats()["size_bytes"]>self.max_mb*1048576:
            with self._connection:self._connection.execute("DELETE FROM raw_events WHERE interaction_id IN (SELECT interaction_id FROM training_examples)")
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._connection.execute("VACUUM")
        return self.stats()
    def close(self):
        if self._closed:return
        self._closed=True
        if self._thread:
            deadline=time.monotonic()+5
            while self._queue.unfinished_tasks and self._thread.is_alive() and time.monotonic()<deadline:time.sleep(.01)
            if self._queue.unfinished_tasks:
                with self._overflow_lock:pending_overflow=len(self._overflow)
                self._dropped_events+=self._queue.qsize()+pending_overflow
                self.log.error("Dataset writer did not drain before shutdown; pending events were counted as dropped")
            if self._thread.is_alive():
                try:self._queue.put_nowait(None)
                except queue.Full:self._queue.put(None,timeout=1)
                self._thread.join(timeout=3)
            if self._thread.is_alive():
                self.log.error("Dataset writer is still alive; leaving SQLite connection open to avoid a close/write race")
                return
            self._thread=None
        if self._connection:
            try:self._cleanup_sync()
            finally:self._connection.close();self._connection=None
    def _cleanup_sync(self):
        cutoff=(datetime.now(timezone.utc)-timedelta(days=self.raw_days)).isoformat()
        with self._connection:self._connection.execute("DELETE FROM raw_events WHERE timestamp<? AND interaction_id IN (SELECT interaction_id FROM training_examples)",(cutoff,))
        size=sum(p.stat().st_size for p in (self.path,Path(str(self.path)+'-wal'),Path(str(self.path)+'-shm')) if p.exists())
        if size>self.max_mb*1048576:
            with self._connection:self._connection.execute("DELETE FROM raw_events WHERE interaction_id IN (SELECT interaction_id FROM training_examples)")
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)");self._connection.execute("VACUUM")

_RECORDER=None
def get_recorder():
    global _RECORDER
    if _RECORDER is None:
        test_root=Path(tempfile.mkdtemp(prefix="jarvis-training-pytest-")) if "pytest" in sys.modules and not os.getenv("TRAINING_DATA_DB_PATH") else None
        test_path=test_root/"training_dataset.sqlite3" if test_root else None
        _RECORDER=DatasetRecorder(test_path)
        if test_root is not None:atexit.register(_cleanup_test_recorder,_RECORDER,test_root)
    return _RECORDER
def _cleanup_test_recorder(recorder,root):
    try:recorder.close()
    finally:shutil.rmtree(root,ignore_errors=True)
