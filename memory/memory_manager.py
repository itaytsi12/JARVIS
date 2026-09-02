from __future__ import annotations
from config.settings import env_float, env_int
import atexit,json,os,re,shutil,sys,tempfile,uuid
from dataclasses import dataclass
from datetime import datetime,timedelta,timezone
from pathlib import Path
from typing import Protocol
from .database import MemoryDatabase

SECRET_PATTERNS=[
    re.compile(r"(?i)\b(password|passcode|api[ _-]?key|token|2fa|verification code)\b\s*(?:is|=|:)?\s*.+?(?=\s+(?:and then|then)\b|$)"),
    re.compile(r"(?i)\b(?:sk-[a-z0-9_-]{12,}|gh[pousr]_[a-z0-9]{12,})\b"),
    re.compile(r"(?i)\bBearer\s+[a-z0-9._~+/-]+=*"),
    re.compile(r"(?i)\b(?:Cookie|Set-Cookie|Authorization)\s*:\s*[^\r\n]+"),
    re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----[\s\S]*?-----END (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----"),
]
SECRET=SECRET_PATTERNS[0]
def utcnow(): return datetime.now(timezone.utc).isoformat()
def redact(value):
    if isinstance(value,str):
        result=SECRET_PATTERNS[0].sub(lambda m:f"{m.group(1)}=<REDACTED>",value)
        for pattern in SECRET_PATTERNS[1:]:result=pattern.sub("<REDACTED>",result)
        return result
    if isinstance(value,dict): return {k:"<REDACTED>" if SECRET.search(str(k)) else redact(v) for k,v in value.items()}
    if isinstance(value,list): return [redact(v) for v in value]
    return value

class ArchiveProvider(Protocol):
    def store_artifact(self,source:Path,artifact_id:str)->str: ...
    def retrieve_artifact(self,artifact_id:str,destination:Path)->Path: ...
    def delete_artifact(self,artifact_id:str)->None: ...
    def exists(self,artifact_id:str)->bool: ...
class LocalFilesystemArchive:
    def __init__(self,root): self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
    def _path(self,i): return self.root/i
    def store_artifact(self,s,i): shutil.copy2(s,self._path(i)); return str(self._path(i))
    def retrieve_artifact(self,i,d): shutil.copy2(self._path(i),d); return Path(d)
    def delete_artifact(self,i): self._path(i).unlink(missing_ok=True)
    def exists(self,i): return self._path(i).exists()
@dataclass
class Resolution:
    status:str; entity:dict|None=None; candidates:list[dict]|None=None; clarification:str|None=None

class MemoryManager:
    def __init__(self,path=None,archive=None):
        configured=path or os.getenv("MEMORY_DB_PATH");self._test_root=None
        if configured is None and "pytest" in sys.modules:
            self._test_root=Path(tempfile.mkdtemp(prefix="jarvis-memory-pytest-"));configured=self._test_root/"memory.sqlite3"
        self.db=MemoryDatabase(configured or Path.cwd()/"data"/"jarvis_memory.sqlite3")
        if self._test_root is not None:atexit.register(self._cleanup_test_memory)
        self.archive=archive; self.max_local_mb=env_int("MEMORY_MAX_LOCAL_MB", 1024); self.raw_retention_days=env_int("MEMORY_RAW_RETENTION_DAYS", 7); self.session_retention_days=env_int("MEMORY_SESSION_RETENTION_DAYS", 30)
    def _cleanup_test_memory(self):
        if self._test_root is None:return
        try:self.db.close()
        except Exception:pass
        shutil.rmtree(self._test_root,ignore_errors=True);self._test_root=None
    def start_session(self):
        i,n=uuid.uuid4().hex,utcnow(); self.db.execute("INSERT INTO sessions(id,started_at) VALUES(?,?)",(i,n)); return i
    def remember_entity(self,entity_type,name,session_id=None,status="active",**metadata):
        i,n=uuid.uuid4().hex,utcnow(); safe=redact(metadata); self.db.execute("INSERT INTO entities VALUES(?,?,?,?,?,?,?,?)",(i,session_id,entity_type,name,status,n,n,json.dumps(safe))); self.db.execute("INSERT INTO memory_search VALUES('entity',?,?)",(i,f"{entity_type} {name}")); return {"id":i,"type":entity_type,"name":name,"status":status,"metadata":safe}
    def update_entity(self,i,status=None,**metadata):
        rows=self.db.query("SELECT metadata_json FROM entities WHERE id=?",(i,));
        if not rows:return
        current=json.loads(rows[0][0]); current.update(redact(metadata)); self.db.execute("UPDATE entities SET status=COALESCE(?,status),metadata_json=?,updated_at=? WHERE id=?",(status,json.dumps(current),utcnow(),i))
    def record_event(self,kind,message,session_id=None,task_id=None,entity_id=None,**metadata):
        msg,meta=redact(message),redact(metadata); cur=self.db.execute("INSERT INTO events(session_id,task_id,entity_id,kind,message,created_at,metadata_json) VALUES(?,?,?,?,?,?,?)",(session_id,task_id,entity_id,kind,msg,utcnow(),json.dumps(meta))); self.db.execute("INSERT INTO memory_search VALUES('event',?,?)",(str(cur.lastrowid),msg))
    def remember(self,kind,text,key=None,importance=1,**metadata):
        if kind not in {"working","session","episodic","semantic","task"}: raise ValueError("Unknown memory kind")
        now=utcnow(); safe=redact(text); cur=self.db.execute("INSERT INTO memories(kind,key,text,importance,created_at,updated_at,metadata_json) VALUES(?,?,?,?,?,?,?)",(kind,key,safe,importance,now,now,json.dumps(redact(metadata)))); self.db.execute("INSERT INTO memory_search VALUES('memory',?,?)",(str(cur.lastrowid),safe)); return cur.lastrowid
    def resolve(self,reference,session_id=None,entity_type=None,verify_live=False):
        text=reference.lower().strip(); inferred=entity_type; name=None
        if "notepad" in text: inferred,name="application","notepad"
        elif "browser" in text or "chrome" in text: inferred,name="application","chrome"
        elif "youtube search" in text or "last search" in text: inferred="search"
        elif "file" in text: inferred="file"
        elif "project" in text: inferred="project"
        elif "task" in text: inferred="task"
        sql="SELECT * FROM entities WHERE status='active'"; args=[]
        if inferred: sql+=" AND type=?"; args.append(inferred)
        if name: sql+=" AND lower(name)=?"; args.append(name)
        if session_id: sql+=" ORDER BY (session_id=?) DESC,updated_at DESC"; args.append(session_id)
        else: sql+=" ORDER BY updated_at DESC"
        found=[self._entity(r) for r in self.db.query(sql+" LIMIT 10",args)]
        if verify_live:
            live=[]
            for entity in found:
                if self._is_live(entity): live.append(entity)
                else: self.update_entity(entity["id"],status="stale")
            found=live
        if not found:return Resolution("not_found")
        if len(found)>1 and any(w in text for w in ("the notepad","that app","it")): return Resolution("ambiguous",candidates=found,clarification=f"Which {inferred or 'item'} do you mean?")
        return Resolution("resolved",entity=found[0])
    @staticmethod
    def _is_live(entity):
        meta=entity["metadata"]
        if entity["type"]=="file": return bool(meta.get("path") and Path(meta["path"]).exists())
        if entity["type"]=="application" and meta.get("pid"):
            try:
                import psutil
                if not psutil.pid_exists(int(meta["pid"])): return False
            except (ValueError,TypeError): return False
        if entity["type"]=="application" and meta.get("hwnd") and os.name=="nt":
            import ctypes
            if not ctypes.windll.user32.IsWindow(int(meta["hwnd"])): return False
        return True
    @staticmethod
    def _entity(r): return {"id":r["id"],"session_id":r["session_id"],"type":r["type"],"name":r["name"],"status":r["status"],"metadata":json.loads(r["metadata_json"])}
    def search(self,q,limit=20): return [dict(r) for r in self.db.query("SELECT kind,reference_id,text FROM memory_search WHERE memory_search MATCH ? LIMIT ?",(q,limit))]
    def add_artifact(self,path,kind="raw",disposable=True,task_id=None):
        p=Path(path); i=uuid.uuid4().hex; self.db.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?,?)",(i,task_id,str(p),kind,p.stat().st_size,int(disposable),None,utcnow(),"{}")); return i
    def usage_bytes(self):
        ps=[self.db.path,Path(str(self.db.path)+"-wal"),Path(str(self.db.path)+"-shm")]+[Path(r[0]) for r in self.db.query("SELECT path FROM artifacts WHERE path IS NOT NULL")]; return sum(p.stat().st_size for p in ps if p.exists())
    def storage_report(self):
        artifacts=[dict(r) for r in self.db.query("SELECT id,path,size_bytes,disposable,created_at FROM artifacts ORDER BY size_bytes DESC LIMIT 20")]
        usage=self.usage_bytes(); return {"usage_bytes":usage,"limit_bytes":self.max_local_mb*1048576,"over_budget":usage>self.max_local_mb*1048576,"largest_artifacts":artifacts}
    def compact(self):
        cutoff=(datetime.now(timezone.utc)-timedelta(days=self.raw_retention_days)).isoformat(); removed=0
        for r in self.db.query("SELECT * FROM artifacts WHERE disposable=1 AND created_at<?",(cutoff,)):
            p=Path(r["path"]); archived=self.archive.store_artifact(p,r["id"]) if self.archive and p.exists() else None
            if p.exists(): p.unlink(); removed+=r["size_bytes"]
            self.db.execute("UPDATE artifacts SET path=NULL,archived_uri=? WHERE id=?",(archived,r["id"]))
        old_events=self.db.query("SELECT kind,COUNT(*) AS count FROM events WHERE created_at<? AND task_id IS NULL GROUP BY kind",(cutoff,))
        if old_events:
            summary="Compacted events: "+", ".join(f"{r['kind']}={r['count']}" for r in old_events)
            self.db.execute("INSERT INTO summaries(subject_type,subject_id,text,created_at,metadata_json) VALUES('maintenance','events',?,?,?)",(summary,utcnow(),json.dumps({"cutoff":cutoff})))
            self.db.execute("DELETE FROM events WHERE created_at<? AND task_id IS NULL",(cutoff,))
        limit=self.max_local_mb*1048576
        for r in self.db.query("SELECT * FROM artifacts WHERE disposable=1 AND path IS NOT NULL ORDER BY created_at"):
            if self.usage_bytes()<=limit: break
            p=Path(r["path"]); archived=self.archive.store_artifact(p,r["id"]) if self.archive and p.exists() else None
            if p.exists(): p.unlink(); removed+=r["size_bytes"]
            self.db.execute("UPDATE artifacts SET path=NULL,archived_uri=? WHERE id=?",(archived,r["id"]))
        usage=self.usage_bytes(); return {"removed_bytes":removed,"usage_bytes":usage,"within_budget":usage<=limit,"old_event_summary":bool(old_events)}
    def export_obsidian(self,vault_path=None):
        if os.getenv("OBSIDIAN_MEMORY_ENABLED","false").lower() not in {"1","true","yes"}:return []
        base=vault_path or os.getenv("OBSIDIAN_VAULT_PATH");
        if not base:raise ValueError("OBSIDIAN_VAULT_PATH is required")
        root=Path(base)/"JARVIS Memory"/"Tasks"; root.mkdir(parents=True,exist_ok=True); out=[]
        for r in self.db.query("SELECT id,goal,status,updated_at,last_result FROM tasks ORDER BY updated_at DESC"):
            p=root/f"{r['id']}.md"; p.write_text(f"# {redact(r['goal'])}\n\nStatus: {r['status']}\nLast worked: {r['updated_at']}\n\n{redact(r['last_result'] or '')}\n",encoding="utf-8"); out.append(p)
        return out
