from __future__ import annotations
import json,os,shlex,subprocess,sys,time,uuid,threading
from datetime import datetime,timezone
from concurrent.futures import Future,ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum,IntEnum
from pathlib import Path
from typing import Protocol
from memory.memory_manager import MemoryManager,redact,utcnow
from tools.windows_process import hidden_process_kwargs
from training_data import EventType,get_recorder
from brain.resource_locks import acquire_named_resource

class TaskStatus(str,Enum):
    QUEUED="QUEUED"; RUNNING="RUNNING"; PAUSED="PAUSED"; BLOCKED="BLOCKED"; WAITING_FOR_APPROVAL="WAITING_FOR_APPROVAL"; COMPLETED="COMPLETED"; FAILED="FAILED"; CANCELLED="CANCELLED"
TERMINAL={TaskStatus.COMPLETED.value,TaskStatus.FAILED.value,TaskStatus.CANCELLED.value}
APPROVAL_TOKENS=("pip install","uninstall",".env","git push","git merge","git reset","git clean","rm ","del ","remove-item")
SAFE_COMMANDS={"pytest","python","py","git"}
READ_ONLY_GIT_COMMANDS={"status","diff","log","show","rev-parse","ls-files","grep"}

class ReadOnlyPriority(IntEnum):
    INTERACTIVE=0
    BACKGROUND=10

class CancellationToken:
    def __init__(self): self._cancelled=threading.Event()
    def cancel(self): self._cancelled.set()
    @property
    def cancelled(self): return self._cancelled.is_set()
    def raise_if_cancelled(self):
        if self.cancelled: raise ReadOnlyTaskCancelled("Read-only task was cancelled")

class ReadOnlyTaskCancelled(RuntimeError): pass

@dataclass
class ReadOnlyTaskHandle:
    id:str
    priority:ReadOnlyPriority
    token:CancellationToken
    future:Future
    def cancel(self):
        self.token.cancel(); self.future.cancel(); return True
    def result(self,timeout=None): return self.future.result(timeout=timeout)
    @property
    def cancelled(self): return self.token.cancelled or self.future.cancelled()
    @property
    def done(self): return self.future.done()

class ReadOnlyTaskScheduler:
    """Independent lanes for interactive and background read-only operations.

    Interactive questions never acquire or modify desktop/runtime locks. A new
    interactive request cooperatively cancels lower-priority read-only work and
    runs in its reserved lane, so a slow background read cannot delay voice.
    """
    def __init__(self):
        self._interactive=ThreadPoolExecutor(max_workers=1,thread_name_prefix="jarvis-question")
        self._background=ThreadPoolExecutor(max_workers=1,thread_name_prefix="jarvis-readonly")
        self._lock=threading.Lock(); self._tasks={}
    def submit(self,operation,priority=ReadOnlyPriority.BACKGROUND,interrupt_lower=False):
        priority=ReadOnlyPriority(priority)
        if interrupt_lower:self.cancel_lower_priority(priority)
        token=CancellationToken(); task_id=uuid.uuid4().hex
        executor=self._interactive if priority is ReadOnlyPriority.INTERACTIVE else self._background
        future=executor.submit(self._run,token,operation)
        handle=ReadOnlyTaskHandle(task_id,priority,token,future)
        with self._lock:self._tasks[task_id]=handle
        future.add_done_callback(lambda _future:self._discard(task_id))
        return handle
    @staticmethod
    def _run(token,operation):
        token.raise_if_cancelled(); result=operation(token); token.raise_if_cancelled(); return result
    def _discard(self,task_id):
        with self._lock:self._tasks.pop(task_id,None)
    def cancel(self,task_id):
        with self._lock:handle=self._tasks.get(task_id)
        return handle.cancel() if handle else False
    def cancel_lower_priority(self,priority):
        with self._lock:handles=list(self._tasks.values())
        cancelled=[]
        for handle in handles:
            if handle.priority>priority and not handle.done:
                handle.cancel(); cancelled.append(handle.id)
        return cancelled
    def cancel_all(self):
        with self._lock:handles=list(self._tasks.values())
        for handle in handles:handle.cancel()
        return len(handles)
    def snapshot(self):
        with self._lock:handles=list(self._tasks.values())
        return [
            {"id":handle.id,"priority":handle.priority.name.lower(),"done":handle.done,"cancelled":handle.cancelled}
            for handle in handles if not handle.done
        ]
    def shutdown(self,wait=True):
        self.cancel_all(); self._interactive.shutdown(wait=wait,cancel_futures=True); self._background.shutdown(wait=wait,cancel_futures=True)

_READ_ONLY_SCHEDULER=None
_READ_ONLY_SCHEDULER_LOCK=threading.Lock()
_INTERACTIVE_TOKENS={}
_INTERACTIVE_TOKENS_LOCK=threading.Lock()
def register_interactive_task(token):
    task_id=uuid.uuid4().hex
    with _INTERACTIVE_TOKENS_LOCK:_INTERACTIVE_TOKENS[task_id]=token
    return task_id
def unregister_interactive_task(task_id):
    with _INTERACTIVE_TOKENS_LOCK:_INTERACTIVE_TOKENS.pop(task_id,None)
def wait_for_interactive_tasks(timeout=2.0):
    deadline=time.perf_counter()+timeout
    while time.perf_counter()<deadline:
        with _INTERACTIVE_TOKENS_LOCK:
            if not _INTERACTIVE_TOKENS:return True
        time.sleep(.01)
    return False
def get_read_only_task_scheduler():
    global _READ_ONLY_SCHEDULER
    if _READ_ONLY_SCHEDULER is None:
        with _READ_ONLY_SCHEDULER_LOCK:
            if _READ_ONLY_SCHEDULER is None:_READ_ONLY_SCHEDULER=ReadOnlyTaskScheduler()
    return _READ_ONLY_SCHEDULER
def cancel_read_only_tasks():
    count=_READ_ONLY_SCHEDULER.cancel_all() if _READ_ONLY_SCHEDULER is not None else 0
    with _INTERACTIVE_TOKENS_LOCK:tokens=list(_INTERACTIVE_TOKENS.values())
    for token in tokens:
        if not token.cancelled:token.cancel();count+=1
    return count
def active_interactive_task_summary():
    scheduled=_READ_ONLY_SCHEDULER.snapshot() if _READ_ONLY_SCHEDULER is not None else []
    with _INTERACTIVE_TOKENS_LOCK:
        action_count=sum(not token.cancelled for token in _INTERACTIVE_TOKENS.values())
    return {"read_only_tasks":scheduled,"interactive_action_count":action_count,"total":len(scheduled)+action_count}

def any_active_interactive_work() -> bool:
    """Is JARVIS actively working on or speaking a response right now?

    This is `brain/router.py`'s task-priority guard: an ambiguous command
    like bare "pause" must resolve to the active JARVIS task instead of a
    media control, but only while there genuinely IS one. Combines every
    signal that answers that -- read-only tasks/questions (this module),
    agent tasks (`tasks/manager.py`, imported lazily the same way
    `brain/agent.py::_active_agent_tasks` already does, so this module
    never takes a hard dependency on the newer `tasks` package), and
    whether JARVIS is currently speaking (`brain/activity_state.py`, set by
    the voice layer). Never raises -- a broken status lookup must not block
    an ordinary command from routing.
    """
    try:
        if active_interactive_task_summary()["total"] > 0:
            return True
    except Exception:
        pass
    try:
        from tasks.manager import get_task_manager
        from tasks.models import TaskStatus
        if any(task.status is TaskStatus.RUNNING for task in get_task_manager().active()):
            return True
    except Exception:
        pass
    try:
        from brain.activity_state import is_speaking
        if is_speaking():
            return True
    except Exception:
        pass
    return False

@dataclass
class ProposedStep:
    kind:str
    path:str|None=None
    content:str|None=None
    command:list[str]|None=None
    hypothesis:str=""

class ReasoningBackend(Protocol):
    def propose_next_step(self,context:dict)->ProposedStep: ...
    def analyze_failure(self,context:dict)->str: ...
    def summarize_iteration(self,context:dict)->str: ...
class DeterministicBackend:
    def __init__(self,steps): self.steps=list(steps)
    def propose_next_step(self,context): return self.steps.pop(0) if self.steps else ProposedStep("complete")
    def analyze_failure(self,context): return "The previous deterministic step failed."
    def summarize_iteration(self,context): return str(context.get("last_result", ""))[:1000]

class NotificationProvider(Protocol):
    def notify_task_completed(self,task): ...
    def notify_task_blocked(self,task): ...
    def notify_approval_required(self,task): ...
class LoggingNotificationProvider:
    def __init__(self,logger=None):
        import logging; self.log=logger or logging.getLogger("jarvis.tasks")
    def notify_task_completed(self,t): self.log.info("Task completed: %s",t["id"])
    def notify_task_blocked(self,t): self.log.warning("Task blocked: %s",t["id"])
    def notify_approval_required(self,t): self.log.warning("Task approval required: %s",t["id"])

class TaskStore:
    def __init__(self,memory:MemoryManager,recorder=None): self.memory=memory;self.recorder=recorder or get_recorder()
    def create(self,goal,workspace,**options):
        i,n=uuid.uuid4().hex,utcnow(); repo=str(Path(workspace).resolve())
        values=(i,redact(goal),TaskStatus.QUEUED.value,n,n,None,None,repo,repo,None,json.dumps(options.get("allowed_paths",[repo])),json.dumps(options.get("forbidden_paths",[".env"])),json.dumps(options.get("allowed_commands",["pytest","python","git"])),options.get("max_runtime",3600),options.get("max_iterations",20),options.get("success_criteria"),0,None,None,None,0,"{}")
        self.memory.db.execute("INSERT INTO tasks VALUES("+",".join("?"*22)+")",values); return self.get(i)
    def get(self,i):
        rows=self.memory.db.query("SELECT * FROM tasks WHERE id=?",(i,)); return self._decode(rows[0]) if rows else None
    def list(self,status=None):
        rows=self.memory.db.query("SELECT * FROM tasks"+(" WHERE status=?" if status else "")+" ORDER BY updated_at DESC",(status,) if status else ()); return [self._decode(r) for r in rows]
    def update(self,i,**fields):
        if not fields:return self.get(i)
        fields["updated_at"]=utcnow(); self.memory.db.execute("UPDATE tasks SET "+",".join(f"{k}=?" for k in fields)+" WHERE id=?",tuple(fields.values())+(i,)); return self.get(i)
    def command(self,verb,task_id=None):
        if verb=="list": return self.list()
        if not task_id:return None
        if verb=="status" or verb=="last_result":return self.get(task_id)
        current=self.get(task_id)
        if current and current["status"] in TERMINAL:return current
        status={"pause":TaskStatus.PAUSED.value,"cancel":TaskStatus.CANCELLED.value}.get(verb)
        if not status:return None
        iid=self.recorder.begin(f"{verb} task",task_id=task_id,metadata={"operation":f"task_{verb}","entrypoint":"task_store_command"});updated=self.update(task_id,status=status);event=EventType.TASK_PAUSED if verb=="pause" else EventType.TASK_CANCELLED;self.recorder.record(event,{"status":updated["status"],"entrypoint":"task_store_command"},iid,task_id);self.recorder.finalize(iid,True,True,task_id,"Task paused." if verb=="pause" else "Task cancelled.");return updated
    @staticmethod
    def _decode(r):
        d=dict(r)
        for k in ("allowed_paths_json","forbidden_paths_json","allowed_commands_json","checkpoint_json"): d[k[:-5]]=json.loads(d.pop(k))
        d["requires_approval"]=bool(d["requires_approval"]); return d

def _resolve_interpreter(args):
    """Run `python ...` with the interpreter JARVIS is ACTUALLY running under.

    Every caller here spells the command `["python", "-m", "pytest", ...]`,
    which resolves through PATH to whatever `python.exe` happens to come
    first -- on this machine, a tool runtime with no pytest installed, so
    every real reproduction, regression check and benchmark run reported
    `No module named pytest` and was scored as a genuine test failure. The
    interpreter running JARVIS is the one that has JARVIS's dependencies,
    and it is the only defensible answer to "which python".

    Only a BARE `python`/`py` is substituted: an explicit path (a
    worktree's own venv, say) is a deliberate choice and is left alone.
    """
    if not args:
        return args
    first = str(args[0])
    if first.lower() in {"python", "py", "python3", "python.exe", "py.exe"} and not os.path.sep in first:
        return [sys.executable, *args[1:]]
    return list(args)


class SafeCommandRunner:
    def run(self,command,workspace,timeout=300,allowed=None):
        args=shlex.split(command) if isinstance(command,str) else list(command); allowed=set(allowed or SAFE_COMMANDS)
        if not args: raise PermissionError("Command is not allowlisted")
        executable=Path(args[0]).name.lower(); executable=executable[:-4] if executable.endswith(".exe") else executable
        if executable not in allowed: raise PermissionError("Command is not allowlisted")
        if executable=="git" and (len(args)<2 or args[1].lower() not in READ_ONLY_GIT_COMMANDS):raise PermissionError("Git subcommand requires approval")
        if executable=="git" and any(arg.lower()=="--ext-diff" or arg.lower().startswith(("--output","--open-files-in-pager")) for arg in args[2:]):raise PermissionError("Git option requires approval")
        joined=" ".join(args).lower()
        if any(token in joined for token in APPROVAL_TOKENS): raise PermissionError("Command requires approval")
        args=_resolve_interpreter(args)
        started=time.perf_counter(); result=subprocess.run(args,cwd=workspace,text=True,capture_output=True,timeout=timeout,**hidden_process_kwargs())
        output=redact((result.stdout+"\n"+result.stderr).strip()); limit=12000
        return {"exit_code":result.returncode,"duration":time.perf_counter()-started,"output":output[-limit:],"truncated":len(output)>limit}

class TaskSupervisor:
    def __init__(self,store,backend,runner=None,notifier=None,recorder=None): self.store=store; self.backend=backend; self.runner=runner or SafeCommandRunner(); self.notifier=notifier or LoggingNotificationProvider(); self.recorder=recorder or get_recorder()
    def pause(self,i):
        current=self.store.get(i)
        if current and current["status"] in TERMINAL:return current
        iid=self.recorder.begin("pause task",task_id=i,metadata={"operation":"task_pause"});task=self.store.update(i,status=TaskStatus.PAUSED.value);self.recorder.record(EventType.TASK_PAUSED,{"status":task["status"]},iid,i);self.recorder.finalize(iid,True,True,i,"Task paused.");return task
    def cancel(self,i):
        current=self.store.get(i)
        if current and current["status"] in TERMINAL:return current
        iid=self.recorder.begin("cancel task",task_id=i,metadata={"operation":"task_cancel"});task=self.store.update(i,status=TaskStatus.CANCELLED.value);self.recorder.record(EventType.TASK_CANCELLED,{"status":task["status"]},iid,i);self.recorder.finalize(iid,True,True,i,"Task cancelled.");return task
    def resume(self,i):
        task=self.store.get(i)
        if task["status"] in TERMINAL: return task
        if task["status"]==TaskStatus.WAITING_FOR_APPROVAL.value and task["requires_approval"]: return task
        iid=self.recorder.begin("resume task",task_id=i,metadata={"operation":"task_resume"});self.recorder.record(EventType.TASK_RESUMED,{"previous_status":task["status"]},iid,i);self.recorder.finalize(iid,True,True,i,"Task resumed.")
        return self.run(i)
    def approve(self,i):
        task=self.store.get(i)
        if task["status"]!=TaskStatus.WAITING_FOR_APPROVAL.value:return task
        iid=self.recorder.begin("approve task",task_id=i,metadata={"operation":"task_approval"});updated=self.store.update(i,status=TaskStatus.PAUSED.value,requires_approval=0,blocked_reason=None);self.recorder.approval(iid,True,{"task_id":i});self.recorder.finalize(iid,True,True,i,"Task approval recorded.");return updated
    def run(self,i):
        task=self.store.get(i);workspace=str(Path(task["workspace"]).resolve()).casefold()
        try:
            with acquire_named_resource(f"repository:{workspace}",timeout=min(30.0,float(task["max_runtime"]))):
                current=self.store.get(i)
                if current["status"] in TERMINAL or current["status"]==TaskStatus.PAUSED.value:return current
                return self._run_locked(i)
        except TimeoutError as exc:
            iid=self.recorder.begin(task["goal"],task_id=i,metadata={"operation":"task_run","workspace":task["workspace"]});blocked=self.store.update(i,status=TaskStatus.BLOCKED.value,blocked_reason="repository_resource_timeout");self.recorder.record(EventType.TASK_BLOCKED,{"reason":"repository_resource_timeout","error":str(exc)},iid,i);self.recorder.finalize(iid,False,False,i,"Task could not acquire its repository resource.");self.notifier.notify_task_blocked(blocked);return blocked
    def _run_locked(self,i):
        task=self.store.get(i); started=time.monotonic(); lifecycle_iid=self.recorder.begin(task["goal"],task_id=i,metadata={"operation":"task_run","workspace":task["workspace"]});self.recorder.record(EventType.TASK_STARTED,{"status":TaskStatus.RUNNING.value},lifecycle_iid,i);self.store.update(i,status=TaskStatus.RUNNING.value,started_at=task["started_at"] or utcnow())
        last_iid=None
        while True:
            task=self.store.get(i)
            if task["status"]!=TaskStatus.RUNNING.value:
                completed=task["status"]==TaskStatus.COMPLETED.value
                if task["status"]==TaskStatus.CANCELLED.value:self.recorder.record(EventType.TASK_CANCELLED,{"success":False,"status":task["status"]},lifecycle_iid,i)
                elif task["status"]==TaskStatus.PAUSED.value:self.recorder.record(EventType.TASK_PAUSED,{"success":False,"status":task["status"]},lifecycle_iid,i)
                self.recorder.finalize(lifecycle_iid,completed,completed,i,f"Task stopped with status {task['status']}.");return task
            if task["current_iteration"]>=task["max_iterations"] or time.monotonic()-started>=task["max_runtime"]:
                failed=self.store.update(i,status=TaskStatus.FAILED.value,blocked_reason="runtime_or_iteration_limit");self.recorder.record(EventType.TASK_FAILED,{"success":False,"reason":"runtime_or_iteration_limit"},lifecycle_iid,i);self.recorder.finalize(lifecycle_iid,False,False,i,"Task reached its runtime or iteration limit.");return failed
            step=self.backend.propose_next_step(task)
            # Planning may take time; a concurrent cancellation or pause must
            # win before any newly proposed action is executed.
            if self.store.get(i)["status"]!=TaskStatus.RUNNING.value:
                continue
            self.recorder.record(EventType.PLAN_CREATED,{"route_type":"task_supervisor","route_source":type(self.backend).__name__,"step":{"kind":step.kind,"path":step.path,"command":step.command},"content_included":False},lifecycle_iid,i)
            if step.kind=="complete":
                done=self.store.update(i,status=TaskStatus.COMPLETED.value,completed_at=utcnow()); self.notifier.notify_task_completed(done)
                checkpoint=done.get("checkpoint") or {};checkpoint_result=checkpoint.get("result") or {};verified=checkpoint.get("step")=="test" and checkpoint_result.get("exit_code")==0
                self.recorder.record(EventType.TASK_COMPLETED,{"success":True,"verified":verified,"message":done.get("last_result")},lifecycle_iid,i);self.recorder.finalize(lifecycle_iid,True,verified,i,"Task completed.")
                return done
            if self._approval(step):
                waiting=self.store.update(i,status=TaskStatus.WAITING_FOR_APPROVAL.value,requires_approval=1,blocked_reason="approval_required");self.recorder.record(EventType.TASK_BLOCKED,{"reason":"approval_required"},lifecycle_iid,i);self.recorder.finalize(lifecycle_iid,False,False,i,"Task requires approval.");self.notifier.notify_approval_required(waiting);return waiting
            previous_checkpoint=task["checkpoint"]
            iteration_iid=previous_checkpoint.get("dataset_interaction_id") if step.kind=="test" else self.recorder.begin(task["goal"],task_id=i,metadata={"category":"coding","iteration":task["current_iteration"]+1,"workspace":task["workspace"]})
            last_iid=iteration_iid
            if step.hypothesis:self.recorder.record(EventType.REASONING_RESPONSE,{"hypothesis":step.hypothesis,"backend":type(self.backend).__name__},iteration_iid,i)
            edit_action_id=None;edit_started=None;edit_started_at=None
            if step.kind=="edit":
                reference=self.recorder.store_code_context(task["workspace"],step.path,step.content)
                self.recorder.record(EventType.CODE_PATCH,{"path":step.path,"code_reference":reference,"hypothesis":step.hypothesis},iteration_iid,i)
                edit_action_id=uuid.uuid4().hex;edit_started=time.perf_counter();edit_started_at=datetime.now(timezone.utc).isoformat()
                self.recorder.record(EventType.TOOL_CALL,{"action_id":edit_action_id,"status":"prepared","name":"code_edit","arguments":{"path":step.path,"code_reference":reference},"execution":{"started_at":edit_started_at}},iteration_iid,i)
            elif step.kind=="test":self.recorder.record(EventType.TEST_RUN,{"command":step.command},iteration_iid,i)
            result=self._execute(task,step); iteration=task["current_iteration"]+1
            if edit_action_id:
                self.recorder.record(EventType.TOOL_RESULT,{"action_id":edit_action_id,"status":"committed" if result.get("success") else "failed","name":"code_edit","success":bool(result.get("success")),"error":result.get("error"),"metadata":{"path":result.get("path"),"verified":False},"execution":{"started_at":edit_started_at,"finished_at":datetime.now(timezone.utc).isoformat(),"latency_ms":round((time.perf_counter()-edit_started)*1000,3)},"automatically_verified":False},iteration_iid,i)
            if step.kind=="test" and result.get("exit_code")!=0 and previous_checkpoint.get("step")=="edit":
                self._rollback(previous_checkpoint); result["regression_rolled_back"]=True
            if step.kind=="test":
                self.recorder.record(EventType.TEST_RESULT,result,iteration_iid,i)
                passed=result.get("exit_code")==0;self.recorder.finalize(iteration_iid,passed,passed,i,response="Candidate passed evaluation" if passed else "Candidate rejected after test regression")
            checkpoint={"step":step.kind,"path":step.path,"result":result,"dataset_interaction_id":iteration_iid}
            self.store.memory.record_event("task_iteration",json.dumps(checkpoint),task_id=i)
            self.store.update(i,current_iteration=iteration,current_hypothesis=redact(step.hypothesis),last_result=redact(json.dumps(result)),checkpoint_json=json.dumps(checkpoint))
            if step.kind=="test" and result.get("exit_code")==0: continue
            if not result.get("success",result.get("exit_code")==0):
                self.store.update(i,current_hypothesis=self.backend.analyze_failure(self.store.get(i)))
    def _approval(self,s):
        text=" ".join(s.command or [])+" "+str(s.path or ""); return any(t in text.lower() for t in APPROVAL_TOKENS) or s.kind in {"delete","merge","push","dependency_upgrade"}
    def _execute(self,task,step):
        root=Path(task["workspace"]).resolve()
        if step.kind=="edit":
            target=(root/step.path).resolve()
            try:relative=target.relative_to(root)
            except ValueError:return {"success":False,"error":"path_not_allowed"}
            forbidden={str(item).replace("\\","/").strip("/").casefold() for item in task.get("forbidden_paths",[])}
            relative_text=str(relative).replace("\\","/").casefold()
            if any(relative_text==item or relative_text.startswith(item+"/") or target.name.casefold()==item for item in forbidden):return {"success":False,"error":"path_not_allowed"}
            before=target.read_text(encoding="utf-8") if target.exists() else None
            target.parent.mkdir(parents=True,exist_ok=True); target.write_text(step.content or "",encoding="utf-8")
            return {"success":True,"before":before,"after":step.content,"path":str(target)}
        if step.kind=="test": return self.runner.run(step.command,root,allowed=task["allowed_commands"])
        return {"success":False,"error":"unknown_step"}
    @staticmethod
    def _rollback(checkpoint):
        result=checkpoint.get("result",{}); path=result.get("path")
        if not path:return
        target=Path(path); before=result.get("before")
        if before is None: target.unlink(missing_ok=True)
        else: target.write_text(before,encoding="utf-8")
def prepare_coding_workspace(repository,task_id):
    repo=Path(repository).resolve()
    status=subprocess.run(["git","status","--porcelain"],cwd=repo,text=True,capture_output=True,**hidden_process_kwargs())
    if status.returncode: raise RuntimeError(status.stderr.strip())
    return {"repository":str(repo),"dirty":bool(status.stdout.strip()),"suggested_branch":f"jarvis/task-{task_id[:8]}","safe":True}

def create_isolated_workspace(repository,destination,task_id,branch=None,base_commit=None):
    repo,dest=Path(repository).resolve(),Path(destination).resolve()
    if dest.exists(): raise FileExistsError(dest)
    inspection=prepare_coding_workspace(repo,task_id)
    branch=branch or inspection["suggested_branch"]
    args=["git","worktree","add","-b",branch,str(dest)]+([base_commit] if base_commit else [])
    result=subprocess.run(args,cwd=repo,text=True,capture_output=True,**hidden_process_kwargs())
    if result.returncode: raise RuntimeError(redact(result.stderr.strip()))
    return {**inspection,"workspace":str(dest),"branch":branch}
