import json,os,tempfile,unittest,threading,time
from pathlib import Path
from memory import MemoryManager
from brain.task_supervisor import DeterministicBackend,ProposedStep,ReadOnlyPriority,ReadOnlyTaskCancelled,ReadOnlyTaskScheduler,SafeCommandRunner,TaskStatus,TaskStore,TaskSupervisor,create_isolated_workspace
from brain.resource_locks import acquire_named_resource
from unittest.mock import patch
from training_data.recorder import DatasetRecorder
from training_data.exporter import export_dataset

class TaskSupervisorTests(unittest.TestCase):
    def test_command_runner_rejects_mutating_git_subcommands_before_launch(self):
        runner=SafeCommandRunner()
        for command in (["git","checkout","--","file.py"],["git","restore","file.py"],["git","commit","-am","change"],["git","branch","-D","work"],["git","diff","--output=report.txt"],["git","diff","--ext-diff"]):
            with self.subTest(command=command),patch("brain.task_supervisor.subprocess.run") as run,self.assertRaises(PermissionError):runner.run(command,self.root)
            run.assert_not_called()

    def test_command_runner_allows_read_only_git_inspection(self):
        completed=type("Result",(),{"stdout":"clean","stderr":"","returncode":0})()
        with patch("brain.task_supervisor.subprocess.run",return_value=completed) as run:
            result=SafeCommandRunner().run(["git","status","--short"],self.root)
        self.assertEqual(result["exit_code"],0);run.assert_called_once()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.memory=MemoryManager(self.root/"m.db"); self.recorder=DatasetRecorder(self.root/"dataset.db",async_writes=False);self.store=TaskStore(self.memory,self.recorder)
    def tearDown(self): self.recorder.close(); self.memory.db.close(); self.temp.cleanup()
    def supervisor(self,backend): return TaskSupervisor(self.store,backend,recorder=self.recorder)
    def test_edit_test_complete_and_persist_restart(self):
        task=self.store.create("fix add",self.root,max_iterations=5)
        backend=DeterministicBackend([ProposedStep("edit","calc.py","def add(a,b): return a+b\n"),ProposedStep("test",command=["python","-c","from calc import add; assert add(1,2)==3"]),ProposedStep("complete")])
        done=self.supervisor(backend).run(task["id"]); self.assertEqual(done["status"],TaskStatus.COMPLETED.value)
        payloads=[json.loads(row[0]) for row in self.recorder._connection.execute("SELECT payload_json FROM raw_events WHERE event_type='CODE_PATCH'")]
        self.assertTrue(payloads[0]["code_reference"]["content_hash"]);self.assertNotIn("content",payloads[0]);self.assertNotIn("def add",json.dumps(payloads[0]))
        edit_events=[(kind,json.loads(payload)) for kind,payload in self.recorder._connection.execute("SELECT event_type,payload_json FROM raw_events WHERE event_type IN ('TOOL_CALL','TOOL_RESULT')") if 'code_edit' in payload]
        self.assertEqual([kind for kind,_ in edit_events],["TOOL_CALL","TOOL_RESULT"]);self.assertTrue(edit_events[1][1]["success"]);self.assertNotIn("def add",json.dumps(edit_events))
        plans=[json.loads(row[0]) for row in self.recorder._connection.execute("SELECT payload_json FROM raw_events WHERE event_type='PLAN_CREATED'")]
        self.assertEqual([plan["step"]["kind"] for plan in plans],["edit","test","complete"]);self.assertTrue(all(not plan["content_included"] for plan in plans));self.assertNotIn("def add",json.dumps(plans))
        self.memory.db.close(); self.memory=MemoryManager(self.root/"m.db"); self.store=TaskStore(self.memory,self.recorder); self.assertEqual(self.store.get(task["id"])["current_iteration"],2)

    def test_cancellation_during_planning_wins_and_is_not_positive_ground_truth(self):
        task=self.store.create("cancel while planning",self.root);holder={}
        class CancellingBackend:
            def propose_next_step(inner,context):holder["supervisor"].cancel(task["id"]);return ProposedStep("edit","should_not_exist.py","bad")
            def analyze_failure(inner,context):return "unused"
        holder["supervisor"]=self.supervisor(CancellingBackend())
        result=holder["supervisor"].run(task["id"])
        self.assertEqual(result["status"],"CANCELLED");self.assertFalse((self.root/"should_not_exist.py").exists())
        examples=list(self.recorder._connection.execute("SELECT quality_label,input_json FROM training_examples"))
        run_labels=[label for label,input_json in examples if json.loads(input_json).get("metadata",{}).get("operation")=="task_run"]
        self.assertEqual(run_labels,["FAILED"])
    def test_cancellation_while_waiting_for_repository_wins(self):
        task=self.store.create("wait then cancel",self.root);supervisor=self.supervisor(DeterministicBackend([ProposedStep("edit","never.py","bad")]))
        holder={};resource=f"repository:{str(self.root.resolve()).casefold()}"
        with acquire_named_resource(resource):
            thread=threading.Thread(target=lambda:holder.setdefault("result",supervisor.run(task["id"])),daemon=True);thread.start();time.sleep(.05);supervisor.cancel(task["id"])
        thread.join(2)
        self.assertEqual(holder["result"]["status"],"CANCELLED");self.assertFalse((self.root/"never.py").exists())
    def test_pause_cancel_approval_and_bounds(self):
        one=self.store.create("pause",self.root); supervisor=self.supervisor(DeterministicBackend([])); self.assertEqual(supervisor.pause(one["id"])["status"],"PAUSED"); self.assertEqual(supervisor.cancel(one["id"])["status"],"CANCELLED")
        approval=self.store.create("dependency",self.root); approval_supervisor=self.supervisor(DeterministicBackend([ProposedStep("dependency_upgrade",command=["pip","install","x"])] )); waiting=approval_supervisor.run(approval["id"]); self.assertEqual(waiting["status"],"WAITING_FOR_APPROVAL"); self.assertEqual(approval_supervisor.resume(approval["id"])["status"],"WAITING_FOR_APPROVAL"); self.assertEqual(approval_supervisor.approve(approval["id"])["status"],"PAUSED")
        bounded=self.store.create("loop",self.root,max_iterations=1); failed=self.supervisor(DeterministicBackend([ProposedStep("edit","x.py","bad"),ProposedStep("edit","x.py","again")])).run(bounded["id"]); self.assertEqual(failed["status"],"FAILED")
        direct=self.store.create("direct command",self.root);self.assertEqual(self.store.command("pause",direct["id"])["status"],"PAUSED")
        self.assertTrue(self.recorder._connection.execute("SELECT 1 FROM raw_events WHERE task_id=? AND event_type='TASK_PAUSED'",(direct["id"],)).fetchone())

    def test_terminal_task_state_cannot_be_overwritten_by_late_controls(self):
        supervisor=self.supervisor(DeterministicBackend([ProposedStep("complete")]))
        task=self.store.create("finish",self.root);done=supervisor.run(task["id"]);self.assertEqual(done["status"],"COMPLETED")
        self.assertEqual(supervisor.pause(task["id"])["status"],"COMPLETED")
        self.assertEqual(supervisor.cancel(task["id"])["status"],"COMPLETED")
        self.assertEqual(self.store.command("cancel",task["id"])["status"],"COMPLETED")
        lifecycle=[row[0] for row in self.recorder._connection.execute("SELECT quality_label FROM training_examples WHERE task_id=?",(task["id"],))]
        self.assertEqual(lifecycle,["SUCCESS"])
    def test_bad_path_is_rejected_and_recorded(self):
        task=self.store.create("bad edit",self.root,max_iterations=1); result=self.supervisor(DeterministicBackend([ProposedStep("edit","../outside.py","bad")])).run(task["id"]); self.assertEqual(result["status"],"FAILED"); self.assertFalse((self.root.parent/"outside.py").exists())

    def test_configured_forbidden_path_policy_is_enforced(self):
        task=self.store.create("protect secret",self.root,max_iterations=1,forbidden_paths=["private/secret.txt"])
        result=self.supervisor(DeterministicBackend([ProposedStep("edit","private/secret.txt","sensitive")])).run(task["id"])
        self.assertEqual(result["status"],"FAILED");self.assertFalse((self.root/"private"/"secret.txt").exists());self.assertIn("path_not_allowed",result["last_result"])
    def test_failed_test_rolls_back_candidate(self):
        target=self.root/"value.py"; target.write_text("GOOD=1\n")
        task=self.store.create("reject regression",self.root,max_iterations=2)
        steps=[ProposedStep("edit","value.py","GOOD=0\n"),ProposedStep("test",command=["python","-c","from value import GOOD; assert GOOD==1"])]
        result=self.supervisor(DeterministicBackend(steps)).run(task["id"])
        self.assertEqual(result["status"],"FAILED"); self.assertEqual(target.read_text(),"GOOD=1\n"); self.assertIn("regression_rolled_back",result["last_result"])
    def test_evaluated_pass_can_be_captured_for_future_training(self):
        task=self.store.create("capture",self.root,max_iterations=3)
        steps=[ProposedStep("edit","ok.py","OK=True\n"),ProposedStep("test",command=["python","-c","from ok import OK; assert OK"]),ProposedStep("complete")]
        self.supervisor(DeterministicBackend(steps)).run(task["id"])
        self.assertEqual(len(self.memory.db.query("SELECT * FROM memories WHERE kind='training_example'")),0)
        self.assertGreaterEqual(self.recorder.stats()["eligible_examples"],1)
    def test_git_worktree_isolation(self):
        import subprocess
        repo=self.root/"repo"; repo.mkdir(); subprocess.run(["git","init","-q"],cwd=repo,check=True); subprocess.run(["git","config","user.email","test@example.invalid"],cwd=repo,check=True); subprocess.run(["git","config","user.name","Test"],cwd=repo,check=True)
        (repo/"a.py").write_text("A=1\n"); subprocess.run(["git","add","a.py"],cwd=repo,check=True); subprocess.run(["git","commit","-qm","initial"],cwd=repo,check=True)
        isolated=create_isolated_workspace(repo,self.root/"worktree","abc12345")
        self.assertTrue(Path(isolated["workspace"]).exists()); self.assertEqual(isolated["branch"],"jarvis/task-abc12345")
    def test_incremental_failure_then_success(self):
        (self.root/"value.py").write_text("VALUE=0\n")
        task=self.store.create("fix two attempts",self.root,max_iterations=6)
        steps=[ProposedStep("edit","value.py","VALUE=1\n"),ProposedStep("test",command=["python","-B","-c","from value import VALUE; assert VALUE==2"]),ProposedStep("edit","value.py","VALUE=2\n"),ProposedStep("test",command=["python","-B","-c","from value import VALUE; assert VALUE==2"]),ProposedStep("complete")]
        done=self.supervisor(DeterministicBackend(steps)).run(task["id"])
        self.assertEqual(done["status"],"COMPLETED"); self.assertEqual((self.root/"value.py").read_text(),"VALUE=2\n"); self.assertEqual(done["current_iteration"],4)
        stats=self.recorder.stats(); self.assertEqual(stats["coding_examples"],2); self.assertEqual(stats["eligible_coding_examples"],1); self.assertEqual(stats["failures"],1)
        coding=self.root/"coding.jsonl"; preferences=self.root/"preferences.jsonl"; self.assertEqual(export_dataset(self.recorder.path,"coding",coding),1); self.assertEqual(export_dataset(self.recorder.path,"preferences",preferences),1)
    def test_interactive_question_coexists_with_background_read(self):
        scheduler=ReadOnlyTaskScheduler(); background_started=threading.Event(); release=threading.Event()
        def background(token):
            background_started.set(); release.wait(2); token.raise_if_cancelled(); return "background"
        slow=scheduler.submit(background,ReadOnlyPriority.BACKGROUND)
        self.assertTrue(background_started.wait(1))
        fast=scheduler.submit(lambda token:"answer",ReadOnlyPriority.INTERACTIVE)
        self.assertEqual(fast.result(1),"answer")
        release.set(); self.assertEqual(slow.result(1),"background"); scheduler.shutdown()
    def test_interactive_question_cancels_lower_priority_read(self):
        scheduler=ReadOnlyTaskScheduler(); started=threading.Event(); observed_cancel=threading.Event()
        def background(token):
            started.set()
            while not token.cancelled:time.sleep(.005)
            observed_cancel.set(); token.raise_if_cancelled()
        slow=scheduler.submit(background,ReadOnlyPriority.BACKGROUND); self.assertTrue(started.wait(1))
        fast=scheduler.submit(lambda token:"priority answer",ReadOnlyPriority.INTERACTIVE,interrupt_lower=True)
        self.assertEqual(fast.result(1),"priority answer"); self.assertTrue(observed_cancel.wait(1)); self.assertTrue(slow.cancelled)
        with self.assertRaises(ReadOnlyTaskCancelled):slow.result(1)
        scheduler.shutdown()

if __name__=="__main__": unittest.main()
