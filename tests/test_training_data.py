import json,os,queue,sqlite3,tempfile,time,unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from unittest.mock import patch

from training_data.exporter import export_dataset,preference_pairs,split_grouped
from training_data.recorder import DatasetRecorder
from training_data.schema import EventType
from training_data.validator import validate_database
from training_data.inspect import inspect_examples
from training_data.readiness import evaluate_readiness
from training_data.sanitizer import privacy_safe_event,sanitize_user_request
from training_data.importer import _legacy_result_metadata
from brain import agent
from brain.models import Action,Plan,ToolResult
from brain.agent_runtime import AgentRuntime
from brain.session_context import SessionContext
from brain.web_answer import WebAnswer

class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.db=self.root/"dataset.db"; self.r=DatasetRecorder(self.db,async_writes=False)
    def tearDown(self):
        if self.r._connection:self.r.close()
        self.temp.cleanup()
    def events(self,iid):
        return list(self.r._connection.execute("SELECT * FROM raw_events WHERE interaction_id=? ORDER BY sequence_number",(iid,)))
    def examples(self):return list(self.r._connection.execute("SELECT * FROM training_examples ORDER BY created_at"))
    def tool_interaction(self,text="open youtube",success=True,verified=True,task_id=None):
        i=self.r.begin(text,task_id=task_id);self.r.record(EventType.PLAN_CREATED,{"tool":"open_website"},i,task_id);self.r.record(EventType.TOOL_CALL,{"name":"open_website","arguments":{"url":"https://youtube.com"}},i,task_id);self.r.record(EventType.TOOL_RESULT,{"success":success},i,task_id);self.r.finalize(i,success,verified,task_id,"done" if success else "failed");return i
    def test_successful_tool_eligible_and_failed_not_positive(self):
        self.tool_interaction();self.tool_interaction("bad",False,False);rows=self.examples();self.assertTrue(rows[0]["training_eligible"]);self.assertEqual(rows[0]["quality_label"],"VERIFIED_SUCCESS");self.assertFalse(rows[1]["training_eligible"]);self.assertEqual(rows[1]["quality_label"],"FAILED")
    def test_explicit_final_failure_overrides_earlier_completed_event(self):
        iid=self.r.begin("contradictory trace");self.r.record(EventType.TASK_COMPLETED,{"success":True},iid);self.r.finalize(iid,False,False,response="final failure")
        example=self.examples()[0];self.assertEqual(example["quality_label"],"FAILED");self.assertFalse(example["training_eligible"])
    def test_coding_pass_regression_and_preference(self):
        task="task1"; bad=self.r.begin("fix calculation",task_id=task);self.r.record(EventType.CODE_PATCH,{"patch":"bad"},bad,task);self.r.record(EventType.TEST_RESULT,{"exit_code":1,"regression_rolled_back":True},bad,task);self.r.finalize(bad,False,False,task,"rejected")
        good=self.r.begin("fix calculation",task_id=task);self.r.record(EventType.CODE_PATCH,{"patch":"good"},good,task);self.r.record(EventType.TEST_RESULT,{"exit_code":0},good,task);self.r.record(EventType.TASK_COMPLETED,{"success":True},good,task);self.r.finalize(good,True,True,task,"tests passed")
        rows=[dict(r) for r in self.examples()];self.assertEqual(rows[0]["quality_label"],"ROLLED_BACK");self.assertEqual(rows[1]["quality_label"],"VERIFIED_SUCCESS");self.assertEqual(len(preference_pairs(rows)),1)
    def test_correction_approval_and_rejection(self):
        i=self.r.begin("open visual studio");self.r.record(EventType.USER_CORRECTION,{"original":"visual studio","correction":"VS Code"},i);self.r.record(EventType.TOOL_RESULT,{"success":True},i);self.r.finalize(i,True,True,response="opened VS Code")
        a=self.r.begin("approve");self.r.approval(a,True);self.r.finalize(a,True,True,response="approved")
        x=self.r.begin("reject");self.r.approval(x,False);self.r.finalize(x,False,False,response="rejected")
        labels=[r["quality_label"] for r in self.examples()];self.assertIn("CORRECTED",labels);self.assertIn("USER_APPROVED",labels);self.assertIn("USER_REJECTED",labels)
    def test_secrets_and_env_are_sanitized_before_write(self):
        i=self.r.begin("password is my very secret phrase then open notepad OPENAI_API_KEY=sk-abcdefghijklmnop Bearer abc.def.ghi")
        self.r.record(EventType.TOOL_CALL,{"arguments":{"password":"secret","Authorization":"Bearer token123","Cookie":"sid=123"},"env":{"OPENAI_API_KEY":"nope"}},i);self.r.finalize(i,False,False,response="failed")
        raw=" ".join(r["payload_json"] for r in self.events(i));self.assertNotIn("very secret phrase",raw);self.assertIn("open notepad",raw);self.assertNotIn("abcdefghijklmnop",raw);self.assertNotIn("token123",raw);self.assertNotIn("sid=123",raw);self.assertNotIn("nope",raw);self.assertEqual(validate_database(self.db),[])
    def test_numeric_token_usage_is_preserved_without_allowing_auth_tokens(self):
        i=self.r.begin("usage");self.r.record(EventType.REASONING_RESPONSE,{"input_tokens":42,"output_tokens":7,"token":"secret-value"},i);self.r.flush();payload=json.loads(self.events(i)[-1]["payload_json"])
        self.assertEqual((payload["input_tokens"],payload["output_tokens"]),(42,7));self.assertEqual(payload["token"],"<REDACTED>")
    def test_type_text_action_arguments_keep_metadata_not_content(self):
        safe=agent._safe_action_arguments(Action("type_text",{"text":"private typed value","delay":.02}))
        self.assertEqual(safe,{"text":"<REDACTED>","delay":.02,"text_length":19})
        whatsapp=agent._safe_action_arguments(Action("send_whatsapp_message",{"recipient":"Alex","message":"private message"}))
        self.assertEqual(whatsapp,{"recipient":"Alex","message":"<REDACTED>","message_length":15})
        browser=agent._safe_action_arguments(Action("browser_type",{"target":"Search","text":"private search"}))
        self.assertEqual(browser,{"target":"Search","text":"<REDACTED>","text_length":14})
    def test_content_bearing_user_requests_keep_intent_not_private_body(self):
        safe=sanitize_user_request("Hey Jarvis, send Alex the private launch code")
        self.assertEqual(safe,"send request <CONTENT_REDACTED; original_length=45>")
        self.assertNotIn("private launch code",safe)
        iid=self.r.begin("type my private diary entry")
        payload=json.loads(self.events(iid)[0]["payload_json"])
        self.assertEqual(payload["original_user_text"],"type request <CONTENT_REDACTED; original_length=27>")
        self.assertEqual(sanitize_user_request("open notepad"),"open notepad")
        self.assertEqual(sanitize_user_request("tell me who created Minecraft"),"tell me who created Minecraft")
        self.assertIn("CONTENT_REDACTED",sanitize_user_request("tell Alex I am late"))
    def test_legacy_events_are_sanitized_at_inspection_and_export_boundary(self):
        request=privacy_safe_event("USER_REQUEST",{"original_user_text":"send Alex legacy private body"})
        call=privacy_safe_event("TOOL_CALL",{"name":"send_whatsapp_message","arguments":{"recipient":"Alex","message":"legacy private body"}})
        plan=privacy_safe_event("PLAN_CREATED",{"actions":[{"tool":"browser_type","arguments":{"target":"Search","text":"legacy private body"}}]})
        serialized=json.dumps([request,call,plan])
        self.assertNotIn("legacy private body",serialized);self.assertIn("message_length",serialized);self.assertIn("text_length",serialized)
        proposal=privacy_safe_event("PLAN_CREATED",{"actions":[{"tool":"send_whatsapp_message","arguments":{"recipient":"Alex","message":"generated private body"}}]})
        self.assertNotIn("generated private body",json.dumps(proposal))
    def test_default_runtime_log_redacts_content_bearing_command(self):
        with patch.dict("os.environ",{"DEBUG_REFERENCE_RESOLUTION":"false"}),self.assertLogs("jarvis.runtime",level="INFO") as logs:
            self.assertIn("content redacted",agent._command_for_log("type private typed value"));agent.runtime_log.info("command=%s",agent._command_for_log("type private typed value"))
        self.assertNotIn("private typed value"," ".join(logs.output))
    def test_model_backed_tool_calls_are_counted_from_structured_results(self):
        results=[ToolResult(True,"analyze_screen","done",{"model_calls":1}),ToolResult(True,"send_whatsapp_message","sent",{"translated":True,"translation_model":"gpt-5-mini"})]
        self.assertEqual(agent._result_model_calls(results),2)
    def test_successful_answer_containing_failed_is_not_mislabeled(self):
        iid=self.r.begin("what happened")
        service=unittest.mock.Mock(model="test-model",timeout=1);service.answer.return_value=WebAnswer("The launch failed before the backup succeeded.",True,model="test-model")
        with patch.object(agent,"get_recorder",return_value=self.r),patch.object(agent,"get_web_answer_service",return_value=service):
            response=agent.run_agent("what happened",route={"type":"question","message":"what happened"},interaction_id=iid)
        self.assertIn("failed",response);events=self.events(iid);self.assertIn("TASK_COMPLETED",[row["event_type"] for row in events]);self.assertNotIn("TASK_FAILED",[row["event_type"] for row in events])
    def test_missing_reference_clarification_is_blocked_not_positive(self):
        context=agent.agent_runtime.context;old=(context.last_assistant_response,context.last_spoken_response);context.last_assistant_response=None;context.last_spoken_response=None;iid=self.r.begin("type what you just said")
        try:
            with patch.object(agent,"get_recorder",return_value=self.r):response=agent.run_agent("type what you just said",interaction_id=iid)
        finally:context.last_assistant_response,context.last_spoken_response=old
        self.assertIn("don't have",response);self.assertIn("TASK_BLOCKED",[row["event_type"] for row in self.events(iid)]);self.assertEqual(self.examples()[0]["quality_label"],"BLOCKED")
    def test_bounded_context_never_contains_pending_message_or_prior_response(self):
        context=agent.agent_runtime.context;old=(context.pending_messaging_message,context.last_assistant_response);context.pending_messaging_message="private message";context.last_assistant_response="private answer"
        try:snapshot=agent._bounded_context(context)
        finally:context.pending_messaging_message,context.last_assistant_response=old
        self.assertNotIn("private message",json.dumps(snapshot));self.assertNotIn("private answer",json.dumps(snapshot));self.assertTrue(snapshot["has_pending_message"]);self.assertTrue(snapshot["has_last_assistant_response"])
    def test_reference_trace_keeps_metadata_not_resolved_or_literal_content(self):
        safe=agent._safe_reference_payload({"segments":["type private literal"],"type_text":[{"type_payload_before_reference_resolution":"private literal","reference_resolution_result":"private answer","detected_reference_phrase":"what you said"}],"resolved_value":"private answer"})
        serialized=json.dumps(safe);self.assertNotIn("private literal",serialized);self.assertNotIn("private answer",serialized);self.assertEqual(safe["segments"],{"count":1});self.assertEqual(safe["type_text"][0]["detected_reference_phrase"],"what you said")
    def test_raw_curated_separation_and_dedup(self):
        self.tool_interaction();self.tool_interaction();self.assertGreater(self.r.stats()["raw_events"],self.r.stats()["curated_examples"]);rows=self.examples();self.assertEqual(rows[0]["duplicate_count"],0);self.assertGreater(rows[1]["duplicate_count"],0)
    def test_cleanup_preserves_curated(self):
        i=self.tool_interaction();old=(datetime.now(timezone.utc)-timedelta(days=90)).isoformat();self.r._connection.execute("UPDATE raw_events SET timestamp=?",(old,));self.r._connection.commit();self.r.raw_days=30;self.r.cleanup();self.assertEqual(len(self.events(i)),0);self.assertEqual(len(self.examples()),1)
    def test_exports_success_only_tools_preference_and_no_overwrite(self):
        self.tool_interaction();self.tool_interaction("failure",False,False);sft=self.root/"sft.jsonl";tools=self.root/"tools.jsonl";export_dataset(self.db,"sft",sft);export_dataset(self.db,"tools",tools);self.assertEqual(len(sft.read_text().splitlines()),1);self.assertIn("trajectory",tools.read_text());self.assertRaises(FileExistsError,export_dataset,self.db,"sft",sft)
    def test_group_split_has_no_task_leakage(self):
        rows=[]
        for task in range(20):
            for iteration in range(2):rows.append({"task_id":f"t{task}","interaction_id":f"i{task}-{iteration}"})
        splits=split_grouped(rows,7,.2,.2);membership={}
        for name,items in splits.items():
            for item in items:membership.setdefault(item["task_id"],set()).add(name)
        self.assertTrue(all(len(v)==1 for v in membership.values()))
    def test_validator_detects_invalid_label_sequence_and_secret(self):
        self.tool_interaction();self.r._connection.execute("UPDATE training_examples SET quality_label='BAD'");self.r._connection.execute("UPDATE raw_events SET sequence_number=99,payload_json='{" + '"password":"visible"' + "}' WHERE event_type='USER_REQUEST'");self.r._connection.commit();errors=validate_database(self.db);self.assertTrue(any("invalid label" in e for e in errors));self.assertTrue(any("invalid sequence" in e for e in errors));self.assertTrue(any("secret" in e for e in errors))
    def test_validator_detects_prepared_action_without_terminal_result(self):
        i=self.r.begin("open notepad");self.r.record(EventType.TOOL_CALL,{"action_id":"unfinished","status":"prepared","name":"open_application"},i);self.r.flush()
        self.assertTrue(any("prepared action without terminal result" in error for error in validate_database(self.db)))
    def test_validator_detects_contradictory_terminal_action_status(self):
        i=self.r.begin("bad result");self.r.record(EventType.TOOL_CALL,{"action_id":"a","status":"prepared","name":"open_application"},i);self.r.record(EventType.TOOL_RESULT,{"action_id":"a","status":"committed","success":False},i);self.r.flush()
        self.assertTrue(any("contradictory action result" in error for error in validate_database(self.db)))
    def test_validator_detects_contradictory_task_and_final_outcomes(self):
        iid=self.r.begin("contradiction");self.r.record(EventType.TASK_COMPLETED,{"success":True},iid);self.r.finalize(iid,False,False,response="failed")
        self.assertTrue(any("contradictory task/final outcome" in error for error in validate_database(self.db)))
    def test_recorder_failure_is_fail_open(self):
        broken=DatasetRecorder(self.root,async_writes=False);self.assertFalse(broken.enabled);self.assertIsNone(broken.begin("still works"))
    def test_audio_capture_default_off(self):self.assertFalse(self.r.capture_audio)
    def test_code_context_is_content_addressed(self):
        first=self.r.store_code_context("project","a.py","print('hello')");second=self.r.store_code_context("project","a.py","print('hello')");self.assertEqual(first["content_hash"],second["content_hash"]);self.assertEqual(self.r._connection.execute("SELECT COUNT(*) FROM content_blobs").fetchone()[0],1)
        self.assertIsNone(self.r.store_code_context("project",".env","OPENAI_API_KEY=secret"))
    def test_legacy_result_import_keeps_hash_metadata_not_content(self):
        private='{"before":"private source code","after":"changed"}';metadata=_legacy_result_metadata(private)
        self.assertNotIn("private source code",json.dumps(metadata));self.assertEqual(metadata["length"],len(private));self.assertEqual(len(metadata["sha256"]),64)
    def test_record_overhead_is_small(self):
        start=time.perf_counter()
        for _ in range(100):self.r.record(EventType.TOOL_RESULT,{"success":True},"perf")
        self.assertLess((time.perf_counter()-start)/100,.01)

    def test_queue_saturation_uses_bounded_overflow_instead_of_silent_drop(self):
        class FullQueue:
            def put_nowait(self,item):raise queue.Full
        original_async,original_queue=self.r._async,self.r._queue
        try:
            self.r._async=True;self.r._queue=FullQueue();self.r._submit("event",{"id":"x"})
            self.assertEqual(len(self.r._overflow),1);self.assertEqual(self.r._dropped_events,0)
        finally:self.r._async=original_async;self.r._queue=original_queue;self.r._overflow.clear()

    def test_async_close_drains_events_before_closing_sqlite(self):
        path=self.root/"async.db";recorder=DatasetRecorder(path,async_writes=True)
        for index in range(100):recorder.record(EventType.TOOL_RESULT,{"success":True,"index":index},"async")
        recorder.close();connection=sqlite3.connect(path)
        try:self.assertEqual(connection.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0],100)
        finally:connection.close()

    def test_runtime_records_prepared_before_execution_and_does_not_fake_verification(self):
        observed=[]
        def execute(action):
            observed.extend(json.loads(row["payload_json"])["status"] for row in self.events(iid) if row["event_type"]=="TOOL_CALL")
            return ToolResult(True,action.tool,"opened",{})
        iid=self.r.begin("open notepad")
        with patch.object(agent,"get_recorder",return_value=self.r),patch.object(agent.executor,"execute_action",side_effect=execute):
            agent.run_agent("open notepad",route={"type":"tool","tool":"open_application","arguments":{"app_name":"notepad"}},interaction_id=iid)
        self.assertEqual(observed,["prepared"])
        events=self.events(iid);self.assertLess(next(i for i,r in enumerate(events) if r["event_type"]=="TOOL_CALL"),next(i for i,r in enumerate(events) if r["event_type"]=="TOOL_RESULT"))
        event_types=[row["event_type"] for row in events]
        self.assertIn("TASK_STARTED",event_types);self.assertIn("TASK_COMPLETED",event_types)
        plan_payload=json.loads(next(row["payload_json"] for row in events if row["event_type"]=="PLAN_CREATED"))
        completed_payload=json.loads(next(row["payload_json"] for row in events if row["event_type"]=="TASK_COMPLETED"))
        self.assertEqual(plan_payload["route_source"],"deterministic_router");self.assertEqual(completed_payload["route_source"],"deterministic_router")
        call_payload=json.loads(next(row["payload_json"] for row in events if row["event_type"]=="TOOL_CALL"));result_payload=json.loads(next(row["payload_json"] for row in events if row["event_type"]=="TOOL_RESULT"))
        self.assertIn("started_at",call_payload["execution"]);self.assertIn("finished_at",result_payload["execution"]);self.assertIn("latency_ms",result_payload["execution"])
        example=self.examples()[0];self.assertEqual(example["quality_label"],"SUCCESS");self.assertFalse(example["training_eligible"])

    def test_plan_event_does_not_reintroduce_typed_content(self):
        iid=self.r.begin("type private typed value")
        with patch.object(agent,"get_recorder",return_value=self.r),patch.object(agent.executor,"execute_action",return_value=ToolResult(True,"type_text","typed",{})):
            agent.run_agent("type private typed value",route={"type":"tool","tool":"type_text","arguments":{"text":"private typed value"}},interaction_id=iid)
        serialized=" ".join(row["payload_json"] for row in self.events(iid))
        self.assertNotIn("private typed value",serialized)
        plan=json.loads(next(row["payload_json"] for row in self.events(iid) if row["event_type"]=="PLAN_CREATED"))
        planned_arguments=plan.get("arguments") or plan["actions"][0]["arguments"]
        self.assertEqual(planned_arguments["text"],"<REDACTED>")

    def test_single_tool_message_route_does_not_persist_message_body(self):
        iid=self.r.begin("send Alex private message body")
        route={"type":"tool","tool":"send_whatsapp_message","arguments":{"recipient":"Alex","message":"private message body"}}
        with patch.object(agent,"get_recorder",return_value=self.r),patch.object(agent.agent_runtime,"execute",return_value=[ToolResult(True,"send_whatsapp_message","sent",{"committed":True,"verified":True})]):
            agent.run_agent("send Alex private message body",route=route,interaction_id=iid)
        serialized=" ".join(row["payload_json"] for row in self.events(iid))
        self.assertNotIn("private message body",serialized)
        self.assertIn("message_length",serialized)

    def test_private_observation_is_returned_live_but_not_persisted(self):
        iid=self.r.begin("read the requested file")
        private="private file contents that should only be returned live"
        result=ToolResult(True,"read_text_file",private,{"contents":private,"verified":True})
        with patch.object(agent,"get_recorder",return_value=self.r),patch.object(agent.executor,"execute_action",return_value=result):
            response=agent.run_agent("read the requested file",route={"type":"tool","tool":"read_text_file","arguments":{"path":"notes.txt"}},interaction_id=iid)
        self.assertEqual(response,private)
        serialized=" ".join(row["payload_json"] for row in self.events(iid))
        self.assertNotIn(private,serialized);self.assertIn("PRIVATE_RESPONSE_REDACTED",serialized)
    def test_ui_observation_keeps_structure_without_control_labels(self):
        private="Private document title";result=ToolResult(True,"inspect_window","inspected",{"controls":[{"name":"Save","control_type":"Button"},{"name":private,"control_type":"Text"}],"verified":True})
        safe=agent._safe_result_metadata(Action("inspect_window",{"app_name":"notepad"}),result);serialized=json.dumps(safe)
        self.assertNotIn("Save",serialized);self.assertNotIn(private,serialized);self.assertEqual(safe["control_count"],2);self.assertEqual(safe["control_type_counts"],{"Button":1,"Text":1})

    def test_unreached_plan_actions_receive_not_executed_terminal_records(self):
        actions=[Action("open_application",{"app_name":"notepad"}),Action("type_text",{"text":"must not run"})]
        prepared=agent._record_prepared_actions(self.r,"stopped",actions,agent.agent_runtime.context)
        agent._record_action_results(self.r,"stopped",actions,[ToolResult(False,"open_application","failed",error="launch_failed")],prepared,agent.agent_runtime.context)
        results=[json.loads(row["payload_json"]) for row in self.events("stopped") if row["event_type"]=="TOOL_RESULT"]
        self.assertEqual([result["status"] for result in results],["failed","not_executed"]);self.assertEqual(results[1]["error"],"prior_action_failed")

    def test_multistep_journal_observes_each_real_action_boundary(self):
        runtime=AgentRuntime(context=SessionContext(),trace=False)
        plan=Plan("open notepad",[Action("open_application",{"app_name":"notepad"}),Action("wait_for_window",{"app_name":"notepad"},depends_on=[0])])
        outcomes=[ToolResult(True,"open_application","opened",{"pid":10,"hwnd":20,"verified":True}),ToolResult(True,"wait_for_window","ready",{"hwnd":20,"verified":True})]
        with patch.object(runtime,"_execute_action",side_effect=outcomes):agent._execute_recorded_plan(self.r,"boundaries",plan,runtime)
        events=[(row["event_type"],json.loads(row["payload_json"])) for row in self.events("boundaries")]
        self.assertEqual([kind for kind,_ in events],["TOOL_CALL","TOOL_RESULT","TOOL_CALL","TOOL_RESULT"])
        self.assertIsNone(events[0][1]["before_state"]["active_app"])
        self.assertEqual(events[1][1]["after_state"]["active_app"],"notepad")
        self.assertEqual(events[2][1]["before_state"]["active_app"],"notepad")

    def test_runtime_trace_redacts_content_without_relying_on_planner_flags(self):
        runtime=AgentRuntime(context=SessionContext(),trace=True);private="private browser field"
        with patch.object(runtime,"_execute_action",return_value=ToolResult(True,"browser_type","typed",{"verified":True})),patch("builtins.print") as output:
            runtime.execute(Plan("fill field",[Action("browser_type",{"target":"Search","text":private})]))
        rendered=" ".join(str(call) for call in output.call_args_list)
        self.assertNotIn(private,rendered);self.assertIn("REDACTED",rendered)

    def test_runtime_exception_closes_prepared_and_unreached_actions(self):
        runtime=AgentRuntime(context=SessionContext(),trace=False);plan=Plan("fail",[Action("open_application",{"app_name":"notepad"}),Action("wait_for_window",{"app_name":"notepad"},depends_on=[0])])
        def explode(plan,**kwargs):
            kwargs["action_observer"]("prepared",0,plan.actions[0],None,runtime.context)
            raise RuntimeError("unexpected failure")
        with patch.object(runtime,"execute",side_effect=explode),self.assertRaises(RuntimeError):agent._execute_recorded_plan(self.r,"exception",plan,runtime)
        results=[json.loads(row["payload_json"]) for row in self.events("exception") if row["event_type"]=="TOOL_RESULT"]
        self.assertEqual([item["status"] for item in results],["failed","not_executed"])
        self.assertEqual(validate_database(self.db),[])

    def test_stats_report_labels_actions_routes_and_disk_usage(self):
        self.tool_interaction();self.tool_interaction("failed",False,False);stats=self.r.stats()
        self.assertEqual(stats["schema_version"],2);self.assertEqual(stats["capture_version"],"1.3");self.assertEqual(stats["verified_records"],1)
        self.assertEqual(stats["successful_records"],1);self.assertEqual(stats["failures"],1)
        self.assertEqual(stats["stored_example_versions"],[{"schema_version":2,"dataset_version":"1.1","capture_version":"1.3","count":2}])
        self.assertEqual(stats["action_types"]["open_website"],2);self.assertGreater(stats["size_bytes"],0);self.assertGreater(stats["average_bytes_per_raw_event"],0);self.assertIsNotNone(stats["oldest_event_at"])

    def test_stats_aggregate_model_calls_and_tokens_by_operation(self):
        i=self.r.begin("question");self.r.record(EventType.REASONING_REQUEST,{"operation":"read_only_web_answer","model_calls":1},i);self.r.record(EventType.REASONING_RESPONSE,{"operation":"read_only_web_answer","input_tokens":80,"output_tokens":12},i);self.r.finalize(i,True,False,response="answer")
        usage=self.r.stats()["model_usage"]["read_only_web_answer"]
        self.assertEqual(usage,{"calls":1,"input_tokens":80,"output_tokens":12})

    def test_bounded_dataset_inspection_filters_actions(self):
        self.tool_interaction("open youtube");self.tool_interaction("failure",False,False)
        items=inspect_examples(self.db,action="open_website",limit=1)
        self.assertEqual(len(items),1);self.assertIn(items[0]["quality_label"],{"VERIFIED_SUCCESS","FAILED"})
        self.assertEqual(inspect_examples(self.db,action="missing",limit=10),[])

    def test_training_readiness_is_evidence_based_and_conservative(self):
        self.tool_interaction()
        report=evaluate_readiness(self.db)
        self.assertFalse(report["ready"]);self.assertEqual(report["recommendation"],"continue_verified_data_collection")
        self.assertEqual(report["metrics"]["eligible_examples"],1)
        permissive={name:0 for name in report["checks"]}
        self.assertTrue(evaluate_readiness(self.db,permissive)["ready"])

if __name__=="__main__":unittest.main()
