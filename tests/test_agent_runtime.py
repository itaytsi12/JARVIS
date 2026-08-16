import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brain.agent_runtime import AgentRuntime
from brain.models import Action, ActionRisk, Plan, PlanStatus, ToolResult
from tools.browser_agent import HumanActionRequired,PageState
from brain.task_planner import create_task_plan,should_use_task_planner
from brain.session_context import SessionContext
from brain.executor import Executor


class FailingRuntime(AgentRuntime):
    def __init__(self):
        super().__init__(trace=False)
        self.calls = 0

    def _execute_action(self, action):
        self.calls += 1
        return ToolResult(False, action.tool, error="not_ready")


class HandoffRuntime(AgentRuntime):
    def __init__(self):
        super().__init__(trace=False)

    def _execute_action(self, action):
        raise HumanActionRequired("SMS verification is required.")


class RuntimeTests(unittest.TestCase):
    def test_type_text_refuses_unverified_global_foreground_target(self):
        runtime=AgentRuntime(context=SessionContext(),trace=False)
        with patch.object(runtime.executor,"execute_action") as global_input:
            result=runtime._execute_action(Action("type_text",{"text":"do not misdirect"}))
        self.assertFalse(result.success);self.assertEqual(result.error,"target_window_unverified");global_input.assert_not_called()

    def test_key_press_requires_and_focuses_captured_window(self):
        runtime=AgentRuntime(context=SessionContext(),trace=False)
        with patch("brain.agent_runtime.press_key") as press:
            missing=runtime._execute_action(Action("press_key",{"key":"enter"}))
        self.assertFalse(missing.success);self.assertEqual(missing.error,"target_window_unverified");press.assert_not_called()
        runtime.context.active_app="notepad";runtime.context.last_hwnd=4242
        with patch("brain.agent_runtime.focus_target",return_value={"success":True,"verified":True,"hwnd":4242}),patch("brain.agent_runtime.press_key",return_value="Pressed enter.") as press:
            sent=runtime._execute_action(Action("press_key",{"key":"enter"}))
        self.assertTrue(sent.success);self.assertFalse(sent.data["verified"]);press.assert_called_once_with("enter")

    def test_type_payload_is_not_logged_without_explicit_reference_debug(self):
        runtime=AgentRuntime(trace=False);secret="private typed content"
        with patch.dict("os.environ",{"DEBUG_REFERENCE_RESOLUTION":"false"}),patch.object(runtime.executor,"execute_action",return_value=ToolResult(True,"type_text","typed")),self.assertLogs("jarvis.runtime",level="INFO") as logs:
            runtime._execute_action(Action("type_text",{"text":secret}))
        output=" ".join(logs.output);self.assertNotIn(secret,output);self.assertIn(str(len(secret)),output)

    def test_malformed_structured_tool_result_cannot_be_false_success(self):
        with patch("brain.executor.execute_tool",return_value={"message":"partial result"}):
            result=Executor().execute_action(Action("get_time",{}))
        self.assertFalse(result.success)

    def test_null_or_false_tool_result_cannot_be_false_success(self):
        for raw in (None,False,""):
            with self.subTest(raw=raw),patch("brain.executor.execute_tool",return_value=raw):
                result=Executor().execute_action(Action("get_time",{}))
            self.assertFalse(result.success);self.assertEqual(result.error,"invalid_tool_result")

    def test_retry_is_bounded_to_two_retries(self):
        runtime = FailingRuntime()
        plan = Plan("test", [Action("wait_for_window")])
        results=runtime.execute(plan)
        self.assertEqual(runtime.calls, 3)
        self.assertEqual(plan.retry_count, 2)
        self.assertEqual(plan.status, PlanStatus.FAILED)
        self.assertEqual(results[0].data["attempts"],3);self.assertEqual(results[0].data["retries"],2)
        self.assertEqual([item["success"] for item in results[0].data["attempt_history"]],[False,False,False])

    def test_direct_file_execution_and_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "hello.txt")
            plan = Plan("create", [Action("create_text_file", {"path": path, "contents": "hello"}), Action("verify_file", {"path": path}, depends_on=[0])])
            runtime = AgentRuntime(trace=False)
            results = runtime.execute(plan)
            self.assertTrue(all(item.success for item in results))
            self.assertEqual(Path(path).read_text(encoding="utf-8"), "hello")
            self.assertEqual(plan.status, PlanStatus.COMPLETED)

    def test_file_observations_and_append_report_direct_verification(self):
        from tools.files import append_text_file,copy_path,exists,list_files
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"note.txt";path.write_text("hello",encoding="utf-8")
            self.assertTrue(exists(str(path))["verified"]);self.assertTrue(list_files(folder)["verified"])
            appended=append_text_file(str(path)," world");self.assertTrue(appended["success"]);self.assertTrue(appended["verified"]);self.assertEqual(path.read_text(encoding="utf-8"),"hello world")
            copied=copy_path(str(path),str(Path(folder)/"copy.txt"));self.assertTrue(copied["success"]);self.assertTrue(copied["verified"])

    def test_caution_action_is_not_retried(self):
        runtime = FailingRuntime()
        plan = Plan("test", [Action("submit_form", risk=ActionRisk.CAUTION)])
        runtime.execute(plan)
        self.assertEqual(runtime.calls, 1)

    def test_side_effecting_safe_actions_are_never_implicitly_retried(self):
        for tool in ("type_text","browser_click","write_text_file","open_application","take_screenshot","volume_up","unknown_future_tool"):
            with self.subTest(tool=tool):
                runtime=FailingRuntime();result=runtime.execute(Plan("side effect",[Action(tool)]))[0]
                self.assertEqual(runtime.calls,1);self.assertEqual(result.data["attempts"],1)

    def test_desktop_resource_wait_is_exposed_as_bounded_metadata(self):
        runtime=AgentRuntime(trace=False);plan=Plan("type",[Action("type_text",{"text":"hello"})])
        with patch.object(runtime,"_execute_action",return_value=ToolResult(True,"type_text","ok",{"verified":True})):
            result=runtime.execute(plan)[0]
        self.assertEqual(result.data["resource"],"desktop_input");self.assertGreaterEqual(result.data["resource_wait_ms"],0)
        self.assertIsInstance(result.data["cross_process_lock"],bool)
        self.assertEqual(result.data["plan_resource"],"action_plan");self.assertGreaterEqual(result.data["plan_resource_wait_ms"],0);self.assertIsInstance(result.data["plan_cross_process_lock"],bool)

    def test_human_verification_pauses_resumably(self):
        runtime = HandoffRuntime()
        plan = Plan("verify", [Action("browser_click", {"target": "Continue"})])
        results = runtime.execute(plan)
        self.assertEqual(plan.status, PlanStatus.PAUSED)
        self.assertEqual(plan.current_action_index, 0)
        self.assertIn("SMS verification", results[0].error)

    def test_contextual_file_create_rename_move_and_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);desktop=root/"Desktop";documents=root/"Documents";desktop.mkdir();documents.mkdir()
            context=SessionContext();runtime=AgentRuntime(context=context,trace=False)
            with patch("brain.task_planner.get_desktop_path",return_value=desktop),patch("brain.task_planner.get_documents_path",return_value=documents):
                create=create_task_plan("Create a file on the Desktop called test.txt",context);self.assertTrue(all(r.success for r in runtime.execute(create)))
                self.assertEqual(Path(context.last_opened_file),desktop/"test.txt")
                rename=create_task_plan("Rename it to ideas.txt",context);self.assertTrue(runtime.execute(rename)[0].success)
                self.assertEqual(Path(context.last_opened_file),desktop/"ideas.txt")
                move=create_task_plan("Move it to Documents",context);self.assertTrue(runtime.execute(move)[0].success)
                self.assertEqual(Path(context.last_opened_file),documents/"ideas.txt")
                (documents/"ideas.txt").write_text("private note",encoding="utf-8")
                read=create_task_plan("Tell me what is inside it",context);result=runtime.execute(read)[0]
                self.assertTrue(result.success);self.assertEqual(result.data["contents"],"private note")

    def test_tell_me_file_request_is_not_whatsapp(self):
        context=SessionContext(last_opened_file=r"C:\Temp\note.txt")
        self.assertTrue(should_use_task_planner("Tell me what is inside it"))
        plan=create_task_plan("Tell me what is inside it",context)
        self.assertEqual([action.tool for action in plan.actions],["read_text_file"])

    def test_browser_url_change_verification_prevents_false_success(self):
        class Browser:
            page=object()
            def __init__(self,after):self.after=after
            def get_current_url(self):return "https://search.example"
            def get_page_state(self):return PageState("Search","https://search.example")
            def click_first_result(self):return PageState("Search",self.after)
        failed=AgentRuntime(browser=Browser("https://search.example"),trace=False)._execute_action(Action("browser_click_first_result",{}))
        passed=AgentRuntime(browser=Browser("https://result.example"),trace=False)._execute_action(Action("browser_click_first_result",{}))
        self.assertFalse(failed.success);self.assertEqual(failed.error,"verification_failed")
        self.assertTrue(passed.success);self.assertTrue(passed.data["verified"])

    def test_browser_button_click_uses_observable_state_change(self):
        class Browser:
            page=object()
            def __init__(self,changed):self.changed=changed
            def get_page_state(self):return PageState("Form","https://example.test",visible_text="before")
            def click_element(self,target,kind):
                return PageState("Form","https://example.test",visible_text="after" if self.changed else "before")
        verified=AgentRuntime(browser=Browser(True),trace=False)._execute_action(Action("browser_click",{"target":"Save","kind":"button"}))
        unverified=AgentRuntime(browser=Browser(False),trace=False)._execute_action(Action("browser_click",{"target":"Save","kind":"button"}))
        self.assertTrue(verified.success);self.assertTrue(verified.data["verified"])
        self.assertTrue(unverified.success);self.assertFalse(unverified.data["verified"])
        self.assertIn("no resulting page change was independently verified",unverified.message)

    def test_browser_link_click_requires_navigation(self):
        class Browser:
            page=object()
            def get_page_state(self):return PageState("Page","https://example.test",visible_text="before")
            def click_element(self,target,kind):return PageState("Changed","https://example.test",visible_text="after")
        result=AgentRuntime(browser=Browser(),trace=False)._execute_action(Action("browser_click",{"target":"Details","kind":"link"}))
        self.assertFalse(result.success);self.assertEqual(result.error,"verification_failed")


if __name__ == "__main__":
    unittest.main()
