import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from brain.task_planner import create_task_plan
from brain.agent_runtime import AgentRuntime
from brain.models import Action, Plan, PlanStatus
from brain import agent
from tools import applications
from tools import system


class VSCodeLaunchTests(unittest.TestCase):
    def test_task_manager_requires_a_visible_responsive_window(self):
        process=Mock(pid=44)
        with patch.object(system.subprocess,"Popen",return_value=process),patch.object(system,"_wait_for_visible_window",return_value=None):failed=system.open_task_manager()
        self.assertFalse(failed["success"]);self.assertFalse(failed["verified"]);self.assertEqual(failed["error"],"application_window_unverified")
        with patch.object(system.subprocess,"Popen",return_value=process),patch.object(system,"_wait_for_visible_window",return_value=55):opened=system.open_task_manager()
        self.assertTrue(opened["success"]);self.assertTrue(opened["verified"]);self.assertEqual(opened["hwnd"],55)

    def test_verified_task_manager_route_updates_shared_window_context(self):
        old=(agent.agent_runtime.context.active_app,agent.agent_runtime.context.last_hwnd,agent.agent_runtime.context.last_pid)
        try:
            with patch.object(agent.executor,"execute_action",return_value=agent.ToolResult(True,"open_task_manager","opened",{"verified":True,"pid":44,"hwnd":55})),patch.object(agent.memory_manager,"remember_entity"):
                agent.run_agent("open task manager",route={"type":"tool","tool":"open_task_manager","arguments":{}})
            self.assertEqual((agent.agent_runtime.context.active_app,agent.agent_runtime.context.last_hwnd,agent.agent_runtime.context.last_pid),("task manager",55,44))
        finally:agent.agent_runtime.context.active_app,agent.agent_runtime.context.last_hwnd,agent.agent_runtime.context.last_pid=old
    def test_exact_start_app_discovery_is_hidden_and_ambiguity_is_rejected(self):
        completed=Mock(returncode=0,stdout='[{"Name":"Apple Music","AppID":"AppleInc.AppleMusic_x!App"},{"Name":"Apple Music Preview","AppID":"preview"}]')
        with patch.object(applications,"_START_APPS_CACHE",None),patch.object(applications.subprocess,"run",return_value=completed) as run,patch.object(applications,"hidden_process_kwargs",return_value={"creationflags":123}):
            command=applications._resolve_start_app_command("apple music")
        self.assertEqual(command,["explorer.exe","shell:AppsFolder\\AppleInc.AppleMusic_x!App"]);self.assertEqual(run.call_args.kwargs["creationflags"],123)
        ambiguous=Mock(returncode=0,stdout='[{"Name":"Music","AppID":"one"},{"Name":"Music","AppID":"two"}]')
        with patch.object(applications,"_START_APPS_CACHE",None),patch.object(applications.subprocess,"run",return_value=ambiguous):self.assertIsNone(applications._resolve_start_app_command("music"))

    def test_start_apps_catalog_is_ttl_cached_in_memory(self):
        completed=Mock(returncode=0,stdout='{"Name":"Example","AppID":"example.id"}')
        with patch.object(applications,"_START_APPS_CACHE",None),patch.object(applications.subprocess,"run",return_value=completed) as run:
            self.assertEqual(applications._resolve_start_app_command("Example"),["explorer.exe","shell:AppsFolder\\example.id"]);self.assertEqual(applications._resolve_start_app_command("Example"),["explorer.exe","shell:AppsFolder\\example.id"])
        self.assertEqual(run.call_count,1)

    def test_start_app_launch_does_not_capture_explorer_helper_pid(self):
        process=Mock(pid=999)
        with patch.object(applications,"_resolve_start_app_command",return_value=["explorer.exe","shell:AppsFolder\\Apple!App"]),patch.object(applications.shutil,"which",return_value=None),patch.object(applications.subprocess,"Popen",return_value=process),patch.object(applications,"_wait_for_visible_window",return_value=77) as wait:
            result=applications.open_application("apple music")
        self.assertTrue(result["success"]);self.assertIsNone(result["pid"]);wait.assert_called_once_with("apple music",None)

    def test_requested_notepad_is_not_hidden(self):
        process = Mock(pid=1234)
        with patch.object(applications.subprocess, "Popen", return_value=process) as popen,patch.object(applications,"_wait_for_visible_window",return_value=99):
            result = applications.open_application("notepad")
        self.assertTrue(result["success"]);self.assertTrue(result["verified"]);self.assertEqual(result["hwnd"],99)
        popen.assert_called_once_with("notepad.exe")

    def test_all_vscode_aliases_use_resolved_executable(self):
        process = Mock(pid=4321)
        with patch.object(applications, "_resolve_vscode_command", return_value=[r"C:\VSCode\Code.exe"]), patch.object(
            applications.subprocess, "Popen", return_value=process
        ) as popen,patch.object(applications,"_wait_for_visible_window",return_value=88):
            for alias in ("vscode", "vs code", "visual studio code", "code"):
                with self.subTest(alias=alias):
                    result = applications.open_application(alias)
                    self.assertTrue(result["success"])
                    self.assertEqual(result["pid"], 4321)
            self.assertEqual(popen.call_count, 4)
            popen.assert_called_with([r"C:\VSCode\Code.exe"])

    def test_process_without_window_is_not_reported_as_open(self):
        process=Mock(pid=1234)
        with patch.object(applications.subprocess,"Popen",return_value=process),patch.object(applications,"_wait_for_visible_window",return_value=None):
            result=applications.open_application("notepad")
        self.assertFalse(result["success"]);self.assertFalse(result["verified"]);self.assertEqual(result["error"],"application_window_unverified")

    def test_path_wrapper_resolves_to_sibling_code_exe(self):
        wrapper = Path(r"C:\Local\Microsoft VS Code\bin\code.cmd")
        expected = wrapper.parent.parent / "Code.exe"
        with patch.object(applications.shutil, "which", return_value=str(wrapper)), patch.object(
            Path, "is_file", autospec=True, side_effect=lambda value: value == expected
        ):
            self.assertEqual(applications._resolve_vscode_command(), [str(expected)])


class DirectSavePlanTests(unittest.TestCase):
    def test_open_known_desktop_and_documents_use_redirected_paths(self):
        from tools import files
        desktop=Path(r"D:\RedirectedDesktop");documents=Path(r"E:\CloudDocuments")
        with patch.object(files,"get_desktop_path",return_value=desktop),patch.object(files,"get_documents_path",return_value=documents),patch.object(files.os,"startfile") as start:
            desktop_result=files.open_known_folder("desktop");documents_result=files.open_known_folder("documents")
        self.assertTrue(desktop_result["success"]);self.assertEqual(desktop_result["path"],str(desktop))
        self.assertTrue(documents_result["success"]);self.assertEqual(documents_result["path"],str(documents))
        self.assertEqual([call.args[0] for call in start.call_args_list],[str(desktop),str(documents)])

    def test_known_content_uses_direct_file_write_and_exact_verification(self):
        desktop = Path(r"D:\RedirectedDesktop")
        with patch("brain.task_planner.get_desktop_path", return_value=desktop):
            plan = create_task_plan(
                "Open Notepad, type hello world, and save it to the Desktop as hello.txt."
            )
        self.assertEqual(
            [action.tool for action in plan.actions],
            ["open_application", "wait_for_window", "type_text", "write_text_file", "verify_file"],
        )
        self.assertEqual(plan.actions[3].args, {"path": str(desktop / "hello.txt"), "contents": "hello world"})
        self.assertEqual(plan.actions[4].args, {"path": str(desktop / "hello.txt"), "expected_content": "hello world"})

    def test_switch_back_plan_reuses_vscode(self):
        plan = create_task_plan("Open VS Code, then open Notepad, then switch back to VS Code")
        self.assertEqual(
            [(action.tool, action.args) for action in plan.actions],
            [
                ("open_application", {"app_name": "vscode"}),
                ("wait_for_window", {"app_name": "vscode"}),
                ("open_application", {"app_name": "notepad"}),
                ("wait_for_window", {"app_name": "notepad"}),
                ("focus_application", {"app_name": "vscode"}),
            ],
        )

    def test_direct_write_and_exact_content_verification_execute(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "hello.txt")
            plan = Plan(
                "save",
                [
                    Action("write_text_file", {"path": path, "contents": "hello world"}),
                    Action("verify_file", {"path": path, "expected_content": "hello world"}, depends_on=[0]),
                ],
            )
            results = AgentRuntime(trace=False).execute(plan)
            self.assertEqual(plan.status, PlanStatus.COMPLETED)
            self.assertTrue(all(result.success for result in results))
            self.assertEqual(Path(path).read_text(encoding="utf-8"), "hello world")


if __name__ == "__main__":
    unittest.main()
