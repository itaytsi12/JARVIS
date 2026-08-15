import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from brain.task_planner import create_task_plan
from brain.agent_runtime import AgentRuntime
from brain.models import Action, Plan, PlanStatus
from tools import applications


class VSCodeLaunchTests(unittest.TestCase):
    def test_requested_notepad_is_not_hidden(self):
        process = Mock(pid=1234)
        with patch.object(applications.subprocess, "Popen", return_value=process) as popen:
            result = applications.open_application("notepad")
        self.assertTrue(result["success"])
        popen.assert_called_once_with("notepad.exe")

    def test_all_vscode_aliases_use_resolved_executable(self):
        process = Mock(pid=4321)
        with patch.object(applications, "_resolve_vscode_command", return_value=[r"C:\VSCode\Code.exe"]), patch.object(
            applications.subprocess, "Popen", return_value=process
        ) as popen:
            for alias in ("vscode", "vs code", "visual studio code", "code"):
                with self.subTest(alias=alias):
                    result = applications.open_application(alias)
                    self.assertTrue(result["success"])
                    self.assertEqual(result["pid"], 4321)
            self.assertEqual(popen.call_count, 4)
            popen.assert_called_with([r"C:\VSCode\Code.exe"])

    def test_path_wrapper_resolves_to_sibling_code_exe(self):
        wrapper = Path(r"C:\Local\Microsoft VS Code\bin\code.cmd")
        expected = wrapper.parent.parent / "Code.exe"
        with patch.object(applications.shutil, "which", return_value=str(wrapper)), patch.object(
            Path, "is_file", autospec=True, side_effect=lambda value: value == expected
        ):
            self.assertEqual(applications._resolve_vscode_command(), [str(expected)])


class DirectSavePlanTests(unittest.TestCase):
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
