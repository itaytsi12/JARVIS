import unittest
from unittest.mock import Mock,patch

from brain.agent_runtime import AgentRuntime
from brain.models import Action
from brain.plan_validator import validate_generated_actions
from brain.tool_router import execute_tool
from brain.router import route_command


class UIPrimitiveTests(unittest.TestCase):
    def test_notepad_editor_selection_excludes_search_hidden_and_nested_edits(self):
        from tools.desktop_agent import _select_notepad_editor
        main={"hwnd":10,"top_hwnd":1,"parent":1,"control_id":15,"visible":True,"enabled":True,"area":800000}
        search={"hwnd":11,"top_hwnd":1,"parent":2,"control_id":23,"visible":True,"enabled":True,"area":12000}
        hidden={"hwnd":12,"top_hwnd":1,"parent":1,"control_id":15,"visible":False,"enabled":True,"area":800000}
        self.assertIs(_select_notepad_editor([search,hidden,main],1000000),main)

    def test_notepad_editor_selection_fails_closed_for_two_document_surfaces(self):
        from tools.desktop_agent import _select_notepad_editor
        candidates=[{"hwnd":value,"top_hwnd":1,"parent":1,"control_id":15,"visible":True,"enabled":True,"area":800000} for value in (10,11)]
        self.assertIsNone(_select_notepad_editor(candidates,1000000))

    def test_inspection_is_available_through_shared_executor(self):
        with patch("brain.agent_runtime.get_controls",return_value={"success":True,"verified":True,"controls":[],"message":"Found 0 controls."}) as inspect:
            result=AgentRuntime(trace=False)._execute_action(Action("inspect_window",{"app_name":"notepad","limit":20}))
        self.assertTrue(result.success);self.assertTrue(result.data["verified"]);inspect.assert_called_once_with("notepad",20,None)

    def test_multistep_ui_click_reuses_verified_session_window(self):
        from brain.session_context import SessionContext
        runtime=AgentRuntime(context=SessionContext(active_app="notepad",last_hwnd=4242),trace=False)
        with patch("brain.agent_runtime.click_control",return_value={"success":True,"verified":False,"message":"Clicked File."}) as click:
            result=runtime._execute_action(Action("click_ui_element",{"app_name":"notepad","name":"File","control_type":"Button"}))
        self.assertTrue(result.success);click.assert_called_once_with("notepad","File","Button",4242)

    def test_stale_expected_window_does_not_fall_back_to_another_instance(self):
        from tools import desktop_agent
        with patch.object(desktop_agent.ctypes.windll.user32,"IsWindow",return_value=False),patch.object(desktop_agent,"find_application_window") as find:
            result=desktop_agent.click_control("notepad","Save","Button",4242)
        self.assertFalse(result["success"]);self.assertEqual(result["error"],"expected_window_unavailable");find.assert_not_called()

    def test_multistep_uia_typing_reuses_verified_session_window(self):
        from brain.session_context import SessionContext
        runtime=AgentRuntime(context=SessionContext(active_app="notepad",last_hwnd=4242),trace=False)
        with patch("brain.agent_runtime.type_into_notepad_native",return_value={"success":True,"verified":True,"message":"typed"}) as type_control:
            result=runtime._execute_action(Action("type_text",{"text":"hello"}))
        self.assertTrue(result.success);type_control.assert_called_once_with("hello",4242)

    def test_failed_verified_window_typing_never_falls_back_to_global_input(self):
        from brain.session_context import SessionContext
        runtime=AgentRuntime(context=SessionContext(active_app="notepad",last_hwnd=4242),trace=False)
        with patch("brain.agent_runtime.type_into_control",return_value={"success":False,"error":"expected_window_unavailable"}),patch.object(runtime.executor,"execute_action") as fallback:
            result=runtime._execute_action(Action("type_text",{"text":"hello"}))
        self.assertFalse(result.success);self.assertEqual(result.error,"expected_window_unavailable");fallback.assert_not_called()

    def test_multistep_close_reuses_verified_session_window(self):
        from brain.session_context import SessionContext
        runtime=AgentRuntime(context=SessionContext(active_app="notepad",last_hwnd=4242),trace=False)
        with patch("brain.agent_runtime.close_application",return_value={"success":True,"verified":True,"message":"closed"}) as close:
            result=runtime._execute_action(Action("close_application",{"app_name":"notepad"}))
        self.assertTrue(result.success);close.assert_called_once_with("notepad",4242)
        runtime._update_context(Action("close_application",{"app_name":"notepad"}),result)
        self.assertIsNone(runtime.context.active_app);self.assertIsNone(runtime.context.last_hwnd)

    def test_switch_back_reuses_per_application_window_identity(self):
        from brain.session_context import SessionContext
        context=SessionContext(active_app="notepad",last_hwnd=222,application_windows={"vscode":111,"notepad":222})
        runtime=AgentRuntime(context=context,trace=False)
        with patch("brain.agent_runtime.focus_target",return_value={"success":True,"verified":True,"hwnd":111,"message":"focused"}) as focus:
            result=runtime._execute_action(Action("focus_application",{"app_name":"vscode"}))
        self.assertTrue(result.success);focus.assert_called_once_with("vscode",111)
        runtime._update_context(Action("focus_application",{"app_name":"vscode"}),result)
        self.assertEqual(context.active_app,"vscode");self.assertEqual(context.last_hwnd,111)

    def test_application_window_identity_map_is_bounded(self):
        runtime=AgentRuntime(trace=False)
        for index in range(20):runtime._remember_app_window(f"app-{index}",index+1)
        self.assertEqual(len(runtime.context.application_windows),16)
        self.assertNotIn("app-0",runtime.context.application_windows);self.assertEqual(runtime.context.application_windows["app-19"],20)

    def test_save_shortcut_requires_focus_on_captured_window(self):
        from brain.session_context import SessionContext
        runtime=AgentRuntime(context=SessionContext(active_app="notepad",last_hwnd=4242),trace=False)
        with patch("brain.agent_runtime.focus_target",return_value={"success":False,"error":"expected_window_unavailable"}),patch("brain.agent_runtime.press_key") as press:
            failed=runtime._execute_action(Action("save_current_document",{}))
        self.assertFalse(failed.success);press.assert_not_called()
        with patch("brain.agent_runtime.focus_target",return_value={"success":True,"verified":True,"hwnd":4242}),patch("brain.agent_runtime.press_key") as press:
            passed=runtime._execute_action(Action("save_current_document",{}))
        self.assertTrue(passed.success);press.assert_called_once_with("ctrl+s")

    def test_planned_window_action_uses_captured_hwnd(self):
        from brain.session_context import SessionContext
        runtime=AgentRuntime(context=SessionContext(active_app="chrome",last_hwnd=4242),trace=False)
        with patch("brain.agent_runtime.maximize_foreground_window",return_value={"success":True,"verified":True,"hwnd":4242,"message":"maximized"}) as maximize:
            result=runtime._execute_action(Action("maximize_window",{}))
        self.assertTrue(result.success);maximize.assert_called_once_with(4242)

    def test_generated_plans_may_inspect_but_may_not_blindly_click(self):
        actions,errors=validate_generated_actions([{"tool":"inspect_window","arguments":{"app_name":"notepad","limit":20}}])
        self.assertFalse(errors);self.assertEqual(actions[0].tool,"inspect_window")
        _,errors=validate_generated_actions([{"tool":"inspect_window","arguments":{"app_name":"notepad","limit":1000}}])
        self.assertTrue(errors)
        _,errors=validate_generated_actions([{"tool":"click_ui_element","arguments":{"app_name":"notepad","name":"Save"}}])
        self.assertTrue(errors)

    def test_semantic_click_uses_existing_shared_tool_path(self):
        with patch("brain.tool_router.click_control",return_value={"success":True,"verified":False,"message":"Clicked Save."}) as click:
            result=execute_tool("click_ui_element",{"app_name":"notepad","name":"Save","control_type":"Button"})
        self.assertTrue(result["success"]);self.assertFalse(result["verified"]);click.assert_called_once_with("notepad","Save","Button")

    def test_explicit_inspect_and_click_commands_are_deterministic(self):
        with patch("brain.router.classify_intent") as model:
            inspect=route_command("what controls are in Notepad")
            click=route_command("click the Save button in Notepad")
        self.assertEqual(inspect,{"type":"tool","tool":"inspect_window","arguments":{"app_name":"notepad","limit":50}})
        self.assertEqual(click,{"type":"tool","tool":"click_ui_element","arguments":{"app_name":"notepad","name":"Save","control_type":"Button"}})
        model.assert_not_called()

    def test_control_preview_is_bounded_for_user_but_structured(self):
        import sys
        from types import SimpleNamespace
        from tools import desktop_agent
        class Control:
            def __init__(self,index):self.element_info=SimpleNamespace(name=f"Control {index}",control_type="Button",automation_id=str(index))
        window=Mock();window.descendants.return_value=[Control(index) for index in range(20)]
        desktop=Mock();desktop.window.return_value=window;module=SimpleNamespace(Desktop=Mock(return_value=desktop))
        with patch.object(desktop_agent,"find_application_window",return_value=1),patch.dict(sys.modules,{"pywinauto":module}):result=desktop_agent.get_controls("notepad",limit=20)
        self.assertEqual(result["control_count"],20);self.assertIn("Control 0",result["message"]);self.assertNotIn("Control 19",result["message"]);self.assertLessEqual(len(result["message"]),500)

    def test_uia_typing_requires_exact_control_value_not_substring(self):
        import sys
        from types import SimpleNamespace
        from tools import desktop_agent
        control=Mock();control.wrapper_object.return_value=control;control.get_value.return_value="stale hello"
        window=Mock();window.child_window.return_value=control
        desktop=Mock();desktop.window.return_value=window;module=SimpleNamespace(Desktop=Mock(return_value=desktop))
        with patch.object(desktop_agent,"find_application_window",return_value=1),patch.dict(sys.modules,{"pywinauto":module}):
            result=desktop_agent.type_into_control("notepad","hello")
        self.assertFalse(result["success"]);self.assertIn("verification failed",result["error"])


if __name__=="__main__":unittest.main()
