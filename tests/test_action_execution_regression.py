import unittest
from unittest.mock import patch

from brain import agent
from brain.models import Action,Plan,ToolResult
from brain.executor import Executor
from brain.tool_router import execute_tool
from brain.request_intent import RequestKind, classify_request_kind
from brain.router import route_command
from brain.web_answer import WebAnswer
from brain.task_supervisor import CancellationToken
from voice.response_formatter import format_spoken_response


class ActionExecutionRegressionTests(unittest.TestCase):
    def test_youtube_actions_reach_executor_once(self):
        for command in ("open youtube", "can you open youtube"):
            with self.subTest(command=command):
                self.assertEqual(classify_request_kind(command).kind, RequestKind.ACTION)
                route = route_command(command)
                self.assertEqual(route["type"], "tool")
                self.assertEqual(route["tool"], "open_website")
                self.assertEqual(route["arguments"], {"url": "https://www.youtube.com"})
                success = ToolResult(True, "open_website", "Opened YouTube.")
                with patch.object(agent.executor, "execute_action", return_value=success) as execute:
                    response = agent.run_agent(command, route=route)
                execute.assert_called_once()
                action = execute.call_args.args[0]
                self.assertEqual(action.tool, "open_website")
                self.assertEqual(action.args, {"url": "https://www.youtube.com"})
                self.assertEqual(response, "Opened YouTube.")

    def test_polite_open_requests_keep_suffixes_out_of_targets(self):
        expected={
            "can you open notepad please":("open_application",{"app_name":"notepad"}),
            "could you open YouTube for me":("open_website",{"url":"https://www.youtube.com"}),
            "please open TikTok":("open_website",{"url":"https://www.tiktok.com"}),
        }
        for command,(tool,arguments) in expected.items():
            with self.subTest(command=command):
                route=route_command(command)
                self.assertEqual(route["type"],"tool");self.assertEqual(route["tool"],tool);self.assertEqual(route["arguments"],arguments)

    def test_legacy_multistep_open_plan_resolves_sites_and_punctuation(self):
        route=route_command("Open Notepad, then open YouTube")
        self.assertEqual(route["type"],"local_plan")
        self.assertEqual([(a.tool,a.args) for a in route["actions"]],[
            ("open_application",{"app_name":"notepad"}),
            ("open_website",{"url":"https://www.youtube.com"}),
        ])

    def test_single_action_updates_shared_session_context_for_followups(self):
        context=agent.agent_runtime.context;old=(context.last_opened_app,context.active_app,context.last_pid,context.last_hwnd,context.browser_active,context.current_url)
        try:
            app_result=ToolResult(True,"open_application","Opened Notepad.",{"pid":123,"hwnd":456,"verified":True})
            with patch.object(agent.executor,"execute_action",return_value=app_result):agent.run_agent("open notepad",route={"type":"tool","tool":"open_application","arguments":{"app_name":"notepad"}})
            self.assertEqual(context.last_opened_app,"notepad");self.assertEqual(context.last_hwnd,456)
            web_result=ToolResult(True,"open_website","Opened YouTube.",{"verified":True})
            with patch.object(agent.executor,"execute_action",return_value=web_result):agent.run_agent("open youtube",route={"type":"tool","tool":"open_website","arguments":{"url":"https://www.youtube.com"}})
            self.assertTrue(context.browser_active);self.assertEqual(context.current_url,"https://www.youtube.com")
        finally:context.last_opened_app,context.active_app,context.last_pid,context.last_hwnd,context.browser_active,context.current_url=old

    def test_failed_action_controls_text_and_spoken_response(self):
        command = "can you open youtube"
        route = route_command(command)
        failure = ToolResult(False, "open_website", "Failed to open Google Chrome.", error="launch failed")
        with patch.object(agent.executor, "execute_action", return_value=failure):
            response = agent.run_agent(command, route=route)
        spoken = format_spoken_response(command, route, response, lang="en")
        self.assertIn("Failed", response)
        self.assertNotIn("opened", spoken.lower())
        self.assertEqual(spoken, "I couldn't complete that action, sir.")

    def test_conversational_answer_containing_failed_is_spoken_normally(self):
        spoken=format_spoken_response("what happened",{"type":"ai","message":"what happened"},"The launch failed before the backup succeeded.",lang="en")
        self.assertEqual(spoken,"The launch failed before the backup succeeded, sir.")

    def test_spoken_formatter_adds_sir_once_only_at_final_layer(self):
        route={"type":"tool","tool":"open_website","arguments":{"url":"https://www.youtube.com"}}
        self.assertEqual(format_spoken_response("open YouTube",route,"Opened YouTube.","en"),"I opened YouTube, sir.")
        self.assertEqual(format_spoken_response("confirm",{"type":"ai"},"Done, sir.","en"),"Done, sir.")
        literal_route={"type":"local_plan","actions":[Action("type_text",{"text":"hello.txt"})]}
        literal_execution={"executed":True,"success":True,"verified":False,"partial":False,"actions":[{"tool":"type_text","arguments":{"text":"hello.txt"},"success":True,"verified":False}]}
        self.assertEqual(format_spoken_response('type "hello.txt"',literal_route,"Text entered.","en",execution=literal_execution),"I typed hello.txt, sir.")

    def test_question_still_never_reaches_executor(self):
        command = "who is the CEO of Nvidia"
        route = route_command(command)
        service = unittest.mock.Mock(model="test-model", timeout=3)
        service.answer.return_value = WebAnswer("Jensen Huang is Nvidia's CEO.", True)
        with patch.object(agent, "get_web_answer_service", return_value=service), patch.object(agent.executor, "execute_action") as execute:
            response = agent.run_agent(command, route=route)
        execute.assert_not_called()
        self.assertEqual(response, "Jensen Huang is Nvidia's CEO.")

    def test_late_read_only_tool_result_is_discarded_after_cancellation(self):
        token=CancellationToken();route={"type":"tool","tool":"analyze_screen","arguments":{"question":"What is open?"}}
        def late_result(action):token.cancel();return ToolResult(True,action.tool,"Late vision answer",{"verified":False})
        with patch.object(agent.executor,"execute_action",side_effect=late_result):response=agent.run_agent("what is on my screen",route=route,cancellation_token=token)
        self.assertNotIn("Late vision answer",response);self.assertIn("cancelled",response.lower())

    def test_missing_focus_and_unknown_tool_are_structured_failures(self):
        with patch("brain.tool_router.find_application_window",return_value=None):
            focus=Executor().execute_action(Action("focus_application",{"app_name":"missing-app"}))
        unknown=Executor().execute_action(Action("not_a_registered_tool",{}))
        self.assertFalse(focus.success);self.assertEqual(focus.error,"window_not_found")
        self.assertFalse(unknown.success);self.assertEqual(unknown.error,"unknown_tool")

    def test_invalid_calculation_is_not_reported_as_success(self):
        result=Executor().execute_action(Action("calculator",{"expression":"not valid"}))
        self.assertFalse(result.success);self.assertEqual(result.error,"invalid_expression")

    def test_website_launch_requires_navigation_verification(self):
        from tools import browser
        process=unittest.mock.Mock(pid=123)
        with patch.object(browser,"_resolve_chrome",return_value=r"C:\Chrome\chrome.exe"),patch.object(browser.subprocess,"Popen",return_value=process),patch.object(browser,"_verify_url_visible",return_value={"success":False,"verified":False,"error":"website_navigation_unverified"}):
            result=browser.open_website("https://www.youtube.com")
        self.assertFalse(result["success"]);self.assertFalse(result["verified"]);self.assertIn("could not verify",result["message"])

    def test_verified_website_launch_is_structured_success(self):
        from tools import browser
        process=unittest.mock.Mock(pid=123);verified={"success":True,"verified":True,"hwnd":456,"evidence":"window_title_or_address"}
        with patch.object(browser,"_resolve_chrome",return_value=r"C:\Chrome\chrome.exe"),patch.object(browser.subprocess,"Popen",return_value=process) as launch,patch.object(browser,"_verify_url_visible",return_value=verified):
            result=browser.open_website("youtube.com")
        self.assertTrue(result["success"]);self.assertTrue(result["verified"]);launch.assert_called_once_with([r"C:\Chrome\chrome.exe","--new-tab","https://youtube.com"])

    def test_optional_isolated_chrome_profile_is_explicit(self):
        from tools import browser
        process=unittest.mock.Mock(pid=123)
        with patch.object(browser,"_resolve_chrome",return_value=r"C:\Chrome\chrome.exe"),patch.dict(browser.os.environ,{"JARVIS_CHROME_USER_DATA_DIR":r"C:\tmp\profile"}),patch.object(browser.subprocess,"Popen",return_value=process) as launch,patch.object(browser,"_verify_url_visible",return_value={"success":True,"verified":True}):
            browser.open_website("https://youtube.com")
        launch.assert_called_once_with([r"C:\Chrome\chrome.exe","--new-tab",r"--user-data-dir=C:\tmp\profile","https://youtube.com"])

    def test_optional_gpu_compatibility_flag_is_explicit(self):
        from tools import browser
        process=unittest.mock.Mock(pid=123)
        with patch.object(browser,"_resolve_chrome",return_value=r"C:\Chrome\chrome.exe"),patch.dict(browser.os.environ,{"JARVIS_CHROME_DISABLE_GPU":"true"}),patch.object(browser.subprocess,"Popen",return_value=process) as launch,patch.object(browser,"_verify_url_visible",return_value={"success":True,"verified":True}):
            browser.open_website("https://youtube.com")
        self.assertIn("--disable-gpu",launch.call_args.args[0])

    def test_navigation_verification_checks_every_visible_chrome_window(self):
        from tools import browser
        with patch.object(browser,"find_application_windows",return_value=[10,20]),patch.object(browser.ctypes.windll.user32,"IsWindowVisible",return_value=True),patch.object(browser.ctypes.windll.user32,"IsHungAppWindow",return_value=False),patch.object(browser,"_window_text",side_effect=["Unrelated - Chrome","YouTube - Chrome"]),patch.object(browser,"_address_values",return_value=[]):
            result=browser._verify_url_visible("https://www.youtube.com",timeout=.1)
        self.assertTrue(result["success"]);self.assertEqual(result["hwnd"],20)

    def test_short_domain_does_not_match_arbitrary_title_character(self):
        from tools import browser
        with patch.object(browser,"find_application_windows",return_value=[10]),patch.object(browser.ctypes.windll.user32,"IsWindowVisible",return_value=True),patch.object(browser.ctypes.windll.user32,"IsHungAppWindow",return_value=False),patch.object(browser,"_window_text",return_value="Example document - Chrome"),patch.object(browser,"_address_values",return_value=["https://example.org"]):
            result=browser._verify_url_visible("https://x.co",timeout=.01)
        self.assertFalse(result["success"])

    def test_stale_route_task_planner_override_never_speaks_false_success(self):
        # Reproduces the real reported bug: route_command() builds a garbage
        # local_plan from punctuation it doesn't understand, but
        # should_use_task_planner() silently overrides that route and runs a
        # completely different (here: deterministically failing) plan. The
        # formatter must describe what actually happened, never the stale plan.
        command = "Open Chrome, go to YouTube, search for Minecraft redstone tutorial, and open the first result."
        stale_route = route_command(command)
        self.assertEqual(stale_route["type"], "local_plan")
        self.assertEqual([a.tool for a in stale_route["actions"]], ["open_application", "open_application"])
        incomplete_plan = Plan(command, [Action("browser_open_url", {"url": "https://www.youtube.com"})], context={})
        with patch.object(agent, "create_task_plan", return_value=incomplete_plan), patch.object(agent, "create_plan", return_value=[]):
            execution_outcome = {}
            response = agent.run_agent(command, route=stale_route, execution_outcome=execution_outcome)
        self.assertEqual(response, "I identified the application request, but I don't have a complete validated UI plan to play that item, so I didn't open anything.")
        # This exact message has no "failed"/"couldn't"/"could not"/"error:" keyword,
        # so the old keyword-only filter would have let it through.
        self.assertNotIn("couldn't", response.lower());self.assertNotIn("could not", response.lower());self.assertNotIn("error:", response.lower())
        self.assertFalse(execution_outcome["executed"])
        spoken = format_spoken_response(command, stale_route, response, lang="en", execution=execution_outcome)
        self.assertNotIn("chrome, go to youtube", spoken.lower())
        self.assertNotIn("opened the first result", spoken.lower())
        self.assertNotIn("opened", spoken.lower())

    def test_zero_execution_evidence_blocks_success_regardless_of_wording(self):
        route = {"type": "tool", "tool": "open_application", "arguments": {"app_name": "chrome"}}
        execution = {"executed": False, "success": False, "verified": False, "partial": False, "actions": []}
        response_text = "I don't have a complete validated plan for that yet."
        spoken = format_spoken_response("open chrome", route, response_text, lang="en", execution=execution)
        self.assertNotIn("opened", spoken.lower())
        self.assertEqual(spoken, "I don't have a complete validated plan for that yet, sir.")

    def test_failed_execution_with_non_keyword_message_does_not_speak_success(self):
        route = {"type": "local_plan", "actions": [Action("open_website", {"url": "https://www.youtube.com"})]}
        execution = {"executed": True, "success": False, "verified": False, "partial": False, "actions": [
            {"tool": "open_website", "arguments": {"url": "https://www.youtube.com"}, "success": False, "verified": False}
        ]}
        response_text = "Website did not launch in time."
        spoken = format_spoken_response("open youtube", route, response_text, lang="en", execution=execution)
        self.assertNotIn("opened", spoken.lower())
        self.assertEqual(spoken, "Website did not launch in time, sir.")

    def test_partial_compound_execution_does_not_claim_full_success(self):
        command = "run my saved routine"
        route = {"type": "local_plan", "actions": [
            Action("open_application", {"app_name": "notepad"}),
            Action("open_website", {"url": "https://www.youtube.com"}),
            Action("type_text", {"text": "hello"}),
        ]}
        results = [
            ToolResult(True, "open_application", "Opened Notepad.", {"verified": True}),
            ToolResult(False, "open_website", "Website did not launch in time.", {"verified": False}),
        ]
        with patch.object(agent, "_execute_recorded_plan", return_value=results):
            execution_outcome = {}
            response = agent.run_agent(command, route=route, execution_outcome=execution_outcome)
        self.assertTrue(execution_outcome["executed"]);self.assertTrue(execution_outcome["partial"]);self.assertFalse(execution_outcome["success"])
        spoken = format_spoken_response(command, route, response, lang="en", execution=execution_outcome)
        self.assertIn("opened notepad", spoken.lower())
        self.assertIn("did not complete", spoken.lower())
        self.assertNotIn("youtube", spoken.lower())

    def test_verified_successful_execution_still_speaks_normal_success(self):
        command = "run my saved routine"
        route = {"type": "local_plan", "actions": [
            Action("open_application", {"app_name": "notepad"}),
            Action("open_website", {"url": "https://www.youtube.com"}),
        ]}
        results = [
            ToolResult(True, "open_application", "Opened Notepad.", {"verified": True}),
            ToolResult(True, "open_website", "Opened YouTube.", {"verified": True}),
        ]
        with patch.object(agent, "_execute_recorded_plan", return_value=results):
            execution_outcome = {}
            response = agent.run_agent(command, route=route, execution_outcome=execution_outcome)
        self.assertTrue(execution_outcome["executed"]);self.assertTrue(execution_outcome["success"]);self.assertTrue(execution_outcome["verified"]);self.assertFalse(execution_outcome["partial"])
        spoken = format_spoken_response(command, route, response, lang="en", execution=execution_outcome)
        self.assertEqual(spoken, "I opened Notepad and opened YouTube, sir.")


if __name__ == "__main__":
    unittest.main()
