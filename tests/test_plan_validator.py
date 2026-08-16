import unittest
from types import SimpleNamespace
from unittest.mock import patch

from brain import agent
from brain.models import Action,Plan,ToolResult
from brain.plan_validator import validate_generated_actions,validate_plan_preflight
from brain.session_context import SessionContext
from brain.task_planner import assess_plan_completeness,create_task_plan


class GeneratedPlanValidationTests(unittest.TestCase):
    def test_incomplete_music_plan_falls_back_with_full_raw_goal(self):
        raw="Hey Jarvis, Open Apple Music and play I Love It"
        proposal=[
            {"tool":"open_application","arguments":{"app_name":"apple music"}},
            {"tool":"wait_for_window","arguments":{"app_name":"apple music"}},
            {"tool":"inspect_window","arguments":{"app_name":"apple music","limit":50}},
            {"tool":"click_ui_element","arguments":{"app_name":"apple music","name":"Search","control_type":"Button"}},
            {"tool":"type_text","arguments":{"text":"I Love It","delay":.02}},
            {"tool":"press_key","arguments":{"key":"enter"}},
        ]
        results=[SimpleNamespace(success=True,error=None,data={"verified":False},message="ok") for _ in proposal]
        with patch.object(agent,"create_plan",return_value=proposal) as cloud,patch.object(agent,"_execute_recorded_plan",return_value=results) as execute:
            agent.run_agent("Open Apple Music and play I Love It",original_user_text=raw)
        cloud.assert_called_once_with(raw);self.assertEqual([action.tool for action in execute.call_args.args[2].actions],[item["tool"] for item in proposal])

    def test_partial_cloud_music_plan_executes_nothing(self):
        proposal=[{"tool":"open_application","arguments":{"app_name":"spotify"}}]
        with patch.object(agent,"create_plan",return_value=proposal),patch.object(agent,"_execute_recorded_plan") as execute:
            response=agent.run_agent("Open Spotify and play I Love It")
        execute.assert_not_called();self.assertIn("complete validated",response)

    def test_ambiguous_music_app_is_clarified_without_guess_or_cloud(self):
        with patch.object(agent,"create_plan") as cloud,patch.object(agent,"_execute_recorded_plan") as execute:
            response=agent.run_agent("open up a music and play I Love It")
        cloud.assert_not_called();execute.assert_not_called();self.assertIn("exact name",response)

    def test_compound_browser_search_and_first_result_completes_locally_regardless_of_provider(self):
        # Generalization check: the local planner must recognize a complete
        # compound browser goal (open browser, navigate/search, click first
        # result) semantically -- via clause coverage, not by matching this
        # one reported sentence -- so it works the same for YouTube, Google,
        # and phrasings without an explicit "open <browser>" clause.
        cases=[
            "Open Chrome, go to YouTube, search for Minecraft redstone tutorial, and open the first result.",
            "Open Chrome, search Google for cute cats, and open the first result.",
            "search youtube for minecraft and open the first video",
        ]
        for command in cases:
            with self.subTest(command=command):
                plan=create_task_plan(command,SessionContext())
                self.assertIsNotNone(plan)
                self.assertEqual([a.tool for a in plan.actions],["browser_open_url","browser_click_first_result"])
                completeness=assess_plan_completeness(command,plan)
                self.assertTrue(completeness["complete"],completeness)
                self.assertEqual(completeness["represented_clause_count"],len(completeness["clauses"]))

    def test_compound_browser_search_runs_locally_without_cloud_planner_or_stale_route(self):
        command="Open Chrome, go to YouTube, search for Minecraft redstone tutorial, and open the first result."
        results=[
            ToolResult(True,"browser_open_url","Opened YouTube search.",{"verified":True}),
            ToolResult(True,"browser_click_first_result","Opened the first result.",{"verified":True}),
        ]
        with patch.object(agent,"create_plan") as cloud,patch.object(agent,"_execute_recorded_plan",return_value=results):
            execution_outcome={}
            response=agent.run_agent(command,execution_outcome=execution_outcome)
        cloud.assert_not_called()
        self.assertTrue(execution_outcome["executed"]);self.assertTrue(execution_outcome["success"])
        self.assertEqual(response,"Opened YouTube search.\nOpened the first result.")

    def test_unrelated_trailing_clause_still_escalates_to_cloud_planner(self):
        # A clause the browser-goal branch genuinely doesn't act on (muting
        # the volume) must not be silently claimed as covered just because it
        # got merged onto the tail of "open the first result" by the coarser
        # sequential-command segmenter.
        command="Open Chrome, go to YouTube, search for Minecraft redstone tutorial, open the first result, and mute the volume."
        plan=create_task_plan(command,SessionContext())
        completeness=assess_plan_completeness(command,plan)
        self.assertFalse(completeness["complete"])
        self.assertLess(completeness["represented_clause_count"],len(completeness["clauses"]))

    def test_complete_plan_preflight_accepts_real_multistep_shape(self):
        plan=Plan("open notepad, type hello, then open youtube",[
            Action("open_application",{"app_name":"notepad"}),
            Action("wait_for_window",{"app_name":"notepad"},depends_on=[0]),
            Action("type_text",{"text":"hello","delay":.02},depends_on=[1]),
            Action("open_website",{"url":"https://www.youtube.com"},depends_on=[2]),
        ],context={"planning_trace":{"segments":["open notepad","type hello","open youtube"]},"represented_clause_count":3})
        self.assertEqual(validate_plan_preflight(plan,SessionContext()),[])

    def test_preflight_rejects_unknown_tool_bad_dependency_missing_context_and_clause(self):
        plans=[
            Plan("unknown",[Action("not_registered",{})]),
            Plan("bad dependency",[Action("open_application",{"app_name":"notepad"},depends_on=[1])]),
            Plan("type",[Action("type_text",{"text":"hello","delay":.02})]),
            Plan("partial",[Action("open_application",{"app_name":"notepad"})],context={"planning_trace":{"segments":["open notepad","open youtube"]},"represented_clause_count":1}),
        ]
        for plan in plans:
            with self.subTest(goal=plan.original_goal):self.assertTrue(validate_plan_preflight(plan,SessionContext()))

    def test_preflight_failure_executes_no_partial_action(self):
        plan=Plan("open then unknown",[Action("open_application",{"app_name":"notepad"}),Action("not_registered",{},depends_on=[0])])
        with patch.object(agent.agent_runtime,"execute") as execute:
            response=agent._execute_recorded_plan(agent.get_recorder(),"preflight-no-partial",plan,agent.agent_runtime)
        execute.assert_not_called();self.assertEqual(response[0].error,"plan_preflight_failed")

    def test_cloud_planner_prompt_forbids_partial_goal_completion(self):
        from brain import planner
        response=SimpleNamespace(output=[])
        with patch.object(planner.client.responses,"create",return_value=response) as create:
            self.assertEqual(planner.create_plan("Open Settings and turn Bluetooth on."),[])
        system=create.call_args.kwargs["input"][0]["content"]
        self.assertIn("entire request",system);self.assertIn("never return a partial plan",system)

    def test_accepts_registered_well_typed_actions(self):
        actions,errors=validate_generated_actions([{"tool":"open_application","arguments":{"app_name":"notepad"}},{"tool":"type_text","arguments":{"text":"hello","delay":.02}}])
        self.assertEqual(errors,[]);self.assertEqual([a.tool for a in actions],["open_application","type_text"])
        self.assertEqual(actions[1].safe_args()["text"],"<REDACTED>")

    def test_rejects_unknown_missing_extra_bad_types_and_unsafe_values(self):
        cases=[
            {"tool":"delete_everything","arguments":{}},
            {"tool":"open_application","arguments":{}},
            {"tool":"open_application","arguments":{"app_name":"notepad","extra":1}},
            {"tool":"volume_up","arguments":{"amount":"many"}},
            {"tool":"volume_up","arguments":{"amount":999}},
            {"tool":"open_website","arguments":{"url":"file:///secret"}},
        ]
        for case in cases:
            with self.subTest(case=case):
                actions,errors=validate_generated_actions([case]);self.assertEqual(actions,[]);self.assertTrue(errors)

    def test_cloud_route_rejection_never_reaches_executor(self):
        route={"type":"tools","actions":[{"tool":"unknown_side_effect","arguments":{}}]}
        with patch.object(agent.executor,"execute_plan") as execute,patch.object(agent.agent_runtime,"execute") as runtime_execute:
            response=agent.run_agent("do unsafe thing",route=route)
        execute.assert_not_called();runtime_execute.assert_not_called();self.assertIn("rejected",response.lower())

    def test_cloud_type_action_cannot_bypass_verified_window_runtime(self):
        context=agent.agent_runtime.context;old=(context.active_app,context.last_hwnd);context.active_app=None;context.last_hwnd=None
        route={"type":"tools","actions":[{"tool":"type_text","arguments":{"text":"do not misdirect","delay":.02}}]}
        try:
            with patch.object(agent.agent_runtime.executor,"execute_action") as global_input:
                response=agent.run_agent("type do not misdirect",route=route)
            global_input.assert_not_called();self.assertIn("complete plan",response)
        finally:context.active_app,context.last_hwnd=old


if __name__=="__main__":unittest.main()
