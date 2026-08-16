import time
import unittest
from unittest.mock import patch

from brain import agent
from brain.agent_runtime import AgentRuntime
from brain.models import ToolResult
from brain.router import route_command
from brain.session_context import SessionContext
from brain.task_planner import create_task_plan,segment_sequential_commands,should_use_task_planner
from voice.text_normalizer import normalize_transcript


class MultiStepContextTests(unittest.TestCase):
    def setUp(self):
        self.context=SessionContext(last_assistant_response="Minecraft was created by Markus Persson.")

    def tools(self,text):
        plan=create_task_plan(text,self.context);self.assertIsNotNone(plan);return plan,[a.tool for a in plan.actions]

    def test_required_segmentation_variants(self):
        for text in (
            "open notepad and then type hello and then save it",
            "open notepad then type hello then save it",
            "open notepad, type hello, then save it",
        ):
            with self.subTest(text=text):
                plan,tools=self.tools(text)
                self.assertEqual(tools,["open_application","wait_for_window","type_text","save_current_document"])
                self.assertEqual(plan.actions[2].args["text"],"hello")
                self.assertEqual(plan.context["model_calls"],0)

    def test_literal_payload_boundaries_are_not_commands(self):
        cases={
            'type "hello and then save it"':"hello and then save it",
            "type rock and roll":"rock and roll",
            'type "open YouTube"':"open YouTube",
            "type who created Minecraft":"who created Minecraft",
        }
        for text,payload in cases.items():
            with self.subTest(text=text):
                route=route_command(text)
                self.assertEqual(route["type"],"tool");self.assertEqual(route["tool"],"type_text");self.assertEqual(route["arguments"]["text"],payload)

    def test_previous_answer_reference_is_local_and_exact(self):
        for text in ("type exactly what you just said","type what you just told me","type what you last told me","type what you last said","type what you told me","type your last answer","type your last response","type that","type the last thing"):
            with self.subTest(text=text):
                self.assertTrue(should_use_task_planner(text))
                plan,_=self.tools(text)
                self.assertEqual(plan.actions[0].args["text"],self.context.last_assistant_response)
                self.assertEqual(plan.context["model_calls"],0)

    def test_only_high_confidence_truncated_reference_is_normalized(self):
        self.assertEqual(normalize_transcript("type exactly what you last-")[0],"type exactly what you last told me")
        self.assertEqual(normalize_transcript('type "what you last-"')[0],'type "what you last-"')
        self.assertEqual(normalize_transcript("type exactly what you like")[0],"type exactly what you like")

    def test_multistep_reference_resolves_before_execution(self):
        plan,tools=self.tools("open notepad and then type exactly what you just said and then save it")
        self.assertEqual(tools,["open_application","wait_for_window","type_text","save_current_document"])
        self.assertEqual(plan.actions[2].args["text"],self.context.last_assistant_response)
        self.assertTrue(plan.context["resolved_references"])

    def test_missing_context_clarifies_without_typing(self):
        plan=create_task_plan("type what you just said",SessionContext())
        self.assertEqual(plan.actions,[])
        self.assertIn("previous answer",plan.context["clarification"])

    def test_execution_order_and_failure_dependency(self):
        plan,_=self.tools("open notepad then type hello then save it")
        runtime=AgentRuntime(context=SessionContext(),trace=False)
        calls=[]
        def succeed(action):calls.append(action.tool);return ToolResult(True,action.tool,"ok",{"pid":1})
        with patch.object(runtime,"_execute_action",side_effect=succeed):results=runtime.execute(plan)
        self.assertEqual(calls,["open_application","wait_for_window","type_text","save_current_document"]);self.assertTrue(all(r.success for r in results))
        failed_plan=create_task_plan("open notepad then type hello then save it",self.context);calls=[]
        def fail_open(action):calls.append(action.tool);return ToolResult(False,action.tool,"failed",error="launch_failed")
        with patch.object(runtime,"_execute_action",side_effect=fail_open):runtime.execute(failed_plan)
        self.assertEqual(calls,["open_application"])

    def test_local_planning_latency_and_no_model_call(self):
        text="open notepad and then type hello and then save it"
        started=time.perf_counter()
        with patch.object(agent,"create_plan") as model_planner,patch.object(agent,"ask_ai") as llm:
            for _ in range(500):create_task_plan(text,self.context)
        average_ms=(time.perf_counter()-started)*1000/500
        self.assertLess(average_ms,2);model_planner.assert_not_called();llm.assert_not_called()

    def test_agent_resolves_previous_answer_without_llm(self):
        previous=agent.agent_runtime.context.last_assistant_response
        previous_spoken=agent.agent_runtime.context.last_spoken_response
        previous_target=(agent.agent_runtime.context.active_app,agent.agent_runtime.context.last_hwnd)
        agent.agent_runtime.context.last_assistant_response=self.context.last_assistant_response
        agent.agent_runtime.context.last_spoken_response=None
        agent.agent_runtime.context.active_app="notepad";agent.agent_runtime.context.last_hwnd=4242
        captured=[]
        def execute(plan,**kwargs):
            captured.append(plan);result=ToolResult(True,"type_text","Typed successfully.");observer=kwargs.get("action_observer")
            if observer:observer("prepared",0,plan.actions[0],None,agent.agent_runtime.context);observer("result",0,plan.actions[0],result,agent.agent_runtime.context)
            return [result]
        try:
            with patch.object(agent.agent_runtime,"execute",side_effect=execute),patch.object(agent,"create_plan") as model_planner,patch.object(agent,"ask_ai") as llm:
                response=agent.run_agent("type exactly what you just said")
            self.assertEqual(captured[0].actions[0].args["text"],self.context.last_assistant_response)
            self.assertIn("Typed",response);model_planner.assert_not_called();llm.assert_not_called()
        finally:
            agent.agent_runtime.context.last_assistant_response=previous
            agent.agent_runtime.context.last_spoken_response=previous_spoken
            agent.agent_runtime.context.active_app,agent.agent_runtime.context.last_hwnd=previous_target

    def test_segmenter_protects_quotes_and_natural_and(self):
        self.assertEqual(segment_sequential_commands('type "hello and then save it"'),['type "hello and then save it"'])
        self.assertEqual(segment_sequential_commands("type rock and roll"),["type rock and roll"])

    def test_real_regression_variants(self):
        previous=self.context.last_assistant_response
        cases={
            "Open Notepad and type exactly what you just said.":["open_application","wait_for_window","type_text"],
            "Open Notepad, type exactly what you last told me, then open YouTube.":["open_application","wait_for_window","type_text","open_website"],
            "Type what you just said.":["type_text"],
            "Type what you last told me.":["type_text"],
            "Type rock and roll, then open YouTube.":["type_text","open_website"],
            'Type "what you last told me, then open YouTube".':["type_text"],
        }
        for text,expected_tools in cases.items():
            with self.subTest(text=text):
                plan=create_task_plan(text,self.context)
                self.assertEqual([a.tool for a in plan.actions],expected_tools)
                typed=next(a.args["text"] for a in plan.actions if a.tool=="type_text")
                if text.startswith('Type "'):
                    self.assertEqual(typed,"what you last told me, then open YouTube")
                elif "rock and roll" in text:
                    self.assertEqual(typed,"rock and roll")
                else:
                    self.assertEqual(typed,previous)

    def test_current_notepad_payload_and_quoted_reference_regressions(self):
        previous="Minecraft was created by Markus Persson."
        context=SessionContext(last_spoken_response=previous,last_assistant_response="different")
        multi=create_task_plan("open notepad, type hello, then open youtube",context)
        self.assertEqual([action.tool for action in multi.actions],["open_application","wait_for_window","type_text","open_website"])
        self.assertEqual(multi.actions[2].args["text"],"hello")
        resolved=create_task_plan("open notepad and type exactly what you just told me",context)
        self.assertEqual(resolved.actions[2].args["text"],previous)
        literal=create_task_plan('open notepad and type "what you just told me"',context)
        self.assertEqual(literal.actions[2].args["text"],"what you just told me")

    def test_voice_router_planner_runtime_path_for_real_regression(self):
        raw="Open Notepad, type exactly what you last told me, then open YouTube."
        normalized,_=normalize_transcript(raw)
        route=route_command(normalized)
        old_assistant=agent.agent_runtime.context.last_assistant_response
        old_spoken=agent.agent_runtime.context.last_spoken_response
        captured=[]
        agent.agent_runtime.context.last_assistant_response=self.context.last_assistant_response
        agent.agent_runtime.context.last_spoken_response="The previously spoken answer."
        def execute(plan,**kwargs):
            captured.append(plan)
            results=[ToolResult(True,action.tool,"ok") for action in plan.actions];observer=kwargs.get("action_observer")
            if observer:
                for index,(action,result) in enumerate(zip(plan.actions,results)):observer("prepared",index,action,None,agent.agent_runtime.context);observer("result",index,action,result,agent.agent_runtime.context)
            return results
        try:
            with patch.object(agent.agent_runtime,"execute",side_effect=execute),patch.object(agent,"create_plan") as model_planner,patch.object(agent,"ask_ai") as llm:
                agent.run_agent(normalized,route=route,original_user_text=raw)
            plan=captured[0]
            self.assertEqual([a.tool for a in plan.actions],["open_application","wait_for_window","type_text","open_website"])
            self.assertEqual(plan.actions[2].args["text"],"The previously spoken answer.")
            self.assertEqual(plan.actions[3].args["url"],"https://www.youtube.com")
            self.assertTrue(plan.context["planning_trace"]["had_last_assistant_response"])
            self.assertTrue(plan.context["planning_trace"]["had_last_spoken_response"])
            self.assertEqual(plan.context["planning_trace"]["type_text"][0]["reference_source"],"last_spoken_response")
            model_planner.assert_not_called();llm.assert_not_called()
        finally:
            agent.agent_runtime.context.last_assistant_response=old_assistant
            agent.agent_runtime.context.last_spoken_response=old_spoken


if __name__=="__main__":unittest.main()
