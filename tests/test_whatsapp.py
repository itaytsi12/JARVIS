import json,sys,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

from brain.session_context import SessionContext
from brain.task_planner import create_task_plan
from tools import whatsapp
from brain.task_supervisor import CancellationToken
from brain import agent
from brain.models import ToolResult
from brain.router import route_command
from brain.task_supervisor import register_interactive_task,unregister_interactive_task

class WhatsAppTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/"contacts.json"
        self.contacts=[
            {"name":"אמא","aliases":["mom","mother","ima"],"phone":"+972500000001","preferred_language":"he"},
            {"name":"Emily","aliases":["em"],"phone":"+15550000002","preferred_language":"en"},
            {"name":"Daniel One","aliases":["daniel"],"phone":"+15550000003","preferred_language":"en"},
            {"name":"Daniel Two","aliases":["daniel"],"phone":"+15550000004","preferred_language":"en"},
        ];self.path.write_text(json.dumps({"contacts":self.contacts},ensure_ascii=False),encoding="utf-8")
        self.env=patch.dict("os.environ",{"JARVIS_WHATSAPP_CONTACTS_PATH":str(self.path)});self.env.start()
    def tearDown(self):self.env.stop();self.temp.cleanup()
    def test_parser_payload_boundaries_and_literals(self):
        context=SessionContext(last_assistant_response="Minecraft was created by Markus Persson.")
        cases={
            "send Daniel on WhatsApp hello":("Daniel","hello",False,1),
            "message Daniel on WhatsApp that I'm outside":("Daniel","I'm outside",False,1),
            "send Daniel hello and then open YouTube":("Daniel","hello",False,2),
            'send Daniel "hello and then open YouTube"':("Daniel","hello and then open YouTube",True,1),
            'send Daniel "open Chrome"':("Daniel","open Chrome",True,1),
            "send Daniel exactly what you just said":("Daniel",context.last_assistant_response,True,1),
        }
        for text,(recipient,message,literal,count) in cases.items():
            with self.subTest(text=text):
                plan=create_task_plan(text,context);self.assertEqual(len([a for a in plan.actions if a.tool!="wait_for_window"]),count)
                action=plan.actions[0];self.assertEqual(action.tool,"send_whatsapp_message");self.assertEqual(action.args,{"recipient":recipient,"message":message,"literal":literal})
    def test_recipient_unique_ambiguous_missing_and_alias(self):
        self.assertEqual(whatsapp.resolve_recipient("mom").status,whatsapp.RecipientStatus.UNIQUE)
        self.assertEqual(whatsapp.resolve_recipient("Daniel").status,whatsapp.RecipientStatus.AMBIGUOUS)
        self.assertEqual(whatsapp.resolve_recipient("Nobody").status,whatsapp.RecipientStatus.NOT_FOUND)
        self.assertEqual(whatsapp.resolve_recipient("").status,whatsapp.RecipientStatus.INVALID)
    def test_contextual_recipient_and_missing_context(self):
        plan=create_task_plan("tell her I'll be late",SessionContext(last_messaging_recipient="Emily"));self.assertEqual(plan.actions[0].args["recipient"],"Emily")
        missing=create_task_plan("tell her I'll be late",SessionContext());self.assertEqual(missing.actions,[]);self.assertIn("recent WhatsApp recipient",missing.context["clarification"])
    def test_translation_only_when_needed_and_literal_bypasses(self):
        response=SimpleNamespace(output_text="אני בדרך.",usage=SimpleNamespace(input_tokens=8,output_tokens=4))
        client=Mock();client.responses.create.return_value=response
        with patch("openai.OpenAI",return_value=client):
            translated,meta=whatsapp._translate("I'm on my way.","he")
        self.assertEqual(translated,"אני בדרך.");self.assertTrue(meta["translated"]);self.assertEqual(client.responses.create.call_count,1)
        with patch("openai.OpenAI") as api:
            same,meta=whatsapp._translate("אני בדרך","he")
        self.assertEqual(same,"אני בדרך");self.assertFalse(meta["translated"]);api.assert_not_called()
        with patch("openai.OpenAI") as api:
            english,meta=whatsapp._translate("I'm on my way.","en")
        self.assertEqual(english,"I'm on my way.");self.assertFalse(meta["translated"]);api.assert_not_called()
    def test_send_commits_exactly_once_after_target_and_payload_verification(self):
        self.contacts[0]["preferred_language"]="en";self.path.write_text(json.dumps({"contacts":self.contacts},ensure_ascii=False),encoding="utf-8")
        class Control:
            def __init__(self,name):self.element_info=SimpleNamespace(name=name);self.value="";self.sent=0
            def set_edit_text(self,text):self.value=text
            def get_value(self):return self.value
            def type_keys(self,key):self.sent+=1
        header=Control("אמא");box=Control("Type a message")
        sent_message=Control("hello")
        window=Mock();window.descendants.side_effect=lambda control_type=None:([header,sent_message] if box.sent else [header]) if control_type=="Text" else [box];window.wait.return_value=None
        desktop=Mock();desktop.window.return_value=window
        fake_module=SimpleNamespace(Desktop=Mock(return_value=desktop))
        with patch.object(whatsapp.subprocess,"Popen"),patch.object(whatsapp,"find_application_window",return_value=123),patch.dict(sys.modules,{"pywinauto":fake_module}):
            result=whatsapp.send_whatsapp_message("mom","hello",literal=True)
        self.assertTrue(result["success"]);self.assertTrue(result["verified"]);self.assertTrue(result["committed"]);self.assertEqual(box.value,"hello");self.assertEqual(box.sent,1)

    def test_unverified_commit_is_not_reported_successful_or_retried(self):
        self.contacts[1]["preferred_language"]="en";self.path.write_text(json.dumps({"contacts":self.contacts},ensure_ascii=False),encoding="utf-8")
        class Control:
            def __init__(self,name):self.element_info=SimpleNamespace(name=name);self.value="";self.sent=0
            def set_edit_text(self,text):self.value=text
            def get_value(self):return self.value
            def type_keys(self,key):self.sent+=1
        header=Control("Emily");box=Control("Type a message");window=Mock();window.wait.return_value=None
        window.descendants.side_effect=lambda control_type=None:[header] if control_type=="Text" else [box]
        desktop=Mock();desktop.window.return_value=window;fake_module=SimpleNamespace(Desktop=Mock(return_value=desktop))
        with patch.object(whatsapp.subprocess,"Popen"),patch.object(whatsapp,"find_application_window",return_value=123),patch.object(whatsapp.time,"sleep"),patch.dict(sys.modules,{"pywinauto":fake_module}):
            result=whatsapp.send_whatsapp_message("Emily","hello",literal=True)
        self.assertFalse(result["success"]);self.assertEqual(result["error"],"send_unverified");self.assertTrue(result["committed"]);self.assertEqual(box.sent,1)
    def test_ambiguous_contact_never_launches_or_sends(self):
        with patch.object(whatsapp.subprocess,"Popen") as launch:
            result=whatsapp.send_whatsapp_message("Daniel","hello")
        self.assertFalse(result["success"]);self.assertEqual(result["recipient_status"],"AMBIGUOUS");launch.assert_not_called()

    def test_contact_name_in_chat_list_is_not_enough_to_verify_active_chat(self):
        self.contacts[1]["preferred_language"]="en";self.path.write_text(json.dumps({"contacts":self.contacts},ensure_ascii=False),encoding="utf-8")
        class Control:
            def __init__(self,name):self.element_info=SimpleNamespace(name=name);self.sent=0
            def set_edit_text(self,text):raise AssertionError("must not type into an unverified chat")
        listed=Control("Emily");duplicate_header=Control("Emily");box=Control("Type a message")
        window=Mock();window.wait.return_value=None;window.descendants.side_effect=lambda control_type=None:[listed,duplicate_header] if control_type=="Text" else [box]
        desktop=Mock();desktop.window.return_value=window;fake_module=SimpleNamespace(Desktop=Mock(return_value=desktop))
        with patch.object(whatsapp.subprocess,"Popen"),patch.object(whatsapp,"find_application_window",return_value=123),patch.dict(sys.modules,{"pywinauto":fake_module}):
            result=whatsapp.send_whatsapp_message("Emily","hello",literal=True)
        self.assertFalse(result["success"]);self.assertEqual(result["error"],"target_verification_failed")

    def test_cancellation_while_waiting_for_window_stops_before_ui_send(self):
        token=CancellationToken();calls=0
        def wait_for_app(_):
            nonlocal calls;calls+=1;token.cancel();return None
        with patch.object(whatsapp.subprocess,"Popen"),patch.object(whatsapp,"find_application_window",side_effect=wait_for_app):
            result=whatsapp.send_whatsapp_message("mom","hello",literal=True,cancellation_token=token)
        self.assertFalse(result["success"]);self.assertEqual(result["error"],"cancelled");self.assertEqual(calls,1)
    def test_cancel_before_commit_prevents_send(self):
        token=CancellationToken();token.cancel()
        with patch.object(whatsapp.subprocess,"Popen") as launch:
            result=whatsapp.send_whatsapp_message("mom","open Chrome",literal=True,cancellation_token=token)
        self.assertFalse(result["success"]);self.assertEqual(result["error"],"cancelled");launch.assert_not_called()
    def test_recipient_correction_cancels_pending_before_new_plan(self):
        context=agent.agent_runtime.context;old=(context.pending_messaging_message,context.messaging_committed)
        context.pending_messaging_message="I'll be late.";context.messaging_committed=False
        token=CancellationToken();task_id=register_interactive_task(token)
        waiter=threading.Thread(target=lambda:(token._cancelled.wait(1),unregister_interactive_task(task_id)),daemon=True);waiter.start()
        captured=[]
        def execute(plan,**kwargs):
            captured.append(plan);result=ToolResult(True,"send_whatsapp_message","sent");observer=kwargs.get("action_observer")
            if observer:observer("prepared",0,plan.actions[0],None,agent.agent_runtime.context);observer("result",0,plan.actions[0],result,agent.agent_runtime.context)
            return [result]
        with patch.object(agent.agent_runtime,"execute",side_effect=execute):
            response=agent.run_agent("send it to Alex instead",route=route_command("send it to Alex instead"))
        waiter.join(1);self.assertTrue(token.cancelled);self.assertEqual(captured[0].actions[0].args["recipient"],"alex");self.assertIn("sent",response)
        context.pending_messaging_message,context.messaging_committed=old
    def test_completed_send_is_not_falsely_undone_or_duplicated(self):
        context=agent.agent_runtime.context;old=(context.pending_messaging_message,context.messaging_committed)
        context.pending_messaging_message="Already sent";context.messaging_committed=True
        with patch.object(agent.agent_runtime,"execute") as execute:
            response=agent.run_agent("send it to Alex instead",route=route_command("send it to Alex instead"))
        self.assertIn("already sent",response);execute.assert_not_called();context.pending_messaging_message,context.messaging_committed=old

if __name__=="__main__":unittest.main()
