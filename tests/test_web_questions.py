import time,unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

from brain import agent
from brain.request_intent import RequestKind,classify_request_kind
from brain.router import route_command
from brain.web_answer import FAILURE,WebAnswer,WebAnswerService
from brain.task_supervisor import CancellationToken,ReadOnlyTaskCancelled
from voice.response_formatter import format_spoken_response

class WebQuestionTests(unittest.TestCase):
    def test_question_action_semantic_classification(self):
        questions=["who is the president of France","what is the capital of Japan","tell me who invented Python","how old is Tom Cruise","what happened with Nvidia today","what is the latest Python version"]
        actions=["open chrome","can you open chrome","open youtube","turn the volume down","could you mute the computer","close notepad"]
        for text in questions:
            with self.subTest(text=text):self.assertEqual(classify_request_kind(text).kind,RequestKind.QUESTION);self.assertEqual(route_command(text)["type"],"question")
        for text in actions:
            with self.subTest(text=text):self.assertEqual(classify_request_kind(text).kind,RequestKind.ACTION)
    def test_existing_local_and_deterministic_routes_win(self):
        expected={"what time is it":"get_time","open Chrome":"open_application","open YouTube":"open_website","mute":"mute_volume","volume up":"volume_up","close Notepad":"close_application","2+2":"calculator"}
        for text,tool in expected.items():self.assertEqual(route_command(text).get("tool"),tool)
    def test_service_uses_official_read_only_web_search_and_extracts_sources(self):
        response=SimpleNamespace(output_text="Jensen Huang is Nvidia's CEO. 【1†source】 ([Nvidia](https://nvidia.com/about))",usage=SimpleNamespace(input_tokens=123,output_tokens=17),model_dump=lambda **_:{"output":[{"content":[{"annotations":[{"url":"https://nvidia.com/about","title":"Nvidia"}]}]}]})
        client=Mock();client.responses.create.return_value=response;service=WebAnswerService(client=client,model="test-model",timeout=3)
        result=service.answer("Who is Nvidia's CEO?");self.assertTrue(result.success);self.assertEqual(result.answer,"Jensen Huang is Nvidia's CEO.");self.assertEqual(result.sources[0]["url"],"https://nvidia.com/about");self.assertEqual((result.input_tokens,result.output_tokens),(123,17))
        kwargs=client.responses.create.call_args.kwargs;self.assertEqual(kwargs["tools"],[{"type":"web_search","search_context_size":"low"}]);self.assertFalse(kwargs["store"]);self.assertEqual(kwargs["timeout"],3)
    def test_timeout_api_failure_and_empty_response_are_safe(self):
        for effect,response,expected in [(TimeoutError("slow"),None,FAILURE),(ValueError("bad"),None,FAILURE),(None,SimpleNamespace(output_text="",model_dump=lambda:{}),FAILURE)]:
            client=Mock();client.responses.create.side_effect=effect
            if response is not None:client.responses.create.return_value=response
            result=WebAnswerService(client=client).answer("question");self.assertFalse(result.success);self.assertEqual(result.answer,expected)
    def test_cancelled_request_never_calls_openai(self):
        client=Mock();token=CancellationToken();token.cancel()
        with self.assertRaises(ReadOnlyTaskCancelled):WebAnswerService(client=client).answer("question",token)
        client.responses.create.assert_not_called()
    def test_question_path_never_invokes_desktop_executor(self):
        service=Mock(model="test-model",timeout=3);service.answer.return_value=WebAnswer("Paris is the capital of France.",True,model="test-model")
        with patch.object(agent,"get_web_answer_service",return_value=service),patch.object(agent.executor,"execute_action") as desktop:
            answer=agent.run_agent("what is the capital of France",route={"type":"question","message":"what is the capital of France"})
        self.assertEqual(answer,"Paris is the capital of France.");desktop.assert_not_called();self.assertEqual(service.answer.call_count,1);self.assertEqual(service.answer.call_args.args[0],"what is the capital of France")
    def test_classified_question_uses_same_read_only_pipeline(self):
        text="who invented the telephone"
        route=route_command(text)
        service=Mock(model="test-model",timeout=3);service.answer.return_value=WebAnswer("Alexander Graham Bell is generally credited with inventing the telephone.",True,model="test-model")
        with patch.object(agent,"get_web_answer_service",return_value=service),patch.object(agent.executor,"execute_action") as desktop:
            answer=agent.run_agent(text,route=route)
        self.assertIn("Alexander",answer);desktop.assert_not_called();self.assertEqual(service.answer.call_count,1);self.assertEqual(service.answer.call_args.args[0],text)
    def test_question_response_uses_existing_spoken_formatter(self):
        route={"type":"question","message":"who invented Python"};spoken=format_spoken_response("who invented Python",route,"Guido van Rossum created Python. https://example.com")
        self.assertEqual(spoken,"Guido van Rossum created Python, sir.")
    def test_intent_classification_overhead_is_negligible(self):
        started=time.perf_counter()
        for _ in range(1000):classify_request_kind("Can you tell me who invented Python?")
        self.assertLess((time.perf_counter()-started)/1000,.001)

    def test_stable_question_cache_is_bounded_and_volatile_questions_bypass_it(self):
        response=SimpleNamespace(output_text="Guido van Rossum created Python.",model_dump=lambda **_: {})
        client=Mock();client.responses.create.return_value=response;service=WebAnswerService(client=client)
        first=service.answer("Who created Python?");second=service.answer("  who CREATED Python? ")
        self.assertFalse(first.cache_hit);self.assertTrue(second.cache_hit);self.assertEqual(client.responses.create.call_count,1)
        service.answer("What is the latest Python version?");service.answer("What is the latest Python version?")
        self.assertEqual(client.responses.create.call_count,3)

    def test_cancelled_token_cannot_receive_cached_late_result(self):
        response=SimpleNamespace(output_text="A stable answer.",model_dump=lambda **_: {})
        client=Mock();client.responses.create.return_value=response;service=WebAnswerService(client=client)
        service.answer("Who invented the telephone?");token=CancellationToken();token.cancel()
        with self.assertRaises(ReadOnlyTaskCancelled):service.answer("Who invented the telephone?",token)

if __name__=="__main__":unittest.main()
