import time,unittest
from unittest.mock import Mock,patch
import numpy as np

from brain import agent
from brain.router import route_command
from voice import text_to_speech
from voice.background_assistant import AlwaysOnAssistant,AssistantState
from test_background_assistant import FakeStream,FakeWakeEngine,StepClock

class BargeInTests(unittest.TestCase):
    def test_tts_stop_reaches_all_providers_and_future_speech_remains_available(self):
        openai=Mock();chatterbox=Mock();engine=Mock()
        with patch.object(text_to_speech,"_openai_provider",openai),patch.object(text_to_speech,"_chatterbox_provider",chatterbox),patch.object(text_to_speech,"_engine",engine):text_to_speech.stop()
        openai.stop.assert_called_once();chatterbox.stop.assert_called_once();engine.stop.assert_called_once()
    def test_wake_word_during_tts_requests_stop_and_enters_listening(self):
        class Capture(AlwaysOnAssistant):
            def _process_capture(self,frames,**kwargs):self._stop.set()
        quiet=np.zeros(1280,dtype=np.int16);loud=np.full(1280,2000,dtype=np.int16)
        states=[];assistant=Capture(wake_engine=FakeWakeEngine(),stream_factory=lambda:FakeStream([quiet,loud]+[quiet]*15),clock=StepClock(),state_callback=lambda state,detail:states.append(state));assistant.state=AssistantState.SPEAKING;assistant.silence_seconds=.2
        with patch("voice.text_to_speech.stop") as stop:assistant._audio_session()
        stop.assert_called_once();self.assertIn(AssistantState.INTERRUPTED_LISTENING,states);self.assertIn(AssistantState.LISTENING,states)
    def test_local_correction_and_resume_use_buffer_without_llm(self):
        context=agent.agent_runtime.context;old=(context.interrupted_response,context.last_spoken_response)
        context.interrupted_response="First important item. Second useful item. Third detail.";context.last_spoken_response=context.interrupted_response
        try:
            with patch.object(agent,"ask_ai") as llm:
                short=agent.run_agent("make it shorter",route=route_command("make it shorter"));resumed=agent.run_agent("continue",route=route_command("continue"))
            self.assertEqual(short,"First important item.");self.assertEqual(resumed,short);llm.assert_not_called()
        finally:context.interrupted_response,context.last_spoken_response=old
    def test_stop_never_mind_and_new_action_routes_stay_deterministic(self):
        self.assertEqual(route_command("stop")["type"],"cancel_read_only_task");self.assertEqual(route_command("never mind")["type"],"cancel_read_only_task");self.assertEqual(route_command("open YouTube")["tool"],"open_website")
    def test_barge_in_new_command_keeps_old_tts_stopped_and_executes_action(self):
        quiet=np.zeros(1280,dtype=np.int16);loud=np.full(1280,2000,dtype=np.int16);executed=[]
        assistant=AlwaysOnAssistant(wake_engine=FakeWakeEngine(),stream_factory=lambda:FakeStream([quiet,loud]+[quiet]*15),clock=StepClock());assistant.state=AssistantState.SPEAKING;assistant.silence_seconds=.2;assistant.cooldown_seconds=0
        with patch("voice.text_to_speech.stop") as stop,patch("voice.speech_to_text.transcribe_audio",return_value="open YouTube"),patch("brain.agent.run_agent",side_effect=lambda *a,**k:(executed.append(a[0]) or "Opened YouTube.")),patch("voice.text_to_speech.speak"):
            assistant._audio_session();deadline=time.perf_counter()+1
            while not executed and time.perf_counter()<deadline:time.sleep(.01)
        stop.assert_called_once();self.assertEqual(executed,["open YouTube"])

if __name__=="__main__":unittest.main()
