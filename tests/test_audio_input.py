import unittest
import time
from pathlib import Path
from unittest.mock import Mock,patch

import numpy as np

from voice.audio_input import NativeMicrophoneStream,_candidate_devices


class FakeRawStream:
    def __init__(self,**kwargs):self.kwargs=kwargs
    def __enter__(self):return self
    def __exit__(self,*_):return None
    def read(self,frames):
        left=np.full(frames,1200,dtype=np.int16);right=np.full(frames,-200,dtype=np.int16)
        return np.column_stack((left,right)).reshape(-1).tobytes(),False


class FakeSoundDevice:
    default=type("Default",(),{"device":[1,4]})()
    def __init__(self):self.opened=[]
    def query_devices(self):return [
        {"name":"Mapper","hostapi":0,"max_input_channels":2,"default_samplerate":44100.0},
        {"name":"Microphone (High Definition Aud","hostapi":0,"max_input_channels":2,"default_samplerate":44100.0},
        {"name":"Microphone (High Definition Audio Device)","hostapi":1,"max_input_channels":2,"default_samplerate":48000.0},
    ]
    def query_hostapis(self,index):return {"name":"MME" if index==0 else "Windows WASAPI"}
    def RawInputStream(self,**kwargs):self.opened.append(kwargs);return FakeRawStream(**kwargs)


class AudioInputTests(unittest.TestCase):
    def test_stable_default_name_selects_equivalent_host_apis(self):
        sd=FakeSoundDevice()
        with patch.dict("os.environ",{},clear=True):
            candidates=_candidate_devices(sd)
        self.assertEqual([item[0] for item in candidates],[1,2])

    def test_similarly_prefixed_different_microphone_is_not_selected(self):
        sd=FakeSoundDevice();devices=sd.query_devices();devices.append({"name":"Microphone (High Definition Other Device)","hostapi":1,"max_input_channels":2,"default_samplerate":48000.0})
        with patch.object(sd,"query_devices",return_value=devices),patch.dict("os.environ",{},clear=True):candidates=_candidate_devices(sd)
        self.assertEqual([item[0] for item in candidates],[1,2])

    def test_native_stereo_is_downmixed_and_resampled_to_pipeline_shape(self):
        sd=FakeSoundDevice();stream=NativeMicrophoneStream.__new__(NativeMicrophoneStream)
        stream.sd=sd;stream.target_rate=16000;stream.target_frame_samples=1280;stream.stream=None;stream.input_rate=None;stream.channels=None;stream.device_identity=None
        with patch.dict("os.environ",{},clear=True):
            with stream:
                raw,overflow=stream.read(1280)
        audio=np.frombuffer(raw,dtype=np.int16)
        self.assertEqual(sd.opened[0]["samplerate"],44100)
        self.assertEqual(sd.opened[0]["channels"],2)
        self.assertEqual(audio.shape,(1280,));self.assertFalse(overflow)
        self.assertGreater(float(audio.mean()),400)

    def test_missing_default_input_uses_valid_host_preference_not_negative_index(self):
        sd=FakeSoundDevice();sd.default=type("Default",(),{"device":[-1,4]})()
        with patch.dict("os.environ",{},clear=True):candidates=_candidate_devices(sd)
        self.assertEqual([item[0] for item in candidates],[2,0,1])

    def test_invalid_zero_rate_input_is_excluded(self):
        sd=FakeSoundDevice();devices=sd.query_devices();devices[0]["default_samplerate"]=0
        with patch.object(sd,"query_devices",return_value=devices),patch.dict("os.environ",{"JARVIS_MICROPHONE_NAME":"Mapper"}):self.assertEqual(_candidate_devices(sd),[])

    def test_push_to_talk_uses_native_microphone_adapter(self):
        from voice.listener import listen_push_to_talk
        class Stream:
            def __enter__(self):return self
            def __exit__(self,*_):return None
            def read(self,samples):time.sleep(.001);return np.full(samples,500,dtype=np.int16).tobytes(),False
        calls=0
        def press_enter(_prompt=""):
            nonlocal calls
            calls+=1
            if calls==2:time.sleep(.02)
            return ""
        path=None
        with patch("voice.audio_input.open_jarvis_microphone",return_value=Stream()) as opened,patch("builtins.input",side_effect=press_enter),patch("voice.listener.sf.write") as write:
            path=listen_push_to_talk()
        try:
            opened.assert_called_once_with(16000,1280)
            self.assertIsNotNone(path);self.assertEqual(write.call_args.args[2],16000)
            self.assertEqual(write.call_args.args[1].ndim,1)
        finally:
            if path:Path(path).unlink(missing_ok=True)

    def test_push_to_talk_ctrl_c_stops_capture_worker(self):
        from voice.listener import listen_push_to_talk
        stream=type("Stream",(),{"__enter__":lambda self:self,"__exit__":lambda self,*_:None})()
        worker=Mock()
        with patch("voice.audio_input.open_jarvis_microphone",return_value=stream),patch("voice.listener.threading.Thread",return_value=worker),patch("builtins.input",side_effect=["",KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):listen_push_to_talk()
        worker.start.assert_called_once_with();worker.join.assert_called_once_with(timeout=2)


if __name__=="__main__":unittest.main()
