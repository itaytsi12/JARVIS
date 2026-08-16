"""Native microphone capture adapted to JARVIS's 16 kHz mono pipeline."""
from __future__ import annotations

import logging
import os
from math import gcd

import numpy as np
from scipy.signal import resample_poly


log = logging.getLogger("jarvis.audio_input")


def _host_name(sd, device: dict) -> str:
    try:return str(sd.query_hostapis(device["hostapi"])["name"])
    except Exception:return f"hostapi:{device.get('hostapi','unknown')}"


def _candidate_devices(sd) -> list[tuple[int, dict, str]]:
    devices=list(sd.query_devices());default_index=int(sd.default.device[0])
    requested_name=os.getenv("JARVIS_MICROPHONE_NAME","").strip().casefold()
    requested_host=os.getenv("JARVIS_MICROPHONE_HOST_API","").strip().casefold()
    available=[]
    for index,device in enumerate(devices):
        if int(device.get("max_input_channels",0))<1 or float(device.get("default_samplerate",0) or 0)<=0:continue
        host=_host_name(sd,device)
        if requested_name and requested_name not in str(device["name"]).casefold():continue
        if requested_host and requested_host not in host.casefold():continue
        available.append((index,device,host))
    preference={"Windows WASAPI":0,"Windows DirectSound":1,"MME":2}
    if requested_name or requested_host:return sorted(available,key=lambda item:preference.get(item[2],9))
    if default_index<0 or default_index>=len(devices) or int(devices[default_index].get("max_input_channels",0))<1:
        return sorted(available,key=lambda item:preference.get(item[2],9))
    default=devices[default_index];default_name=str(default["name"]).casefold()
    # PortAudio truncates some MME names. Match the stable physical-name prefix
    # so the same endpoint can be tried through WASAPI/DirectSound as well.
    equivalents=[]
    for item in available:
        candidate_name=str(item[1]["name"]).casefold()
        if item[0]==default_index or candidate_name.startswith(default_name) or default_name.startswith(candidate_name):equivalents.append(item)
    return sorted(equivalents,key=lambda item:(0 if item[0]==default_index else 1,preference.get(item[2],9)))


class NativeMicrophoneStream:
    """Expose native hardware capture as fixed-size 16 kHz mono int16 frames."""
    def __init__(self,target_rate:int,target_frame_samples:int):
        import sounddevice as sd
        self.sd=sd;self.target_rate=int(target_rate);self.target_frame_samples=int(target_frame_samples)
        self.stream=None;self.input_rate=None;self.channels=None;self.device_identity=None

    def __enter__(self):
        failures=[]
        for index,device,host in _candidate_devices(self.sd):
            rate=int(round(float(device["default_samplerate"])))
            channel_options=[]
            for channels in (min(2,int(device["max_input_channels"])),1):
                if channels not in channel_options:channel_options.append(channels)
            for channels in channel_options:
                native_frames=max(1,round(self.target_frame_samples*rate/self.target_rate))
                try:
                    stream=self.sd.RawInputStream(device=index,samplerate=rate,blocksize=native_frames,channels=channels,dtype="int16")
                    stream.__enter__()
                    self.stream=stream;self.input_rate=rate;self.channels=channels
                    self.device_identity=(str(device["name"]),host)
                    log.info("Microphone opened: name=%r host_api=%r rate=%s channels=%s target_rate=%s",self.device_identity[0],host,rate,channels,self.target_rate)
                    return self
                except Exception as exc:
                    failures.append(f"{device['name']} [{host}] {rate}Hz/{channels}ch: {exc}")
        raise RuntimeError("No compatible microphone input could be opened. " + " | ".join(failures))

    def __exit__(self,*args):
        if self.stream is not None:return self.stream.__exit__(*args)
        return None

    def read(self,samples:int):
        native_frames=max(1,round(samples*self.input_rate/self.target_rate))
        raw,overflowed=self.stream.read(native_frames)
        audio=np.frombuffer(raw,dtype=np.int16)
        if self.channels>1:
            audio=audio.reshape(-1,self.channels).astype(np.int32).mean(axis=1).astype(np.int16)
        divisor=gcd(self.input_rate,self.target_rate)
        converted=resample_poly(audio.astype(np.float32),self.target_rate//divisor,self.input_rate//divisor)
        if converted.size<samples:converted=np.pad(converted,(0,samples-converted.size))
        converted=np.clip(np.rint(converted[:samples]),-32768,32767).astype(np.int16)
        return converted.tobytes(),overflowed


def open_jarvis_microphone(target_rate:int,target_frame_samples:int):
    return NativeMicrophoneStream(target_rate,target_frame_samples)
