"""Microphone listener (push-to-talk) using `sounddevice`.

This module provides a simple push-to-talk recorder that writes a temporary
WAV file and returns its path. It uses a module-level default samplerate and
keeps recordings short. The temporary file is caller-managed and removed after
use by the voice controller.
"""

from __future__ import annotations

import tempfile
import time
import os
import threading
from typing import Optional

_SD_AVAILABLE = True
try:
	import sounddevice as sd
	import numpy as np
	import soundfile as sf
except Exception:  # pragma: no cover - optional dependency
	sd = None
	np = None
	sf = None
	_SD_AVAILABLE = False


def is_available() -> bool:
	return _SD_AVAILABLE


def listen_push_to_talk(samplerate: int = 16000, channels: int = 1, dtype="int16") -> Optional[str]:
	"""Record audio in push-to-talk style.

	Usage:
	  - Call this function; it will prompt to press Enter to start and Enter to stop.
	  - Returns path to a temporary WAV file, or None on failure.
	"""
	if not _SD_AVAILABLE:
		raise ImportError("sounddevice/soundfile are required for recording")

	input("Press Enter to start recording...")

	frames = []
	stop_capture = threading.Event()
	capture_error = []
	frame_samples = max(1, int(samplerate * 0.08))

	def capture(stream):
		try:
			while not stop_capture.is_set():
				raw, _overflowed = stream.read(frame_samples)
				frames.append(np.frombuffer(raw, dtype=np.int16).copy())
		except Exception as exc:
			capture_error.append(exc)
			stop_capture.set()

	worker = None
	try:
		from .audio_input import open_jarvis_microphone
		with open_jarvis_microphone(samplerate, frame_samples) as stream:
			worker = threading.Thread(target=capture, args=(stream,), name="jarvis-push-to-talk-capture", daemon=True)
			worker.start()
			try:
				print("Recording... press Enter to stop.")
				input()
			finally:
				stop_capture.set()
				worker.join(timeout=2)

	except Exception as e:
		print(f"Recording failed: {e}")
		return None
	if capture_error:
		print(f"Recording failed: {capture_error[0]}")
		return None

	if not frames:
		return None

	audio = np.concatenate(frames, axis=0)

	# Ensure float32 for soundfile; convert if necessary
	if np.issubdtype(audio.dtype, np.integer):
		# numpy int16 -> float32 in range -1..1
		audio = audio.astype("float32") / 32768.0
	else:
		audio = audio.astype("float32")

	tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
	tmp_name = tmp.name
	tmp.close()

	try:
		sf.write(tmp_name, audio, samplerate)
		return tmp_name
	except Exception as e:
		try:
			os.unlink(tmp_name)
		except Exception:
			pass
		print(f"Failed to write audio file: {e}")
		return None

