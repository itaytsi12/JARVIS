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
	recording = True

	def callback(indata, frames_count, time_info, status):
		# copy to avoid referencing same buffer
		frames.append(indata.copy())

	try:
		with sd.InputStream(samplerate=samplerate, channels=channels, dtype=dtype, callback=callback):
			print("Recording... press Enter to stop.")
			input()
			# stream context manager ends here

	except Exception as e:
		print(f"Recording failed: {e}")
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

