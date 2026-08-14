"""TTS wrapper.

Attempt to use `kokoro` TTS if available; otherwise fall back to `pyttsx3`.
This module keeps a single model/engine instance for reuse and exposes
`speak(text)`.
"""

from __future__ import annotations
import importlib
import sys
from typing import Optional

# Try Kokoro (best-effort). If unavailable, fall back to pyttsx3.
_kokoro = None
_pyttsx3 = None
_engine = None
_kokoro_available = False
_pyttsx3_available = False

try:
	_kokoro = importlib.import_module("kokoro")
	_kokoro_available = True
except Exception:
	_kokoro = None

if not _kokoro_available:
	try:
		_pyttsx3 = importlib.import_module("pyttsx3")
		_pyttsx3_available = True
	except Exception:
		_pyttsx3 = None


def _init_pyttsx3():
	global _engine
	if _engine is None and _pyttsx3_available:
		e = _pyttsx3.init()
		try:
			e.setProperty('rate', 160)
		except Exception:
			pass
		_engine = e
	return _engine


class KokoroWrapper:
	"""Thin wrapper around kokoro if present.

	This is intentionally defensive: kokoro may not be installed in the
	user's environment. If installed, users should ensure the model files
	are downloaded beforehand (see README). The wrapper loads the model
	once and reuses it.
	"""
	def __init__(self):
		self.model = None

	def _ensure_model(self):
		if self.model is None:
			# kokoro API is not standardized here; attempt common patterns.
			# If kokoro is installed and provides a `TTS` class or `load_model` function,
			# try to use them. This is defensive; if it fails, we raise ImportError
			# and fall back to pyttsx3.
			if hasattr(_kokoro, 'load_model'):
				self.model = _kokoro.load_model()
			elif hasattr(_kokoro, 'TTS'):
				self.model = _kokoro.TTS()
			else:
				raise ImportError("Kokoro appears installed but no known loader found")

	def speak(self, text: str) -> None:
		self._ensure_model()
		# Try common synthesize API
		if hasattr(self.model, 'synthesize'):
			audio = self.model.synthesize(text)
			# If audio is a numpy array or bytes, attempt playback via simple player
			try:
				import sounddevice as sd
				import numpy as np
				if isinstance(audio, bytes):
					# unknown bytes format — fallback to writing temp wav
					import tempfile, wave
					with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
						f.write(audio)
						tmp = f.name
					import subprocess
					if sys.platform.startswith('win'):
						subprocess.Popen(['powershell', 'Start-Process', tmp])
					else:
						subprocess.Popen(['xdg-open', tmp])
				else:
					sd.play(audio, samplerate=22050)
					sd.wait()
				return
			except Exception:
				pass
		# If model has speak/play API
		if hasattr(self.model, 'speak'):
			return self.model.speak(text)
		raise RuntimeError("Kokoro model loaded but no known speak API")


_kokoro_wrapper: Optional[KokoroWrapper] = None


def speak(text: str, max_speech_chars: int = 300) -> None:
	"""Speak `text` locally. Prefer Kokoro if available, otherwise pyttsx3.

	For long texts, speak a short summary and leave full text for the terminal.
	"""
	if not text:
		return

	if len(text) > max_speech_chars:
		short = text[:max_speech_chars].rsplit(' ', 1)[0]
		print(f"Jarvis (long): {text}")
		to_speak = short
	else:
		to_speak = text

	# Try Kokoro
	if _kokoro_available:
		global _kokoro_wrapper
		if _kokoro_wrapper is None:
			_kokoro_wrapper = KokoroWrapper()
		try:
			_kokoro_wrapper.speak(to_speak)
			return
		except Exception as e:
			print(f"Kokoro TTS failed: {e} -- falling back to pyttsx3 if available")

	# Fallback to pyttsx3
	if _pyttsx3_available:
		engine = _init_pyttsx3()
		if engine is None:
			print(f"(TTS unavailable) Jarvis: {text}")
			return
		engine.say(to_speak)
		engine.runAndWait()
		return

	# Last resort: print
	print(f"(TTS unavailable) Jarvis: {text}")


