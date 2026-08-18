"""Configurable JARVIS TTS providers with deterministic fallback order."""

from __future__ import annotations
import importlib
import os
import threading
import time
from typing import Optional
from brain.resource_locks import acquire_action_resource

_pyttsx3 = None
_engine = None
_pyttsx3_available = False
_chatterbox_provider = None
_chatterbox_available = False
_openai_provider = None
_elevenlabs_provider = None
_elevenlabs_available = False
_SPEAKER_LOCK=threading.Lock()

try:
	_openai_provider = importlib.import_module('voice.tts.openai_tts')
except Exception:
	_openai_provider = None

try:
	_elevenlabs_provider = importlib.import_module('voice.tts.elevenlabs_tts')
	_elevenlabs_available = True
except Exception:
	_elevenlabs_provider = None

# Try chatterbox provider module under voice/tts
try:
	_chb_mod = importlib.import_module('voice.tts.chatterbox_tts')
	_chatterbox_provider = _chb_mod
	_chatterbox_available = True
except Exception:
	_chatterbox_provider = None

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


def _configured_provider() -> str:
	value = os.getenv("TTS_PROVIDER", "auto").strip().lower()
	if value not in {"auto", "openai", "chatterbox", "pyttsx3", "elevenlabs"}:
		print(f"Unknown TTS_PROVIDER={value!r}; using auto")
		return "auto"
	return value


def _openai_is_available() -> bool:
	return bool(
		_openai_provider is not None
		and _openai_provider.is_available()
	)


def _elevenlabs_is_available() -> bool:
	return bool(_elevenlabs_available and _elevenlabs_provider is not None and _elevenlabs_provider.is_available())


def _provider_order() -> list[str]:
	configured = _configured_provider()
	if configured == "elevenlabs":
		return ["elevenlabs", "openai", "chatterbox", "pyttsx3"]
	if configured == "openai":
		return ["openai", "pyttsx3"]
	if configured == "chatterbox":
		return ["chatterbox", "pyttsx3"]
	if configured == "pyttsx3":
		return ["pyttsx3"]

	providers = []
	if _elevenlabs_is_available():
		providers.append("elevenlabs")
	if (
		_chatterbox_available
		and _chatterbox_provider is not None
		and _chatterbox_provider.is_low_latency_ready()
	):
		providers.append("chatterbox")
	if _openai_is_available():
		providers.append("openai")
	providers.append("pyttsx3")
	return providers


def _active_provider() -> str:
	for provider in _provider_order():
		if provider == "elevenlabs" and _elevenlabs_is_available():
			return provider
		if provider == "openai" and _openai_is_available():
			return provider
		if provider == "chatterbox" and _chatterbox_available:
			return provider
		if provider == "pyttsx3" and _pyttsx3_available:
			return provider
	return "pyttsx3"


def _provider_label(provider: str) -> str:
	return {
		"elevenlabs": "ElevenLabs streaming",
		"openai": "OpenAI cedar",
		"chatterbox": "Chatterbox local",
		"pyttsx3": "pyttsx3 fallback",
	}[provider]


def start_background() -> None:
	"""Warm local TTS when appropriate and announce the selected provider."""
	configured = _configured_provider()
	warm_auto=os.getenv("JARVIS_CHATTERBOX_WARM_AUTO","false").lower() in {"1","true","yes","on"}
	if (configured == "chatterbox" or configured=="auto" and warm_auto) and _chatterbox_available and _chatterbox_provider is not None:
		_chatterbox_provider.start_service_background()
	print(f"TTS provider: {_provider_label(_active_provider())}")


def speak(text: str, max_speech_chars: int = 300, lang: Optional[str] = None, on_first_audio_chunk=None) -> dict:
	started=time.perf_counter()
	with acquire_action_resource("speak_response") as resource_info:
		with _SPEAKER_LOCK:
			wait_ms=(time.perf_counter()-started)*1000
			result=_speak_unlocked(text,max_speech_chars,lang,on_first_audio_chunk)
			result["resource"]="speaker";result["resource_wait_ms"]=round(wait_ms,3);result["cross_process_lock"]=bool(resource_info.get("cross_process"))
			return result


def _speak_unlocked(text: str, max_speech_chars: int = 300, lang: Optional[str] = None, on_first_audio_chunk=None) -> dict:
	"""Speak locally with Chatterbox, using pyttsx3 only on failure.

	For long texts, speak a short summary and leave full text for the terminal.
	"""
	if not text:
		return {"success":False,"provider":None,"error":"empty_text","spoken_chars":0}

	if len(text) > max_speech_chars:
		short = text[:max_speech_chars].rsplit(' ', 1)[0]
		print(f"Jarvis (long): {text}")
		to_speak = short
	else:
		to_speak = text

	attempted=[]
	for provider in _provider_order():
		if provider == "elevenlabs" and _elevenlabs_is_available():
			attempted.append(provider)
			try:
				_elevenlabs_provider.speak(to_speak, lang=lang or 'en', on_first_audio_chunk=on_first_audio_chunk)
				return {"success":True,"provider":"elevenlabs","spoken_chars":len(to_speak),"attempted_providers":attempted,"fallback_from":attempted[:-1]}
			except Exception as e:
				print(f"ElevenLabs TTS failed: {e} -- falling back")
		elif provider == "chatterbox" and _chatterbox_available and _chatterbox_provider is not None:
			attempted.append(provider)
			try:
				_chatterbox_provider.speak(to_speak, lang=lang or 'en')
				return {"success":True,"provider":"chatterbox","spoken_chars":len(to_speak),"attempted_providers":attempted,"fallback_from":attempted[:-1]}
			except Exception as e:
				print(f"Chatterbox TTS failed: {e} -- falling back to pyttsx3")
		elif provider == "openai" and _openai_is_available():
			attempted.append(provider)
			try:
				_openai_provider.speak(to_speak)
				return {"success":True,"provider":"openai","spoken_chars":len(to_speak),"attempted_providers":attempted,"fallback_from":attempted[:-1]}
			except Exception as e:
				print(f"OpenAI TTS failed: {e} -- trying local speech")
		elif provider == "pyttsx3" and _pyttsx3_available:
			attempted.append(provider)
			engine = _init_pyttsx3()
			if engine is not None:
				engine.say(to_speak)
				engine.runAndWait()
				return {"success":True,"provider":"pyttsx3","spoken_chars":len(to_speak),"attempted_providers":attempted,"fallback_from":attempted[:-1]}

	# Last resort: print
	print(f"(TTS unavailable) Jarvis: {text}")
	return {"success":False,"provider":None,"error":"tts_unavailable","spoken_chars":0,"attempted_providers":attempted,"fallback_from":attempted}


def stop() -> None:
	"""Stop current playback without disabling future speech."""
	if _elevenlabs_provider is not None and hasattr(_elevenlabs_provider,"stop"):
		_elevenlabs_provider.stop()
	if _openai_provider is not None and hasattr(_openai_provider,"stop"):
		_openai_provider.stop()
	if _chatterbox_provider is not None and hasattr(_chatterbox_provider,"stop"):
		_chatterbox_provider.stop()
	if _engine is not None:
		try:_engine.stop()
		except Exception:pass


