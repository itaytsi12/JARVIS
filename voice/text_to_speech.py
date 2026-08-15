"""Configurable JARVIS TTS providers with deterministic fallback order."""

from __future__ import annotations
import importlib
import os
from typing import Optional

_pyttsx3 = None
_engine = None
_pyttsx3_available = False
_chatterbox_provider = None
_chatterbox_available = False
_openai_provider = None

try:
	_openai_provider = importlib.import_module('voice.tts.openai_tts')
except Exception:
	_openai_provider = None

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
	if value not in {"auto", "openai", "chatterbox", "pyttsx3"}:
		print(f"Unknown TTS_PROVIDER={value!r}; using auto")
		return "auto"
	return value


def _openai_is_available() -> bool:
	return bool(
		_openai_provider is not None
		and _openai_provider.is_available()
	)


def _provider_order() -> list[str]:
	configured = _configured_provider()
	if configured == "openai":
		return ["openai", "pyttsx3"]
	if configured == "chatterbox":
		return ["chatterbox", "pyttsx3"]
	if configured == "pyttsx3":
		return ["pyttsx3"]

	providers = []
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
		if provider == "openai" and _openai_is_available():
			return provider
		if provider == "chatterbox" and _chatterbox_available:
			return provider
		if provider == "pyttsx3" and _pyttsx3_available:
			return provider
	return "pyttsx3"


def _provider_label(provider: str) -> str:
	return {
		"openai": "OpenAI cedar",
		"chatterbox": "Chatterbox local",
		"pyttsx3": "pyttsx3 fallback",
	}[provider]


def start_background() -> None:
	"""Warm local TTS when appropriate and announce the selected provider."""
	configured = _configured_provider()
	if configured in {"auto", "chatterbox"} and _chatterbox_available and _chatterbox_provider is not None:
		_chatterbox_provider.start_service_background()
	print(f"TTS provider: {_provider_label(_active_provider())}")


def speak(text: str, max_speech_chars: int = 300, lang: Optional[str] = None) -> None:
	"""Speak locally with Chatterbox, using pyttsx3 only on failure.

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

	for provider in _provider_order():
		if provider == "chatterbox" and _chatterbox_available and _chatterbox_provider is not None:
			try:
				_chatterbox_provider.speak(to_speak, lang=lang or 'en')
				return
			except Exception as e:
				print(f"Chatterbox TTS failed: {e} -- falling back to pyttsx3")
		elif provider == "openai" and _openai_is_available():
			try:
				_openai_provider.speak(to_speak)
				return
			except Exception as e:
				print(f"OpenAI TTS failed: {e} -- trying local speech")
		elif provider == "pyttsx3" and _pyttsx3_available:
			engine = _init_pyttsx3()
			if engine is not None:
				engine.say(to_speak)
				engine.runAndWait()
				return

	# Last resort: print
	print(f"(TTS unavailable) Jarvis: {text}")


