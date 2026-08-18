"""Speech-to-text using local faster-whisper.

This module provides a simple, lazy-loaded wrapper around faster-whisper's
WhisperModel. The model is loaded once on first use and reused for subsequent
transcriptions.

If faster-whisper is not installed, the module raises ImportError when used.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

_MODEL = None
_MODEL_LOCK = threading.Lock()
_AVAILABLE = True

try:
	from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional dependency
	WhisperModel = None
	_AVAILABLE = False


def _get_model(model_size: Optional[str] = None):
	global _MODEL

	if not _AVAILABLE:
		raise ImportError("faster-whisper is not installed")

	if _MODEL is None:
		with _MODEL_LOCK:
			if _MODEL is not None:
				return _MODEL
			_MODEL = _create_model(model_size)

	return _MODEL


def _create_model(model_size: Optional[str] = None):
		from .voice_language import get_voice_language, language_name
		voice_language = get_voice_language()
		# English keeps the historical English-optimized small model. Hebrew
		# (or any future non-English mode) MUST use a multilingual model --
		# the ".en" variants are English-only and cannot transcribe Hebrew at
		# all, no matter what `language=` is passed to .transcribe(). Only
		# ever loads ONE model per process (the module-level singleton below
		# is keyed by whichever size this resolves to), never both.
		default_model = "small.en" if voice_language == "en" else "small"
		model_size = model_size or os.getenv("WHISPER_MODEL", default_model)

		# Prefer CPU by default to avoid cublas/cuda DLL errors on Windows
		# machines that don't have CUDA configured. Users can still override
		# by setting WHISPER_COMPUTE_TYPE or WHISPER_MODEL environment variables.
		compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

		model = None
		try:
			# Avoid a network/cache metadata check on the common installed-model path.
			# A normal lookup remains the fallback so first-time setup still works.
			model = WhisperModel(model_size, device="cpu", compute_type=compute_type, local_files_only=True)
		except Exception as e:
			try:
				model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
			except Exception as e2:
				try:
					model = WhisperModel(model_size, device="auto", compute_type=compute_type)
				except Exception as e3:
					raise RuntimeError(
						"Failed to initialize faster-whisper model. "
						"Local CPU error: %s. CPU error: %s. Auto error: %s. "
						"Set WHISPER_MODEL to a smaller model (e.g. tiny.en) or install required dependencies." % (e, e2, e3)
					)

		# Print a concise startup message once when the model is first created.
		if model is not None:
			print(f"[STT] Whisper model: {model_size}")
			print(f"[STT] Language: {language_name(voice_language)}")

		return model


def warm_up() -> None:
	"""Load the configured model once without transcribing audio."""
	_get_model()


_ENGLISH_INITIAL_PROMPT = "Jarvis commands may include: Open Notepad. Type exactly what you last told me. Type exactly what you just said. Then open YouTube."


def transcribe_audio(path: str, model_size: Optional[str] = None) -> str:
	"""Transcribe an audio file at `path` to text using a local model.

	Returns the recognized text (empty string on no speech).
	"""
	from .voice_language import get_voice_language

	model = _get_model(model_size)
	voice_language = get_voice_language()
	# The English initial_prompt is an English-phrasing hint -- steering
	# Whisper toward it in Hebrew/auto mode would bias output back toward
	# English, exactly what a non-English-only mode must not do.
	initial_prompt = _ENGLISH_INITIAL_PROMPT if voice_language == "en" else None
	# faster-whisper's `language=` expects a real ISO code or None (meaning
	# "detect per-segment") -- "auto" is this project's own mode name, not
	# a real Whisper language code, so it must never be passed through
	# literally (that would raise/misbehave inside faster-whisper).
	whisper_language = None if voice_language == "auto" else voice_language

	segments, info = model.transcribe(
		path,
		language=whisper_language,
		task="transcribe",
		initial_prompt=initial_prompt,
		beam_size=5,
		temperature=0.0,
		vad_filter=True,
		condition_on_previous_text=False,
	)

	texts = []

	for segment in segments:
		text = segment.text.strip()

		if text:
			texts.append(text)

	return " ".join(texts).strip()


def is_available() -> bool:
	return _AVAILABLE

