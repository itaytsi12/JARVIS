"""Speech-to-text using local faster-whisper.

This module provides a simple, lazy-loaded wrapper around faster-whisper's
WhisperModel. The model is loaded once on first use and reused for subsequent
transcriptions.

If faster-whisper is not installed, the module raises ImportError when used.
"""

from __future__ import annotations

import os
from typing import Optional

_MODEL = None
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
		# Default to a small CPU-friendly model for fast first-run and to avoid
		# attempting GPU/CUDA when the system lacks the required DLLs.
		model_size = model_size or os.getenv("WHISPER_MODEL", "tiny.en")

		# Prefer CPU by default to avoid cublas/cuda DLL errors on Windows
		# machines that don't have CUDA configured. Users can still override
		# by setting WHISPER_COMPUTE_TYPE or WHISPER_MODEL environment variables.
		compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

		try:
			_MODEL = WhisperModel(model_size, device="cpu", compute_type=compute_type)
		except Exception as e:
			# If CPU init fails, try auto as a last resort and provide an
			# informative error if both fail.
			try:
				_MODEL = WhisperModel(model_size, device="auto", compute_type=compute_type)
			except Exception as e2:
				raise RuntimeError(
					"Failed to initialize faster-whisper model. "
					"CPU attempt error: %s. Auto attempt error: %s. "
					"Set WHISPER_MODEL to a smaller model (e.g. tiny.en) or install required dependencies." % (e, e2)
				)

	return _MODEL


def transcribe_audio(path: str, model_size: Optional[str] = None) -> str:
	"""Transcribe an audio file at `path` to text using a local model.

	Returns the recognized text (empty string on no speech).
	"""
	model = _get_model(model_size)

	# faster-whisper returns segments; join them into a single string.
	segments, info = model.transcribe(path, beam_size=5)

	texts = []
	for segment in segments:
		texts.append(segment.text)

	return "".join(texts).strip()


def is_available() -> bool:
	return _AVAILABLE

