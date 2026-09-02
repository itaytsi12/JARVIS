"""Speech-to-text using local faster-whisper.

This module provides a simple, lazy-loaded wrapper around faster-whisper's
WhisperModel. The model is loaded once on first use and reused for subsequent
transcriptions.

If faster-whisper is not installed, the module raises ImportError when used.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

log = logging.getLogger("jarvis.stt")

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
		from .voice_language import expected_input_languages, get_voice_language, language_name
		voice_language = get_voice_language()
		# The English-optimized ".en" model is only safe when English is the
		# ONLY expected language: those variants cannot transcribe Hebrew at
		# all, whatever `language=` is passed to .transcribe(). Any mode that
		# expects more than one language gets the multilingual model. Only
		# ever loads ONE model per process (the module-level singleton below
		# is keyed by whichever size this resolves to), never both.
		default_model = "base.en" if expected_input_languages(voice_language) == ("en",) else "small"
		# Treat an explicitly blank value like an unset value. `.env.example`
		# intentionally leaves this blank so language-aware defaults apply.
		configured_model = os.getenv("WHISPER_MODEL", "").strip()
		model_size = (model_size or "").strip() or configured_model or default_model

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
	from .voice_language import detect_input_language, expected_input_languages, get_voice_language, stt_language_code

	model = _get_model(model_size)
	voice_language = get_voice_language()
	# faster-whisper's `language=` expects a real ISO code or None (meaning
	# "detect per-segment"). `stt_language_code` is the SINGLE shared rule
	# both STT providers use: force a code only when the configuration
	# expects exactly one language. This is what stops the Whisper FALLBACK
	# from being stricter than the ElevenLabs primary -- confirmed live,
	# where `VOICE_LANGUAGE=he` forced `language="he"` here and turned
	# English commands into Hebrew transliterations no route could match.
	# ("auto" is this project's own mode name, never a Whisper code, and
	# is never passed through literally.)
	whisper_language = stt_language_code(voice_language)
	# The English initial_prompt is an English-phrasing hint. It is only
	# safe when English is the sole expected language; in a bilingual mode
	# it would bias output back toward English.
	initial_prompt = _ENGLISH_INITIAL_PROMPT if whisper_language == "en" else None
	log.info("[STT] whisper_fallback voice_language_mode=%s forced_language=%s", voice_language, whisper_language or "auto-detect")

	segments, info = model.transcribe(
		path,
		language=whisper_language,
		task="transcribe",
		initial_prompt=initial_prompt,
		beam_size=max(1, int(os.getenv("WHISPER_BEAM_SIZE", "1") or "1")),
		temperature=0.0,
		vad_filter=True,
		condition_on_previous_text=False,
	)

	texts = []

	for segment in segments:
		text = segment.text.strip()

		if text:
			texts.append(text)

	text = " ".join(texts).strip()

	# Whisper's own unconstrained auto-detect (`language=None`, used
	# whenever more than one language is expected) picks from its full
	# ~100-language vocabulary, not just the languages JARVIS is actually
	# configured to recognize -- confirmed live: plain English commands
	# committed as Dutch, and "stop" committed as Russian. JARVIS has no
	# route for any language outside `expected_input_languages`, so
	# accepting that text is never useful even when the detection happens
	# to be technically correct. Re-transcribing once, constrained to
	# whichever of the configured candidates the mistranscribed text looks
	# more like (script-based, via the same Hebrew-block check
	# `voice_language.py` already uses -- no cloud LLM call), is cheap and
	# strictly better than keeping text no local route can ever match. Only
	# applies when the language was genuinely unconstrained (`whisper_language
	# is None`); a forced single-language mode is never second-guessed.
	detected_language = getattr(info, "language", None)
	expected = expected_input_languages(voice_language)
	if whisper_language is None and text and isinstance(detected_language, str) and detected_language and detected_language not in expected:
		fallback_language = "he" if detect_input_language(text) == "he" and "he" in expected else "en"
		log.info(
			"[STT] whisper_fallback implausible_detected_language=%s (expected one of %s); re-transcribing constrained to %s",
			detected_language, expected, fallback_language,
		)
		retry_segments, _retry_info = model.transcribe(
			path,
			language=fallback_language,
			task="transcribe",
			initial_prompt=_ENGLISH_INITIAL_PROMPT if fallback_language == "en" else None,
			beam_size=max(1, int(os.getenv("WHISPER_BEAM_SIZE", "1") or "1")),
			temperature=0.0,
			vad_filter=True,
			condition_on_previous_text=False,
		)
		retried_text = " ".join(segment.text.strip() for segment in retry_segments if segment.text.strip()).strip()
		if retried_text:
			text = retried_text

	return text


def is_available() -> bool:
	return _AVAILABLE

