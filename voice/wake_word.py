"""Local streaming wake-word detection using openWakeWord's Jarvis model."""
from __future__ import annotations

import os
from config.settings import env_float, env_int
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "wake_word"


class WakeWordUnavailable(RuntimeError):
    pass


class OpenWakeWordEngine:
    sample_rate = 16000
    frame_samples = 1280

    def __init__(self, threshold: float | None = None, model_dir: Path | None = None):
        self.threshold = threshold if threshold is not None else env_float("WAKE_WORD_THRESHOLD", 0.5)
        self.model_dir = Path(model_dir or MODEL_DIR)
        self.model = None
        self.model_name = "hey_jarvis_v0.1"

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise WakeWordUnavailable("openwakeword is not installed") from exc
        wake_model = self.model_dir / "hey_jarvis_v0.1.onnx"
        mel_model = self.model_dir / "melspectrogram.onnx"
        embedding_model = self.model_dir / "embedding_model.onnx"
        missing = [path.name for path in (wake_model, mel_model, embedding_model) if not path.is_file()]
        if missing:
            raise WakeWordUnavailable(f"Missing wake model files: {', '.join(missing)}")
        self.model = Model(
            wakeword_models=[str(wake_model)],
            inference_framework="onnx",
            melspec_model_path=str(mel_model),
            embedding_model_path=str(embedding_model),
        )

    def process(self, pcm_int16) -> tuple[bool, float]:
        self.load()
        scores = self.model.predict(pcm_int16)
        score = float(scores.get(self.model_name, max(scores.values(), default=0.0)))
        return score >= self.threshold, score

    def reset(self) -> None:
        if self.model is not None:
            self.model.reset()
