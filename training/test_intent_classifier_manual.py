from __future__ import annotations

import json

import joblib
from sentence_transformers import SentenceTransformer

from intent_classifier_common import EMBEDDING_CACHE_DIR, MODEL_DIR


CASES = [
    ("Open YouTube", "open_website"),
    ("Search YouTube for Jude Law", "youtube_search"),
    ("I want to watch Jude Law on YouTube", "youtube_search"),
    ("Open Notepad", "open_application"),
    ("Could you launch Discord for me", "open_application"),
    ("Close Spotify", "close_application"),
    ("Search the web for Python decorators", "web_search"),
    ("Google neural networks", "web_search"),
    ("Turn the volume down", "volume_down"),
    ("Make it a little louder", "volume_up"),
    ("Mute the sound", "mute_volume"),
    ("Turn the sound back on", "unmute_volume"),
    ("Type hello world", "type_text"),
    ("Press escape", "press_key"),
    ("Press F5", "press_key"),
    ("Press control shift escape", "hotkey"),
    ("Switch to Chrome", "switch_window"),
    ("Minimize this window", "minimize_window"),
    ("Maximize this window", "maximize_window"),
    ("Close this window", "close_window"),
    ("Take a screenshot", "take_screenshot"),
    ("Open resume.pdf", "open_file"),
    ("Open my downloads folder", "open_folder"),
    ("Run ipconfig", "run_command"),
]


def main() -> None:
    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    classifier = joblib.load(MODEL_DIR / "classifier.joblib")
    embedder = SentenceTransformer(
        config["embedding_model"], device="cpu", cache_folder=str(EMBEDDING_CACHE_DIR)
    )
    texts = [text for text, _ in CASES]
    embeddings = embedder.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    probabilities = classifier.predict_proba(embeddings)
    failures = 0
    for (text, expected), row in zip(CASES, probabilities):
        index = int(row.argmax())
        predicted = str(classifier.classes_[index])
        confidence = float(row[index])
        passed = predicted == expected
        failures += not passed
        print(
            f"{'PASS' if passed else 'FAIL'} | confidence={confidence:.4f} | "
            f"expected={expected} | predicted={predicted} | {text}"
        )
    print(f"\nManual tests: {len(CASES) - failures}/{len(CASES)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
