from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "training" / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "intent_classifier"
EMBEDDING_CACHE_DIR = PROJECT_ROOT / "models" / "embedding_cache"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "text" not in row or "label" not in row:
                raise ValueError(f"{path}:{line_number}: expected text and label fields")
            text = row["text"]
            label = row["label"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: text must be a non-empty string")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{path}:{line_number}: label must be a non-empty string")
            texts.append(text.strip())
            labels.append(label.strip())
    if not texts:
        raise ValueError(f"{path}: dataset is empty")
    return texts, labels


def encode_texts(model, texts: list[str], batch_size: int = 128) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)


def threshold_metrics(y_true, y_pred, probabilities, threshold: float) -> dict:
    confidence = np.max(probabilities, axis=1)
    accepted = confidence >= threshold
    accepted_count = int(accepted.sum())
    accepted_accuracy = (
        float(accuracy_score(np.asarray(y_true)[accepted], np.asarray(y_pred)[accepted]))
        if accepted_count
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "accepted_accuracy": accepted_accuracy,
        "rejection_rate": float(1.0 - accepted.mean()),
        "accepted_count": accepted_count,
        "error_rate_accepted": float(1.0 - accepted_accuracy) if accepted_count else 1.0,
    }


def choose_threshold(y_true, y_pred, probabilities) -> tuple[float, list[dict]]:
    candidates = [round(value, 2) for value in np.arange(0.80, 0.91, 0.01)]
    rows = [threshold_metrics(y_true, y_pred, probabilities, value) for value in candidates]
    qualifying = [row for row in rows if row["accepted_accuracy"] >= 0.98]
    if qualifying:
        chosen = min(qualifying, key=lambda row: row["threshold"])
    else:
        chosen = max(rows, key=lambda row: (row["accepted_accuracy"], row["accepted_count"]))
    return chosen["threshold"], rows


def evaluate_predictions(y_true, y_pred, probabilities, labels: list[str]) -> dict:
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    errors = []
    pairs = Counter()
    for text_index, (expected, predicted) in enumerate(zip(y_true, y_pred)):
        if expected == predicted:
            continue
        confidence = float(np.max(probabilities[text_index]))
        errors.append(
            {
                "index": text_index,
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence,
            }
        )
        pairs[(expected, predicted)] += 1
    errors.sort(key=lambda item: item["confidence"], reverse=True)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": matrix,
        "errors": errors,
        "confusion_pairs": pairs.most_common(),
    }


def print_evaluation(name: str, metrics: dict, labels: list[str], texts: list[str]) -> None:
    print(f"\n{name} accuracy: {metrics['accuracy']:.6f}")
    print(f"{name} macro F1: {metrics['macro_f1']:.6f}")
    print(f"{name} wrong predictions: {len(metrics['errors'])}")
    print("\nPer-intent precision / recall / F1:")
    for label in labels:
        row = metrics["classification_report"][label]
        print(
            f"{label:24} precision={row['precision']:.4f} "
            f"recall={row['recall']:.4f} f1={row['f1-score']:.4f} support={int(row['support'])}"
        )
    print("\nConfusion matrix label order:", labels)
    print(metrics["confusion_matrix"])
    print("\nTop confusion pairs:")
    for (expected, predicted), count in metrics["confusion_pairs"][:15]:
        print(f"{expected:24} -> {predicted:24} {count}")
    print("\nHighest-confidence wrong predictions:")
    for error in metrics["errors"][:20]:
        print(
            f"{error['confidence']:.4f} expected={error['expected']} "
            f"predicted={error['predicted']} text={texts[error['index']]!r}"
        )
