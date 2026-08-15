from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import psutil
from sentence_transformers import SentenceTransformer

from intent_classifier_common import (
    DATA_DIR,
    EMBEDDING_CACHE_DIR,
    MODEL_DIR,
    encode_texts,
    evaluate_predictions,
    load_jsonl,
    print_evaluation,
    threshold_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--update-test-metadata", action="store_true")
    args = parser.parse_args()

    config_path = MODEL_DIR / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    classifier = joblib.load(MODEL_DIR / "classifier.joblib")
    embedder = SentenceTransformer(
        config["embedding_model"], device="cpu", cache_folder=str(EMBEDDING_CACHE_DIR)
    )
    texts, true_labels = load_jsonl(DATA_DIR / f"{args.split}.jsonl")
    print(f"Loaded {args.split}: {len(texts)}")
    embeddings = encode_texts(embedder, texts, args.batch_size)
    probabilities = classifier.predict_proba(embeddings)
    predictions = classifier.classes_[probabilities.argmax(axis=1)]
    labels = config["labels"]
    metrics = evaluate_predictions(true_labels, predictions, probabilities, labels)
    print_evaluation(args.split.title(), metrics, labels, texts)
    accepted = threshold_metrics(
        true_labels, predictions, probabilities, config["confidence_threshold"]
    )
    print("\nConfidence-threshold metrics:", accepted)

    process = psutil.Process()
    timings = []
    for text in texts[:100]:
        start = time.perf_counter()
        embedding = embedder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        after_embedding = time.perf_counter()
        classifier.predict_proba(embedding)
        end = time.perf_counter()
        timings.append((after_embedding - start, end - after_embedding, end - start))
    means = np.mean(timings, axis=0)
    print(
        f"\nAverage single-command embedding={means[0]*1000:.3f} ms "
        f"classifier={means[1]*1000:.3f} ms total={means[2]*1000:.3f} ms"
    )
    print(f"Approximate process RSS: {process.memory_info().rss / (1024**2):.1f} MiB")

    if args.split == "test" and args.update_test_metadata:
        config["test_accuracy"] = metrics["accuracy"]
        config["test_macro_f1"] = metrics["macro_f1"]
        config["test_threshold_metrics"] = accepted
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
