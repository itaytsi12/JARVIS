from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import sklearn
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from intent_classifier_common import (
    DATA_DIR,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_MODEL_NAME,
    MODEL_DIR,
    choose_threshold,
    encode_texts,
    evaluate_predictions,
    load_jsonl,
    print_evaluation,
    threshold_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--c", type=float, default=4.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    train_texts, train_labels = load_jsonl(DATA_DIR / "train.jsonl")
    validation_texts, validation_labels = load_jsonl(DATA_DIR / "validation.jsonl")
    test_texts, test_labels = load_jsonl(DATA_DIR / "test.jsonl")
    labels = sorted(set(train_labels))
    if set(validation_labels) - set(labels) or set(test_labels) - set(labels):
        raise ValueError("Validation/test contains labels absent from training")
    print(
        f"Loaded train={len(train_texts)} validation={len(validation_texts)} "
        f"test={len(test_texts)} labels={len(labels)}"
    )

    print(f"Loading embedding model on CPU: {EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(
        EMBEDDING_MODEL_NAME, device="cpu", cache_folder=str(EMBEDDING_CACHE_DIR)
    )
    print("Encoding training texts...")
    train_embeddings = encode_texts(embedder, train_texts, args.batch_size)
    print("Encoding validation texts...")
    validation_embeddings = encode_texts(embedder, validation_texts, args.batch_size)

    classifier = LogisticRegression(
        C=args.c,
        max_iter=args.max_iter,
        solver="lbfgs",
        random_state=42,
        n_jobs=None,
    )
    print(f"Training LogisticRegression(C={args.c}, max_iter={args.max_iter})...")
    classifier.fit(train_embeddings, train_labels)

    validation_probabilities = classifier.predict_proba(validation_embeddings)
    validation_predictions = classifier.classes_[validation_probabilities.argmax(axis=1)]
    metrics = evaluate_predictions(
        validation_labels, validation_predictions, validation_probabilities, labels
    )
    print_evaluation("Validation", metrics, labels, validation_texts)
    threshold, threshold_rows = choose_threshold(
        validation_labels, validation_predictions, validation_probabilities
    )
    print("\nValidation confidence threshold sweep:")
    for row in threshold_rows:
        print(
            f"threshold={row['threshold']:.2f} accepted_accuracy={row['accepted_accuracy']:.4f} "
            f"rejection_rate={row['rejection_rate']:.4f}"
        )
    chosen = threshold_metrics(
        validation_labels, validation_predictions, validation_probabilities, threshold
    )
    print(f"Chosen threshold: {threshold:.2f} ({chosen})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_DIR / "classifier.joblib", compress=3)
    config = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "classifier_type": "sklearn.linear_model.LogisticRegression",
        "labels": labels,
        "normalize_embeddings": True,
        "confidence_threshold": threshold,
        "validation_accuracy": metrics["accuracy"],
        "validation_macro_f1": metrics["macro_f1"],
        "test_accuracy": None,
        "train_size": len(train_texts),
        "validation_size": len(validation_texts),
        "test_size": len(test_texts),
        "logistic_regression": {"C": args.c, "max_iter": args.max_iter, "solver": "lbfgs"},
        "scikit_learn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (MODEL_DIR / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved classifier: {MODEL_DIR / 'classifier.joblib'}")
    print(f"Saved metadata: {MODEL_DIR / 'config.json'}")


if __name__ == "__main__":
    main()
