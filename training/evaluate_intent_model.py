import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from setfit import SetFitModel


MODEL_DIR = "models/intent"
TEST_FILE = "training/data/test.jsonl"

MAX_ERRORS_TO_SHOW = 50


def main():
    print("Loading model...")
    model = SetFitModel.from_pretrained(MODEL_DIR)

    print("Loading test dataset...")
    dataset = load_dataset(
        "json",
        data_files={"test": TEST_FILE},
    )["test"]

    texts = dataset["text"]
    true_labels = dataset["label"]

    print(f"Test samples: {len(texts)}")
    print("Running predictions...")

    predicted_labels = model.predict(texts)

    # SetFit may return numpy arrays / tensors depending on version.
    if hasattr(predicted_labels, "tolist"):
        predicted_labels = predicted_labels.tolist()

    predicted_labels = [str(x) for x in predicted_labels]

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    print("\n" + "=" * 70)
    print(f"TEST ACCURACY: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 70)

    labels = sorted(set(true_labels) | set(predicted_labels))

    print("\nCLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=labels,
            digits=4,
            zero_division=0,
        )
    )

    print("\nCONFUSION MATRIX")
    print("-" * 70)

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=labels,
    )

    # Header
    print("Labels:")
    for index, label in enumerate(labels):
        print(f"{index}: {label}")

    print("\nMatrix:")
    print(cm)

    # ------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------

    print("\nGetting prediction probabilities...")

    probabilities = None

    try:
        probabilities = model.predict_proba(texts)

        if hasattr(probabilities, "cpu"):
            probabilities = probabilities.cpu().numpy()
        elif hasattr(probabilities, "numpy"):
            probabilities = probabilities.numpy()
        else:
            probabilities = np.asarray(probabilities)

    except Exception as exc:
        print(f"Could not get probabilities: {exc}")

    # ------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------

    errors = []

    for i, (text, expected, predicted) in enumerate(
        zip(texts, true_labels, predicted_labels)
    ):
        if expected == predicted:
            continue

        confidence = None

        if probabilities is not None:
            try:
                confidence = float(np.max(probabilities[i]))
            except Exception:
                pass

        errors.append(
            {
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence,
            }
        )

    print("\n" + "=" * 70)
    print(f"TOTAL ERRORS: {len(errors)} / {len(texts)}")
    print("=" * 70)

    # Sort highest-confidence mistakes first.
    errors.sort(
        key=lambda item: (
            item["confidence"]
            if item["confidence"] is not None
            else -1
        ),
        reverse=True,
    )

    print(
        f"\nShowing up to {MAX_ERRORS_TO_SHOW} "
        "highest-confidence mistakes:\n"
    )

    for number, error in enumerate(
        errors[:MAX_ERRORS_TO_SHOW],
        start=1,
    ):
        print(f"[{number}]")
        print(f'Text:       "{error["text"]}"')
        print(f'Expected:   {error["expected"]}')
        print(f'Predicted:  {error["predicted"]}')

        if error["confidence"] is not None:
            print(f'Confidence: {error["confidence"]:.4f}')

        print("-" * 70)

    # ------------------------------------------------------------
    # Save errors for later dataset improvement
    # ------------------------------------------------------------

    output_path = Path("training/data/test_errors.jsonl")

    with output_path.open("w", encoding="utf-8") as file:
        for error in errors:
            file.write(
                json.dumps(
                    error,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"\nErrors saved to: {output_path}")

    # ------------------------------------------------------------
    # Per confusion pair
    # ------------------------------------------------------------

    confusion_pairs = {}

    for error in errors:
        pair = (
            error["expected"],
            error["predicted"],
        )

        confusion_pairs[pair] = (
            confusion_pairs.get(pair, 0) + 1
        )

    print("\nMOST COMMON CONFUSIONS")
    print("-" * 70)

    sorted_pairs = sorted(
        confusion_pairs.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for (expected, predicted), count in sorted_pairs[:20]:
        print(
            f"{expected:25} -> "
            f"{predicted:25} : {count}"
        )


if __name__ == "__main__":
    main()