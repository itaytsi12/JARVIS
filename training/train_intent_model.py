from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments

MODEL_NAME = "intfloat/multilingual-e5-small"

DATA_FILES = {
    "train": "training/data/train.jsonl",
    "validation": "training/data/validation.jsonl",
}

OUTPUT_DIR = "models/intent"


def main():
    print("Loading dataset...")

    dataset = load_dataset(
        "json",
        data_files=DATA_FILES
    )

    print(dataset)

    print("Loading base model...")

    model = SetFitModel.from_pretrained(
        MODEL_NAME,
        labels=sorted(
            list(set(dataset["train"]["label"]))
        )
    )

    args = TrainingArguments(
        batch_size=8,
        num_epochs=1,
        num_iterations=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        output_dir=OUTPUT_DIR,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        metric="accuracy",
        column_mapping={
            "text": "text",
            "label": "label"
        }
    )

    print("Starting training...")

    trainer.train()

    print("Evaluating...")

    metrics = trainer.evaluate()
    print(metrics)

    print("Saving model...")

    model.save_pretrained(OUTPUT_DIR)

    print(f"Done. Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()