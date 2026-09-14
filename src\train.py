"""Train and save a three-class financial sentiment model."""

from __future__ import annotations

import argparse
import random

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .settings import settings


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {value: key.capitalize() for key, value in LABEL_TO_ID.items()}


class FinancialDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {name: tensor.squeeze(0) for name, tensor in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def load_data(path) -> tuple[list[str], list[int]]:
    try:
        frame = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="latin-1")
    required = {"Sentence", "Sentiment"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")

    # Preserve repeated rows to reproduce the dataset treatment used by the
    # published experiment. A new benchmark should use grouped, deduplicated
    # splits, but that would no longer be a strict paper reproduction.
    frame = frame[["Sentence", "Sentiment"]].dropna()
    frame["Sentence"] = frame["Sentence"].astype(str).str.strip()
    frame["Sentiment"] = frame["Sentiment"].astype(str).str.strip().str.lower()
    frame = frame[frame["Sentence"].ne("") & frame["Sentiment"].isin(LABEL_TO_ID)]
    if frame.empty:
        raise ValueError("No valid training rows were found.")

    return frame["Sentence"].tolist(), frame["Sentiment"].map(LABEL_TO_ID).tolist()


def evaluate(model, dataloader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch in dataloader:
            batch = {name: value.to(device) for name, value in batch.items()}
            predictions = model(**batch).logits.argmax(dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return correct / total if total else 0.0


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    texts, labels = load_data(settings.data_file)
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id={label: index for index, label in ID_TO_LABEL.items()},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_data = FinancialDataset(train_texts, train_labels, tokenizer, settings.max_length)
    validation_data = FinancialDataset(
        validation_texts, validation_labels, tokenizer, settings.max_length
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            total_loss += output.loss.item()

        average_loss = total_loss / len(train_loader)
        accuracy = evaluate(model, validation_loader, device)
        print(f"Epoch {epoch}/{args.epochs} - loss={average_loss:.4f} - val_accuracy={accuracy:.4f}")

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(settings.model_dir)
    tokenizer.save_pretrained(settings.model_dir)
    print(f"Saved model to {settings.model_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
