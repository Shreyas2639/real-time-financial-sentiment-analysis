"""Fine-tune the report-specified models using the published configuration."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .model_registry import MODEL_BY_KEY, MODEL_SPECS, is_model_ready
from .settings import settings
from .train import FinancialDataset, ID_TO_LABEL, LABEL_TO_ID, load_data


def evaluate(model, dataloader, device) -> tuple[dict[str, object], np.ndarray]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    with torch.inference_mode():
        for batch in dataloader:
            batch = {name: value.to(device) for name, value in batch.items()}
            predicted = model(**batch).logits.argmax(dim=-1)
            predictions.extend(predicted.cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    report = classification_report(
        labels,
        predictions,
        labels=[0, 1, 2],
        target_names=[ID_TO_LABEL[index] for index in range(3)],
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "classification_report": report,
    }
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    return metrics, matrix


def train_one(spec, args, train_texts, validation_texts, train_labels, validation_labels):
    if is_model_ready(spec) and not args.force:
        print(f"Skipping {spec.display_name}: checkpoint already exists at {spec.output_dir}")
        return None

    print(f"\nTraining {spec.display_name} from {spec.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(spec.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.checkpoint,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id={label: index for index, label in ID_TO_LABEL.items()},
        ignore_mismatched_sizes=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = FinancialDataset(train_texts, train_labels, tokenizer, settings.max_length)
    validation_dataset = FinancialDataset(
        validation_texts, validation_labels, tokenizer, settings.max_length
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"{spec.key} epoch {epoch}/{args.epochs}")
        for batch in progress:
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        metrics, matrix = evaluate(model, validation_loader, device)
        metrics["epoch"] = epoch
        metrics["training_loss"] = total_loss / len(train_loader)
        metrics["confusion_matrix"] = matrix.tolist()
        history.append(metrics)
        print(
            f"Epoch {epoch}: loss={metrics['training_loss']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}"
        )

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(spec.output_dir)
    tokenizer.save_pretrained(spec.output_dir)
    print(f"Saved epoch {args.epochs} checkpoint to {spec.output_dir}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"final": history[-1], "history": history}


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    texts, labels = load_data(settings.data_file)
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )

    unknown = sorted(set(args.models).difference(MODEL_BY_KEY))
    if unknown:
        raise ValueError("Unknown model keys: " + ", ".join(unknown))

    selected = [MODEL_BY_KEY[key] for key in args.models]
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = settings.reports_dir / "training_metrics.json"
    all_metrics = {}
    if metrics_path.is_file():
        all_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    for spec in selected:
        result = train_one(
            spec,
            args,
            train_texts,
            validation_texts,
            train_labels,
            validation_labels,
        )
        if result is not None:
            all_metrics[spec.key] = result
            metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    print(f"Training stage complete. Metrics: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[spec.key for spec in MODEL_SPECS],
        help="Model keys to train; existing checkpoints are skipped by default.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Retrain existing checkpoints.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
