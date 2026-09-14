"""Evaluate all individual models and their soft-voting ensemble."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .model_registry import MODEL_SPECS, is_model_ready
from .settings import settings
from .train import FinancialDataset, ID_TO_LABEL, load_data


def metrics(labels: list[int], predictions: list[int]) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1, 2],
            target_names=[ID_TO_LABEL[index] for index in range(3)],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
    }


def save_confusion_plot(matrix: list[list[int]], title: str, path) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    labels = [ID_TO_LABEL[index] for index in range(3)]
    axis.set(xticks=range(3), yticks=range(3), xticklabels=labels, yticklabels=labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    for row in range(3):
        for column in range(3):
            axis.text(column, row, matrix[row][column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def predict_logits(spec, texts, labels, device) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(spec.output_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.output_dir, local_files_only=True
    ).to(device)
    model.eval()
    dataset = FinancialDataset(texts, labels, tokenizer, settings.max_length)
    loader = DataLoader(dataset, batch_size=16)
    batches = []
    with torch.inference_mode():
        for batch in loader:
            labels_tensor = batch.pop("labels")
            inputs = {name: value.to(device) for name, value in batch.items()}
            batches.append(model(**inputs).logits.cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(batches, axis=0)


def main() -> None:
    missing = [spec.display_name for spec in MODEL_SPECS if not is_model_ready(spec)]
    if missing:
        raise RuntimeError("Cannot evaluate ensemble. Missing: " + ", ".join(missing))

    texts, labels = load_data(settings.data_file)
    _, evaluation_texts, _, evaluation_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logit_sets = []
    results = {"individual_models": {}}

    plots_dir = settings.reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for spec in MODEL_SPECS:
        print(f"Evaluating {spec.display_name}")
        logits = predict_logits(
            spec, evaluation_texts, evaluation_labels, device
        )
        logit_sets.append(logits * spec.weight)
        predictions = logits.argmax(axis=1).tolist()
        model_metrics = metrics(evaluation_labels, predictions)
        results["individual_models"][spec.key] = model_metrics
        save_confusion_plot(
            model_metrics["confusion_matrix"],
            f"{spec.display_name} Confusion Matrix",
            plots_dir / f"{spec.key}_confusion_matrix.png",
        )

    total_weight = sum(spec.weight for spec in MODEL_SPECS)
    ensemble_logits = np.sum(logit_sets, axis=0) / total_weight
    ensemble_predictions = ensemble_logits.argmax(axis=1).tolist()
    results["soft_voting_ensemble"] = metrics(evaluation_labels, ensemble_predictions)
    save_confusion_plot(
        results["soft_voting_ensemble"]["confusion_matrix"],
        "Soft Voting Ensemble Confusion Matrix",
        plots_dir / "ensemble_confusion_matrix.png",
    )

    output_path = settings.reports_dir / "ensemble_evaluation.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["soft_voting_ensemble"], indent=2))
    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()
