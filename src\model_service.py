"""Loading and inference logic for one trained sentiment model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}


@dataclass(frozen=True)
class Prediction:
    sentiment: str
    confidence: float
    probabilities: dict[str, float]


class SentimentModel:
    def __init__(self, model_dir: Path, max_length: int = 128) -> None:
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        ).to(self.device)
        self.model.eval()

    def predict_logits(self, text: str) -> list[float]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits[0].cpu().tolist()
        return [float(value) for value in logits]

    @staticmethod
    def prediction_from_logits(logits: list[float]) -> Prediction:
        scores = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).tolist()
        probabilities = {
            LABELS.get(index, f"Label {index}"): round(float(score), 4)
            for index, score in enumerate(scores)
        }
        best_index = max(range(len(scores)), key=scores.__getitem__)
        return Prediction(
            sentiment=LABELS.get(best_index, f"Label {best_index}"),
            confidence=round(float(scores[best_index]), 4),
            probabilities=probabilities,
        )

    def predict(self, text: str) -> Prediction:
        return self.prediction_from_logits(self.predict_logits(text))
