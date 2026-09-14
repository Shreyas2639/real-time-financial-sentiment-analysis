"""Soft-voting inference across all five report-specified models."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .model_registry import MODEL_SPECS, ModelSpec, is_model_ready
from .model_service import LABELS, SentimentModel
from .settings import settings


class EnsembleNotReadyError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnsemblePrediction:
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    individual_predictions: dict[str, dict[str, object]]


class EnsembleSentimentModel:
    def __init__(self) -> None:
        self._models: dict[str, SentimentModel] = {}

    @staticmethod
    def status() -> dict[str, dict[str, object]]:
        return {
            spec.key: {
                "name": spec.display_name,
                "checkpoint": spec.checkpoint,
                "path": str(spec.output_dir),
                "ready": is_model_ready(spec),
            }
            for spec in MODEL_SPECS
        }

    def _require_all_models(self) -> None:
        missing = [spec.display_name for spec in MODEL_SPECS if not is_model_ready(spec)]
        if missing:
            raise EnsembleNotReadyError(
                "The five-model ensemble is not ready. Missing: " + ", ".join(missing)
            )

    def _get_model(self, spec: ModelSpec) -> SentimentModel:
        if spec.key not in self._models:
            self._models[spec.key] = SentimentModel(spec.output_dir, settings.max_length)
        return self._models[spec.key]

    def predict(self, text: str) -> EnsemblePrediction:
        self._require_all_models()
        logit_total = torch.zeros(len(LABELS), dtype=torch.float32)
        total_weight = sum(spec.weight for spec in MODEL_SPECS)
        individual: dict[str, dict[str, object]] = {}

        for spec in MODEL_SPECS:
            model = self._get_model(spec)
            logits = model.predict_logits(text)
            prediction = model.prediction_from_logits(logits)
            individual[spec.key] = asdict(prediction)
            logit_total += torch.tensor(logits, dtype=torch.float32) * spec.weight

        scores = torch.softmax(logit_total / total_weight, dim=-1).tolist()
        probabilities = {
            LABELS[index]: round(float(score), 4) for index, score in enumerate(scores)
        }
        sentiment = max(probabilities, key=probabilities.get)
        return EnsemblePrediction(
            sentiment=sentiment,
            confidence=probabilities[sentiment],
            probabilities=probabilities,
            individual_predictions=individual,
        )
