"""FastAPI entry point for single-model and ensemble predictions."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .ensemble_service import EnsembleNotReadyError, EnsembleSentimentModel
from .model_service import SentimentModel
from .settings import settings


app = FastAPI(
    title="Financial Sentiment Analysis API",
    description="Single FinBERT and five-model soft-voting sentiment predictions.",
    version="2.0.0",
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict[str, float]


class EnsemblePredictionResponse(PredictionResponse):
    individual_predictions: dict[str, dict[str, object]]


@lru_cache(maxsize=1)
def get_model() -> SentimentModel:
    return SentimentModel(settings.model_dir, settings.max_length)


@lru_cache(maxsize=1)
def get_ensemble() -> EnsembleSentimentModel:
    return EnsembleSentimentModel()


def _validated_text(request: PredictionRequest) -> str:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text cannot be blank.")
    return text


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Financial Sentiment Analysis API",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
    }


@app.get("/health")
def health() -> dict[str, object]:
    try:
        model = get_model()
    except Exception as exc:
        return {"status": "not_ready", "detail": str(exc)}
    return {"status": "ready", "device": str(model.device)}


@app.get("/models")
def models() -> dict[str, dict[str, object]]:
    return get_ensemble().status()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    text = _validated_text(request)
    try:
        result = get_model().predict(text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
    return PredictionResponse(**result.__dict__)


@app.post("/predict/ensemble", response_model=EnsemblePredictionResponse)
def predict_ensemble(request: PredictionRequest) -> EnsemblePredictionResponse:
    text = _validated_text(request)
    try:
        result = get_ensemble().predict(text)
    except EnsembleNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Ensemble prediction failed.") from exc
    return EnsemblePredictionResponse(**result.__dict__)
