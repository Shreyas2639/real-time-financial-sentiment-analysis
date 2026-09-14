"""Single source of truth for the five models specified in the report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .settings import settings


@dataclass(frozen=True)
class ModelSpec:
    key: str
    checkpoint: str
    output_dir: Path
    display_name: str
    weight: float = 1.0


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "yiyang_finbert",
        "yiyanghkust/finbert-tone",
        settings.report_models_dir / "yiyang_finbert",
        "YiyangHKUST FinBERT Tone",
    ),
    ModelSpec(
        "twitter_roberta",
        "cardiffnlp/twitter-roberta-base-sentiment",
        settings.report_models_dir / "twitter_roberta",
        "Twitter RoBERTa",
    ),
    ModelSpec(
        "prosus_finbert",
        "ProsusAI/finbert",
        settings.report_models_dir / "prosus_finbert",
        "ProsusAI FinBERT",
    ),
    ModelSpec(
        "distilbert",
        "distilbert-base-uncased",
        settings.report_models_dir / "distilbert",
        "DistilBERT",
    ),
    ModelSpec(
        "electra",
        "google/electra-small-discriminator",
        settings.report_models_dir / "electra",
        "ELECTRA Small",
    ),
)

MODEL_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}


def is_model_ready(spec: ModelSpec) -> bool:
    return (spec.output_dir / "config.json").is_file()
