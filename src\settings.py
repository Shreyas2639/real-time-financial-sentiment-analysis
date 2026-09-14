"""Application settings resolved relative to the project directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    model_dir: Path = _resolve_path(os.getenv("MODEL_DIR", "saved_model"))
    report_models_dir: Path = _resolve_path(
        os.getenv("REPORT_MODELS_DIR", "report_models")
    )
    data_file: Path = _resolve_path(os.getenv("DATA_FILE", "data/financial_news.csv"))
    reports_dir: Path = _resolve_path(os.getenv("REPORTS_DIR", "reports"))
    max_length: int = int(os.getenv("MAX_LENGTH", "128"))


settings = Settings()
