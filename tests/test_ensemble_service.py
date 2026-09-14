from dataclasses import dataclass

from src import ensemble_service
from src.model_service import Prediction


@dataclass(frozen=True)
class FakeSpec:
    key: str
    display_name: str
    checkpoint: str = "fake"
    output_dir: str = "fake"
    weight: float = 1.0


class FakeModel:
    def __init__(self, logits):
        self.logits = logits

    def predict_logits(self, text):
        return self.logits

    def prediction_from_logits(self, logits):
        return ensemble_service.SentimentModel.prediction_from_logits(logits)


def test_soft_voting_averages_logits(monkeypatch):
    specs = (
        FakeSpec("one", "One"),
        FakeSpec("two", "Two"),
    )
    logits = {
        "one": [2.0, 0.0, 0.0],
        "two": [0.0, 0.0, 1.0],
    }
    monkeypatch.setattr(ensemble_service, "MODEL_SPECS", specs)
    monkeypatch.setattr(ensemble_service, "is_model_ready", lambda spec: True)
    service = ensemble_service.EnsembleSentimentModel()
    monkeypatch.setattr(service, "_get_model", lambda spec: FakeModel(logits[spec.key]))

    result = service.predict("Financial text")

    assert result.sentiment == "Negative"
    assert result.probabilities == {"Negative": 0.5065, "Neutral": 0.1863, "Positive": 0.3072}
    assert set(result.individual_predictions) == {"one", "two"}
