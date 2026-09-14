from fastapi.testclient import TestClient

from src import api
from src.model_service import Prediction


class FakeModel:
    device = "cpu"

    def predict(self, text: str) -> Prediction:
        return Prediction(
            sentiment="Positive",
            confidence=0.9,
            probabilities={"Negative": 0.02, "Neutral": 0.08, "Positive": 0.9},
        )


def test_root() -> None:
    client = TestClient(api.app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_predict(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_model", lambda: FakeModel())
    client = TestClient(api.app)
    response = client.post("/predict", json={"text": "Revenue increased."})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "Positive"


def test_blank_text_is_rejected() -> None:
    client = TestClient(api.app)
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422

