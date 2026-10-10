"""
test_app.py
-----------
Basic smoke tests. These run automatically in CI on every push.
Run locally with: pytest
"""

from fastapi.testclient import TestClient

from app import IrrigationInput, app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_class():
    sample = IrrigationInput.Config.json_schema_extra["example"]
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    body = response.json()
    assert body["irrigation_need"] in {"Low", "Medium", "High"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01


def test_predict_rejects_bad_input():
    bad_sample = {"Soil_Type": "NotARealSoil"}
    response = client.post("/predict", json=bad_sample)
    assert response.status_code == 422  # validation error, not a crash
