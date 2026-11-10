import math
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from api import app

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_real_transaction():
    row = pd.read_csv(DATA_PATH, nrows=1).iloc[0]
    expected_label = int(row["Class"])

    payload = row.drop(labels=["Class"]).to_dict()
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in ["FRAUD", "LEGITIMATE"]
    assert isinstance(result["is_fraud"], bool)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["decision_threshold"] == 0.25

    print(f"Actual label: {expected_label}")
    print(f"Predicted label: {result['prediction']}")
    print(f"Fraud probability: {result['fraud_probability']}")


def test_rejects_negative_amount():
    row = pd.read_csv(DATA_PATH, nrows=1).iloc[0]
    payload = row.drop(labels=["Class"]).to_dict()
    payload["Amount"] = -10

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_rejects_missing_feature():
    row = pd.read_csv(DATA_PATH, nrows=1).iloc[0]
    payload = row.drop(labels=["Class"]).to_dict()
    payload.pop("V1")

    response = client.post("/predict", json=payload)
    assert response.status_code == 422



def test_feast_features_endpoint():
    response = client.get("/features/0")
    

    assert response.status_code == 200

    result = response.json()
    assert result["transaction_id"] == 0
    assert result["source"] == "historical_feast_store"

    features = result["features"]
    expected_feature_names = {
        "amount",
        "amount_log1p",
        "transaction_time",
        "time_since_start",
    }
    assert set(features) == expected_feature_names

    assert features["amount"] == 149.62
    assert features["amount_log1p"] == math.log1p(149.62)
    assert features["transaction_time"] == 0.0
    assert features["time_since_start"] == 0.0


def test_feast_features_reject_negative_transaction_id():
    response = client.get("/features/-1")

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "transaction_id must be non-negative"
    )


def test_feast_features_missing_transaction_returns_404():
    response = client.get("/features/999999999")

    assert response.status_code == 404
    assert "No complete Feast feature record" in response.json()["detail"]


if __name__ == "__main__":
    test_health()
    test_real_transaction()
    test_rejects_negative_amount()
    test_rejects_missing_feature()
    test_feast_features_endpoint()
    test_feast_features_reject_negative_transaction_id()
    test_feast_features_missing_transaction_returns_404()
    print("All API checks passed.")
