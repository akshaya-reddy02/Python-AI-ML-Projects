
import json
import traceback
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------
# 1. Paths and model loading
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Model metadata not found: {METADATA_PATH}")

model = joblib.load(MODEL_PATH)

with METADATA_PATH.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

THRESHOLD = float(metadata.get("decision_threshold", 0.25))


# --------------------------------------------------
# 2. FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Financial Fraud Detection API",
    description=(
        "Scores credit-card transactions and retrieves historical "
        "features from the local Feast feature store."
    ),
    version="1.1.0",
)


# --------------------------------------------------
# 3. Transaction request validation
# --------------------------------------------------

class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Time: float = Field(ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)


# --------------------------------------------------
# 4. Root and health endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Financial Fraud Detection API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# --------------------------------------------------
# 5. Fraud prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        values = transaction.model_dump()

        features = pd.DataFrame(
            [[values[column] for column in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )

        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)

        if 1 not in classes:
            raise RuntimeError(
                "Loaded model has no fraud class (Class=1)."
            )

        fraud_index = classes.index(1)
        fraud_probability = float(probabilities[fraud_index])
        is_fraud = fraud_probability >= THRESHOLD

        return {
            "prediction": "FRAUD" if is_fraud else "LEGITIMATE",
            "is_fraud": is_fraud,
            "fraud_probability": round(fraud_probability, 6),
            "decision_threshold": THRESHOLD,
            "model": metadata.get("model_name", "Unknown"),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction failed: {type(exc).__name__}",
        ) from exc


# --------------------------------------------------
# 6. Historical Feast feature endpoint
# --------------------------------------------------

@app.get("/features/{transaction_id}")
def get_historical_features(transaction_id: int):
    """Retrieve features for a historical dataset row ID."""

    if transaction_id < 0:
        raise HTTPException(
            status_code=422,
            detail="transaction_id must be non-negative",
        )

    try:
        from feature_store.feast_service import FeastFeatureService

        service = FeastFeatureService()
        features = service.get_transaction_features(transaction_id)

        return {
            "transaction_id": transaction_id,
            "source": "historical_feast_store",
            "note": (
                "This is a synthetic row ID from the historical dataset, "
                "not an ID from a live transaction feed."
            ),
            "features": features,
        }

    except HTTPException:
        raise

    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                "No complete Feast feature record was found "
                "for this transaction ID."
            ),
        ) from exc

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # Temporary diagnostic: print the underlying exception.
        traceback.print_exc()

        raise HTTPException(
            status_code=503,
            detail=(
                "Historical feature store unavailable: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc
