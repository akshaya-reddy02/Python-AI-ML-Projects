
"""Local real-time transaction simulation engine."""

from pathlib import Path
from datetime import datetime, timezone
import json
import time
import uuid

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
LOG_PATH = BASE_DIR / "reports" / "realtime_transactions.csv"


def load_engine():
    """Load the saved model, metadata, and sample transactions."""

    for path in (DATA_PATH, MODEL_PATH, METADATA_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    data = pd.read_csv(DATA_PATH)
    data = data.drop_duplicates().reset_index(drop=True)

    features = metadata["feature_columns"]

    missing = [
        column for column in features
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing model features: {missing}"
        )

    model = joblib.load(MODEL_PATH)

    # Return exactly three values.
    return model, metadata, data[features]


def score_transaction(model, metadata, transaction):
    """Score one transaction using the saved model."""

    features = metadata["feature_columns"]
    row = pd.DataFrame([transaction])[features]

    probability = float(model.predict_proba(row)[0, 1])
    threshold = float(metadata["decision_threshold"])
    prediction = int(probability >= threshold)

    return {
        "event_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "amount": float(transaction["Amount"]),
        "fraud_probability": probability,
        "threshold": threshold,
        "prediction": prediction,
        "status": (
            "FRAUD_ALERT" if prediction == 1 else "APPROVED"
        ),
    }


def append_event(event):
    """Append one valid prediction to the CSV event log."""

    if not isinstance(event, dict) or not event:
        raise ValueError("Event must be a non-empty dictionary.")

    required_fields = [
        "event_id",
        "timestamp_utc",
        "amount",
        "fraud_probability",
        "threshold",
        "prediction",
        "status",
    ]

    missing_fields = [
        field for field in required_fields
        if field not in event or event[field] is None
    ]

    if missing_fields:
        raise ValueError(
            f"Event is missing required fields: {missing_fields}"
        )

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([event], columns=required_fields).to_csv(
        LOG_PATH,
        mode="a",
        header=not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0,
        index=False,
    )




def main():
    """Run a continuous local transaction simulation."""

    model, metadata, transactions = load_engine()

    print("=" * 55)
    print("FINANCIAL FRAUD DETECTION - REAL-TIME SIMULATOR")
    print(
        "Simulation only: no real bank transactions are connected."
    )
    print(
        f"Decision threshold: {metadata['decision_threshold']}"
    )
    print("Press Ctrl+C to stop.")
    print("=" * 55)

    try:
        while True:
            transaction = transactions.sample(
                n=1
            ).iloc[0].to_dict()

            event = score_transaction(
                model, metadata, transaction
            )
            append_event(event)

            label = (
                "FRAUD" if event["prediction"] == 1
                else "LEGITIMATE"
            )

            print(
                f"{event['timestamp_utc']} | "
                f"Amount: {event['amount']:.2f} | "
                f"Fraud probability: "
                f"{event['fraud_probability']:.2%} | "
                f"Result: {label}"
            )

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nSimulator stopped safely.")
        print(f"Event log: {LOG_PATH}")


if __name__ == "__main__":
    main()
