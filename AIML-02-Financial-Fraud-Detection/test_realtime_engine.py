
"""Tests for the real-time fraud detection engine."""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from realtime_engine import load_engine, score_transaction


def main():
    print("\n=== REAL-TIME FRAUD ENGINE TESTS ===")

    # load_engine returns exactly three values.
    model, metadata, transactions = load_engine()

    feature_columns = metadata["feature_columns"]

    assert not transactions.empty, "Transaction data is empty."
    assert all(
        column in transactions.columns
        for column in feature_columns
    ), "Required features are missing."

    print("PASS: Model, metadata, and data loaded.")

    # Test a sample transaction.
    transaction = transactions.iloc[0].to_dict()
    result = score_transaction(model, metadata, transaction)

    required_fields = [
        "event_id",
        "timestamp_utc",
        "amount",
        "fraud_probability",
        "threshold",
        "prediction",
        "status",
    ]

    for field in required_fields:
        assert field in result, f"Missing field: {field}"

    print("PASS: Transaction scoring works.")

    probability = float(result["fraud_probability"])
    threshold = float(result["threshold"])
    prediction = int(result["prediction"])

    assert 0 <= probability <= 1
    assert 0 <= threshold <= 1
    assert prediction in (0, 1)

    print("PASS: Probability and threshold are valid.")

    expected_prediction = int(probability >= threshold)
    expected_status = (
        "FRAUD_ALERT" if expected_prediction else "APPROVED"
    )

    assert prediction == expected_prediction
    assert result["status"] == expected_status

    print("PASS: Prediction and status are consistent.")

    print("\nSample result:")
    print(f"Amount: {result['amount']:.2f}")
    print(f"Fraud probability: {probability:.4%}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Prediction: {prediction}")
    print(f"Status: {result['status']}")

    print("\nALL TESTS PASSED!")


if __name__ == "__main__":
    main()
