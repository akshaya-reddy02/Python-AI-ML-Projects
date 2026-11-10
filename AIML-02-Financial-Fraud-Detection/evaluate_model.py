"""Reproduce the saved model's final held-out evaluation."""

from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
REPORTS_DIR = BASE_DIR / "reports"


def main():
    for path in (DATA_PATH, MODEL_PATH, METADATA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required project file not found: {path}")

    REPORTS_DIR.mkdir(exist_ok=True)

    print("Loading dataset and saved evaluation settings...")
    df = pd.read_csv(DATA_PATH)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' target column.")

    df = df.drop_duplicates().copy()
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata.get("feature_columns", [])
    if feature_columns and list(X.columns) != feature_columns:
        raise ValueError("Dataset features do not match model metadata.")

    # Recreate the stratified final test split used by the project scripts.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=float(metadata.get("test_size", 0.20)),
        stratify=y,
        random_state=int(metadata.get("random_state", 42)),
    )

    model = joblib.load(MODEL_PATH)
    threshold = float(metadata.get("decision_threshold", 0.50))
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "average_precision": average_precision_score(y_test, probabilities),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraudulent"],
        zero_division=0,
        digits=4,
    )

    lines = [
        "FINANCIAL FRAUD DETECTION - FINAL TEST EVALUATION",
        "=" * 56,
        f"Selected model: {metadata.get('model_name', 'Saved classifier')}",
        f"Test transactions: {len(y_test):,}",
        f"Decision threshold: {threshold:.2f}",
        "",
        "FINAL TEST METRICS",
        "-" * 30,
    ]
    lines.extend(f"{name}: {value:.6f}" for name, value in metrics.items())
    lines.extend([
        "",
        "CONFUSION MATRIX COUNTS",
        f"True negatives: {tn}",
        f"False positives: {fp}",
        f"False negatives: {fn}",
        f"True positives: {tp}",
        "",
        "CLASSIFICATION REPORT",
        report.rstrip(),
        "",
        "Threshold and model selection were determined using validation data. "
        "This script reproduces the held-out test report; do not use the test "
        "results to tune the model or threshold.",
    ])

    report_path = REPORTS_DIR / "final_evaluation_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame([{
        "model": metadata.get("model_name", "Saved classifier"),
        "threshold": threshold,
        **metrics,
    }]).to_csv(REPORTS_DIR / "final_test_metrics.csv", index=False)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Legitimate", "Fraudulent"],
    )
    display.plot(values_format="d")
    plt.title("Final Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print("\n".join(lines))
    print(f"\nReport saved to: {report_path}")
    print(f"Confusion matrix saved to: {REPORTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
