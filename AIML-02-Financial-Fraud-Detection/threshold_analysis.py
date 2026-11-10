
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/creditcard.csv")
MODEL_PATH = Path("models/fraud_detection_model.joblib")
REPORTS_DIR = Path("reports")

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_FROM_REMAINING = 0.25

# Thresholds are compared using validation data only.
THRESHOLDS = [
    0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75, 0.80,
]


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().copy()

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # Recreate the same stratified split used by train_model.py.
    X_development, _, y_development, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=VALIDATION_SIZE_FROM_REMAINING,
        stratify=y_development,
        random_state=RANDOM_STATE,
    )

    print("Loading the selected model...")
    model = joblib.load(MODEL_PATH)

    # Refit a separate in-memory copy on training data only.
    # Do not change the saved production/demo model.
    print("Fitting a temporary model on training data...")
    model.fit(X_train, y_train)

    print("Calculating validation probabilities...")
    probabilities = model.predict_proba(X_validation)[:, 1]

    rows = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "precision": precision_score(
                y_validation, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_validation, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_validation, predictions, zero_division=0
            ),
            "predicted_fraud_count": int(predictions.sum()),
        })

    results = pd.DataFrame(rows)

    results["average_precision"] = average_precision_score(
        y_validation, probabilities
    )

    output_csv = REPORTS_DIR / "threshold_analysis.csv"
    results.to_csv(output_csv, index=False)

    # Illustrative selection rule:
    # among thresholds with at least 80% precision, choose the one
    # with the highest recall. If none qualifies, report that clearly.
    eligible = results[results["precision"] >= 0.80]

    report_path = REPORTS_DIR / "threshold_analysis_report.txt"

    with report_path.open("w", encoding="utf-8") as file:
        file.write("FRAUD DETECTION - VALIDATION THRESHOLD ANALYSIS\n")
        file.write("=" * 60 + "\n\n")
        file.write(f"Validation transactions: {len(X_validation):,}\n")
        file.write(f"Validation fraud cases: {int(y_validation.sum()):,}\n")
        file.write(
            "Average precision (threshold-independent): "
            f"{average_precision_score(y_validation, probabilities):.4f}\n\n"
        )
        file.write("RESULTS BY THRESHOLD\n")
        file.write(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        file.write("\n\n")

        if not eligible.empty:
            best = eligible.sort_values(
                ["recall", "f1_score"],
                ascending=False,
            ).iloc[0]

            file.write("ILLUSTRATIVE CANDIDATE\n")
            file.write(
                "Selection rule: highest recall among thresholds "
                "with validation precision >= 0.80.\n"
            )
            file.write(f"Threshold: {best['threshold']:.2f}\n")
            file.write(f"Precision: {best['precision']:.4f}\n")
            file.write(f"Recall: {best['recall']:.4f}\n")
            file.write(f"F1-score: {best['f1_score']:.4f}\n")
            file.write(
                f"Predicted fraud cases: "
                f"{int(best['predicted_fraud_count'])}\n"
            )
        else:
            file.write(
                "No tested threshold achieved validation precision >= 0.80. "
                "Review the full table and choose an appropriate trade-off.\n"
            )

        file.write(
            "\nIMPORTANT: This analysis uses validation data only. "
            "It does not evaluate or modify the final test set or saved model. "
            "Any chosen threshold must be evaluated on the untouched test set "
            "once, after the selection rule is finalized.\n"
        )

    # Save validation curves.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(results["threshold"], results["precision"], marker="o",
            label="Precision")
    ax.plot(results["threshold"], results["recall"], marker="o",
            label="Recall")
    ax.plot(results["threshold"], results["f1_score"], marker="o",
            label="F1-score")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics by Decision Threshold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "threshold_analysis.png", dpi=160)
    plt.close(fig)

    print("\nValidation threshold analysis completed.")
    print(f"Results table: {output_csv}")
    print(f"Report: {report_path}")
    print(f"Chart: {REPORTS_DIR / 'threshold_analysis.png'}")
    print("\nCandidate selection rule: maximize recall with precision >= 0.80.")
    print("Read the report before changing the dashboard threshold.")


if __name__ == "__main__":
    main()
