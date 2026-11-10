
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/creditcard.csv")
MODEL_PATH = Path("models/fraud_detection_model.joblib")
METADATA_PATH = Path("models/model_metadata.json")
REPORTS_DIR = Path("reports")

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25
MINIMUM_VALIDATION_PRECISION = 0.80

THRESHOLDS = [
    0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75, 0.80,
]


def get_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Loading and preparing dataset...")
    df = pd.read_csv(DATA_PATH)
    original_rows = len(df)
    duplicate_rows_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # Reserve the final test set first.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Reproduce the development/validation split.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=VALIDATION_SIZE,
        stratify=y_development,
        random_state=RANDOM_STATE,
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Final test rows: {len(X_test):,}")

    # Use the Random Forest selected in the prior model comparison.
    print("\nFitting Random Forest on training data...")
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    rows = []
    for threshold in THRESHOLDS:
        metrics = get_metrics(
            y_validation,
            validation_probabilities,
            threshold,
        )
        rows.append({
            "threshold": threshold,
            **metrics,
        })

    validation_results = pd.DataFrame(rows)
    validation_results.to_csv(
        REPORTS_DIR / "threshold_analysis.csv",
        index=False,
    )

    eligible = validation_results[
        validation_results["precision"] >= MINIMUM_VALIDATION_PRECISION
    ]

    if eligible.empty:
        raise RuntimeError(
            "No tested threshold met the minimum validation precision. "
            "No model files were changed."
        )

    # Primary objective: maximize recall.
    # Tie-breakers: higher F1, then higher precision.
    chosen = eligible.sort_values(
        ["recall", "f1_score", "precision"],
        ascending=False,
    ).iloc[0]

    chosen_threshold = float(chosen["threshold"])

    print("\nThreshold selected using validation data only:")
    print(f"Threshold: {chosen_threshold:.2f}")
    print(f"Validation precision: {chosen['precision']:.4f}")
    print(f"Validation recall: {chosen['recall']:.4f}")
    print(f"Validation F1-score: {chosen['f1_score']:.4f}")

    # Refit on all development data, leaving the test set untouched.
    print("\nRefitting model on training + validation data...")
    model.fit(X_development, y_development)

    # Evaluate on the reserved test set exactly once at the selected threshold.
    print("\nEvaluating selected threshold on the final test set...")
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= chosen_threshold
    ).astype(int)

    test_metrics = get_metrics(
        y_test,
        test_probabilities,
        chosen_threshold,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_predictions, labels=[0, 1]
    ).ravel()

    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraudulent"],
        zero_division=0,
        digits=4,
    )

    print("\nFINAL TEST RESULTS")
    print("=" * 50)
    print(f"Selected threshold: {chosen_threshold:.2f}")
    for key, value in test_metrics.items():
        print(f"{key.replace('_', ' ').title():<22}: {value:.4f}")

    print("\nConfusion matrix counts:")
    print(f"True negatives:  {tn:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    print(f"True positives:  {tp:,}")

    print("\nClassification report:")
    print(report)

    # Save the model and the exact threshold used for its evaluation.
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": "Random Forest",
        "decision_threshold": chosen_threshold,
        "feature_columns": list(X.columns),
        "duplicate_rows_removed": duplicate_rows_removed,
        "original_dataset_rows": original_rows,
        "rows_after_duplicate_removal": len(df),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_fraction_of_development": VALIDATION_SIZE,
        "selection_metric": "validation average precision",
        "threshold_selection_rule": (
            "Maximize validation recall among thresholds with "
            "precision >= 0.80; ties broken by F1-score, then precision."
        ),
        "validation_threshold_metrics": {
            "precision": float(chosen["precision"]),
            "recall": float(chosen["recall"]),
            "f1_score": float(chosen["f1_score"]),
        },
        "final_test_metrics": test_metrics,
        "final_test_confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "test_set_used_for_model_selection": False,
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    with (REPORTS_DIR / "final_evaluation_report.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("FINANCIAL FRAUD DETECTION - FINAL TEST EVALUATION\n")
        file.write("=" * 60 + "\n\n")
        file.write("Model: Random Forest\n")
        file.write(f"Test transactions: {len(X_test):,}\n")
        file.write(f"Decision threshold: {chosen_threshold:.2f}\n\n")
        file.write("FINAL TEST METRICS\n")
        for key, value in test_metrics.items():
            file.write(f"{key}: {value:.6f}\n")

        file.write("\nCONFUSION MATRIX COUNTS\n")
        file.write(f"True negatives: {tn}\n")
        file.write(f"False positives: {fp}\n")
        file.write(f"False negatives: {fn}\n")
        file.write(f"True positives: {tp}\n\n")
        file.write("CLASSIFICATION REPORT\n")
        file.write(report)
        file.write(
            "\nThreshold selected using validation data only. "
            "The final test set was not used to select the threshold.\n"
        )

    pd.DataFrame([{
        "model": "Random Forest",
        "threshold": chosen_threshold,
        **test_metrics,
    }]).to_csv(
        REPORTS_DIR / "final_test_metrics.csv",
        index=False,
    )

    print("\nSaved updated model, metadata, and final test report.")
    print("Run the dashboard again to display the selected threshold.")


if __name__ == "__main__":
    main()
