
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/creditcard.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_FROM_REMAINING = 0.25
DECISION_THRESHOLD = 0.50


def calculate_metrics(y_true, probabilities, threshold):
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


def save_confusion_matrix(y_true, predictions, filename, title):
    matrix = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Legitimate", "Fraudulent"],
    )
    display.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / filename, dpi=160)
    plt.close()


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Make sure data/creditcard.csv exists."
        )

    print("=" * 65)
    print("FINANCIAL FRAUD DETECTION - MODEL TRAINING")
    print("=" * 65)

    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    if df["Class"].isna().any():
        raise ValueError("The target column contains missing values.")

    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("The Class column must contain only 0 and 1.")

    original_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    if y.nunique() != 2:
        raise ValueError("Both legitimate and fraudulent classes are required.")

    print(f"Original rows: {original_rows:,}")
    print(f"Duplicate rows removed: {duplicate_rows:,}")
    print(f"Rows retained: {len(df):,}")
    print(f"Legitimate transactions: {(y == 0).sum():,}")
    print(f"Fraudulent transactions: {(y == 1).sum():,}")
    print(f"Fraud rate: {y.mean() * 100:.4f}%")

    # First reserve the final test set. It will not be used to select a model.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Split the remaining development data into training and validation.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=VALIDATION_SIZE_FROM_REMAINING,
        stratify=y_development,
        random_state=RANDOM_STATE,
    )

    print("\nData split:")
    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Final test rows: {len(X_test):,}")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    validation_results = []
    trained_models = {}

    print("\n" + "=" * 65)
    print("PHASE 1: TRAINING AND VALIDATION")
    print("=" * 65)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        validation_probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        metrics = calculate_metrics(
            y_validation,
            validation_probabilities,
            DECISION_THRESHOLD,
        )

        validation_results.append({
            "model": name,
            **metrics,
        })

        print(f"Validation precision:         {metrics['precision']:.4f}")
        print(f"Validation recall:            {metrics['recall']:.4f}")
        print(f"Validation F1-score:          {metrics['f1_score']:.4f}")
        print(f"Validation average precision: {metrics['average_precision']:.4f}")
        print(f"Validation ROC-AUC:           {metrics['roc_auc']:.4f}")

    validation_df = pd.DataFrame(validation_results)
    validation_df.to_csv(
        REPORTS_DIR / "validation_model_comparison.csv",
        index=False,
    )

    # Choose the model using validation average precision only.
    best_row = validation_df.sort_values(
        "average_precision", ascending=False
    ).iloc[0]

    best_name = str(best_row["model"])

    print("\nValidation comparison:")
    print(validation_df.to_string(index=False))
    print(f"\nSelected model using validation data: {best_name}")

    # Refit the selected model on all development data.
    # The reserved final test set remains untouched until this point.
    print("\nRefitting selected model on training + validation data...")
    final_model = models[best_name]
    final_model.fit(X_development, y_development)

    print("\n" + "=" * 65)
    print("PHASE 2: FINAL TEST EVALUATION")
    print("=" * 65)

    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= DECISION_THRESHOLD
    ).astype(int)

    final_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        DECISION_THRESHOLD,
    )

    for metric, value in final_metrics.items():
        print(f"{metric.replace('_', ' ').title():<25}: {value:.4f}")

    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraudulent"],
        zero_division=0,
        digits=4,
    )

    print("\nClassification report:")
    print(report)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_predictions, labels=[0, 1]
    ).ravel()

    print("Confusion matrix counts:")
    print(f"True negatives:  {tn:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    print(f"True positives:  {tp:,}")

    save_confusion_matrix(
        y_test,
        test_predictions,
        "confusion_matrix.png",
        "Final Test Confusion Matrix",
    )

    # Save final model and metadata.
    model_path = MODELS_DIR / "fraud_detection_model.joblib"
    joblib.dump(final_model, model_path)

    metadata = {
        "model_name": best_name,
        "decision_threshold": DECISION_THRESHOLD,
        "feature_columns": list(X.columns),
        "duplicate_rows_removed": duplicate_rows,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_fraction_of_development": (
            VALIDATION_SIZE_FROM_REMAINING
        ),
        "selection_metric": "validation average precision",
        "validation_metrics": {
            key: float(value)
            for key, value in best_row.items()
            if key != "model"
        },
        "final_test_metrics": final_metrics,
        "final_test_confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "test_set_used_for_model_selection": False,
        "threshold_note": (
            "Fixed at 0.50 for this baseline. "
            "Threshold tuning should use validation data only."
        ),
    }

    with (MODELS_DIR / "model_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    # Save a readable final report.
    report_path = REPORTS_DIR / "final_evaluation_report.txt"

    with report_path.open("w", encoding="utf-8") as file:
        file.write("FINANCIAL FRAUD DETECTION - FINAL TEST EVALUATION\n")
        file.write("=" * 60 + "\n\n")
        file.write(f"Selected model: {best_name}\n")
        file.write(f"Test transactions: {len(X_test):,}\n")
        file.write(f"Decision threshold: {DECISION_THRESHOLD:.2f}\n\n")
        file.write("FINAL TEST METRICS\n")
        file.write("-" * 30 + "\n")

        for metric, value in final_metrics.items():
            file.write(f"{metric}: {value:.6f}\n")

        file.write("\nCONFUSION MATRIX COUNTS\n")
        file.write(f"True negatives: {tn}\n")
        file.write(f"False positives: {fp}\n")
        file.write(f"False negatives: {fn}\n")
        file.write(f"True positives: {tp}\n\n")
        file.write("CLASSIFICATION REPORT\n")
        file.write(report)
        file.write(
            "\nMethodology: selected using validation average precision; "
            "final test set was not used for model selection.\n"
        )

    # Record validation and final-test results separately.
    final_comparison = pd.DataFrame([{
        "model": best_name,
        **final_metrics,
    }])

    final_comparison.to_csv(
        REPORTS_DIR / "final_test_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("SAVED PROJECT OUTPUTS")
    print("=" * 65)
    print(f"Selected model: {best_name}")
    print(f"Model: {model_path}")
    print(f"Metadata: {MODELS_DIR / 'model_metadata.json'}")
    print(f"Validation comparison: {REPORTS_DIR / 'validation_model_comparison.csv'}")
    print(f"Final test metrics: {REPORTS_DIR / 'final_test_metrics.csv'}")
    print(f"Evaluation report: {report_path}")
    print(f"Confusion matrix: {REPORTS_DIR / 'confusion_matrix.png'}")
    print("\nTraining and evaluation completed successfully.")


if __name__ == "__main__":
    main()
