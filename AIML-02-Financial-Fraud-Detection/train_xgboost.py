from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
REPORT_DIR = ROOT / "reports"
MLFLOW_DB = ROOT / "mlflow.db"

EXPERIMENT_NAME = "financial_fraud_xgboost"
RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading dataset...")
    original = pd.read_csv(DATA_PATH)

    if "Class" not in original.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    original_rows = len(original)
    df = original.drop_duplicates().reset_index(drop=True)
    duplicate_rows_removed = original_rows - len(df)

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError("Expected binary target labels 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    if positives == 0:
        raise ValueError("Training split contains no fraud cases.")

    scale_pos_weight = negatives / positives

    params = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": 4,
    }

    # Use SQLite tracking rather than MLflow's legacy file-store backend.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Training XGBoost and recording the MLflow run...")

    with mlflow.start_run(run_name="xgboost_baseline") as run:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(int)

        metrics = {
            "average_precision": float(
                average_precision_score(y_test, probabilities)
            ),
            "roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
            "precision_at_0_50": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall_at_0_50": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1_at_0_50": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
        }

        mlflow.log_params(params)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("decision_threshold", THRESHOLD)
        mlflow.log_param("duplicate_rows_removed", duplicate_rows_removed)
        mlflow.log_metric("training_rows", len(X_train))
        mlflow.log_metric("test_rows", len(X_test))
        mlflow.log_metric("fraud_test_rows", int(y_test.sum()))
        mlflow.log_metrics(metrics)

        # Save the model artifact inside the MLflow run.
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / "xgboost_evaluation.txt"

        with report_path.open("w", encoding="utf-8") as report:
            report.write("XGBoost Baseline Evaluation\n")
            report.write("===========================\n\n")
            report.write(f"MLflow run ID: {run.info.run_id}\n")
            report.write(f"Dataset rows before deduplication: {original_rows}\n")
            report.write(f"Duplicate rows removed: {duplicate_rows_removed}\n")
            report.write(f"Rows after deduplication: {len(df)}\n")
            report.write(f"Features: {list(X.columns)}\n")
            report.write(f"Decision threshold: {THRESHOLD}\n\n")

            for name, value in metrics.items():
                report.write(f"{name}: {value:.6f}\n")

        print("\nXGBoost training completed.")
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"MLflow database: {MLFLOW_DB}")
        print(f"Evaluation report: {report_path}")
        print("\nEvaluation metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()