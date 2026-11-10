from pathlib import Path
import json

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
REPORT_DIR = BASE_DIR / "reports"

REFERENCE_SIZE = 10000
CURRENT_SIZE = 1000

EXCLUDED_COLUMNS = {"Class"}


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH)

    if "Class" not in data.columns:
        raise ValueError("Dataset must contain the Class target column.")

    feature_columns = [
        column for column in data.columns
        if column not in EXCLUDED_COLUMNS
    ]

    if len(data) < REFERENCE_SIZE + CURRENT_SIZE:
        raise ValueError("Not enough rows for the configured drift comparison.")

    # Use separate chronological portions as historical reference and
    # comparison batches. These are not live production transactions.
    reference = data.iloc[:REFERENCE_SIZE][feature_columns].copy()
    current = data.iloc[-CURRENT_SIZE:][feature_columns].copy()

    # Keep column names and numeric types consistent.
    reference = reference.astype("float64")
    current = current.astype("float64")

    print(f"Reference rows: {len(reference)}")
    print(f"Comparison rows: {len(current)}")
    print(f"Features monitored: {len(feature_columns)}")

    report = Report([
        DataDriftPreset(),
    ])

    snapshot = report.run(
        reference_data=reference,
        current_data=current,
    )

    html_path = REPORT_DIR / "drift_report.html"
    json_path = REPORT_DIR / "drift_report.json"

    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))

    print("\nDrift report generated successfully.")
    print(f"HTML report: {html_path}")
    print(f"JSON report: {json_path}")
    print(
        "\nNote: This compares historical data batches. "
        "It does not monitor live transactions or trigger retraining."
    )


if __name__ == "__main__":
    main()