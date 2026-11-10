
from pathlib import Path

import pandas as pd


# Project paths
DATA_PATH = Path("data") / "creditcard.csv"
REPORTS_DIR = Path("reports")


def main():
    print("=" * 60)
    print("FINANCIAL FRAUD DETECTION - DATA PREPROCESSING")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the transaction dataset
    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset shape: {df.shape}")
    print(f"Total transactions: {len(df):,}")
    print(f"Total features: {df.shape[1] - 1}")

    # Validate the target column
    if "Class" not in df.columns:
        raise ValueError("The dataset must contain a 'Class' column.")

    if not set(df["Class"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Unexpected labels found in the Class column.")

    # Inspect data quality
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    print(f"Missing values: {missing_values:,}")
    print(f"Duplicate rows: {duplicate_rows:,}")

    # Summarize legitimate and fraudulent transactions
    class_counts = df["Class"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    class_summary = pd.DataFrame({
        "transaction_type": ["Legitimate", "Fraudulent"],
        "count": [int(class_counts.loc[0]), int(class_counts.loc[1])]
    })

    class_summary["percentage"] = (
        class_summary["count"] / len(df) * 100
    ).round(4)

    print("\nTransaction class summary:")
    print(class_summary.to_string(index=False))

    # Save a report without modifying the original dataset
    report_path = REPORTS_DIR / "data_quality_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("FINANCIAL FRAUD DETECTION - DATA QUALITY REPORT\n")
        report.write("=" * 50 + "\n")
        report.write(f"Dataset shape: {df.shape}\n")
        report.write(f"Missing values: {missing_values}\n")
        report.write(f"Duplicate rows: {duplicate_rows}\n\n")
        report.write(class_summary.to_string(index=False))
        report.write("\n")

    print(f"\nData quality report saved to: {report_path}")
    print("\nPreprocessing inspection completed successfully.")


if __name__ == "__main__":
    main()
