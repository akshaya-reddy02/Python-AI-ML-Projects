from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
INPUT_FILE = PROJECT_DIR / "data" / "creditcard.csv"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "transaction_features.parquet"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read the existing dataset without modifying the original CSV.
    df = pd.read_csv(INPUT_FILE)

    required = {"Time", "Amount", "Class"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Use a stable row identifier only. The dataset has no customer ID.
    features = pd.DataFrame({
        "transaction_id": np.arange(len(df), dtype=np.int64),
        "event_timestamp": (
            pd.Timestamp("2023-01-01", tz="UTC")
            + pd.to_timedelta(df["Time"], unit="s")
        ),
        "amount": df["Amount"].astype("float64"),
        "transaction_time": df["Time"].astype("float64"),
        "amount_log1p": np.log1p(df["Amount"].clip(lower=0)).astype("float64"),
        "time_since_start": df["Time"].astype("float64"),
        "Class": df["Class"].astype("int64"),
    })

    # Label is retained for evaluation only, never included in the Feast
    # feature view used for model inputs.
    features.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved {len(features):,} rows to: {OUTPUT_FILE}")
    print(f"Fraud rows: {int(features['Class'].sum()):,}")
    print("Model features: amount, transaction_time, amount_log1p, time_since_start")
    print("Customer/device/location features were not created because the source has no such identifiers.")


if __name__ == "__main__":
    main()