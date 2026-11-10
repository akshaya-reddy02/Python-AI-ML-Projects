from pathlib import Path

import pandas as pd
import pytest
from feast import FeatureStore


PROJECT_DIR = Path(__file__).resolve().parent
store = FeatureStore(repo_path=str(PROJECT_DIR))

FEATURES = [
    "transaction_features:amount",
    "transaction_features:amount_log1p",
    "transaction_features:transaction_time",
    "transaction_features:time_since_start",
]

EXPECTED_COLUMNS = [
    "transaction_id",
    "amount",
    "amount_log1p",
    "transaction_time",
    "time_since_start",
]


def test_historical_features_are_retrieved():
    entity_df = pd.DataFrame({
        "transaction_id": [0],
        "event_timestamp": [
            pd.Timestamp("2023-01-01T00:00:00Z")
        ],
    })

    result = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURES,
    ).to_df()

    assert not result.empty, "Feast returned no historical features."
    assert set(EXPECTED_COLUMNS).issubset(result.columns)

    selected = result[EXPECTED_COLUMNS]
    assert not selected.isnull().any().any(), (
        "Feast returned missing feature values."
    )


def test_amount_log1p_is_correct():
    entity_df = pd.DataFrame({
        "transaction_id": [0],
        "event_timestamp": [
            pd.Timestamp("2023-01-01T00:00:00Z")
        ],
    })

    result = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURES,
    ).to_df()

    expected_amount = 149.62
    expected_log_amount = float(
        __import__("math").log1p(expected_amount)
    )

    assert result.loc[0, "amount"] == pytest.approx(
        expected_amount
    )
    assert result.loc[0, "amount_log1p"] == pytest.approx(
        expected_log_amount
    )
