from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64


# Entity identifier for this dataset.
# This is a synthetic row identifier, NOT a real customer ID.
transaction = Entity(
    name="transaction_id",
    join_keys=["transaction_id"],
    description="Synthetic identifier for a transaction row",
)

# Source data prepared by the feature pipeline.
transaction_source = FileSource(
    name="transaction_features_source",
    path="data/transaction_features.parquet",
    timestamp_field="event_timestamp",
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="amount", dtype=Float64),
        Field(name="transaction_time", dtype=Float64),
        Field(name="amount_log1p", dtype=Float64),
        Field(name="time_since_start", dtype=Float64),
    ],
    source=transaction_source,
    online=True,
    description=(
        "Transaction-level features derived from the public credit-card "
        "fraud dataset. No customer, device, or location identifiers exist "
        "in the source dataset."
    ),
)