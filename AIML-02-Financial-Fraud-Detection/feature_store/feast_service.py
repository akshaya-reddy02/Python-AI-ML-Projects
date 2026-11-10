
from pathlib import Path

from feast import FeatureStore


PROJECT_DIR = Path(__file__).resolve().parent
FEATURES = [
    "transaction_features:amount",
    "transaction_features:amount_log1p",
    "transaction_features:transaction_time",
    "transaction_features:time_since_start",
]


class FeastFeatureService:
    """Retrieve transaction features from the local Feast online store."""

    def __init__(self):
        self.store = FeatureStore(repo_path=str(PROJECT_DIR))

    def get_transaction_features(self, transaction_id: int) -> dict:
        if isinstance(transaction_id, bool) or not isinstance(
            transaction_id, int
        ):
            raise TypeError("transaction_id must be an integer")

        if transaction_id < 0:
            raise ValueError("transaction_id must be non-negative")

        result = self.store.get_online_features(
            features=FEATURES,
            entity_rows=[{"transaction_id": transaction_id}],
        ).to_dict()

        values = {
            name: result[name][0]
            for name in (
                "amount",
                "amount_log1p",
                "transaction_time",
                "time_since_start",
            )
        }

        if any(value is None for value in values.values()):
            raise LookupError(
                f"Feast returned a missing feature for transaction "
                f"{transaction_id}"
            )

        return values


if __name__ == "__main__":
    service = FeastFeatureService()
    transaction_id = 0
    features = service.get_transaction_features(transaction_id)

    print(f"Transaction ID: {transaction_id}")
    print("Retrieved Feast features:")
    for name, value in features.items():
        print(f"  {name}: {value}")
