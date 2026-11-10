
import json
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "fraud-transactions"
MAX_TRANSACTIONS = 20
DELAY_SECONDS = 1


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, nrows=MAX_TRANSACTIONS)

    # Class is deliberately excluded from messages sent for prediction.
    feature_columns = [
        column for column in data.columns if column != "Class"
    ]

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
        retries=3,
    )

    try:
        for index, row in data.iterrows():
            transaction = {
                "transaction_id": int(index),
                "features": {
                    column: float(row[column])
                    for column in feature_columns
                },
            }

            try:
                metadata = producer.send(TOPIC, value=transaction).get(
                    timeout=10
                )
                print(
                    f"Sent transaction {index} to "
                    f"{metadata.topic}, partition {metadata.partition}, "
                    f"offset {metadata.offset}"
                )
            except KafkaError as exc:
                print(f"Failed to send transaction {index}: {exc}")
                raise

            time.sleep(DELAY_SECONDS)

    finally:
        producer.flush()
        producer.close()

    print("Producer finished.")


if __name__ == "__main__":
    main()
