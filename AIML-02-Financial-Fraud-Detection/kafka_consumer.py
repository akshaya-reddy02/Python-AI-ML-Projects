
import json
import urllib.error
import urllib.request

from kafka import KafkaConsumer

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "fraud-transactions"
API_URL = "http://127.0.0.1:8000/predict"

REQUIRED_FEATURES = {
    "Time",
    *(f"V{i}" for i in range(1, 29)),
    "Amount",
}


def score_transaction(transaction):
    transaction_id = transaction.get("transaction_id")
    features = transaction.get("features")

    if not isinstance(features, dict):
        raise ValueError(
            f"Transaction {transaction_id}: message must contain a features object."
        )

    missing = REQUIRED_FEATURES - features.keys()
    if missing:
        raise ValueError(
            f"Transaction {transaction_id}: missing required features: "
            f"{', '.join(sorted(missing))}"
        )

    payload = json.dumps(
        {name: features[name] for name in REQUIRED_FEATURES}
    ).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))

        print(f"Transaction ID: {transaction_id}")
        print(f"Prediction result: {result}")
        print("-" * 50)
        return result

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Prediction API rejected transaction {transaction_id} "
            f"(HTTP {exc.code}): {error_body}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot connect to prediction API: {exc.reason}"
        ) from exc


def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id="fraud-detection-consumer-v1",
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )

    print(f"Listening to Kafka topic: {TOPIC}")
    print(f"Sending transactions to: {API_URL}")
    print("Press Ctrl+C to stop.")

    try:
        for message in consumer:
            # Commit only after a successful prediction.
            score_transaction(message.value)
            consumer.commit()

    except KeyboardInterrupt:
        print("\nConsumer stopped by user.")

    finally:
        consumer.close()


if __name__ == "__main__":
    main()
