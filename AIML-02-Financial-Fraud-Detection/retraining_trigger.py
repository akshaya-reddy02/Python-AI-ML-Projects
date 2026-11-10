from pathlib import Path
import json
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "reports" / "drift_report.json"
OUTPUT_PATH = BASE_DIR / "reports" / "retraining_recommendation.json"

DRIFT_SHARE_THRESHOLD = 0.50


def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Drift report not found: {REPORT_PATH}. "
            "Run drift_monitor.py first."
        )

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    metrics = report.get("metrics", [])
    drift_metric = next(
        (
            metric for metric in metrics
            if metric.get("metric_name", "").startswith(
                "DriftedColumnsCount("
            )
        ),
        None,
    )

    if drift_metric is None:
        raise ValueError(
            "Could not find DriftedColumnsCount in the Evidently report. "
            "Inspect the report before proceeding."
        )

    value = drift_metric.get("value", {})
    drifted_count = int(value["count"])
    drift_share = float(value["share"])

    trigger = drift_share >= DRIFT_SHARE_THRESHOLD

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": str(REPORT_PATH.relative_to(BASE_DIR)),
        "drifted_features": drifted_count,
        "total_features": len(
            [
                metric for metric in metrics
                if metric.get("metric_name", "").startswith("ValueDrift(")
            ]
        ),
        "drift_share": drift_share,
        "drift_share_threshold": DRIFT_SHARE_THRESHOLD,
        "retraining_recommended": trigger,
        "recommendation": (
            "Review data quality and model performance before "
            "approving retraining."
            if trigger
            else "No retraining recommendation from this drift threshold."
        ),
        "automatic_retraining": False,
        "model_overwritten": False,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print(json.dumps(result, indent=4))
    print(f"\nRecommendation saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()