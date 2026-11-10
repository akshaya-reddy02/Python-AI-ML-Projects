
"""FraudGuard - Real-Time Monitoring and Manual Prediction Dashboard."""

from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from realtime_engine import load_engine, score_transaction


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
LOG_PATH = BASE_DIR / "reports" / "realtime_transactions.csv"


st.set_page_config(
    page_title="FraudGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

@st.cache_resource
def get_engine():
    return load_engine()


@st.cache_data
def get_sample_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)


try:
    model, metadata, transactions = get_engine()
    sample_data = get_sample_data()
except Exception as exc:
    st.error(f"Could not load the fraud detection system: {exc}")
    st.stop()


threshold = float(metadata["decision_threshold"])
feature_columns = metadata["feature_columns"]

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🛡️ FraudGuard")
st.sidebar.caption("Financial Fraud Detection")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Live Monitoring",
        "Manual Prediction",
        "Model Information",
        "Analyst Case Management",
    ],
)

st.sidebar.divider()
st.sidebar.metric("Decision threshold", f"{threshold:.0%}")
st.sidebar.caption(
    "Prototype only — predictions are not confirmation of fraud."
)

# --------------------------------------------------
# PAGE 1: LIVE MONITORING
# --------------------------------------------------

if page == "Live Monitoring":
    st.title("🛡️ FraudGuard — Live Transaction Monitor")
    st.write(
        "Monitor model predictions from the simulated transaction stream."
    )

    st_autorefresh(interval=5000, key="fraudguard_refresh")

    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
    else:
        df = pd.DataFrame(
            columns=[
                "event_id",
                "timestamp_utc",
                "amount",
                "fraud_probability",
                "threshold",
                "prediction",
                "status",
            ]
        )

    if not df.empty:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["fraud_probability"] = pd.to_numeric(
            df["fraud_probability"], errors="coerce"
        )
        df["prediction"] = pd.to_numeric(
            df["prediction"], errors="coerce"
        )

        alerts = df[df["prediction"] == 1]
        latest = df.tail(10).iloc[::-1].copy()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Transactions recorded", f"{len(df):,}")
        col2.metric("Fraud alerts", f"{len(alerts):,}")
        col3.metric(
            "Alert rate",
            f"{len(alerts) / len(df):.2%}" if len(df) else "0%",
        )
        col4.metric(
            "Decision threshold",
            f"{threshold:.0%}",
        )

        st.divider()

        left, right = st.columns(2)

        with left:
            st.subheader("Recent fraud alerts")

            if alerts.empty:
                st.success("No fraud alerts in the current log.")
            else:
                display_alerts = alerts.tail(10).iloc[::-1].copy()
                display_alerts["fraud_probability"] = (
                    display_alerts["fraud_probability"].map(
                        lambda value: f"{value:.2%}"
                        if pd.notna(value) else "N/A"
                    )
                )
                st.table(
                    display_alerts[
                        [
                            "timestamp_utc",
                            "amount",
                            "fraud_probability",
                            "status",
                        ]
                    ]
                 )

            st.table(
                latest[
                    [
                        "timestamp_utc",
                        "amount",
                        "fraud_probability",
                        "status",
                    ]
                ]
                
            )

        st.subheader("Fraud probability distribution")

        distribution_values = df["fraud_probability"].dropna()
        if not distribution_values.empty:
            bins = [0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.000001]
            labels = [
                "0–5%",
                "5–10%",
                "10–25%",
                "25–50%",
                "50–75%",
                "75–100%",
            ]

            ranges = pd.cut(
                distribution_values.clip(0, 1),
                bins=bins,
                labels=labels,
                include_lowest=True,
                right=False,
            )

            distribution = (
                ranges.value_counts()
                .reindex(labels, fill_value=0)
                .rename_axis("Probability range")
                .reset_index(name="Transaction count")
            )

            st.bar_chart(
                distribution,
                x="Probability range",
                y="Transaction count",
            )

        st.divider()

        st.subheader("Export transaction history")
        st.download_button(
            label="Download transaction log CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="realtime_transactions.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "No simulated transactions have been recorded yet. "
            "Start the simulator in a separate CMD window to populate "
            "this dashboard."
        )

    st.caption(
        "Prototype limitation: transactions are simulated from historical "
        "data. This dashboard is not connected to a bank or payment gateway."
    )

# --------------------------------------------------
# PAGE 2: MANUAL PREDICTION
# --------------------------------------------------

elif page == "Manual Prediction":
    st.title("🔎 Manual Fraud Prediction")
    st.write(
        "Choose a historical sample transaction, optionally change its "
        "amount, and score it with the saved Random Forest model."
    )

    st.warning(
        "This uses an existing dataset row for the required features. "
        "Changing the amount alone does not create a new, realistic "
        "transaction. The result is a model prediction, not proof of fraud."
    )

    with st.form("manual_prediction_form"):
        sample_index = st.number_input(
            "Sample transaction row",
            min_value=0,
            max_value=len(sample_data) - 1,
            value=0,
            step=1,
            help="Choose a row from the existing credit-card dataset.",
        )

        original_transaction = (
            sample_data.iloc[int(sample_index)][feature_columns].to_dict()
        )

        st.write(
            f"Original dataset amount: "
            f"{float(original_transaction['Amount']):.2f}"
        )

        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=max(0.0, float(original_transaction["Amount"])),
            step=10.0,
            format="%.2f",
        )

        submitted = st.form_submit_button(
            "Predict transaction",
            type="primary",
        )

    if submitted:
        transaction = original_transaction.copy()
        transaction["Amount"] = float(amount)

        try:
            result = score_transaction(
                model,
                metadata,
                transaction,
            )

            st.subheader("Prediction result")

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Fraud probability",
                f"{result['fraud_probability']:.2%}",
            )
            c2.metric(
                "Decision threshold",
                f"{result['threshold']:.0%}",
            )
            c3.metric(
                "Transaction amount",
                f"{result['amount']:.2f}",
            )

            if result["prediction"] == 1:
                st.error(
                    "🚨 FRAUD ALERT — The model flagged this transaction."
                )
            else:
                st.success(
                    "✅ APPROVED — The model did not flag this transaction."
                )

            st.write(
                f"Prediction status: **{result['status']}**"
            )

            st.caption(
                "Manual predictions are not appended to the simulated "
                "transaction history. Other features came from the selected "
                "historical row and were not changed."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# --------------------------------------------------
# PAGE 3: MODEL INFORMATION
# --------------------------------------------------

elif page == "Model Information":
    st.title("📊 Model Information")

    st.write("The dashboard uses the saved Random Forest classifier.")

    st.metric("Decision threshold", f"{threshold:.0%}")

    st.write("**Model input features**")
    st.write(
        "The model expects Time, V1 through V28, and Amount — "
        f"{len(feature_columns)} numerical features in total."
    )

    st.code(", ".join(feature_columns), language="text")

    st.write("**Final held-out test results**")

    metrics_path = BASE_DIR / "reports" / "final_test_metrics.csv"

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.table(metrics)
    else:
        st.info(
            "The final test metrics CSV was not found. "
            "See reports/final_evaluation_report.txt instead."
        )

    st.caption(
        "The test metrics describe evaluation on a held-out dataset. "
        "They do not guarantee the same performance on real transactions."
    )


# --------------------------------------------------
# PAGE 4: ANALYST CASE MANAGEMENT
# --------------------------------------------------

elif page == "Analyst Case Management":
    st.title("🗂️ Analyst Case Management")
    st.write("Review flagged transactions and record analyst decisions.")

    REVIEW_PATH = BASE_DIR / "reports" / "analyst_reviews.csv"

    if LOG_PATH.exists():
        review_data = pd.read_csv(LOG_PATH)
    else:
        review_data = pd.DataFrame()

    if review_data.empty or "event_id" not in review_data.columns:
        st.info("No transaction events are available to review yet.")
    else:
        review_data["prediction"] = pd.to_numeric(
            review_data["prediction"], errors="coerce"
        )
        flagged = review_data[review_data["prediction"] == 1].copy()

        if flagged.empty:
            st.info("No flagged transactions are available for review.")
        else:
            if REVIEW_PATH.exists():
                saved_reviews = pd.read_csv(REVIEW_PATH)
            else:
                saved_reviews = pd.DataFrame(
                    columns=[
                        "event_id",
                        "review_status",
                        "feedback_label",
                        "analyst_notes",
                        "updated_at_utc",
                    ]
                )

            flagged["event_id"] = flagged["event_id"].astype(str)
            saved_reviews["event_id"] = (
                saved_reviews["event_id"].astype(str)
            )

            st.metric("Flagged transactions", len(flagged))

            event_ids = flagged["event_id"].tolist()
            selected_id = st.selectbox(
                "Select a flagged transaction",
                event_ids,
            )

            selected = flagged[
                flagged["event_id"] == selected_id
            ].iloc[0]

            existing = saved_reviews[
                saved_reviews["event_id"] == selected_id
            ]

            if not existing.empty:
                previous = existing.iloc[-1]
                default_status = previous["review_status"]
                default_feedback = previous["feedback_label"]
                default_notes = previous["analyst_notes"]
            else:
                default_status = "Open"
                default_feedback = "Needs Investigation"
                default_notes = ""

            st.write(f"**Transaction ID:** {selected_id}")
            st.write(
                f"**Amount:** {selected.get('amount', 'N/A')}"
            )
            st.write(
                f"**Fraud probability:** "
                f"{selected.get('fraud_probability', 'N/A')}"
            )

            statuses = ["Open", "Under Review", "Closed"]
            feedback_options = [
                "Needs Investigation",
                "Confirmed Fraud",
                "Legitimate",
            ]

            status_index = (
                statuses.index(default_status)
                if default_status in statuses else 0
            )
            feedback_index = (
                feedback_options.index(default_feedback)
                if default_feedback in feedback_options else 0
            )

            with st.form("analyst_review_form"):
                new_status = st.selectbox(
                    "Case status",
                    statuses,
                    index=status_index,
                )
                new_feedback = st.selectbox(
                    "Analyst feedback",
                    feedback_options,
                    index=feedback_index,
                )
                notes = st.text_area(
                    "Analyst notes",
                    value="" if pd.isna(default_notes) else str(default_notes),
                )
                save_review = st.form_submit_button(
                    "Save review",
                    type="primary",
                )

            if save_review:
                from datetime import datetime, timezone

                REVIEW_PATH.parent.mkdir(
                    parents=True, exist_ok=True
                )

                saved_reviews = saved_reviews[
                    saved_reviews["event_id"] != selected_id
                ]

                new_review = pd.DataFrame([{
                    "event_id": selected_id,
                    "review_status": new_status,
                    "feedback_label": new_feedback,
                    "analyst_notes": notes,
                    "updated_at_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }])

                saved_reviews = pd.concat(
                    [saved_reviews, new_review],
                    ignore_index=True,
                )
                saved_reviews.to_csv(REVIEW_PATH, index=False)

                st.success("Analyst review saved successfully.")
                st.rerun()

            st.divider()
            st.subheader("Saved analyst reviews")

            if not saved_reviews.empty:
                st.dataframe(
                    saved_reviews,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No analyst reviews have been saved yet.")