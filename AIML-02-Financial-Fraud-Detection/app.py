
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
REPORT_PATH = BASE_DIR / "reports" / "final_evaluation_report.txt"
MATRIX_PATH = BASE_DIR / "reports" / "confusion_matrix.png"

st.set_page_config(
    page_title="FraudGuard | Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# FILE LOADING
# ============================================================

@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


@st.cache_data
def load_dataset(path, modified_time):
    return pd.read_csv(path)


@st.cache_data
def load_metadata(path, modified_time):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_percentage(value):
    return f"{float(value) * 100:.2f}%"


for required_path in [DATA_PATH, MODEL_PATH, METADATA_PATH]:
    if not required_path.exists():
        st.error(f"Required project file not found: {required_path}")
        if required_path == MODEL_PATH:
            st.info("Run `python train_model.py` to train the model.")
        st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    df = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    metadata = load_metadata(
        str(METADATA_PATH), METADATA_PATH.stat().st_mtime
    )
except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# ============================================================
# VALIDATE DATA AND MODEL
# ============================================================

if "Class" not in df.columns:
    st.error("The dataset must contain a Class target column.")
    st.stop()

feature_columns = [column for column in df.columns if column != "Class"]
saved_features = metadata.get("feature_columns", [])

if saved_features and feature_columns != saved_features:
    st.error(
        "Dataset features do not match the saved model metadata. "
        "Check the dataset and retrain the model if necessary."
    )
    st.stop()

if not hasattr(model, "predict"):
    st.error("The saved model does not support prediction.")
    st.stop()

if not hasattr(model, "predict_proba"):
    st.error("The saved model does not support probability estimates.")
    st.stop()

if not set(df["Class"].dropna().unique()).issubset({0, 1}):
    st.error("The Class column must contain only 0 and 1.")
    st.stop()

threshold = float(metadata.get("decision_threshold", 0.5))
model_name = metadata.get("model_name", "Trained classifier")
final_metrics = metadata.get("final_test_metrics", {})
test_counts = metadata.get("final_test_confusion_matrix", {})


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🛡️ FraudGuard")
    st.caption("Financial transaction risk analysis")
    st.divider()

    st.subheader("Model Information")
    st.write(f"**Algorithm:** {model_name}")
    st.write(f"**Decision threshold:** {threshold:.2f}")
    st.write("**Dataset:** Credit Card Fraud Detection")

    st.divider()
    st.subheader("Navigation")
    st.caption(
        "Scroll through the dashboard to explore the dataset, "
        "analyze individual transactions, and review model evaluation."
    )

    st.divider()
    st.caption("Educational and portfolio project")


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ Financial Fraud Detection System")
st.markdown(
    """
    Explore historical credit-card transactions, assess potential
    fraud risk, and review machine-learning model performance.
    """
)

st.warning(
    "Educational demonstration only. This project uses historical data, "
    "not a live bank feed. A model prediction is a risk estimate, not proof "
    "of fraud or proof that a transaction is safe."
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("📊 Dataset Overview")

total_transactions = len(df)
fraud_count = int((df["Class"] == 1).sum())
legitimate_count = int((df["Class"] == 0).sum())
fraud_rate = (
    fraud_count / total_transactions * 100
    if total_transactions else 0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", f"{total_transactions:,}")
c2.metric("Legitimate Transactions", f"{legitimate_count:,}")
c3.metric("Fraudulent Transactions", f"{fraud_count:,}")
c4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

left, right = st.columns(2)

with left:
    st.subheader("Transaction Class Distribution")
    class_counts = pd.DataFrame({
        "Transaction Type": ["Legitimate", "Fraudulent"],
        "Transactions": [legitimate_count, fraud_count],
    }).set_index("Transaction Type")
    st.bar_chart(class_counts)

with right:
    st.subheader("Transaction Amounts")
    st.write("Distribution of transaction amounts in the complete dataset.")
    amount_data = df["Amount"].describe()
    st.dataframe(
        amount_data.rename("Amount").to_frame(),
        use_container_width=True,
    )

st.caption(
    "Fraud is a minority class in this dataset. Precision, recall, "
    "F1-score, and average precision are therefore important metrics."
)

st.divider()


# ============================================================
# TRANSACTION ANALYSIS
# ============================================================

# ============================================================
# TRANSACTION ANALYSIS
# ============================================================

st.header("🔎 Analyze a Transaction")
st.write(
    "Choose any historical transaction by its dataset row number "
    "and let the model estimate its fraud risk."
)

with st.form("transaction_analysis"):
    category = st.radio(
        "Transaction category",
        [
            "Any transaction",
            "Legitimate-labeled example",
            "Fraudulent-labeled example",
        ],
        horizontal=True,
    )

    if category == "Legitimate-labeled example":
        eligible = df.index[df["Class"] == 0].tolist()
    elif category == "Fraudulent-labeled example":
        eligible = df.index[df["Class"] == 1].tolist()
    else:
        eligible = df.index.tolist()

    st.caption(
        f"Available rows: 0 to {len(df) - 1:,} "
        f"({len(eligible):,} eligible for the selected category)."
    )

    row_number = st.number_input(
        "Enter any dataset row number",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
        help=f"Enter a whole number between 0 and {len(df) - 1:,}.",
    )

    analyze = st.form_submit_button(
        "Analyze Transaction",
        type="primary",
        use_container_width=True,
    )




# ============================================================
# PREDICTION RESULTS
# ============================================================

if analyze:
    row_number = int(row_number)
    transaction = df.iloc[[row_number]]
    
    
    actual_label = int(transaction["Class"].iloc[0])

    if category == "Legitimate-labeled example" and actual_label != 0:
        st.error(
            "The selected row is not labeled legitimate. "
            "Choose another row or select Any transaction."
        )
    elif category == "Fraudulent-labeled example" and actual_label != 1:
        st.error(
            "The selected row is not labeled fraudulent. "
            "Choose another row or select Any transaction."
        )
    else:
        X_transaction = transaction[feature_columns]

        try:
            prediction = int(model.predict(X_transaction)[0])
            probabilities = model.predict_proba(X_transaction)[0]
            fraud_probability = float(probabilities[1])

            st.subheader("Prediction Result")

            result_col, probability_col = st.columns([1, 1])

            with result_col:
                if prediction == 1:
                    st.error("⚠️ Potential Fraud")
                    st.write(
                        "The estimated risk meets or exceeds "
                        "the model's decision threshold."
                    )
                else:
                    st.success("✅ Below Fraud Threshold")
                    st.write(
                        "The estimated risk is below the model's "
                        "current decision threshold."
                    )

            with probability_col:
                st.metric(
                    "Estimated Fraud Probability",
                    format_percentage(fraud_probability),
                )
                st.progress(
                    min(1.0, max(0.0, fraud_probability))
                )
                st.caption(
                    "This is a model estimate and may not be a "
                    "well-calibrated real-world probability."
                )

            actual_text = (
                "Fraudulent" if actual_label == 1 else "Legitimate"
            )
            predicted_text = (
                "Potential fraud" if prediction == 1
                else "Below fraud threshold"
            )

            st.write(f"**Dataset label:** {actual_text}")
            st.write(f"**Model result:** {predicted_text}")

            if prediction == actual_label:
                st.success("The prediction matches the dataset label.")
            else:
                st.warning("The prediction differs from the dataset label.")

            transaction_details = pd.DataFrame({
                "Field": [
                    "Dataset row",
                    "Transaction amount",
                    "Transaction time",
                    "Dataset label",
                    "Model prediction",
                    "Estimated fraud probability",
                    "Decision threshold",
                ],
                "Value": [
                    row_number,
                    f"{float(transaction['Amount'].iloc[0]):.2f}",
                    f"{float(transaction['Time'].iloc[0]):.0f}"
                    if "Time" in transaction.columns else "Not available",
                    actual_text,
                    predicted_text,
                    format_percentage(fraud_probability),
                    format_percentage(threshold),
                ],
            })

            with st.expander("View transaction details and model features"):
                st.dataframe(
                    transaction_details,
                    hide_index=True,
                    use_container_width=True,
                )
                st.write("Model input features")
                st.dataframe(
                    X_transaction,
                    hide_index=True,
                    use_container_width=True,
                )

            st.caption(
                "Checking a selected historical record is a demonstration. "
                "It does not establish overall model performance."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("📈 Final Model Evaluation")
st.write(
    "The following results are loaded from the final evaluation "
    "performed by `train_model.py`."
)

if final_metrics:
    metric_cols = st.columns(4)

    metric_names = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-score"),
    ]

    for column, (key, label) in zip(metric_cols, metric_names):
        if key in final_metrics:
            column.metric(label, format_percentage(final_metrics[key]))
        else:
            column.metric(label, "N/A")

    extra_cols = st.columns(2)
    extra_cols[0].metric(
        "Average Precision",
        format_percentage(final_metrics["average_precision"])
        if "average_precision" in final_metrics else "N/A",
    )
    extra_cols[1].metric(
        "ROC-AUC",
        f"{float(final_metrics['roc_auc']):.4f}"
        if "roc_auc" in final_metrics else "N/A",
    )

    st.caption(
        "These metrics are from a held-out test set. The model was selected "
        "using validation average precision, not the final test set."
    )
else:
    st.info(
        "Final evaluation metrics are not present in the metadata. "
        "Run `python train_model.py` to regenerate the model and reports."
    )

if test_counts:
    st.subheader("Final Test Confusion Matrix")

    matrix_df = pd.DataFrame(
        [
            [
                test_counts.get("true_negatives", 0),
                test_counts.get("false_positives", 0),
            ],
            [
                test_counts.get("false_negatives", 0),
                test_counts.get("true_positives", 0),
            ],
        ],
        index=["Actual legitimate", "Actual fraudulent"],
        columns=["Predicted legitimate", "Predicted fraudulent"],
    )

    st.dataframe(matrix_df, use_container_width=True)

    tn = test_counts.get("true_negatives", 0)
    fp = test_counts.get("false_positives", 0)
    fn = test_counts.get("false_negatives", 0)
    tp = test_counts.get("true_positives", 0)

    a, b, c, d = st.columns(4)
    a.metric("True Negatives", f"{tn:,}")
    b.metric("False Positives", f"{fp:,}")
    c.metric("False Negatives", f"{fn:,}")
    d.metric("True Positives", f"{tp:,}")

if REPORT_PATH.exists():
    with st.expander("Read the full evaluation report"):
        st.text(REPORT_PATH.read_text(encoding="utf-8"))

if MATRIX_PATH.exists():
    with st.expander("View confusion matrix image"):
        st.image(
            str(MATRIX_PATH),
            caption="Confusion matrix from the final test evaluation",
            use_container_width=True,
        )

st.divider()


# ============================================================
# DATASET PREVIEW
# ============================================================

st.header("📋 Dataset Preview")

preview_count = st.slider(
    "Records to display",
    min_value=5,
    max_value=30,
    value=10,
)

preview = df.head(preview_count).copy()
preview["Class"] = preview["Class"].map({
    0: "Legitimate",
    1: "Fraudulent",
})

st.dataframe(
    preview,
    hide_index=True,
    use_container_width=True,
)

csv_data = preview.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download displayed preview as CSV",
    data=csv_data,
    file_name="transaction_preview.csv",
    mime="text/csv",
)


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "FraudGuard | Python · Pandas · Scikit-learn · Streamlit | "
    "Educational machine-learning project"
)
