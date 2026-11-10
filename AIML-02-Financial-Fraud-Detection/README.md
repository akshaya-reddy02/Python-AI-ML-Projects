# Financial Fraud Detection Platform

An end-to-end educational machine-learning project for credit-card fraud-risk analysis. It combines historical transaction analysis, classification models, feature engineering, dashboards, API-based scoring, Kafka event streaming, and exploratory drift monitoring.

> **Important:** This is an educational prototype. Transactions and streaming demonstrations use historical data or simulated events. The system is not connected to a bank or live payment network and must not make real financial decisions.

## 1. Project Objectives

- Explore and preprocess an imbalanced credit-card dataset.
- Compare Logistic Regression, Random Forest, and XGBoost.
- Select a decision threshold using validation data.
- Evaluate fraud detection using precision, recall, F1-score, average precision, and ROC-AUC.
- Demonstrate transaction scoring through a FastAPI service.
- Visualize historical and simulated transactions using Streamlit.
- Experiment with Feast feature retrieval and graph-based fraud analysis.
- Demonstrate Kafka event processing and data-drift reporting.
- Document limitations, testing, and responsible use.

## 2. Dataset

The project uses the public credit-card fraud dataset commonly distributed through Kaggle.

| Characteristic | Value |
|---|---:|
| Original rows | 284,807 |
| Original columns | 31 |
| Duplicate rows removed during preprocessing | 1,081 |
| Rows after deduplication | 283,726 |
| Legitimate transactions after deduplication | 283,253 |
| Fraudulent transactions after deduplication | 473 |

The target column is `Class`: `0` represents legitimate transactions and `1` represents fraud in the labeled dataset. `Time`, `Amount`, and anonymized features `V1` through `V28` are used as inputs.

The original dataset contains no real customer, device, or location identifiers. Consequently, this project cannot demonstrate genuine customer-velocity features, device changes, or customer relationship networks from the original data alone.

The dataset is not included in the source repository. Obtain it separately and place it at `data/creditcard.csv`.

**Dashboard and evaluation note:** The historical dashboard may display the original dataset totals (284,807 transactions and 492 fraudulent transactions), while the model training and evaluation workflow uses the deduplicated dataset (283,726 transactions and 473 fraudulent transactions). The reported test metrics refer to the held-out test split from the model evaluation workflow.

## 3. Technologies

- Python, Pandas, NumPy
- Scikit-learn and XGBoost
- PyTorch and PyTorch Geometric
- Feast
- FastAPI and Uvicorn
- Apache Kafka and `kafka-python`
- Streamlit
- MLflow and Evidently for experimentation or monitoring where configured
- Matplotlib and Seaborn
- Joblib, PyArrow, and PyYAML
- Docker and Docker Compose
- Pytest

The presence of a dependency in the environment does not by itself establish that every related feature is production-ready.

## 4. Main Project Components

### Data preparation and model training

- `data_preprocessing.py`
- `train_model.py`
- `threshold_analysis.py`
- `finalize_threshold.py`
- `evaluate_model.py`
- `train_xgboost.py`
- `train_gnn.py`
- `evaluate_blend.py`

The Random Forest model is the primary saved model. Additional XGBoost and graph-based experiments are exploratory comparisons.

### Historical analysis dashboard

Start the dashboard from the project root:

```bat
python -m streamlit run app.py
```

### Simulated transaction monitoring dashboard

```bat
python -m streamlit run realtime_dashboard.py
```

The monitoring demonstration uses historical transactions or generated events rather than a live bank feed.

## 5. FastAPI Prediction Service

The API provides health checks and transaction scoring. Requests must contain the expected numerical features in the format required by the API.

Start the API from the project root:

```bat
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Basic health and prediction behavior was tested, including invalid or incomplete request handling.

A successful prediction from this educational model is not proof that a transaction is fraudulent or legitimate in the real world.

## 6. Random Forest Evaluation

The following results are taken from `reports/final_evaluation_report.txt` for the saved Random Forest model at decision threshold `0.25`.

| Metric | Result |
|---|---:|
| Test transactions | 56,746 |
| Decision threshold | 0.25 |
| Accuracy | 99.9524% |
| Precision | 90.4762% |
| Recall | 80.0000% |
| F1-score | 84.9162% |
| Average precision | 0.803334 |
| ROC-AUC | 0.947420 |
| True negatives | 56,643 |
| False positives | 8 |
| False negatives | 19 |
| True positives | 76 |

The test set contains 56,651 legitimate transactions and 95 fraudulent transactions.

At the documented threshold, the model identified 76 of the 95 fraudulent transactions and incorrectly flagged 8 legitimate transactions. It missed 19 fraudulent transactions.

Precision measures how many flagged transactions were actually fraud in the labeled evaluation set. Recall measures how many fraudulent transactions were detected.

Accuracy alone is insufficient for imbalanced fraud detection. Do not use the held-out test set to choose a model or tune its threshold.

Re-run `evaluate_model.py` after intentionally rebuilding the model to confirm that the reports match the saved artifacts.

## 7. Experimental Model Comparison

The recorded XGBoost evaluation reports:

- Average precision: approximately `0.8202`
- ROC-AUC: approximately `0.9744`
- Precision at threshold `0.50`: approximately `0.9157`
- Recall at threshold `0.50`: `0.8000`

The recorded XGBoost and GraphSAGE blend reports average precision of approximately `0.8235`.

These are experimental results, and comparisons must account for threshold and evaluation differences. The graph uses feature similarity rather than verified links between customers, cards, or devices. It should not be interpreted as evidence of actual fraud rings.

## 8. Feast Feature Store

The Feast configuration and feature definitions are in `feature_store/`.

Prepare the feature data:

```bat
python feature_store\prepare_features.py
```

Test feature retrieval from the feature-store directory:

```bat
cd feature_store
python test_feast.py
cd ..
```

The current feature pipeline prepares `amount`, `transaction_time`, `amount_log1p`, and `time_since_start`. Its `transaction_id` is a synthetic row identifier, not a real customer ID.

The feature-store workflow is a local demonstration and does not establish a production-grade real-time feature pipeline.

## 9. Kafka Integration Status

The Docker Compose configuration starts a Kafka broker. The producer successfully sent 20 transaction messages to the `fraud-transactions` topic.

The Kafka broker and prediction API containers were confirmed running during testing. Two Python Kafka consumer processes were also found running.

However, successful end-to-end processing of the messages by the consumer and confirmation of the corresponding API predictions have not been established from the available output. Therefore, the Kafka-to-API workflow is documented as **partially verified**, not fully validated.

Do not describe all 20 transactions as successfully scored by the consumer unless the consumer's prediction output confirms that result.

Start Kafka when needed:

```bat
docker compose up -d
docker compose ps
```

Keep the API and consumer running while testing the producer-to-consumer workflow. Do not start duplicate consumers unnecessarily.

Stop Kafka when finished:

```bat
docker compose down
```

## 10. Drift Monitoring and Retraining Recommendation

Generate the historical data-drift report:

```bat
python drift_monitor.py
```

Generate a retraining recommendation:

```bat
python retraining_trigger.py
```

The drift script compares selected batches from the historical dataset. A recorded report indicated that 27 of 30 monitored features exceeded the configured drift criterion.

The trigger applies a configured drift-share threshold and writes a recommendation report. The result is a monitoring signal, not proof that predictive performance has degraded.

The retraining recommendation does not automatically retrain or overwrite the saved model.

## 11. Testing

Run the real-time engine test:

```bat
python test_realtime_engine.py
```

Run the API tests from the activated virtual environment:

```bat
python -m pytest -v test_api.py
```

The recorded automated test-suite result was **9 passing tests**. The API test suite previously passed four tests in the project environment, and the real-time engine test was also reported as passing.

The saved Random Forest evaluation report was verified separately. Docker API startup and basic prediction functionality were tested.

Kafka end-to-end consumer processing and Kubernetes health and prediction checks remain unverified from the available evidence.

These results describe checks performed in the project environment; they do not constitute a full production-readiness audit.

## 12. Dependencies and Setup

Use a Python version compatible with the installed dependencies. The project has been exercised in a Windows virtual environment; some scientific and graph-library dependencies may have Python-version-specific compatibility requirements.

From the project root in Windows Command Prompt:

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the dataset at:

```text
data/creditcard.csv
```

Run the relevant scripts from the project root, except for the Feast test, which should run from the `feature_store` directory as documented above.

## 13. Docker Deployment

The API can be packaged into a Docker image.

Build the image from the project root:

```bat
docker build -t fraud-detection-api:1.0 .
```

Run the container only if an instance is not already running on port `8000`:

```bat
docker run --rm -d --name fraud-api-test -p 8000:8000 fraud-detection-api:1.0
```

Check the health endpoint:

```bat
curl http://127.0.0.1:8000/health
```

The expected response indicates that the service is healthy and the model is loaded.

Stop the container when finished, if you started it with the command above:

```bat
docker stop fraud-api-test
```

The current `docker-compose.yml` configures Kafka only. The API and Streamlit dashboards run separately. Therefore, the current Compose configuration is a Kafka development setup, not a complete containerized deployment of the entire platform.

A full deployment would require service definitions, configuration, health checks, networking, and appropriate secret management for the API and other components.

## 14. Kubernetes Deployment Status

Kubernetes deployment is a possible extension for container orchestration. Successful deployment, health checks, and prediction checks should be reported only after their actual output has been verified in the target Kubernetes environment.

The Kubernetes workflow and commands should be documented against the current deployment manifest and actual cluster configuration before being presented as a verified deployment.

## 15. Limitations and Responsible Use

- The application uses historical data and simulated events, not live financial transactions.
- Anonymized features limit interpretation and prevent authentic customer, device, and location feature engineering.
- The graph experiment does not establish real fraud-ring relationships.
- The drift report compares historical batches, not a live production stream.
- A drift threshold is a heuristic; feature drift alone does not prove that model performance has degraded.
- The retraining trigger recommends review but does not automatically retrain the model.
- Model scores are not guaranteed to be calibrated real-world probabilities.
- A flagged transaction is not proof of fraud, and an unflagged transaction is not proof of legitimacy.
- Kafka end-to-end scoring remains unverified.
- Kubernetes deployment and its health and prediction checks remain unverified from the available evidence.
- Production use would require secure ingestion, access controls, operational monitoring, independent validation, drift and performance monitoring, alert workflows, and additional integration testing.

Model predictions should support investigation and human review rather than automatically determining financial outcomes.

## 16. Future Improvements

- Validate feature-store behavior with additional automated tests.
- Validate the complete Kafka producer-consumer-API workflow.
- Verify and document Kubernetes deployment and service checks.
- Add authentic entity identifiers if an appropriate dataset becomes available.
- Improve model comparison with consistent validation and threshold policies.
- Add live drift monitoring and model-performance checks.
- Develop a controlled retraining and approval workflow.
- Containerize and test the complete application stack.
- Add deployment documentation, observability, and security controls.
- Record a clear demonstration of verified workflows.

These are potential enhancements, not claims that all features already exist.

## 17. Conclusion

This project demonstrates an educational workflow for credit-card fraud-risk analysis, including historical model evaluation, transaction scoring, interactive dashboards, local feature-store retrieval, graph-model experiments, and drift-based retraining recommendations.

The documented Random Forest test results are based on the saved evaluation report. The project also distinguishes verified behavior from integrations that still require confirmation.

It emphasizes reproducibility, the difference between simulated and live systems, careful interpretation of imbalanced-classification metrics, and responsible use.

## 18. Submission Notes

Before submitting, verify that the ZIP contains the source files, required model artifacts, reports, tests, README, dependency list, Dockerfile, Docker Compose configuration, and any deployment manifest that is actually part of the project.

Do not include the virtual environment, Python cache directories, temporary files, or unnecessary backup copies. The dataset should be obtained separately as described in the Dataset section.

Ensure that the README's testing statements match the actual evidence available at submission time.