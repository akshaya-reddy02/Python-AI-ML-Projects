# Financial Fraud Detection Platform — Architecture

## 1. Overview

The platform demonstrates an educational machine-learning workflow for credit-card fraud-risk analysis using historical data and simulated transaction events.

## 2. Architecture Flow

```text
                 Historical Credit Card Dataset
                              |
                              v
                    Data Preprocessing
                              |
                              v
                    Feature Engineering
                              |
                              v
                  Model Training and Evaluation
                              |
                              v
                    Saved Random Forest Model
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          FastAPI Prediction         Streamlit Dashboard
              Service                Historical Analysis
                 |
                 v
          Transaction Risk Score

Simulated Transaction Events
              |
              v
         Kafka Producer
              |
              v
       Kafka Broker / Topic
              |
              v
        Kafka Consumer
              |
              v
      Processing Demonstration

Historical Data Batches
              |
              v
        Drift Monitoring
              |
              v
    Retraining Recommendation
```

## 3. Main Components

- **Data preprocessing:** Removes duplicate records and prepares features.
- **Model training:** Compares classification algorithms and saves the primary Random Forest model.
- **FastAPI:** Provides health checks and transaction prediction endpoints.
- **Streamlit:** Displays historical transaction analysis and simulated monitoring.
- **Kafka:** Demonstrates transaction event publishing and processing.
- **Drift monitoring:** Compares historical data batches and generates recommendations.

## 4. Data Flow

1. Historical transaction data is obtained separately.
2. Data preprocessing prepares the model input features.
3. Classification models are trained and evaluated.
4. The saved Random Forest model is used by the prediction service.
5. The dashboard displays historical analysis.
6. Simulated events demonstrate Kafka integration.
7. Drift monitoring generates a report and retraining recommendation.

## 5. Deployment and Limitations

The Streamlit dashboard is deployed separately. The API and Kafka components have separate execution and deployment configurations.

The Kafka-to-API workflow is partially verified. The system does not connect to live banking or payment networks and does not establish genuine fraud-ring relationships.

This architecture represents an educational prototype, not a production financial fraud detection system.