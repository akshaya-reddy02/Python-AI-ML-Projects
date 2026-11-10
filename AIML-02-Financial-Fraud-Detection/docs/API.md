# Financial Fraud Detection Platform — API Documentation

## 1. Overview

The project provides a FastAPI service for educational credit-card fraud-risk prediction and historical feature retrieval.

**Base URL (local):**

`http://127.0.0.1:8000`

Interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

## 2. Start the API

From the project root, activate the virtual environment and run:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

## 3. API Endpoints

### 3.1 Root Endpoint

- **Method:** GET
- **Path:** `/`

Returns basic API information, version, documentation path, and health endpoint.

Example response:

```json
{
  "message": "Financial Fraud Detection API",
  "version": "1.0",
  "docs": "/docs",
  "health": "/health"
}
```

### 3.2 Health Check

- **Method:** GET
- **Path:** `/health`

Checks whether the API model is loaded.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3.3 Fraud Prediction

- **Method:** POST
- **Path:** `/predict`
- **Content-Type:** `application/json`

Accepts a transaction JSON object matching the `Transaction` request schema defined in `api.py`.

The `Amount` field must be non-negative.

The API:
1. Converts the request into model input features.
2. Obtains class probabilities from the loaded model.
3. Extracts the probability for fraud class `1`.
4. Compares the probability with the configured decision threshold.
5. Returns the prediction and model information.

Example response:

```json
{
  "prediction": "LEGITIMATE",
  "is_fraud": false,
  "fraud_probability": 0.012345,
  "decision_threshold": 0.25,
  "model": "Random Forest"
}
```

The response values above are illustrative, not a recorded prediction.

### 3.4 Historical Feature Retrieval

- **Method:** GET
- **Path:** `/features/{transaction_id}`

Retrieves historical transaction features through the Feast feature service.

Example:

`GET /features/100`

The transaction ID is a synthetic row identifier from the historical dataset, not an identifier from a live transaction feed.

The response includes the transaction ID, source information, explanatory note, and retrieved features.

## 4. HTTP Error Handling

| Status code | Meaning |
|---|---|
| 404 | No complete historical Feast feature record found |
| 422 | Invalid input or negative transaction ID |
| 503 | Prediction or historical feature service unavailable |

Prediction failures are returned with an HTTP 503 response. Historical feature retrieval also handles missing records and validation errors.

## 5. Input and Model Requirements

- Prediction requests must match the `Transaction` schema in `api.py`.
- The input fields must match the model's expected feature columns.
- The `Amount` value must be greater than or equal to zero.
- The model must be loaded successfully for prediction.
- The fraud class must be available in the loaded model.

## 6. Responsible Use

This API is an educational demonstration using a historical credit-card fraud dataset.

It is not connected to a live bank or payment network. Predictions are model-based risk indicators, not proof of fraud or legitimacy. Human review is necessary before any real-world decision.


## 7. Automated Test Results

The API test suite was executed using the project's activated Python virtual environment.

**Command:** `python -m pytest -v test_api.py`

**Result:** 7 tests passed.

This result applies to the API tests in the local project environment.