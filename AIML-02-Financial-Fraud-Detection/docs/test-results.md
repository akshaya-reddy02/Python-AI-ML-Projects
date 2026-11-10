## Real-Time Fraud Engine Test Results

**Command:** `python test_realtime_engine.py`

**Result:** All 4 checks passed.

- Model, metadata, and data loaded successfully.
- Transaction scoring works.
- Probability and threshold are valid.
- Prediction and status are consistent.

**Sample transaction:**
- Amount: 149.62
- Fraud probability: 0.0000%
- Threshold: 0.25
- Prediction: 0 (Approved)

**Status:** All standalone real-time engine checks passed.