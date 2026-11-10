FROM python:3.14-slim

WORKDIR /app

RUN python -m pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    joblib \
    fastapi \
    uvicorn \
    "feast==0.67.0"

COPY api.py .
COPY models/ ./models/
COPY feature_store/ ./feature_store/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]