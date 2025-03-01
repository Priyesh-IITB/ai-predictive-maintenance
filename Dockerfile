FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt xgboost==2.0.3

# Copy application code
COPY . .

# Create validated dummy model in JSON format
RUN mkdir -p /app/models && \
    python -c "import pandas as pd; import numpy as np; from xgboost import XGBClassifier; \
    model = XGBClassifier(n_estimators=10, max_depth=3); \
    X = pd.DataFrame([[0.5,100.0,60.0],[0.8,110.0,70.0],[1.2,120.0,75.0], \
               [1.5,130.0,80.0],[1.8,140.0,90.0],[2.0,150.0,95.0]], \
               columns=['vibration', 'temperature', 'pressure']); \
    y = np.array([0,0,0,1,1,1]); \
    model.fit(X, y); \
    model.save_model('/app/models/dummy_model.json'); \
    print('Successfully created dummy model at /app/models/dummy_model.json')"



# Environment configuration
ENV USE_DUMMY_MODEL=1 \
    RANDOMIZE_PREDICTIONS=1 \
    DUMMY_MODEL_PATH=/app/models/dummy_model.json \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Healthcheck and startup command
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "scripts.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "60"]