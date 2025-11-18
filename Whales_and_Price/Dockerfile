FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements FIRST (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py data_pipeline.py ./
COPY models/ ./models/
COPY whale_prices_ml_ready.csv .

# Fixed line - copy cache file if it exists
COPY prediction_cache.json* /tmp/
RUN cp /tmp/prediction_cache.json* ./ 2>/dev/null || true

EXPOSE 9696

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]