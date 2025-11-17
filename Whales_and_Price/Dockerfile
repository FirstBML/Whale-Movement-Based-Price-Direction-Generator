FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by numpy/pandas/scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9696

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]
