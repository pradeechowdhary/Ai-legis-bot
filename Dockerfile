# Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    UVICORN_WORKERS=1 \
    STRICT_STATE=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY web ./web

# Build the CSV and FAISS index during image build (fastembed downloads ONNX here)
RUN python scripts/prepare_json_to_csv.py && python scripts/build_index.py

EXPOSE 8000
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
