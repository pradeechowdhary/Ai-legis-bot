# Dockerfile
FROM python:3.11-slim

# (Faiss wheel needs OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    EMB_MODEL_LOCAL=/app/models/all-MiniLM-L6-v2 \
    GROQ_MODEL=llama-3.3-70b-versatile \
    STRICT_STATE=1 \
    PORT=8000

WORKDIR /app

# Copy and install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app ./app
COPY scripts ./scripts
COPY data ./data
# (optional) if your index.html lives at repo root, also copy it:
# COPY index.html ./public/index.html

# Fetch embedder + build data/index during image build
RUN python scripts/fetch_model.py \
 && python scripts/prepare_json_to_csv.py \
 && python scripts/build_index.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
