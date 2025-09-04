# scripts/fetch_model.py
import os
from sentence_transformers import SentenceTransformer

target = os.environ.get("EMB_MODEL_LOCAL", "/app/models/all-MiniLM-L6-v2")
os.makedirs(target, exist_ok=True)
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").save(target)
print("Saved model to", target)
