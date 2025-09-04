# scripts/build_index.py
"""
Build sentence-transformer embeddings and a FAISS index from data/bills.csv.
Writes:
- vector_store/bills.npy      (embeddings)
- vector_store/bills.faiss    (FAISS index)
- vector_store/meta.csv       (bill metadata for lookup)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "bills.csv"
VEC_DIR = ROOT / "vector_store"
VEC_DIR.mkdir(parents=True, exist_ok=True)

INDEX = VEC_DIR / "bills.faiss"
EMBS = VEC_DIR / "bills.npy"
META = VEC_DIR / "meta.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    assert CSV.exists(), f"Missing {CSV}"
    df = pd.read_csv(CSV, dtype=str).fillna("")
    texts = df["text"].astype(str).tolist()

    model = SentenceTransformer(MODEL_NAME)
    X = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    # Save embeddings
    np.save(EMBS, X)

    # Build FAISS index (inner product for normalized vectors == cosine)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(INDEX))

    # Save lookup metadata
    meta_cols = ["id", "title", "state", "category", "date", "url", "status"]
    df[meta_cols].to_csv(META, index=False)

    print(f"Index size: {index.ntotal}, dim: {X.shape[1]}")
    print(f"Wrote {EMBS}, {INDEX}, {META}")

if __name__ == "__main__":
    main()
