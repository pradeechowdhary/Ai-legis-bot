# scripts/build_index.py
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bills.csv"
VEC_DIR = ROOT / "vector_store"
INDEX = VEC_DIR / "bills.faiss"
META  = VEC_DIR / "meta.csv"

def _norm(a):
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return a / n

def main():
    assert DATA.exists(), f"Missing {DATA}"
    VEC_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA, dtype=str).fillna("")
    texts = df["text"].astype(str).tolist()

    emb = TextEmbedding("intfloat/e5-small-v2")  # small, no torch
    vecs = np.array(list(emb.embed(texts, batch_size=256)), dtype="float32")
    vecs = _norm(vecs)
    dim = vecs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(INDEX))

    # Just the lightweight metadata you need at runtime
    meta = df[["id","title","state","category","date","url"]].copy()
    meta.to_csv(META, index=False)

    print(f"Built FAISS index {INDEX} with {len(df)} items, dim={dim}")

if __name__ == "__main__":
    main()
