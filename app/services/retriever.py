# app/services/retriever.py
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from .utils import safe_float, as_str, normalize_state

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "bills.csv"
VEC_DIR = ROOT / "vector_store"
INDEX = VEC_DIR / "bills.faiss"
EMBS = VEC_DIR / "bills.npy"
META = VEC_DIR / "meta.csv"

_MODEL = None
_INDEX = None
_META = None
_EMBS = None
_BILLS = None

def _load_all():
    global _MODEL, _INDEX, _META, _EMBS, _BILLS
    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    if _INDEX is None:
        _INDEX = faiss.read_index(str(INDEX))
    if _META is None:
        _META = pd.read_csv(META, dtype=str).fillna("")
    if _EMBS is None:
        _EMBS = np.load(EMBS).astype("float32")
    if _BILLS is None:
        _BILLS = pd.read_csv(DATA, dtype=str).fillna("")  # same row order as META/EMBS

def _snippet(text: str, max_chars: int = 600) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    # avoid cutting mid-word
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"

def _row_to_dict(i: int, sim):
    row = _META.iloc[i]
    text = _BILLS.iloc[i].get("text", "")
    return {
        "bill_id": as_str(row.get("id", "")),
        "title":   as_str(row.get("title", "")),
        "state":   as_str(row.get("state", "")),
        "category":as_str(row.get("category", "")),
        "date":    as_str(row.get("date", "")),
        "url":     as_str(row.get("url", "")),
        "snippet": _snippet(text),
        "sim":     safe_float(sim),
        "row_idx": i,
    }

def knn(query: str, k: int = 12):
    _load_all()
    qv = _MODEL.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, idxs = _INDEX.search(qv, k)
    return [_row_to_dict(int(i), float(s)) for s, i in zip(scores[0], idxs[0])]

def knn_state(query: str, state: str, k: int = 12):
    _load_all()
    state_code = normalize_state(state)
    if not state_code:
        return []
    mask = _META["state"].fillna("").apply(lambda x: normalize_state(x) == state_code).to_numpy()
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return []
    qv = _MODEL.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")[0]
    Xs = _EMBS[idxs]
    sims = Xs @ qv
    k_eff = min(k, idxs.size)
    top = np.argpartition(-sims, k_eff - 1)[:k_eff]
    top_sorted = top[np.argsort(-sims[top])]
    return [_row_to_dict(int(idxs[int(ti)]), float(sims[int(ti)])) for ti in top_sorted]
