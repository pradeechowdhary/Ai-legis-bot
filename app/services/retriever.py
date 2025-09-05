# app/services/retriever.py
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from fastembed import TextEmbedding
from .utils import safe_float, as_str, normalize_state

ROOT   = Path(__file__).resolve().parents[2]
DATA   = ROOT / "data" / "bills.csv"     # for snippets
VEC_DIR= ROOT / "vector_store"
INDEXP = VEC_DIR / "bills.faiss"
METAP  = VEC_DIR / "meta.csv"

_EMBED = None
_INDEX = None
_META  = None
_BILLS = None
_DIM   = None

def _load_all():
    global _EMBED, _INDEX, _META, _BILLS, _DIM
    if _EMBED is None:
        _EMBED = TextEmbedding("intfloat/e5-small-v2")
    if _INDEX is None:
        _INDEX = faiss.read_index(str(INDEXP))
        _DIM = _INDEX.d  # query vector dimension
    if _META is None:
        _META = pd.read_csv(METAP, dtype=str).fillna("")
    if _BILLS is None:
        _BILLS = pd.read_csv(DATA, dtype=str).fillna("")

def _norm(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def _snippet(text: str, max_chars: int = 600) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= max_chars: return s
    cut = s[:max_chars]
    if " " in cut: cut = cut.rsplit(" ", 1)[0]
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
        "row_idx": int(i),
    }

def _encode_one(text: str) -> np.ndarray:
    v = np.array(list(_EMBED.embed([text], batch_size=1)), dtype="float32")
    return _norm(v)

def knn(query: str, k: int = 12):
    _load_all()
    qv = _encode_one(query)
    scores, idxs = _INDEX.search(qv, k)
    return [_row_to_dict(int(i), float(s)) for s, i in zip(scores[0], idxs[0])]

def knn_state(query: str, state: str, k: int = 12):
    _load_all()
    code = normalize_state(state)
    if not code:
        return []
    # Search wider, then filter by state
    qv = _encode_one(query)
    widen = max(k * 10, 200)
    scores, idxs = _INDEX.search(qv, min(widen, _INDEX.ntotal))
    out = []
    for s, i in zip(scores[0], idxs[0]):
        if normalize_state(_META.iloc[int(i)].get("state", "")) == code:
            out.append(_row_to_dict(int(i), float(s)))
            if len(out) >= k: break
    return out
