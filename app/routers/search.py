# app/routers/search.py
from fastapi import APIRouter, Query
from typing import List
from app.models import SearchHit
from app.services.retriever import knn
from app.services.utils import safe_float

router = APIRouter()

@router.get("", response_model=List[SearchHit])
def search(q: str = Query(...), top_k: int = 8):
    raw = knn(q, k=top_k)
    out = []
    for r in raw:
        out.append(SearchHit(
            bill_id=r["bill_id"],
            title=r["title"],
            state=r["state"],
            category=r["category"],
            date=r.get("date", ""),
            url=r.get("url", ""),
            score=safe_float(r["sim"]),
        ))
    return out
