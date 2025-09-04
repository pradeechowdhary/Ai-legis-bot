# app/services/ranker.py
from datetime import datetime
import math

def recency_boost(date_str: str) -> float:
    if not date_str:
        return 0.0
    try:
        d = datetime.fromisoformat(date_str).date()
    except Exception:
        return 0.0
    days = (datetime.utcnow().date() - d).days
    # ~18-month half-life
    return math.exp(-days / 540.0)

def final_score(sim: float, same_state: bool, cat_match: bool, date_str: str) -> float:
    # Stronger jurisdiction preference so user’s state ranks first
    return (
        0.45 * sim
        + 0.40 * (1.0 if same_state else 0.0)
        + 0.10 * (1.0 if cat_match else 0.0)
        + 0.05 * recency_boost(date_str)
    )
