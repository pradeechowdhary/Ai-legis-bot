# app/services/utils.py
import math

# app/services/utils.py (append)
US_STATE_MAP = {
    "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA","colorado":"CO",
    "connecticut":"CT","delaware":"DE","florida":"FL","georgia":"GA","hawaii":"HI","idaho":"ID",
    "illinois":"IL","indiana":"IN","iowa":"IA","kansas":"KS","kentucky":"KY","louisiana":"LA",
    "maine":"ME","maryland":"MD","massachusetts":"MA","michigan":"MI","minnesota":"MN","mississippi":"MS",
    "missouri":"MO","montana":"MT","nebraska":"NE","nevada":"NV","new hampshire":"NH","new jersey":"NJ",
    "new mexico":"NM","new york":"NY","north carolina":"NC","north dakota":"ND","ohio":"OH","oklahoma":"OK",
    "oregon":"OR","pennsylvania":"PA","rhode island":"RI","south carolina":"SC","south dakota":"SD",
    "tennessee":"TN","texas":"TX","utah":"UT","vermont":"VT","virginia":"VA","washington":"WA",
    "west virginia":"WV","wisconsin":"WI","wyoming":"WY","district of columbia":"DC","dc":"DC"
}
def normalize_state(s: str | None) -> str:
    if not s: return ""
    s = str(s).strip()
    if len(s) == 2 and s.isalpha():
        return s.upper()
    key = s.lower()
    return US_STATE_MAP.get(key, s.upper())  # fallback to uppercased original


def safe_float(x, default=0.0) -> float:
    """Return a finite float; coerce NaN/Inf/err to default."""
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)

def is_nan_like(x) -> bool:
    if x is None:
        return True
    try:
        if isinstance(x, float) and math.isnan(x):
            return True
    except Exception:
        pass
    try:
        import pandas as pd  # optional
        return bool(pd.isna(x))
    except Exception:
        # Fallback string tests
        s = str(x).strip().lower()
        return s in {"nan", "none", "null", ""}
        
def as_str(x: object, default: str = "") -> str:
    """Normalize any value to a clean string without NaN."""
    return default if is_nan_like(x) else str(x).strip()
