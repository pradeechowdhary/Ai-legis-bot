# scripts/prepare_json_to_csv.py
"""
Convert structured_aibills.json --> data/bills.csv
Creates a clean, deduped table with columns:
id,title,state,category,date,url,text,status

- Stronger title fallback: uses bill_id or first words of text
- Flexible date parsing with dateutil (handles many formats)
- Dedupes by (bill_id,state) or (title,state,date), keeping the newest date
- Normalizes NaN/None to empty strings
- Flattens categories to a semicolon-separated string
- Includes bill status when present
- Pulls best URL from common fields or nested link arrays
- Works if JSON root is a list, {"records":[...]}, or JSONL (one JSON per line)
"""

import json
import csv
import re
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from dateutil import parser as dtparser

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "structured_aibills.json"
OUT = ROOT / "data" / "bills.csv"

# Candidate keys that often occur across sources
DATE_KEYS = [
    "version_date", "introduced_date", "last_action_date", "date",
    "updated_at", "created_at", "effective_date", "signed_date", "enacted_date"
]
TEXT_KEYS = ["text", "bill_text", "body", "full_text", "content"]
TITLE_KEYS = ["title", "bill_title", "name", "short_title"]
URL_KEYS = ["url", "source_url", "document_url", "link", "bill_url", "pdf_url", "html_url"]
STATE_KEYS = ["state", "jurisdiction", "us_state", "region"]
ID_KEYS = ["bill_id", "id", "bill_number", "number", "slug"]
CATEGORY_KEYS = ["categories", "category", "tags", "labels"]
STATUS_KEYS = ["status", "bill_status", "stage", "current_status"]

CUT_TOKENS = ["Author:", "Version:", "Version Date:", "HOUSE", "SENATE", "ASSEMBLY", "STATE"]

def clean_title_for_csv(bill_id: str, title: str) -> str:
    t = (title or "").strip()
    if bill_id:
        # drop duplicated bill id prefixes like "S 2530 — S 2530 — ..."
        t = re.sub(rf"^{re.escape(bill_id)}\s*—\s*", "", t)
        t = re.sub(rf"^{re.escape(bill_id)}\b[:,\s-]*", "", t)
    # cut noisy metadata tails
    for tok in CUT_TOKENS:
        pos = t.find(tok)
        if pos > 0:
            t = t[:pos]
            break
    return t.strip(" ,;—-")

def is_nan_like(x: Any) -> bool:
    if x is None:
        return True
    try:
        if isinstance(x, float) and math.isnan(x):
            return True
    except Exception:
        pass
    s = str(x).strip().lower()
    return s in {"nan", "none", "null", ""}

def as_str(x: Any, default: str = "") -> str:
    return default if is_nan_like(x) else str(x).strip()

def coalesce(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and not is_nan_like(d[k]):
            return d[k]
    return None

def parse_date_any(v: Any) -> Optional[str]:
    if is_nan_like(v):
        return None
    s = str(v)
    try:
        return dtparser.parse(s, fuzzy=True).date().isoformat()
    except Exception:
        m = re.search(r"(19|20)\d{2}[-/\.](\d{1,2})[-/\.](\d{1,2})", s)
        if m:
            y = int(m.group(0)[0:4])
            mo = int(m.group(2))
            da = int(m.group(3))
            return f"{y:04d}-{mo:02d}-{da:02d}"
    return None

def best_date(rec: Dict[str, Any]) -> Optional[str]:
    for k in DATE_KEYS:
        if k in rec and not is_nan_like(rec[k]):
            d = parse_date_any(rec[k])
            if d:
                return d
    return None

def clean_text(x: Any) -> str:
    t = as_str(x)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def first_words(text: str, n: int = 12) -> str:
    parts = text.split()
    return " ".join(parts[:n])

def ensure_list(x: Any) -> List[str]:
    if is_nan_like(x):
        return []
    if isinstance(x, list):
        return [as_str(i) for i in x if not is_nan_like(i)]
    return [as_str(x)]

def normalize_categories(rec: Dict[str, Any]) -> List[str]:
    for k in CATEGORY_KEYS:
        if k in rec and not is_nan_like(rec[k]):
            cats = ensure_list(rec[k])
            cats = [c.strip().lower() for c in cats if c.strip()]
            return cats
    return []

def title_fallback(bill_id: str, text: str) -> str:
    head = first_words(text, 12) if text else ""
    if bill_id and head:
        return f"{bill_id} — {head}"
    if bill_id:
        return bill_id
    if head:
        return head
    return "(untitled)"

def best_url(rec: Dict[str, Any]) -> str:
    """Pick the best URL from common fields or nested link arrays."""
    u = coalesce(rec, URL_KEYS)
    if u:
        return as_str(u)
    # also check nested lists like links/documents/versions
    for k in ("links", "urls", "documents", "versions"):
        v = rec.get(k)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    u2 = coalesce(item, ["url", "href", "link"])
                    if u2:
                        return as_str(u2)
    return ""

def iter_records_from_source(path: Path) -> Iterable[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
            for rec in data["records"]:
                if isinstance(rec, dict):
                    yield rec
            return
        if isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    yield rec
            return
        if isinstance(data, dict):
            yield data
            return
    except json.JSONDecodeError:
        pass
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    yield rec
            except Exception:
                continue

def main():
    assert SRC.exists(), f"Missing input file: {SRC}"
    OUT.parent.mkdir(parents=True, exist_ok=True)

    seen: Dict[Any, Dict[str, str]] = {}
    count_in = 0
    count_kept = 0

    for rec in iter_records_from_source(SRC):
        count_in += 1

        bill_id = as_str(coalesce(rec, ID_KEYS), "")
        title_raw = as_str(coalesce(rec, TITLE_KEYS), "")
        state = as_str(coalesce(rec, STATE_KEYS), "")
        url = best_url(rec)                         # <-- INSIDE the loop
        text_raw = coalesce(rec, TEXT_KEYS)
        text = clean_text(text_raw)
        if not text:
            continue

        date_iso = best_date(rec) or ""
        cats = normalize_categories(rec)
        category = ";".join(sorted(set(cats))) if cats else ""
        status = as_str(coalesce(rec, STATUS_KEYS), "")

        title = title_raw if title_raw and title_raw.lower() != "(untitled)" else title_fallback(bill_id, text)
        title = clean_title_for_csv(bill_id, title)

        # Dedup key: prefer (bill_id, state); else (title.lower(), state, date)
        if bill_id:
            key = ("id", bill_id, state)
        else:
            key = ("ttl", title.lower(), state, date_iso)

        prev = seen.get(key)
        if prev is None:
            seen[key] = {
                "id": bill_id or f"{title[:40]}_{state}",
                "title": title,
                "state": state,
                "category": category,
                "date": date_iso,
                "url": url,
                "text": text,
                "status": status,
            }
            count_kept += 1
        else:
            old_date = prev.get("date") or ""
            if date_iso and (not old_date or date_iso > old_date):
                prev.update({
                    "id": bill_id or prev["id"],
                    "title": title or prev["title"],
                    "state": state or prev["state"],
                    "category": category or prev["category"],
                    "date": date_iso,
                    "url": url or prev["url"],
                    "text": text or prev["text"],
                    "status": status or prev.get("status", ""),
                })

    rows = list(seen.values())

    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "state", "category", "date", "url", "text", "status"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "id": r.get("id", ""),
                "title": r.get("title", ""),
                "state": r.get("state", ""),
                "category": r.get("category", ""),
                "date": r.get("date", ""),
                "url": r.get("url", ""),
                "text": r.get("text", ""),
                "status": r.get("status", ""),
            })

    print(f"Wrote {len(rows)} rows to {OUT} (from {count_in} input records)")

if __name__ == "__main__":
    main()
