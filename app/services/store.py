# app/services/store.py
import json
import uuid
from pathlib import Path

DB = Path(__file__).resolve().parents[2] / "data" / "sessions.json"
DB.parent.mkdir(parents=True, exist_ok=True)
if not DB.exists():
    DB.write_text("{}", encoding="utf-8")

def save_profile(data: dict) -> str:
    sid = str(uuid.uuid4())
    all_ = json.loads(DB.read_text(encoding="utf-8"))
    all_[sid] = data
    DB.write_text(json.dumps(all_, ensure_ascii=False), encoding="utf-8")
    return sid

def get_profile(session_id: str) -> dict | None:
    all_ = json.loads(DB.read_text(encoding="utf-8"))
    return all_.get(session_id)
