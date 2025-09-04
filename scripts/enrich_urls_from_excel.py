# scripts/enrich_urls_from_excel.py
# Merge links from data/bill_urls.csv into data/bills.csv (overwrite bills.csv)
import pandas as pd, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BILLS = ROOT / "data" / "bills.csv"
URLS  = ROOT / "data" / "bill_urls.csv"   # put your Excel-extracted CSV here

def norm_state(x: str) -> str:
    return (x or "").strip().upper()

def base_bill(x: str) -> str:
    # normalize "S 1588", "H 890", "A 4030", etc. → "S1588", "H890", "A4030"
    s = (x or "").upper().replace("—"," ").replace("-", " ")
    m = re.search(r"\b([ASHEB]{1,2})\s*(\d{1,6})\b", s)  # A,S,H,SB,HB, etc.
    if m:
        return f"{m.group(1)}{m.group(2)}"
    # fallback: strip spaces
    return re.sub(r"\s+", "", s)

def main():
    assert BILLS.exists(), f"missing {BILLS}"
    assert URLS.exists(),  f"missing {URLS}"

    df = pd.read_csv(BILLS, dtype=str).fillna("")
    linkmap = pd.read_csv(URLS, dtype=str).fillna("")

    # bills.csv uses id for bill number/title-ish; build join keys
    df["state_key"] = df["state"].map(norm_state)
    df["bill_key"]  = df["id"].map(base_bill)

    # Excel file has bill_id column; build join keys
    if "bill_id" not in linkmap.columns:
        raise SystemExit("bill_urls.csv must have a 'bill_id' column")
    linkmap["state_key"] = linkmap["state"].map(norm_state) if "state" in linkmap.columns else ""
    linkmap["bill_key"]  = linkmap["bill_id"].map(base_bill)

    # Drop any empty keys to avoid bad joins
    linkmap = linkmap[(linkmap["bill_key"] != "") & (linkmap["state_key"] != "")]

    merged = df.merge(
        linkmap[["state_key","bill_key","url"]],
        on=["state_key","bill_key"],
        how="left",
        suffixes=("","_excel")
    )

    # prefer existing url if present, else take Excel url
    def pick(a, b): 
        a = (a or "").strip(); b = (b or "").strip()
        return a if a else b

    merged["url"] = [pick(a, b) for a, b in zip(merged.get("url",""), merged.get("url_excel",""))]
    merged = merged.drop(columns=["state_key","bill_key","url_excel"])

    merged.to_csv(BILLS, index=False)  # overwrite bills.csv so the rest of the pipeline picks it up
    print(f"Updated {BILLS} with URLs for { (merged['url']!='').sum() } rows out of {len(merged)}")

if __name__ == "__main__":
    main()
