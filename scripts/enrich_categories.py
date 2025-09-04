# scripts/enrich_categories.py
"""
Heuristically fill/augment the `category` column in data/bills.csv.
Writes back to the same CSV. Then re-run build_index.py to refresh meta.csv.
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "bills.csv"

RULES = [
    # Employment / AEDT
    (r"\b(hiring|employment|aedt|screening|recruit|background check|bias audit|employment decision tool)\b",
     {"Effect on Labor/Employment", "Private Sector Use", "Impact Assessment"}),

    # Oversight / governance bodies, councils, commissions
    (r"\b(advisory council|council|commission|board|task force|oversight|governance)\b",
     {"Oversight/Governance"}),

    # Definitions / general AI regulation in code
    (r"\b(define|definition of artificial intelligence|regulation of artificial intelligence|ai technology)\b",
     {"Definitions", "Oversight/Governance"}),

    # Studies / reports
    (r"\b(study|report|tacir|working group|recommendations)\b",
     {"Studies"}),

    # Education sector
    (r"\b(education|school|student|university|district|curriculum)\b",
     {"Education Use"}),

    # Synthetic media / deepfakes / provenance
    (r"\b(deepfake|synthetic media|content credential|watermark|provenance)\b",
     {"Provenance", "Notification"}),

    # Content safety / child exploitation
    (r"\b(sexual exploitation of children|child sexual|csam|obscenity)\b",
     {"Content Safety", "Criminal"}),

    # Cybersecurity
    (r"\b(cybersecurity|breach|critical infrastructure|ransomware)\b",
     {"Cybersecurity"}),

    # Government use
    (r"\b(government use|agency use|public body)\b",
     {"Government Use"}),

    # Notification / disclosures generally
    (r"\b(notification|notice to (consumers|applicants)|disclosure)\b",
     {"Notification"}),

    # Impact assessments generally
    (r"\b(impact assessment|risk assessment|consequence assessment)\b",
     {"Impact Assessment"}),

    # Appropriations
    (r"\b(appropriation|appropriated|budget)\b",
     {"Appropriations"}),

    # Private right of action
    (r"\b(private right of action|cause of action|civil action)\b",
     {"Private Right of Action"}),
]

def categorize(text: str) -> set[str]:
    s = (text or "").lower()
    cats: set[str] = set()
    for pat, tags in RULES:
        if re.search(pat, s):
            cats |= tags
    return cats

def merge(existing: str, inferred: set[str]) -> str:
    exist = {c.strip() for c in (existing or "").split(";") if c.strip()}
    allcats = sorted(exist | inferred)
    return ";".join(allcats)

def main():
    df = pd.read_csv(CSV, dtype=str).fillna("")
    out = []
    for _, r in df.iterrows():
        blob = f"{r.get('title','')} {r.get('text','')}"
        inferred = categorize(blob)
        r["category"] = merge(r.get("category",""), inferred)
        out.append(r)
    newdf = pd.DataFrame(out)
    newdf.to_csv(CSV, index=False)
    print(f"Updated categories in {CSV}; rows: {len(newdf)}")

if __name__ == "__main__":
    main()
