"""
Generate label mapping files from data/train_clean.csv
Writes:
 - data/label_map.json  (dict: label_index -> language_name)
 - data/labels.txt      (one language per line, ordered by label index)

Run:
 python3 scripts/generate_label_map.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "train_clean.csv"
OUT_JSON = ROOT / "data" / "label_map.json"
OUT_TXT = ROOT / "data" / "labels.txt"

if not TRAIN_CSV.exists():
    raise SystemExit(f"Missing {TRAIN_CSV}. Run this script from project root.")

mapping = {}
# Keep first seen language for each label
with TRAIN_CSV.open("r", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        # Expecting columns: client_id,path,language,label
        lang = row.get("language")
        label = row.get("label")
        if label is None or lang is None:
            continue
        try:
            li = int(label)
        except Exception:
            continue
        if li not in mapping:
            mapping[li] = lang

if not mapping:
    raise SystemExit("No label mapping found in train_clean.csv")

# Normalize: ensure consecutive ordering in labels.txt by label index ascending
ordered = [mapping[i] if i in mapping else "" for i in sorted(mapping.keys())]

# Write JSON as {"0":"Mangolian", "1":"Russian", ...}
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with OUT_JSON.open("w", encoding="utf-8") as jfh:
    json.dump({str(k): v for k, v in mapping.items()}, jfh, ensure_ascii=False, indent=2)

# Write labels.txt: index order by numeric label
with OUT_TXT.open("w", encoding="utf-8") as tfh:
    for idx in sorted(mapping.keys()):
        tfh.write(f"{mapping[idx]}\n")

print(f"Wrote {OUT_JSON} and {OUT_TXT} with {len(mapping)} labels")
