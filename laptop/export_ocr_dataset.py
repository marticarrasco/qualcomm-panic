#!/usr/bin/env python3
"""Rebuild the train/validation JSONL files from OCR results already in cache.

Only problems whose OCR has been run are exported. The default split matches
prepare_m3kang.py: a stable SHA-256 hash of the problem ID assigns 15% to val.
The export replaces data/train.jsonl and data/val.jsonl so eval.py and
build_index.py use the accumulated dataset by default.
"""
import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", help="local M3Kang parquet; defaults to cached high_quality shard 0")
    parser.add_argument("--languages", default="", help="comma-separated languages; default is all cached languages")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="share of problem IDs assigned to validation (default: 0.15)")
    parser.add_argument("--out", default="data", help="directory for train.jsonl and val.jsonl")
    return parser.parse_args()


def find_parquet(value):
    if value:
        path = Path(value).resolve()
        if not path.is_file():
            raise SystemExit(f"No existeix el parquet: {path}")
        return path
    cache = Path.home() / ".cache/huggingface/hub/datasets--qualcomm--M3Kang/snapshots"
    paths = sorted(cache.rglob("high_quality-00000-of-00004.parquet"))
    if not paths:
        raise SystemExit("No trobo el shard local; descarrega M3Kang o passa --parquet <fitxer>.")
    return paths[-1]


def read_caches():
    caches = {}
    paths = set(DATA.rglob("*ocr_cache.jsonl"))
    for path in sorted(paths):
        for raw in path.read_bytes().splitlines():
            if not raw.strip():
                continue
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                line = raw.decode("cp1252")
            row = json.loads(line)
            caches[row["key"]] = row.get("options") or {}
    return caches


def read_corrections():
    path = DATA / "ocr_review.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as source:
        review_data = json.load(source)
    return {
        key: row.get("options") or {}
        for key, row in review_data.get("reviews", {}).items()
        if row.get("reviewed") and row.get("corrected")
    }


def atomic_jsonl(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def export_dataset(parquet_path=None, languages=None, val_fraction=0.15, out_dir=None):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction ha d'estar entre 0 i 1")
    parquet_path = find_parquet(parquet_path)
    cache = read_caches()
    corrections = read_corrections()
    wanted_languages = set(languages or ())

    table = pq.read_table(parquet_path, columns=[
        "id", "language", "level", "text_only", "label", "text",
    ])
    rows = []
    unique_keys = set()
    cache_success = set()
    labels_by_split = Counter()
    for i in range(table.num_rows):
        row = {name: table.column(name)[i].as_py() for name in table.column_names}
        language = row["language"]
        key = f"{row['id']}|{language}"
        if (wanted_languages and language not in wanted_languages) or key not in cache:
            continue
        if key in unique_keys:
            continue
        unique_keys.add(key)
        options = corrections.get(key, cache[key])
        if options:
            cache_success.add(key)
        year = int(row["id"].split("_")[1])
        bucket = int(hashlib.sha256(row["id"].encode()).hexdigest(), 16) % 1000
        split = "val" if bucket < val_fraction * 1000 else "train"
        record = {
            "id": row["id"], "language": language, "level": row["level"],
            "year": year, "text_only": row["text_only"], "text": row["text"].strip(),
            "options": options, "label": row["label"],
        }
        rows.append((split, record))
        labels_by_split[(split, language)] += 1

    if not rows:
        raise SystemExit("No hi ha resultats OCR de la memòria cau per als idiomes demanats.")

    out_dir = Path(out_dir or DATA).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train = [row for split, row in rows if split == "train"]
    val = [row for split, row in rows if split == "val"]
    atomic_jsonl(out_dir / "train.jsonl", train)
    atomic_jsonl(out_dir / "val.jsonl", val)

    by_language = Counter(row["language"] for _, row in rows)
    print(f"OCR al parquet: {parquet_path}")
    print(f"Problemes amb OCR exportats: {len(rows)} ({len(cache_success)} amb opcions llegides, {len(rows)-len(cache_success)} sense opcions)")
    for language in sorted(by_language):
        print(f"  {language}: train={labels_by_split[('train', language)]}, val={labels_by_split[('val', language)]}")
    print(f"Escrit: {out_dir / 'train.jsonl'} ({len(train)})")
    print(f"Escrit: {out_dir / 'val.jsonl'} ({len(val)})")
    return len(rows), len(train), len(val)


def main():
    args = parse_args()
    languages = set(filter(None, args.languages.split(",")))
    export_dataset(args.parquet, languages, args.val_fraction, args.out)


if __name__ == "__main__":
    main()
