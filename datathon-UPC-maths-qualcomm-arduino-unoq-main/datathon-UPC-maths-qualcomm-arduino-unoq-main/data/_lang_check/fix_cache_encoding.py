#!/usr/bin/env python3
"""Rewrite ocr_cache.jsonl as clean UTF-8 (older lines were written with cp1252)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(HERE, "ocr_cache.jsonl")

lines, bad = [], 0
raw = open(CACHE_FILE, "rb").read()
for chunk in raw.split(b"\n"):
    if not chunk.strip():
        continue
    for enc in ("utf-8", "cp1252"):
        try:
            text = chunk.decode(enc)
            if enc != "utf-8":
                bad += 1
            break
        except UnicodeDecodeError:
            continue
    else:
        text = chunk.decode("utf-8", "replace")
        bad += 1
    lines.append(text)

# validate + dedupe by key, keep last
seen = {}
n_ok = 0
for text in lines:
    try:
        rec = json.loads(text)
        seen[rec["key"]] = rec
        n_ok += 1
    except Exception:
        bad += 1

with open(CACHE_FILE, "w", encoding="utf-8") as f:
    for rec in seen.values():
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"rewrote {len(seen)} unique entries ({n_ok} parsed lines, {bad} problem lines fixed/dropped)")
