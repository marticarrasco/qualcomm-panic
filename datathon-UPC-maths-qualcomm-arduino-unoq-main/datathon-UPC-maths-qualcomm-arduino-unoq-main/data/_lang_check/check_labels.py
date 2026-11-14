#!/usr/bin/env python3
"""Check whether the same M3Kang problem id has the same label across languages.

Reads only id/language/label/text_only/level/text columns from the cached
high_quality shard(s) (the image column is skipped, so this is fast).
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import pyarrow.parquet as pq

CACHE = os.path.expanduser(r"~\.cache\huggingface\hub\datasets--qualcomm--M3Kang\snapshots")
shards = glob.glob(os.path.join(CACHE, "*", "data", "*.parquet"))
print("Shards found:", [os.path.basename(p) for p in shards])

rows = []
for path in shards:
    t = pq.read_table(path, columns=["id", "language", "label", "level", "text_only"])
    d = t.to_pylist()
    print(f"{os.path.basename(path)}: {len(d)} rows")
    rows.extend(d)
print("Total rows in cached shards:", len(rows))

langs = Counter(r["language"] for r in rows)
print("Languages:", len(langs))
print(sorted(langs.items(), key=lambda x: -x[1])[:20])

# label consistency per id
by_id = defaultdict(dict)
for r in rows:
    by_id[r["id"]].setdefault(r["language"], r["label"])

ids_multi = {i: d for i, d in by_id.items() if len(d) > 1}
conflicts = {i: d for i, d in ids_multi.items() if len(set(d.values())) > 1}
print(f"\nIDs seen in >1 language: {len(ids_multi)} / {len(by_id)}")
print(f"IDs with CONFLICTING labels across languages: {len(conflicts)}")
if conflicts:
    for i, d in list(conflicts.items())[:10]:
        print("  CONFLICT", i, d)

# per-language coverage of a few languages of interest
focus = ["eng", "cat", "fra", "esp", "deu", "ita"]
for lang in focus:
    subset = {i for i, d in ids_multi.items() if lang in d}
    print(f"  {lang}: {langs.get(lang, 0)} rows in shard(s)")

# candidate sample: problems present in eng,cat,fra,esp (+deu if possible),
# spread over levels/years, mix of text_only, and probably numeric options
cands = []
for i, d in sorted(ids_multi.items()):
    if all(l in d for l in ("eng", "cat", "fra", "esp")):
        cands.append(i)
print(f"\nIDs present in eng+cat+fra+esp: {len(cands)}")

# spread: pick 10 with distinct (level, year), preferring more languages present
meta = {r["id"]: r for r in rows if r["language"] == "eng"}
picked, seen = [], set()
for i in cands:
    m = meta.get(i)
    if not m:
        continue
    key = (m["level"], i.split("_")[1])
    if key in seen:
        continue
    seen.add(key)
    picked.append((i, m["level"], i.split("_")[1], m["text_only"], len(by_id[i])))
    if len(picked) >= 14:
        break
print("\nCandidate sample (id, level, year, text_only, #langs):")
for p in picked:
    print("  ", p)
