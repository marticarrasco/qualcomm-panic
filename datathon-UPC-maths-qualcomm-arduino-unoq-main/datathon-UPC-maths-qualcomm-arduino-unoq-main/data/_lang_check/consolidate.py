#!/usr/bin/env python3
"""Consolidate the full OCR cache: option equality across languages + label-letter check.

Recomputes everything from ocr_cache.jsonl (no new OCR), classifies option content,
and dumps contact sheets for any id whose label-letter value differs across languages.
"""
import io
import json
import os
import re
import sys
import glob
from collections import Counter, defaultdict

import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
LANGS = ["eng", "cat", "fra", "esp", "deu"]

SYMBOLS = r"0-9+\-*/=()\[\],.:;%€$°²³·–—"
UNITS = r"(cm|km|kg|g|m|s|mm|l|ml|h|min|º)"

def norm(v):
    s = re.sub(r"[\s\u00a0\u202f€$]", "", v).replace(",", ".").rstrip("%")
    s = s.replace("\u2212", "-").replace("\u2116", "no").replace("№", "no")
    s = s.replace("[", "").replace("]", "").rstrip(".")
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return f"{float(s):.6f}".rstrip("0").rstrip(".")
    return s.lower()

def kind(v):
    s = re.sub(rf"[{SYMBOLS}\s\u00a0\u202f]", "", v)
    s = re.sub(UNITS, "", s, flags=re.IGNORECASE)
    return "TEXT" if re.search(r"[A-Za-z\u00c0-\u017f]", s) else "num"

by_id = defaultdict(dict)
for line in open(os.path.join(HERE, "ocr_cache.jsonl"), encoding="utf-8", errors="replace"):
    r = json.loads(line)
    pid, lang = r["key"].split("|")
    by_id[pid][lang] = r["options"]

cache = os.path.expanduser(r"~\.cache\huggingface\hub\datasets--qualcomm--M3Kang\snapshots")
t = pq.read_table(glob.glob(os.path.join(cache, "*", "data", "*.parquet"))[0],
                  columns=["id", "language", "label"])
labels = {}
for r in t.to_pylist():
    labels[(r["id"], r["language"])] = r["label"]

n_full, n_same, n_diff, n_labdiff = 0, 0, 0, 0
text_ids = Counter()
labdiffs, diffs = [], []
for pid, per_lang in by_id.items():
    if any(len(o) != 5 or not o for o in per_lang.values()):
        continue
    if any(l not in per_lang for l in LANGS):
        continue
    n_full += 1
    ref = {L: norm(v) for L, v in per_lang["eng"].items()}
    lab = labels.get((pid, "eng"))
    same = True
    for lang in LANGS[1:]:
        cur = {L: norm(v) for L, v in per_lang[lang].items()}
        if cur != ref:
            same = False
            bad = {L: (per_lang["eng"][L], per_lang[lang][L]) for L in "ABCDE" if cur[L] != ref[L]}
            if lab and lab in bad:
                n_labdiff += 1
                labdiffs.append((pid, lang, lab, bad[lab][0], bad[lab][1]))
            diffs.append({"id": pid, "lang": lang, "label": lab, "diff": bad})
    n_same += same
    for lang, o in per_lang.items():
        if any(kind(v) == "TEXT" for v in o.values()):
            text_ids[f"{pid}|{lang}"] += 1
n_diff = n_full - n_same

print(f"ids fully OCR-parsed in all {len(LANGS)} languages: {n_full}")
print(f"  all 5 options normalised-identical across languages: {n_same}")
print(f"  some option differs in some language:                {n_diff}")
print(f"  where the *label letter value* differs:              {n_labdiff}")
vals = Counter()
for pid, per_lang in by_id.items():
    if any(len(o) != 5 for o in per_lang.values()) or any(l not in per_lang for l in LANGS):
        continue
    lab = labels.get((pid, "eng"))
    vs = {norm(per_lang[l][lab]) for l in LANGS if (pid, l) in labels or l in per_lang}
    vals[len(vs) == 1] += 1
print(f"  label-letter value identical across langs (all cached ids): {vals[True]}/{sum(vals.values())}")

print(f"\nids with any TEXT-looking option value (per id|lang): {len(text_ids)}")
for k in sorted(text_ids):
    print("  ", k)

print("\nlabel-letter value mismatches (eng vs other):")
for d in labdiffs:
    print("  ", d)

print("\nall option diffs (maybe OCR noise):", len(diffs))
with open(os.path.join(HERE, "diffs.json"), "w", encoding="utf-8") as f:
    json.dump(diffs, f, ensure_ascii=False, indent=1)
for d in diffs[:20]:
    print("  ", json.dumps(d, ensure_ascii=False))
