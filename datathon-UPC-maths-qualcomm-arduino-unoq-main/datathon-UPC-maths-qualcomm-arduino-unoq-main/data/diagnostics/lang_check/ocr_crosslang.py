#!/usr/bin/env python3
"""OCR the options of text_only problems in 5 languages and compare them per letter.

Uses the team's own parse_options from laptop/prepare_m3kang.py, so the comparison
is exactly what their pipeline would read for each language.
"""
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, r"C:\Users\marti\qualcomm-panic\datathon-UPC-maths-qualcomm-arduino-unoq-main\datathon-UPC-maths-qualcomm-arduino-unoq-main\laptop")
from prepare_m3kang import parse_options  # noqa: E402

from rapidocr_onnxruntime import RapidOCR  # noqa: E402

CACHE = os.path.expanduser(r"~\.cache\huggingface\hub\datasets--qualcomm--M3Kang\snapshots")
LANGS = ["eng", "cat", "fra", "esp", "deu"]
N_IDS = 60

shard = glob.glob(os.path.join(CACHE, "*", "data", "*.parquet"))[0]
t = pq.read_table(shard, columns=["id", "language", "label", "text_only", "image"])
rows = [r for r in t.to_pylist() if r["text_only"]]

have = defaultdict(dict)
for r in rows:
    if r["language"] in LANGS:
        have[r["id"]][r["language"]] = r
full_ids = [i for i, d in have.items() if len(d) == len(LANGS)]
print(f"text_only ids with all {len(LANGS)} langs: {len(full_ids)}")

random.seed(42)
sample = random.sample(full_ids, min(N_IDS, len(full_ids)))

ocr = RapidOCR()
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_cache.jsonl")
parsed_cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        for line in f:
            r = json.loads(line)
            parsed_cache[r["key"]] = r["options"]
cache_f = open(CACHE_FILE, "a")

def ocr_options(pid, lang, img):
    key = f"{pid}|{lang}"
    if key in parsed_cache:
        return parsed_cache[key]
    boxes, _ = ocr(img)
    options = parse_options(boxes or [])
    parsed_cache[key] = options
    cache_f.write(json.dumps({"key": key, "options": options}, ensure_ascii=False) + "\n")
    cache_f.flush()
    return options

def norm(v):
    s = re.sub(r"[\s€$]", "", v).replace(",", ".").rstrip("%")
    s = s.rstrip(".")
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        f = float(s)
        return f"{f:.6f}".rstrip("0").rstrip(".")
    return s.lower()

n_ok, n_mismatch, n_notparsed = 0, 0, 0
mismatches = []
label_value_match = Counter()
for k, pid in enumerate(sample):
    parsed = {}
    for lang in LANGS:
        parsed[lang] = ocr_options(pid, lang, have[pid][lang]["image"]["bytes"])
    if any(len(o) != 5 for o in parsed.values()):
        n_notparsed += 1
        continue
    ref = {L: norm(v) for L, v in parsed["eng"].items()}
    same = True
    for lang in LANGS[1:]:
        cur = {L: norm(v) for L, v in parsed[lang].items()}
        if cur != ref:
            same = False
            diffs = {L: (parsed["eng"][L], parsed[lang][L]) for L in "ABCDE" if cur[L] != ref[L]}
            mismatches.append({"id": pid, "lang": lang, "diff": diffs,
                               "label": have[pid]["eng"]["label"]})
    # does option[label] agree across languages?
    lab = have[pid]["eng"]["label"]
    vals = {lang: norm(parsed[lang][lab]) for lang in LANGS if len(parsed[lang]) == 5}
    label_value_match[len(set(vals.values())) == 1] += 1
    n_ok += same
    if (k + 1) % 10 == 0:
        print(f"  {k + 1}/{len(sample)} checked", flush=True)

print(f"\nCompared {len(sample) - n_notparsed} text_only ids in {len(LANGS)} langs (same OCR parser):")
print(f"  not fully OCR-parsed in every language: {n_notparsed}")
print(f"  options identical (normalised) in all languages: {n_ok}")
print(f"  at least one language with different options: {len(mismatches)}")
print(f"  value at the label letter identical across languages: {label_value_match[True]}/{sum(label_value_match.values())}")
if mismatches:
    print("\nMismatches (first 15):")
    for m in mismatches[:15]:
        print(json.dumps(m, ensure_ascii=False))
