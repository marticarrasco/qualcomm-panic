#!/usr/bin/env python3
"""Classify cached cross-language OCR options: numeric vs text-like, per language."""
import json
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(HERE, "ocr_cache.jsonl")

SYMBOLS = r"0-9+\-*/=()\[\],.:;%€$°²³·–—"
UNITS = r"(cm|km|kg|g|m|s|mm|l|ml|h|min|º)"
def kind(v):
    s = re.sub(rf"[{SYMBOLS}\s]", "", v)
    s = re.sub(UNITS, "", s, flags=re.IGNORECASE)
    if s.lower().replace("l", "").strip("√¼½¾") == "":
        return "num"
    return "TEXT" if re.search(r"[A-Za-z\u00c0-\u017f]", s) else "num"

by_id = defaultdict(dict)
for line in open(CACHE_FILE, encoding="utf-8", errors="replace"):
    r = json.loads(line)
    pid, lang = r["key"].split("|")
    by_id[pid][lang] = r["options"]

counts = Counter()
text_examples = []
for pid, langs in by_id.items():
    for lang, o in langs.items():
        if not o:
            counts.setdefault("unparsed", Counter())[lang] += 1
            continue
        kinds = {L: kind(v) for L, v in o.items()}
        if any(k == "TEXT" for k in kinds.values()):
            counts.setdefault("has-text", Counter())[lang] += 1
            text_examples.append((pid, lang, o))
        else:
            counts.setdefault("numeric", Counter())[lang] += 1

print("cached (id, lang) pairs:", sum(counts[c][l] for c in counts for l in counts[c]))
for cat in ("numeric", "has-text", "unparsed"):
    print(f"  {cat:9}: " + ", ".join(f"{l}={n}" for l, n in sorted(counts.get(cat, {}).items())))

if text_examples:
    print("\nTEXT-containing option sets:")
    for pid, lang, o in text_examples:
        print(f"  {pid} [{lang}]: {json.dumps(o, ensure_ascii=False)}")
