#!/usr/bin/env python3
"""Pixel-level comparison of M3Kang images across languages for figure problems.

For each sample id, compares each non-English image with the English one:
  - top strip    (mostly translated question text) -> should differ;
  - middle strip (figure)                          -> should be identical;
  - bottom strip (answer options)                  -> should be identical.
Reports mean absolute pixel difference (0-255 scale) per strip.
"""
import glob
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

CACHE = os.path.expanduser(r"~\.cache\huggingface\hub\datasets--qualcomm--M3Kang\snapshots")
LANGS = ["eng", "cat", "fra", "esp", "deu"]
N = 12

shard = glob.glob(os.path.join(CACHE, "*", "data", "*.parquet"))[0]
t = pq.read_table(shard, columns=["id", "language", "text_only", "image"])
rows = [r for r in t.to_pylist() if not r["text_only"] and r["language"] in LANGS]

have = defaultdict(dict)
for r in rows:
    have[r["id"]][r["language"]] = r
ids = [i for i, d in have.items() if len(d) == len(LANGS)]
random.seed(3)
sample = random.sample(ids, min(N, len(ids)))

agg = {"top": [], "mid": [], "bot": []}
print(f"{'id':16} {'lang':4} {'top':>7} {'mid':>7} {'bot':>7}")
for pid in sample:
    ref = Image.open(__import__("io").BytesIO(have[pid]["eng"]["image"]["bytes"])).convert("L")
    w, h = ref.size
    zones = {
        "top": ref.crop((0, 0, w, int(h * 0.35))),
        "mid": ref.crop((0, int(h * 0.35), w, int(h * 0.70))),
        "bot": ref.crop((0, int(h * 0.70), w, h)),
    }
    refa = {k: np.asarray(v, dtype=np.int16) for k, v in zones.items()}
    for lang in LANGS[1:]:
        if lang not in have[pid]:
            continue
        im = Image.open(__import__("io").BytesIO(have[pid][lang]["image"]["bytes"])).convert("L")
        if im.size != ref.size:
            im = im.resize(ref.size)
        for k, z in (("top", (0, 0, w, int(h * 0.35))),
                     ("mid", (0, int(h * 0.35), w, int(h * 0.70))),
                     ("bot", (0, int(h * 0.70), w, h))):
            cur = np.asarray(im.crop(z), dtype=np.int16)
            mad = float(np.abs(cur - refa[k]).mean())
            agg[k].append(mad)
            print(f"{pid:16} {lang:4} {mad:7.2f}", end="" if k == "top" else "")
            if k == "bot":
                print()
    print()

print("\nmean abs pixel diff (0-255): top=%.1f  mid=%.1f  bottom=%.1f  (n=%d id-lang pairs)" % (
    sum(agg["top"]) / len(agg["top"]), sum(agg["mid"]) / len(agg["mid"]),
    sum(agg["bot"]) / len(agg["bot"]), len(agg["bot"])))
