#!/usr/bin/env python3
"""Dump labeled contact sheets (grid layout) for 10 M3Kang problems x 5 languages."""
import glob
import io
import os
import sys
from collections import defaultdict

import pyarrow.parquet as pq
from PIL import Image, ImageDraw

CACHE = os.path.expanduser(r"~\.cache\huggingface\hub\datasets--qualcomm--M3Kang\snapshots")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sheets")
LANGS = ["eng", "cat", "fra", "esp", "deu"]
N = 10

shard = glob.glob(os.path.join(CACHE, "*", "data", "*.parquet"))[0]
t = pq.read_table(shard, columns=["id", "language", "label", "level", "text_only", "text", "image"])

rows = t.to_pylist()
have = defaultdict(set)
for r in rows:
    if r["language"] in LANGS:
        have[r["id"]].add(r["language"])
full_ids = [i for i, s in have.items() if len(s) == len(LANGS)]

meta = {r["id"]: r for r in rows if r["language"] == "eng"}
# pick one id per (level, text_only) combination, spread over years
picks = []
seen = set()
for i in sorted(full_ids):
    m = meta.get(i)
    if not m:
        continue
    key = (m["level"], m["text_only"])
    if key in seen:
        continue
    seen.add(key)
    picks.append((i, m))
picks = picks[0::2] + [picks[1], picks[-1]]  # 8 figure problems (one per level) + 2 text-only, levels 0..7
print("picked:")
for i, m in picks:
    print(f"  {i}  level={m['level']} text_only={m['text_only']} year={i.split('_')[1]} langs={len(have[i])}")
data = {}
for r in rows:
    if r["id"] in [p[0] for p in picks] and r["language"] in LANGS:
        data[(r["id"], r["language"])] = r

os.makedirs(OUT, exist_ok=True)
for pid, m in picks:
    cells = []
    for lang in LANGS:
        r = data[(pid, lang)]
        im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        canvas = Image.new("RGB", (im.width, im.height + 22), "black")
        canvas.paste(im, (0, 22))
        ImageDraw.Draw(canvas).text((4, 4), f"{lang}  label={r['label']}", fill="yellow")
        cells.append(canvas)
    h = min(c.height for c in cells)
    cells = [c.resize((round(c.width * h / c.height), h)) for c in cells]
    # grid: 2 columns per row
    cols = 2
    wmax = max(c.width for c in cells)
    rows_n = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * wmax + 8 * (cols - 1), rows_n * h + 8 * (rows_n - 1)), "white")
    for k, c in enumerate(cells):
        cx, cy = k % cols, k // cols
        sheet.paste(c, (cx * (wmax + 8), cy * (h + 8)))
    p = os.path.join(OUT, pid + ".png")
    sheet.save(p, optimize=True)
    print("sheet:", p, sheet.size)
