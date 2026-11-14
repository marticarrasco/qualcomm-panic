#!/usr/bin/env python3
"""Contact sheets for the suspected problems: label-letter diffs and TEXT option ids."""
import glob
import io
import json
import os

import pyarrow.parquet as pq
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sheets")
IDS = ["lvl-7_2021_9", "lvl-7_2017_8", "lvl-6_2018_16", "lvl-6_2014_30",
       "lvl-3_2017_26", "lvl-4_2017_10", "lvl-5_2009_30", "lvl-5_2016_25",
       "lvl-6_2015_25", "lvl-6_2020_6", "lvl-7_2019_21"]
LANGS = ["eng", "cat", "fra", "esp", "deu"]

cache = os.path.expanduser(r"~\.cache\huggingface\hub\datasets--qualcomm--M3Kang\snapshots")
t = pq.read_table(glob.glob(os.path.join(cache, "*", "data", "*.parquet"))[0],
                  columns=["id", "language", "label", "text", "image"])
data = {}
for r in t.to_pylist():
    if r["id"] in IDS and r["language"] in LANGS:
        data[(r["id"], r["language"])] = r

for pid in IDS:
    cells = []
    for lang in LANGS:
        r = data.get((pid, lang))
        if not r:
            print("missing:", pid, lang)
            continue
        im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        c = Image.new("RGB", (im.width, im.height + 22), "black")
        c.paste(im, (0, 22))
        ImageDraw.Draw(c).text((4, 4), f"{lang}  label={r['label']}", fill="yellow")
        cells.append(c)
    if not cells:
        continue
    h = min(c.height for c in cells)
    cells = [c.resize((round(c.width * h / c.height), h)) for c in cells]
    cols = 2
    wmax = max(c.width for c in cells)
    rows_n = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * wmax + 8 * (cols - 1), rows_n * h + 8 * (rows_n - 1)), "white")
    for k, c in enumerate(cells):
        cx, cy = k % cols, k // cols
        sheet.paste(c, (cx * (wmax + 8), cy * (h + 8)))
    p = os.path.join(OUT, "chk_" + pid + ".png")
    sheet.save(p, optimize=True)
    print("sheet:", p, sheet.size)
