#!/usr/bin/env python3
"""Crop the figure out of an M3Kang problem strip (question text + options + figure).

    .venv/bin/python experiments/figure_crop.py --n 80 --sheets   # overlays in results_local/figcrop/

Laptop only (needs pillow, numpy, scipy, pyarrow). Heuristic, no OCR:
  1. ink = pixels darker than 235; smear it 15 px sideways so letters join into words/lines;
  2. connected blocks lower than TEXT_H px are text (question lines, "A) 10" options);
     a wide block glued to the top-left corner is the question paragraph;
  3. everything else that is tall is figure; the crop is the bounding box of those blocks,
     plus small labels sitting inside / right next to it.
crop_figure() returns (PIL image or None, box or None). None means "no figure found".
"""
import argparse
import glob
import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHARD = "~/.cache/huggingface/hub/datasets--qualcomm--M3Kang/snapshots/*/data/high_quality-00000-of-00004.parquet"
TEXT_H = 48      # a block lower than this is a line of text
SMEAR = 15       # horizontal smear in px
NEAR = 24        # labels this close to the figure box are kept
PAD = 6


def find_box(image):
    """Returns (box, text_boxes): the figure box and the text lines inside it that should be painted white."""
    gray = np.asarray(image.convert("L"))
    h, w = gray.shape
    ink = gray < 235
    smeared = ndimage.binary_dilation(ink, structure=np.ones((1, SMEAR), bool))
    labels, count = ndimage.label(smeared, structure=np.ones((3, 3), bool))
    tall, small = [], []
    for index, sl in enumerate(ndimage.find_objects(labels), 1):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        pixels = int((ink[sl] & (labels[sl] == index)).sum())
        if pixels < 12:
            continue
        height, width = y1 - y0, x1 - x0
        # question lines that touch each other form one wide block glued to the top-left corner
        paragraph = y0 < 25 and x0 < 0.2 * w and width > 0.3 * w and height < 0.75 * h
        (small if height < TEXT_H or paragraph else tall).append((x0, y0, x1, y1))
    if not tall:
        return None, []
    x0 = min(b[0] for b in tall); y0 = min(b[1] for b in tall)
    x1 = max(b[2] for b in tall); y1 = max(b[3] for b in tall)
    # question text = wide lines, plus shorter lines starting at the same left margin above the last wide line
    wide = [b for b in small if b[2] - b[0] > 0.3 * w]
    text = list(wide)
    if wide:
        margin = min(b[0] for b in wide)
        bottom = max(b[3] for b in wide)
        text += [b for b in small if b not in wide and abs(b[0] - margin) < 25 and b[1] < bottom + 40
                 and b[2] - b[0] > 60]
    box = [x0, y0, x1, y1]
    for block in small:   # labels over the figure's columns and close to it (one pass)
        if block in text:
            continue
        sx0, sy0, sx1, sy1 = block
        overlap = min(sx1, x1) - max(sx0, x0)
        inside_x = overlap >= 0.5 * (sx1 - sx0) and (sx1 - sx0) <= 1.5 * (x1 - x0)
        close_y = sy0 >= y0 - NEAR - (sy1 - sy0) and sy1 <= y1 + NEAR + (sy1 - sy0)
        if inside_x and close_y:
            box = [min(box[0], sx0), min(box[1], sy0), max(box[2], sx1), max(box[3], sy1)]
    box = (max(0, box[0] - PAD), max(0, box[1] - PAD), min(w, box[2] + PAD), min(h, box[3] + PAD))
    inside = [b for b in text if b[2] > box[0] and b[0] < box[2] and b[3] > box[1] and b[1] < box[3]]
    return box, inside


def crop_figure(raw, min_side=0):
    """raw: image bytes. Returns (cropped PIL image, box) or (None, None).

    Question text that falls inside the box (figure top right + picture options below) is painted white,
    then the crop is trimmed to what is left."""
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    box, text = find_box(image)
    if box is None:
        return None, None
    if text:
        draw = ImageDraw.Draw(image)
        for x0, y0, x1, y1 in text:
            draw.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), fill=(255, 255, 255))
        ink = np.asarray(image.convert("L").crop(box)) < 235
        ys, xs = np.where(ink)
        if len(xs):
            box = (max(0, box[0] + xs.min() - PAD), max(0, box[1] + ys.min() - PAD),
                   min(image.width, box[0] + xs.max() + PAD), min(image.height, box[1] + ys.max() + PAD))
    crop = image.crop(box)
    if min_side and max(crop.size) < min_side:      # tiny figures: enlarge so the encoder gets enough pixels
        scale = min_side / max(crop.size)
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
    return crop, box


def to_jpeg(image, quality=92):
    out = io.BytesIO()
    image.save(out, "JPEG", quality=quality)
    return out.getvalue()


def load_problems(paths=("data/val.jsonl",), language="eng"):
    problems = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            problems += [json.loads(line) for line in f if line.strip()]
    return [x for x in problems if x["language"] == language and not x["text_only"]]


def load_images(keys):
    import pyarrow.parquet as pq
    parquet = sorted(glob.glob(os.path.expanduser(SHARD)))[-1]
    meta = pq.read_table(parquet, columns=["id", "language"]).to_pylist()
    wanted = [i for i, row in enumerate(meta) if (row["id"], row["language"]) in keys]
    table = pq.read_table(parquet, columns=["id", "language", "image"]).take(wanted)
    return {(row["id"], row["language"]): row["image"]["bytes"] for row in table.to_pylist()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--out", default="results_local/figcrop")
    p.add_argument("--sheets", action="store_true", help="write contact sheets: strip on the left, crop on the right")
    p.add_argument("--save", type=int, default=12, help="save this many crops as jpg")
    args = p.parse_args()
    problems = load_problems()[:args.n]
    images = load_images({(x["id"], x["language"]) for x in problems})
    os.makedirs(f"{args.out}/crops", exist_ok=True)
    rows, none = [], 0
    for n, problem in enumerate(problems):
        raw = images[(problem["id"], problem["language"])]
        crop, box = crop_figure(raw)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        if box is None:
            none += 1
        else:
            ImageDraw.Draw(image).rectangle(box, outline=(255, 0, 0), width=4)
            if n < args.save:
                crop.save(f"{args.out}/crops/{problem['id']}.jpg", quality=92)
        rows.append((n, problem["id"], image, crop))
        print(n, problem["id"], image.size, box)
    print(f"{none}/{len(problems)} with no figure found")
    if args.sheets:
        per = 10
        for start in range(0, len(rows), per):
            sheet = Image.new("RGB", (1400, per * 146), (120, 120, 120))
            for k, (n, pid, image, crop) in enumerate(rows[start:start + per]):
                left = image.copy(); left.thumbnail((820, 140))
                sheet.paste(left, (0, k * 146))
                if crop is not None:
                    right = crop.copy(); right.thumbnail((560, 140))
                    sheet.paste(right, (830, k * 146))
                ImageDraw.Draw(sheet).text((2, k * 146 + 2), f"{n} {pid}" + ("" if crop else " NO FIGURE"), fill=(0, 0, 255))
            sheet.save(f"{args.out}/sheet-{start // per:02d}.jpg", quality=88)


if __name__ == "__main__":
    main()
