"""Second OCR pass for problems whose options the first pass missed.

For each problem with empty options it re-reads the bottom of the image,
cropped and enlarged, and then copies options between languages when they
don't depend on the language (numbers, units). Results are appended to
data/.ocr_cache.jsonl, so running prepare_m3kang.py again rebuilds
train.jsonl and val.jsonl with them.

    python laptop/fix_options.py                # text-only problems
    python laptop/fix_options.py --all          # also problems with figures
"""
import argparse
import collections
import json
import os
import re
import shutil
import sys

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(__file__))
from prepare_m3kang import parse_options  # noqa: E402

# (bottom fraction of the image to keep, enlargement)
VARIANTS = [(0.5, 2), (0.5, 3), (0.35, 3), (1.0, 2), (0.7, 2)]
WORD = re.compile(r"[^\W\d_]{3,}")  # 3+ letters in a row: language-dependent


def language_free(options):
    return not any(WORD.search(v) for v in options.values())


def read_options(ocr, path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    for frac, scale in VARIANTS:
        crop = image.crop((0, int(h * (1 - frac)), w, h))
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
        crop = ImageOps.expand(crop, border=40, fill="white")
        boxes, _ = ocr(np.array(crop))
        options = parse_options(boxes or [])
        if options and len(set(options.values())) == 5:
            return options, f"crop{frac}x{scale}"
    return {}, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--all", action="store_true", help="also problems with figures")
    ap.add_argument("--primary", default="eng", help="language to OCR first")
    args = ap.parse_args()

    rows = []
    for name in ("train.jsonl", "val.jsonl"):
        with open(os.path.join(args.data, name), encoding="utf-8") as f:
            rows += [json.loads(line) for line in f]
    by_id = collections.defaultdict(dict)
    for r in rows:
        by_id[r["id"]][r["language"]] = r
    todo = [i for i, d in by_id.items() if (args.all or next(iter(d.values()))["text_only"])
            and any(not r["options"] for r in d.values())]
    print(f"{len(todo)} problems with missing options", flush=True)

    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    fixes, stats = [], collections.Counter()

    def image_path(pid, lang):
        return os.path.join(args.data, "images", f"{pid}_{lang}.png")

    for n, pid in enumerate(todo, 1):
        langs = by_id[pid]
        order = sorted(langs, key=lambda l: l != args.primary)
        for lang in order:
            r = langs[lang]
            if r["options"]:
                continue
            # 1. copy from another language, if the options are language-free
            donor = next((o for o in langs.values() if o["options"] and language_free(o["options"])), None)
            if donor:
                r["options"], how = dict(donor["options"]), f"copy:{donor['language']}"
            else:
                r["options"], how = read_options(ocr, image_path(pid, lang))
            if r["options"]:
                fixes.append({"key": f"{pid}|{lang}", "options": r["options"], "how": how})
                stats[how.split(":")[0].split("x")[0] if how.startswith("copy") else "ocr"] += 1
            else:
                stats["still_missing"] += 1
        if n % 25 == 0:
            print(f"  {n}/{len(todo)} {dict(stats)}", flush=True)

    cache = os.path.join(args.data, ".ocr_cache.jsonl")
    shutil.copyfile(cache, cache + ".bak")
    with open(cache, "a", encoding="utf-8") as f:
        for fix in fixes:
            f.write(json.dumps({"key": fix["key"], "options": fix["options"]}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.data, "ocr_fixes.jsonl"), "w", encoding="utf-8") as f:
        for fix in fixes:
            f.write(json.dumps(fix, ensure_ascii=False) + "\n")
    print(f"Done: {dict(stats)}. {len(fixes)} fixes appended to {cache} (backup: .bak)")
    print("Now run prepare_m3kang.py again to rebuild train.jsonl and val.jsonl.")


if __name__ == "__main__":
    main()
