"""Val problems that are a twin of a training problem (same year, same five options, same answer, another level),
and the val scores of each model without them.

Usage, from the Datathon folder:  <venv python> -X utf8 mesures/ft_twins.py
"""
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "datathon-UPC-maths-qualcomm-arduino-unoq"))
sys.path.insert(0, ROOT)
from copilot import pot  # noqa: E402


def load(p):
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def words(text):
    return set(re.sub(r"[^a-z0-9àèéíïòóúüç]+", " ", re.sub(r"^\s*\d+\s*[.)]\s*", "", text.lower())).split())


def signature(r):
    nums = pot.numeric_options(r)
    if nums is None:
        return None
    return (r["year"], tuple(sorted(nums.values())), nums[r["label"]])


pairs = load(os.path.join(ROOT, "data", "pot_pairs.jsonl"))
train = {(r["id"], r["language"]): r for r in load(os.path.join(ROOT, "data", "train.jsonl"))}
val = {(r["id"], r["language"]): r for r in load(os.path.join(ROOT, "data", "val.jsonl"))}
pair_ids = {r["id"] for r in pairs}
by_sig = {}
for (pid, lang), r in train.items():
    if pid in pair_ids and lang == "eng":
        sig = signature(r)
        if sig:
            by_sig.setdefault(sig, []).append(r)

twins = {}
for (pid, lang), r in val.items():
    if lang != "eng" or not r.get("text_only"):
        continue
    sig = signature(r)
    for t in by_sig.get(sig, []) if sig else []:
        a, b = words(r["text"]), words(t["text"])
        j = len(a & b) / max(len(a | b), 1)
        if j >= 0.3:
            twins[pid] = (t["id"], round(j, 2))
asked = {k for k, r in val.items() if r.get("text_only") and pot.numeric_options(r) is not None}
asked_ids = {k[0] for k in asked}
print(f"val problems asked: {len(asked_ids)}; with a twin among the training pairs: {len(set(twins) & asked_ids)}")
for pid, (tid, j) in sorted(twins.items())[:8]:
    print(f"   {pid} ~ {tid} (word Jaccard {j})")

cache = json.load(open(os.path.join(HERE, "ft-run-cache.json"), encoding="utf-8"))
print("\nval, smart sandbox: right / wrong, on twins and on the rest")
for tag in ("ft1", "ft2", "ft3"):
    c = {True: Counter(), False: Counter()}
    for o in load(os.path.join(HERE, f"ft-{tag}-val.jsonl")):
        key = (o["id"], o["language"])
        printed, _ = cache["smart\x00" + pot.extract_code(o["reply"])]
        nums = pot.numeric_options(val[key])
        letters = []
        for part in (printed or "").split("\t")[::-1]:
            letters = pot._match_one(nums, part)
            if len(letters) == 1:
                break
        c[o["id"] in twins]["right" if letters == [o["label"]] else "wrong" if len(letters) == 1 else "none"] += 1
    t, r = c[True], c[False]
    print(f"  {tag}: twins {t['right']}/{t['wrong']} of {sum(t.values())} rows | not twins {r['right']}/{r['wrong']} of "
          f"{sum(r.values())} rows, precision {100 * r['right'] / max(r['right'] + r['wrong'], 1):.0f}%")
