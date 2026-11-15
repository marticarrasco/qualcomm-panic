"""Audit of the fine-tuning data: overlap with val and the practice set, coverage and shape of the programs.

Read-only. Usage, from the Datathon folder:
  <venv python> -X utf8 mesures/ft_audit_data.py
"""
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datathon-UPC-maths-qualcomm-arduino-unoq"))
sys.path.insert(0, ROOT)
from copilot import pot  # noqa: E402

D = os.path.join(ROOT, "data")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(text):
    text = re.sub(r"^\s*\d+\s*[.)]\s*", "", text.lower())
    return re.sub(r"[^a-z0-9àèéíïòóúüç]+", " ", text).strip()


def shingles(text, n=5):
    w = norm(text).split()
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 1))}


pairs = load(os.path.join(D, "pot_pairs.jsonl"))
programs = load(os.path.join(D, "pot_programs.jsonl"))
train = load(os.path.join(D, "train.jsonl"))
val = load(os.path.join(D, "val.jsonl"))
demo = load(os.path.join(D, "demo", "train.jsonl"))

pair_ids = {r["id"] for r in pairs}
print(f"pairs {len(pairs)} rows, {len(pair_ids)} problems; by language {dict(Counter(r['language'] for r in pairs))}")
print("ids shared with val:", len(pair_ids & {r['id'] for r in val}), "| with the practice set:",
      len(pair_ids & {r['id'] for r in demo}), "| not in train:", len(pair_ids - {r['id'] for r in train}))

# Same statement under another id (Kangaroo repeats problems across levels).
for name, rows in (("val", val), ("practice set", demo)):
    exact = Counter()
    pair_norm = {}
    for r in pairs:
        pair_norm.setdefault(norm(r["text"]), r["id"])
    near = []
    pair_sh = [(r["id"], r["language"], shingles(r["text"])) for r in pairs]
    for r in rows:
        if norm(r["text"]) in pair_norm:
            exact[r["id"]] += 1
            continue
        s = shingles(r["text"])
        if len(s) < 4:
            continue
        best = max(((len(s & t) / len(s | t), pid) for pid, lang, t in pair_sh if lang == r["language"]), default=(0, ""))
        if best[0] >= 0.5:
            near.append((round(best[0], 2), r["id"], best[1]))
    asked = [r for r in rows if r.get("text_only") and pot.numeric_options(r) is not None]
    asked_ids = {r["id"] for r in asked}
    print(f"\n{name}: {len(rows)} rows, {len({r['id'] for r in rows})} problems; asked (text only, five numeric options) "
          f"{len(asked)} rows, {len(asked_ids)} problems")
    print(f"  same statement as a training pair: {len(exact)} problems ({sum(1 for i in exact if i in asked_ids)} of them asked)")
    print(f"  near-duplicates (5-word shingle Jaccard >= 0.5): {len({n[1] for n in near})} problems "
          f"({len({n[1] for n in near} & asked_ids)} of them asked)")
    for n in sorted(near, reverse=True)[:6]:
        print("    ", n)
    why = Counter()
    for r in rows:
        if not r.get("text_only"):
            why["has a figure"] += 1
        elif not r.get("options"):
            why["no options"] += 1
        elif pot.numeric_options(r) is None:
            why["options are not five numbers"] += 1
        else:
            why["asked"] += 1
    print("  rows by gate:", dict(why))
    bad = [r["options"] for r in rows if r.get("text_only") and r.get("options") and pot.numeric_options(r) is None]
    for o in bad[:8]:
        print("    not numeric:", json.dumps(o, ensure_ascii=False)[:150])

# Coverage of train: how many problems could have a program and don't.
by_id = {}
for r in train:
    by_id.setdefault(r["id"], []).append(r)
eligible = {pid for pid, rows in by_id.items()
            if any(r.get("text_only") and pot.numeric_options(r) is not None for r in rows)}
text_only = {pid for pid, rows in by_id.items() if any(r.get("text_only") for r in rows)}
with_opts = {pid for pid, rows in by_id.items() if any(r.get("options") for r in rows)}
print(f"\ntrain: {len(by_id)} problems; text only {len(text_only)}; with options {len(with_opts)}; "
      f"text only with five numeric options {len(eligible)}; of those with a program {len(eligible & pair_ids)}, "
      f"without {len(eligible - pair_ids)}")
print("  pairs by level:", dict(sorted(Counter(r["level"] for r in programs).items())))
print("  eligible without program, by level:", dict(sorted(Counter(by_id[p][0]["level"] for p in eligible - pair_ids).items())))

# Shape of the programs.
codes = {r["id"]: r["code"] for r in programs}
n_lines = Counter(len(c.strip().splitlines()) for c in codes.values())
print(f"\nprograms {len(codes)}; lines: {dict(sorted(n_lines.items()))}")
feat = Counter()
for c in codes.values():
    feat["has a loop or comprehension"] += bool(re.search(r"\bfor\b|\bwhile\b", c))
    feat["no loop: plain arithmetic"] += not re.search(r"\bfor\b|\bwhile\b", c)
    feat["uses permutations/combinations/product"] += bool(re.search(r"permutations|combinations|product\(", c))
    feat["uses str()"] += "str(" in c
    feat["uses Fraction"] += "Fraction" in c
    feat["ends with print(ans)"] += c.strip().endswith("print(ans)")
print("  ", dict(feat))
# A program that only restates the answer: the printed value appears as a literal and no number of the text does.
hard = 0
for r in programs:
    text = next((x["text"] for x in by_id.get(r["id"], []) if x["language"] == "eng"), "")
    nums_text = set(re.findall(r"\d+", text))
    nums_code = set(re.findall(r"\d+", r["code"]))
    ans = str(r.get("printed") or "").strip().split(".")[0].lstrip("-")
    if ans and len(ans) >= 2 and ans in nums_code and ans not in nums_text:
        hard += 1
print(f"  programs where the answer (2+ digits) is a literal of the code and not of the statement: {hard}")
dup = Counter(codes.values())
print(f"  distinct programs {len(dup)}; most repeated: {dup.most_common(3)[0][1]} times")
