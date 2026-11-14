"""Build the manual-review queue for ambiguous classifications plus a
spot-check sample of the confident ones."""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
recs = [json.loads(l) for l in open(ROOT / "data" / "train_problem_categories.jsonl", encoding="utf-8")]
flagged = [r for r in recs if r["needs_review"]]
confident = [r for r in recs if not r["needs_review"]]


def key(r):
    m = re.match(r"lvl-(\d+)_(\d+)_(\d+)", r["id"])
    return tuple(map(int, m.groups()))


flagged.sort(key=key)
confident.sort(key=key)


def sc(r):
    pairs = sorted(r["scores"].items(), key=lambda kv: -kv[1])
    return " ".join(f"{k[:4]}:{int(v) if v == int(v) else v}" for k, v in pairs if v > 0) or "-"


def block(r):
    tag = f"{'txt' if r['text_only'] else 'img'} L{r['level']} {r['year']}"
    return f"{r['id']} [{tag}] guess={r['topic']} ({sc(r)})\n{' '.join(r['text'].split())}"


(ROOT / "analysis" / "review_queue.txt").write_text("\n".join(block(r) for r in flagged), encoding="utf-8")
random.seed(7)
sample = random.sample(confident, 40)
(ROOT / "analysis" / "review_sample_confident.txt").write_text("\n".join(block(r) for r in sample), encoding="utf-8")
print(f"flagged: {len(flagged)}  confident: {len(confident)}")
