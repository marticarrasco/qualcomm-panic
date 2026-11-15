"""Generate the clean review queue (no keyword guess, to avoid anchoring)
for subagent review, split into batches, plus a spot-check sample of the
strong-keyword problems."""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
recs = [json.loads(l) for l in open(ROOT / "data" / "train_problem_categories.jsonl", encoding="utf-8")]

flagged = [r for r in recs if r["needs_review"]]
weak = [r for r in recs if not r["needs_review"] and not (r["scores"][r["topic"]] >= 6 and r["margin"] >= 3)]
strong = [r for r in recs if not r["needs_review"] and (r["scores"][r["topic"]] >= 6 and r["margin"] >= 3)]
queue = sorted(flagged + weak, key=lambda r: int(r["id"].split("_")[-1]))


def block(r):
    tag = f"{'txt' if r['text_only'] else 'img'} L{r['level']} {r['year']}"
    return f"# {r['id']} [{tag}]\n{' '.join(r['text'].split())}"


# Batches of at most 156 problems
batches = [queue[i:i + 156] for i in range(0, len(queue), 156)]
out = ROOT / "analysis" / "review"
out.mkdir(exist_ok=True)
ranges = []
line = 1
for k, b in enumerate(batches, 1):
    text = "\n".join(block(r) for r in b)
    (out / f"batch_{k:02d}.txt").write_text(text, encoding="utf-8")
    n_lines = len(text.splitlines())
    ranges.append((k, len(b), line, line + n_lines - 1))
    line += n_lines

random.seed(11)
sample = random.sample(strong, 60)
(ROOT / "analysis" / "review" / "strong_sample.txt").write_text(
    "\n".join(block(r) for r in sample), encoding="utf-8")

print(f"queue: {len(queue)} problems in {len(batches)} batches; strong: {len(strong)}, sample 60")
for k, n, a, b in ranges:
    print(f"  batch_{k:02d}.txt  {n} problems  lines {a}-{b}")
