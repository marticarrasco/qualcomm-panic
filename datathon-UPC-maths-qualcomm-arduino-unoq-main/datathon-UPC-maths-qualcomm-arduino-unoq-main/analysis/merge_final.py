"""Merge the final problem categories.

Priority: manual verdicts (strong-sample audit) > subagent review batches >
keyword first pass. Validates coverage (all 1,466 ids exactly once) and the
topic vocabulary, then rewrites data/train_problem_categories.jsonl with a
`source` field and prints the final distribution.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tutor.topics import TOPICS  # noqa: E402

# My verdicts on the 60-problem audit of the strong keyword set
# (analysis/audit_strong_sample.py holds the reasoning).
from analysis.audit_strong_sample import mine  # noqa: E402


def main():
    recs = [json.loads(l) for l in open(ROOT / "data" / "train_problem_categories.jsonl", encoding="utf-8")]
    final = {r["id"]: r for r in recs}

    # 1. subagent review batches (analysis/review/batch_XX.json)
    # Each JSON must list exactly the ids of its batch file, in order.
    # A batch may be split into chunk files batch_XXa/b/c.json (retry after
    # an output-limit truncation); they are concatenated in glob order.
    reviewed = Counter()
    for k in range(1, 8):
        root = ROOT / "analysis" / "review"
        files = sorted(root.glob(f"batch_{k:02d}.json")) or sorted(root.glob(f"batch_{k:02d}[a-z].json"))
        if not files:
            raise SystemExit(f"missing review output for batch_{k:02d}")
        items = [obj for f in files for obj in json.loads(f.read_text(encoding="utf-8"))]
        expected = []
        for line in (root / f"batch_{k:02d}.txt").read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                expected.append(line[2:].split()[0])
        got = [item["id"] for item in items]
        if got != expected:
            raise SystemExit(f"batch_{k:02d}: ids/order mismatch with batch file "
                             f"({len(got)} vs {len(expected)} entries)")
        for item in items:
            pid, topic = item["id"], item["topic"]
            if pid not in final:
                raise SystemExit(f"unknown id {pid} in {path.name}")
            if topic not in TOPICS:
                raise SystemExit(f"bad topic {topic!r} for {pid} in {path.name}")
            final[pid]["topic"] = topic
            final[pid]["source"] = "review"
            reviewed[k] += 1
    print("reviewed per batch:", dict(reviewed))

    # 2. manual audit verdicts win over everything
    for pid, (topic, _alts) in mine.items():
        final[pid]["topic"] = topic
        final[pid]["source"] = "audit"

    ids = [r["id"] for r in final.values()]
    assert len(ids) == len(set(ids)) == 1466, (len(ids), len(set(ids)))

    out = ROOT / "data" / "train_problem_categories.jsonl"
    dist = Counter()
    src = Counter()
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for pid in sorted(final):
            r = final[pid]
            r.setdefault("source", "keyword")
            dist[r["topic"]] += 1
            src[r["source"]] += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("sources:", dict(src))
    for t in TOPICS:
        print(f"  {t:20s} {dist.get(t, 0)}")
    print("total:", sum(dist.values()))


if __name__ == "__main__":
    main()
