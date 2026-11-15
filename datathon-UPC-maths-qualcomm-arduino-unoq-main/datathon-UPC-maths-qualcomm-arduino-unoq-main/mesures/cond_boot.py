"""95% intervals (resampling problem ids) for the two-attempt policy: search line, then program, then D.
Usage: python -I cond_boot.py <line.jsonl> <pot-rerun.jsonl> <data.jsonl>"""
import json
import random
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def answer(row):
    m = (row or {}).get("matches") or []
    return m[0] if row and row["status"] != "skipped" and len(m) == 1 else None


line = {(r["id"], r["language"]): r for r in load(sys.argv[1])}
pot = {(r["id"], r["language"]): r for r in load(sys.argv[2])}
data = [r for r in load(sys.argv[3]) if (r["id"], r["language"]) in line]
policies = {
    "always D": lambda k: "D",
    "program only": lambda k: answer(pot.get(k)) or "D",
    "search line only": lambda k: answer(line.get(k)) or "D",
    "search line, then program": lambda k: answer(line.get(k)) or answer(pot.get(k)) or "D",
}
ids = sorted({r["id"] for r in data})
hits = {name: {i: [] for i in ids} for name in policies}
for r in data:
    k = (r["id"], r["language"])
    for name, pick in policies.items():
        hits[name][r["id"]].append(pick(k) == r["label"])
rng = random.Random(42)
samples = [[ids[rng.randrange(len(ids))] for _ in ids] for _ in range(4000)]


def acc(name, sample):
    h = [x for i in sample for x in hits[name][i]]
    return sum(h) / len(h)


print(f"{len(data)} rows, {len(ids)} problems")
for name in policies:
    a = sorted(acc(name, s) for s in samples)
    gain = sorted(acc(name, s) - acc("always D", s) for s in samples)
    point = acc(name, ids)
    print(f"  {name:28} {100 * point:5.1f}%  [{100 * a[100]:.0f}, {100 * a[3899]:.0f}]   gain over always D "
          f"{100 * (point - acc('always D', ids)):+.0f} [{100 * gain[100]:+.0f}, {100 * gain[3899]:+.0f}]")
both = sorted(acc("search line, then program", s) - acc("program only", s) for s in samples)
print(f"  two attempts minus program only: [{100 * both[100]:+.0f}, {100 * both[3899]:+.0f}] points")
