"""Compare a cond_probe.py run with the free-form program run on the same rows.
Usage: python -I cond_compare.py <cond.jsonl> <pot-rerun.jsonl> <data.jsonl>"""
import json
import math
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def answer(row):
    m = row.get("matches") or []
    return m[0] if len(m) == 1 else None


def by_id_p(rows):
    """One row per problem id: right / matched, and the chance of that many if matches were luck."""
    seen, n, k = set(), 0, 0
    for r in rows:
        if answer(r) is None or r["id"] in seen:
            continue
        seen.add(r["id"])
        n += 1
        k += answer(r) == r["label"]
    p = sum(math.comb(n, j) * 0.2 ** j * 0.8 ** (n - j) for j in range(k, n + 1)) if n else float("nan")
    return k, n, p


cond = {(r["id"], r["language"]): r for r in load(sys.argv[1])}
pot = {(r["id"], r["language"]): r for r in load(sys.argv[2])}
data = [r for r in load(sys.argv[3]) if (r["id"], r["language"]) in cond]
keys = [k for k in cond if cond[k]["status"] != "skipped" and k in pot and pot[k]["status"] != "skipped"]

c_ok = {k for k in keys if answer(cond[k]) == cond[k]["label"]}
p_ok = {k for k in keys if answer(pot[k]) == pot[k]["label"]}
c_m = {k for k in keys if answer(cond[k])}
p_m = {k for k in keys if answer(pot[k])}
print(f"rows asked in both runs: {len(keys)}")
print(f"  this form:         matched {len(c_m):3}, right {len(c_ok):3}")
print(f"  free-form program: matched {len(p_m):3}, right {len(p_ok):3}")
print(f"  right in both {len(c_ok & p_ok)}, only this form {len(c_ok - p_ok)}, only the program {len(p_ok - c_ok)}, "
      f"in either {len(c_ok | p_ok)}")
k, n, p = by_id_p([cond[x] for x in keys])
print(f"  this form, one row per problem: {k}/{n} matches right (p = {p:.2g} if matches were luck)")
k, n, p = by_id_p([pot[x] for x in keys])
print(f"  free-form program, one row per problem: {k}/{n} (p = {p:.2g})")

# Policies over every row of the data that the runs covered
total = len(data)


def score(name, pick):
    hits = sum(pick((r["id"], r["language"])) == r["label"] for r in data)
    print(f"  {name:46} {hits:3}/{total} = {100 * hits / total:.1f}%")


def first(*runs):
    def pick(key):
        for run in runs:
            row = run.get(key)
            if row and row["status"] != "skipped" and answer(row):
                return answer(row)
        return "D"
    return pick


print(f"accuracy on the {total} rows covered, D when nothing matches:")
score("always D", lambda key: "D")
score("this form only", first(cond))
score("free-form program only", first(pot))
score("this form, then the program", first(cond, pot))
score("the program, then this form", first(pot, cond))
both = [k for k in keys if answer(cond[k]) and answer(pot[k])]
agree = [k for k in both if answer(cond[k]) == answer(pot[k])]
print(f"both give an answer on {len(both)} rows; they agree on {len(agree)}, and "
      f"{sum(answer(cond[k]) == cond[k]['label'] for k in agree)} of those are right")
