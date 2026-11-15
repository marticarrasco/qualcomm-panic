"""Combine a PoT run with the letter dump of the same data file.
Usage: python -I pot_analyze.py <pot.jsonl> <lp.jsonl> <data.jsonl> [--codes N]"""
import json
import math
import random
import sys
from collections import Counter, defaultdict

LETTERS = "ABCDE"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def boot(ids, hits_by_id, n_boot=2000):
    rng = random.Random(42)
    acc = []
    for _ in range(n_boot):
        s = [ids[rng.randrange(len(ids))] for _ in ids]
        h = [x for i in s for x in hits_by_id[i]]
        acc.append(sum(h) / len(h))
    acc.sort()
    return acc[int(0.025 * n_boot)], acc[int(0.975 * n_boot)]


def main():
    pot, lp, data = load(sys.argv[1]), load(sys.argv[2]), load(sys.argv[3])
    n_codes = int(sys.argv[sys.argv.index("--codes") + 1]) if "--codes" in sys.argv else 0
    key = lambda r: (r["id"], r["language"])
    pot_by = {key(r): r for r in pot}
    letter = {key(r): r.get("answer") for r in lp if "id" in r and r.get("status") == "ok"}
    rows = [r for r in data if key(r) in pot_by]  # the rows the PoT run covered (asked or skipped)
    asked = [r for r in pot if r["status"] != "skipped"]
    matched = [r for r in asked if len(r.get("matches") or []) == 1]
    right = [r for r in matched if r["matches"][0] == r["label"]]

    print(f"rows covered {len(rows)} ({len({r['id'] for r in rows})} ids); asked {len(asked)}; "
          f"unique match {len(matched)}; right {len(right)}")
    ids_m = sorted({r["id"] for r in matched})
    ids_r = sorted({r["id"] for r in right})
    print(f"  per problem id: matched in some language {len(ids_m)}, right in some language {len(ids_r)}")
    # precision against the 20% a no-signal match would get, by problem id (first language only)
    seen, k, n = set(), 0, 0
    for r in matched:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        n += 1
        k += r["matches"][0] == r["label"]
    tail = sum(math.comb(n, j) * 0.2 ** j * 0.8 ** (n - j) for j in range(k, n + 1)) if n else float("nan")
    print(f"  one row per id: {k}/{n} right; P(this many or more | matches were luck, 20%) = {tail:.2g}")

    by_level = defaultdict(lambda: [0, 0, 0])
    for r in asked:
        by_level[r["level"]][0] += 1
        by_level[r["level"]][1] += len(r.get("matches") or []) == 1
        by_level[r["level"]][2] += len(r.get("matches") or []) == 1 and r["matches"][0] == r["label"]
    print("  by level (asked/matched/right): " + "  ".join(f"L{lv}:{a}/{m}/{c}" for lv, (a, m, c) in sorted(by_level.items())))

    # nearest option when the program printed a number that matches nothing
    opts = {key(r): r.get("options") or {} for r in data}
    near_n = near_ok = 0
    for r in asked:
        if r.get("value") is None or r.get("matches"):
            continue
        import re
        vals = {}
        for k2, v in opts[key(r)].items():
            s = str(v).strip().replace(",", ".").replace("−", "-").replace("一", "-")
            s = "0" if s in ("O", "o") else s
            try:
                vals[k2] = float(s) if "/" not in s else float(s.split("/")[0]) / float(s.split("/")[1])
            except (ValueError, ZeroDivisionError):
                pass
        if len(vals) == 5:
            near_n += 1
            best = min(vals, key=lambda q: abs(vals[q] - r["value"]))
            near_ok += best == r["label"]
    print(f"  printed a number that matches no option: {near_n} rows; nearest option right in {near_ok}")

    # policies over every covered row
    def score(name, answers):
        hits_by_id = defaultdict(list)
        for r in rows:
            hits_by_id[r["id"]].append(answers[key(r)] == r["label"])
        ids = sorted(hits_by_id)
        hit = sum(sum(h) for h in hits_by_id.values())
        lo, hi = boot(ids, hits_by_id)
        c = Counter(answers[key(r)] for r in rows)
        print(f"  {name:46} {hit:3}/{len(rows)} = {100 * hit / len(rows):5.1f}%   [{100 * lo:4.1f}, {100 * hi:4.1f}]   "
              + "/".join(str(c[l]) for l in LETTERS))

    base = {key(r): (letter.get(key(r)) or "D") for r in rows}
    always_d = {key(r): "D" for r in rows}
    pot_or = lambda fb: {key(r): (pot_by[key(r)]["matches"][0] if len(pot_by[key(r)].get("matches") or []) == 1 else fb[key(r)])
                         for r in rows}
    print(f"\npolicies on the {len(rows)} covered rows   (accuracy, 95% interval by problem, letters given)")
    score("always D", always_d)
    score("baseline letter", base)
    score("PoT match, else always D", pot_or(always_d))
    score("PoT match, else baseline letter", pot_or(base))
    truth = Counter(r["label"] for r in rows)
    print(f"  {'truth':46} {'':36}" + "/".join(str(truth[l]) for l in LETTERS))

    st = Counter()
    for r in asked:
        s = r["status"]
        st["ok" if s == "ok" else "timeout" if s == "timeout" else "blocked" if s.startswith("blocked") else "error: " + s.split(":")[0][:28]] += 1
    print("\nstatus of the generated programs:", dict(st))
    print("truncated at max_tokens:", sum(1 for r in asked if r.get("finish") == "length"))
    if n_codes:
        print("\n--- examples of programs")
        for title, group in (("RIGHT", right), ("WRONG MATCH", [r for r in matched if r not in right]),
                             ("FAILED", [r for r in asked if r["status"] != "ok"])):
            for r in group[:n_codes]:
                print(f"\n[{title}] {r['id']} {r['language']} status={r['status']} printed={r.get('printed')!r} truth={r['label']}")
                print("    " + r["code"].replace("\n", "\n    ")[:420])


if __name__ == "__main__":
    main()
