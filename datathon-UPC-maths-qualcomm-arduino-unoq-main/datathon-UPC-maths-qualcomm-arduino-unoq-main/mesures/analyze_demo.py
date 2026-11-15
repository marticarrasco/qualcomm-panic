"""Score every way of turning the model's five letter log-probabilities into an answer.
Usage: python -I analyze_demo.py <lp-dump.jsonl> [--balanced]
No labels are used to fit anything except where stated."""
import json
import math
import random
import sys
from collections import Counter, defaultdict

LETTERS = "ABCDE"

# The investigation's hard-coded table (section 5.1), keyed by the dump's option_kind
INV_PRIORS = {
    "num_inc": {"A": 0.081, "B": 0.218, "C": 0.286, "D": 0.294, "E": 0.120},
    "num_dec": {"A": 0.080, "B": 0.340, "C": 0.160, "D": 0.340, "E": 0.080},
    "num_unsorted": {"A": 0.213, "B": 0.163, "C": 0.216, "D": 0.195, "E": 0.213},
    "other": {"A": 0.169, "B": 0.197, "C": 0.197, "D": 0.230, "E": 0.207},
}
ALPHA, BETA = 0.65, 0.85  # the constants in the investigation's code


def fill(lp):
    found = [v for v in lp.values() if v is not None]
    floor = min(found) - 2.0 if found else 0.0
    return [lp[l] if lp.get(l) is not None else floor for l in LETTERS]


def log_softmax(v):
    top = max(v)
    total = top + math.log(sum(math.exp(x - top) for x in v))
    return [x - total for x in v]


def pick(v):
    return LETTERS[max(range(5), key=lambda i: v[i])]


def hungarian(cost):
    """Minimum-cost assignment, square matrix (e-maxx). Returns column index per row."""
    n = len(cost)
    u, v, p, way = [0.0] * (n + 1), [0.0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out = [0] * n
    for j in range(1, n + 1):
        out[p[j] - 1] = j - 1
    return out


def balanced(rows, z):
    """Each letter used the same number of times, maximising the total log-probability."""
    n = len(rows)
    per = n // 5
    slots = [l for l in range(5) for _ in range(per)] + list(range(n - 5 * per))
    cost = [[-z[i][slots[j]] for j in range(n)] for i in range(n)]
    cols = hungarian(cost)
    return [LETTERS[slots[c]] for c in cols]


def boot(rows, answers, n_boot=2000):
    ids = sorted({r["id"] for r in rows})
    by = defaultdict(list)
    for r, a in zip(rows, answers):
        by[r["id"]].append(a == r["label"])
    rng = random.Random(42)
    acc = []
    for _ in range(n_boot):
        s = [ids[rng.randrange(len(ids))] for _ in ids]
        h = [x for i in s for x in by[i]]
        acc.append(sum(h) / len(h))
    acc.sort()
    return acc[int(0.025 * n_boot)], acc[int(0.975 * n_boot)]


def perm_test(rows, z, n_perm=5000):
    """Mean log-probability the bias-removed model gives the true letter, against shuffled labels."""
    q = [log_softmax(v) for v in z]
    labels = [LETTERS.index(r["label"]) for r in rows]
    real = sum(q[i][labels[i]] for i in range(len(rows))) / len(rows)
    rng = random.Random(42)
    ge = 0
    for _ in range(n_perm):
        sh = labels[:]
        rng.shuffle(sh)
        ge += sum(q[i][sh[i]] for i in range(len(rows))) / len(rows) >= real - 1e-12
    return real, (1 + ge) / (1 + n_perm)


def report(title, rows, null):
    print(f"\n=== {title}: {len(rows)} rows, {len({r['id'] for r in rows})} problems")
    raw = [fill(r["logprobs"]) for r in rows]
    z = [log_softmax(v) for v in raw]
    truth = Counter(r["label"] for r in rows)
    mean = [sum(v[i] for v in z) / len(z) for i in range(5)]
    by_lang = defaultdict(list)
    for r, v in zip(rows, z):
        by_lang[r["language"]].append(v)
    lang_mean = {l: [sum(v[i] for v in vs) / len(vs) for i in range(5)] for l, vs in by_lang.items()}
    nul = fill(null)
    variants = {}
    variants["always D"] = ["D"] * len(rows)
    variants["baseline (model's letter)"] = [r["answer"] for r in rows]
    variants["null-prompt calibration, alpha=1"] = [pick([v[i] - nul[i] for i in range(5)]) for v in raw]
    variants["investigation: alpha .65, beta .85"] = [
        pick([v[i] - ALPHA * nul[i] + BETA * math.log(INV_PRIORS[r["option_kind"] if r["option_kind"] in INV_PRIORS else "other"][LETTERS[i]])
              for i in range(5)]) for r, v in zip(rows, raw)]
    variants["investigation prior alone"] = [
        pick([math.log(INV_PRIORS[r["option_kind"] if r["option_kind"] in INV_PRIORS else "other"][LETTERS[i]]) for i in range(5)])
        for r in rows]
    variants["batch calibration (minus batch mean)"] = [
        pick([v[i] - lang_mean[r["language"]][i] for i in range(5)]) for r, v in zip(rows, z)]
    if "--balanced" in sys.argv:
        ans = [None] * len(rows)
        for lang in by_lang:
            idx = [i for i, r in enumerate(rows) if r["language"] == lang]
            got = balanced([rows[i] for i in idx], [z[i] for i in idx])
            for i, a in zip(idx, got):
                ans[i] = a
        variants["balanced assignment (equal letter counts)"] = ans
    print(f"  {'':44} accuracy   95% interval (by problem)   letters given A/B/C/D/E")
    for name, ans in variants.items():
        hit = sum(a == r["label"] for a, r in zip(ans, rows))
        lo, hi = boot(rows, ans)
        c = Counter(ans)
        print(f"  {name:44} {hit:3}/{len(rows)} = {100 * hit / len(rows):5.1f}%   {100 * lo:5.1f} to {100 * hi:5.1f}        "
              + "/".join(str(c[l]) for l in LETTERS))
    print(f"  {'truth':44} {'':35}" + "/".join(str(truth[l]) for l in LETTERS))
    print("  mean log-prob per letter (the model's letter bias): " + "  ".join(f"{l}:{m:.2f}" for l, m in zip(LETTERS, mean)))
    print("  null-prompt log-probs (normalised):                 " + "  ".join(f"{l}:{m:.2f}" for l, m in zip(LETTERS, log_softmax(nul))))
    for lang in sorted(by_lang):
        idx = [i for i, r in enumerate(rows) if r["language"] == lang]
        zz = [[z[i][k] - lang_mean[lang][k] for k in range(5)] for i in idx]
        real, p = perm_test([rows[i] for i in idx], zz)
        print(f"  signal test, {lang}: mean log-prob of the true letter after bias removal {real:.3f} "
              f"(chance {math.log(0.2):.3f}); permutation p = {p:.3f}")
    full = sum(r["letters_found"] == 5 for r in rows)
    top5 = sum(r.get("letters_in_top5") == 5 for r in rows)
    mass = sum(r["letter_mass"] for r in rows) / len(rows)
    print(f"  all five letters in the model's top 100 tokens: {full}/{len(rows)}; in its top 5 tokens: {top5}/{len(rows)}; "
          f"probability mass on A-E: {100 * mass:.1f}%")
    cached = [r["cache_n"] for r in rows if r.get("cache_n") is not None]
    read = [r["prompt_n"] for r in rows if r.get("prompt_n") is not None]
    if read:
        print(f"  tokens read per problem: mean {sum(read) / len(read):.0f}, max {max(read)}; "
              f"reused from cache: mean {sum(cached) / max(len(cached), 1):.1f} (min {min(cached) if cached else 'n/a'}, max {max(cached) if cached else 'n/a'})")
    return variants


def main():
    rows, null = [], None
    with open(sys.argv[1], encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "null_logprobs" in row:
                null = row["null_logprobs"]
                null = {l: null.get(l) for l in LETTERS}
            elif row.get("status") == "ok":
                rows.append(row)
    report("all rows", rows, null)
    for key, title in (("text_only", "text-only"),):
        report("text-only problems", [r for r in rows if r["text_only"]], null)
        fig = [r for r in rows if not r["text_only"]]
        if fig:
            print(f"\n=== figure problems: baseline {sum(r['answer'] == r['label'] for r in fig)}/{len(fig)}")
    print("\n=== baseline by level")
    by = defaultdict(list)
    for r in rows:
        by[r["level"]].append(r["answer"] == r["label"])
    print("  " + "   ".join(f"L{lv}: {sum(h)}/{len(h)}" for lv, h in sorted(by.items())))
    by = defaultdict(list)
    for r in rows:
        by[r["language"]].append(r["answer"] == r["label"])
    print("  " + "   ".join(f"{lang}: {sum(h)}/{len(h)}" for lang, h in sorted(by.items())))
    # agreement between the two languages of the same problem
    ans = defaultdict(dict)
    for r in rows:
        ans[r["id"]][r["language"]] = r["answer"]
    both = [d for d in ans.values() if len(d) == 2]
    if both:
        same = sum(d["eng"] == d["cat"] for d in both)
        print(f"  same letter in eng and cat for the same problem: {same}/{len(both)}")


if __name__ == "__main__":
    main()
