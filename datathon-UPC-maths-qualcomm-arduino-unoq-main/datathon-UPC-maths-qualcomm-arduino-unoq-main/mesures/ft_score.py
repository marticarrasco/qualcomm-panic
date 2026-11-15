"""Score the programs written by ft_gen.py: per model and set, with the current sandbox and with a plain one.

Also: paired comparison between models, the conditional Pass@3 of copilot/llm.py, voting over five programs,
and what to answer when nothing matches. Only the standard library is used.

Usage, from the Datathon folder:
  <venv python> -X utf8 mesures/ft_score.py
"""
import json
import math
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "datathon-UPC-maths-qualcomm-arduino-unoq"))
sys.path.insert(0, ROOT)
from copilot import pot  # noqa: E402

SMART = pot.RUNNER
PLAIN = SMART
for old, new in (
    ('safe["sum"] = smart_sum', 'safe["sum"] = real_sum'),
    ('safe["int"] = smart_int', 'safe["int"] = real_int'),
    ('safe["str"] = smart_str', 'safe["str"] = real_str'),
    ('"permutations": _wrap_iter(itertools.permutations)', '"permutations": itertools.permutations'),
    ('"combinations": _wrap_iter(itertools.combinations)', '"combinations": itertools.combinations'),
    ('"product": smart_product', '"product": itertools.product'),
    ('"combinations_with_replacement": _wrap_iter(itertools.combinations_with_replacement)',
     '"combinations_with_replacement": itertools.combinations_with_replacement'),
    (r'''    if len(printed) > 1 and printed[0] != printed[-1]:
        real_print(printed[0] + "\t" + printed[-1])
    else:
        real_print(printed[-1])''', '    real_print(printed[-1])'),
):
    assert old in PLAIN, old
    PLAIN = PLAIN.replace(old, new)
RUNNERS = {"smart": SMART, "plain": PLAIN}

CACHE_PATH = os.path.join(HERE, "ft-run-cache.json")
CACHE = json.load(open(CACHE_PATH, encoding="utf-8")) if os.path.exists(CACHE_PATH) else {}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def execute(runner, code):
    try:
        r = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", RUNNERS[runner]], input=code, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=4.0)
    except subprocess.TimeoutExpired:
        return [None, "timeout"]
    if r.returncode != 0:
        return [None, ((r.stderr or "").strip().splitlines() or ["error"])[-1][:80]]
    out = r.stdout.strip().splitlines()
    return [(out[-1].strip() if out else None), "ok"]


def run_all(jobs):
    todo = sorted({(runner, code) for runner, code in jobs if runner + "\x00" + code not in CACHE})
    if todo:
        with ThreadPoolExecutor(8) as pool:
            for (runner, code), res in zip(todo, pool.map(lambda j: execute(*j), todo)):
                CACHE[runner + "\x00" + code] = res
        json.dump(CACHE, open(CACHE_PATH, "w", encoding="utf-8"))


def letters_of(runner, code, problem):
    """(letters that match, printed, status), as pot.run_and_match does."""
    printed, status = CACHE[runner + "\x00" + code]
    nums = pot.numeric_options(problem)
    if printed and "\t" in printed:
        first_p, last_p = printed.split("\t", 1)
        letters = pot._match_one(nums, last_p)
        if len(letters) == 1:
            return letters, last_p, status
        first_letters = pot._match_one(nums, first_p)
        if len(first_letters) == 1:
            return first_letters, first_p, status
        return letters, last_p, status
    return pot.match_printed(nums, printed), printed, status


def mcnemar(b, c):
    """Exact two-sided p for b wins against c wins."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


SETS = {"val": os.path.join(ROOT, "data", "val.jsonl"), "demo": os.path.join(ROOT, "data", "demo", "train.jsonl")}
problems = {name: {(r["id"], r["language"]): r for r in load(path)} for name, path in SETS.items()}
gen = {}
for tag in ("stock", "ft1", "ft2", "ft3"):
    for name in SETS:
        path = os.path.join(HERE, f"ft-{tag}-{name}.jsonl")
        if os.path.exists(path):
            gen[(tag, name)] = load(path)
samples = {}
for tag in ("ft1", "ft2"):
    for name in SETS:
        path = os.path.join(HERE, f"ft-{tag}-{name}-samples.jsonl")
        if os.path.exists(path):
            samples[(tag, name)] = load(path)

for rows in list(gen.values()) + list(samples.values()):
    for o in rows:
        o["code"] = pot.extract_code(o["reply"])
run_all([(runner, o["code"]) for rows in gen.values() for o in rows for runner in RUNNERS])
run_all([("smart", o["code"]) for rows in samples.values() for o in rows])

verdict = {}   # (tag, set, runner) -> {(id, lang): "right" | "wrong" | "none"}
print("=" * 100)
print("GREEDY, ONE PROGRAM. Laptop, llama.cpp 9d25e00 (the board runs 2cbe0ec): confirm on the board.")
for name in SETS:
    allrows = problems[name]
    print(f"\n--- {name}: {len(allrows)} rows, {len({k[0] for k in allrows})} problems ---")
    for tag in ("stock", "ft1", "ft2", "ft3"):
        if (tag, name) not in gen:
            continue
        rows = gen[(tag, name)]
        for runner in ("plain", "smart"):
            v, answered = {}, {}
            printed_n = 0
            for o in rows:
                key = (o["id"], o["language"])
                letters, printed, status = letters_of(runner, o["code"], allrows[key])
                printed_n += printed is not None
                if len(letters) == 1:
                    v[key] = "right" if letters[0] == o["label"] else "wrong"
                    answered[key] = letters[0]
                else:
                    v[key] = "none"
            verdict[(tag, name, runner)] = v
            right = [k for k, x in v.items() if x == "right"]
            wrong = [k for k, x in v.items() if x == "wrong"]
            score_d = sum(answered.get(k, "D") == r["label"] for k, r in allrows.items())
            expected = len(right) + 0.2 * (len(allrows) - len(right) - len(wrong))
            ids_right = {k[0] for k in right}
            board = [o["read_n"] / 11 + o["predicted_n"] / 5.5 for o in rows[1:]]
            print(f"{tag:5s} {runner:5s}: asked {len(rows)}, printed {printed_n}, matched {len(right) + len(wrong)}, "
                  f"RIGHT {len(right)} (eng {sum(k[1] == 'eng' for k in right)}, cat {sum(k[1] == 'cat' for k in right)}; "
                  f"{len(ids_right)} problems), wrong {len(wrong)}, precision "
                  f"{100 * len(right) / max(len(right) + len(wrong), 1):.0f}% | score with D {score_d} "
                  f"({100 * score_d / len(allrows):.1f}%), with a 20% fallback {100 * expected / len(allrows):.1f}% | "
                  f"cut {sum(o['finish'] == 'length' for o in rows)}, tokens written {sum(o['predicted_n'] for o in rows) / len(rows):.0f}, "
                  f"board est. mean {sum(board) / len(board):.0f} s worst {max(board):.0f} s")

print("\n" + "=" * 100)
print("PAIRED, same rows, smart sandbox: rows only the first gets right / only the second (exact McNemar p)")
for name in SETS:
    for a, b in (("ft2", "ft3"), ("ft2", "ft1"), ("ft1", "ft3"), ("ft2", "stock"), ("ft3", "stock")):
        if (a, name, "smart") not in verdict or (b, name, "smart") not in verdict:
            continue
        va, vb = verdict[(a, name, "smart")], verdict[(b, name, "smart")]
        shared = set(va) & set(vb)
        va, vb = {k: va[k] for k in shared}, {k: vb[k] for k in shared}
        only_a = sum(va[k] == "right" and vb[k] != "right" for k in va)
        only_b = sum(vb[k] == "right" and va[k] != "right" for k in va)
        # by problem: a problem counts for a model if it gets it right in at least one language
        ids = {k[0] for k in va}
        pa = {i for i in ids if any(va.get((i, lang)) == "right" for lang in ("eng", "cat"))}
        pb = {i for i in ids if any(vb.get((i, lang)) == "right" for lang in ("eng", "cat"))}
        print(f"  {name:4s} {a} vs {b}: rows {only_a} / {only_b} (p = {mcnemar(only_a, only_b):.2f}); "
              f"problems {len(pa - pb)} / {len(pb - pa)} (p = {mcnemar(len(pa - pb), len(pb - pa)):.2f})")

print("\nBY LEVEL, smart sandbox: right / wrong / asked")
for (tag, name, runner), v in sorted(verdict.items()):
    if runner != "smart" or tag == "stock":
        continue
    lv = {}
    for k, x in v.items():
        lv.setdefault(problems[name][k]["level"], Counter())[x] += 1
    print(f"  {name:4s} {tag}: " + "  ".join(f"L{level} {c['right']}/{c['wrong']}/{sum(c.values())}" for level, c in sorted(lv.items())))
print("\nWHY NOTHING MATCHED, smart sandbox")
for (tag, name), rows in sorted(gen.items()):
    why = Counter()
    for o in rows:
        key = (o["id"], o["language"])
        if verdict[(tag, name, "smart")][key] != "none":
            continue
        letters, printed, status = letters_of("smart", o["code"], problems[name][key])
        if status != "ok":
            why["python error: " + status.split(":")[0][:28]] += 1
        elif printed is None:
            why["printed nothing"] += 1
        elif pot.to_value(printed) is None:
            why["printed no number"] += 1
        elif len(letters) > 1:
            why["matches several options"] += 1
        else:
            why["number that is not an option"] += 1
    print(f"  {name:4s} {tag:5s}: {dict(why.most_common())}")

print("\nSMART vs PLAIN sandbox, per model: rows gained / lost as right, and new wrong matches")
for (tag, name, runner), v in sorted(verdict.items()):
    if runner != "smart":
        continue
    plain = verdict[(tag, name, "plain")]
    print(f"  {name:4s} {tag:5s}: right +{sum(v[k] == 'right' and plain[k] != 'right' for k in v)} "
          f"-{sum(plain[k] == 'right' and v[k] != 'right' for k in v)}; wrong +"
          f"{sum(v[k] == 'wrong' and plain[k] != 'wrong' for k in v)} -{sum(plain[k] == 'wrong' and v[k] != 'wrong' for k in v)}")

print("\n" + "=" * 100)
print("MORE THAN ONE PROGRAM (smart sandbox). Samples: temperature 0.45, top_p 0.9, seeds 43 and 44")
print("\nAGREEMENT BETWEEN MODELS on the greedy program: letters that two or three models give")
for name in SETS:
    tags = [t for t in ("ft1", "ft2", "ft3") if (t, name) in gen]
    if len(tags) < 3:
        continue
    allrows = problems[name]
    letter = {}
    for t in tags:
        for o in gen[(t, name)]:
            key = (o["id"], o["language"])
            ls, _, _ = letters_of("smart", o["code"], allrows[key])
            letter[(t, key)] = ls[0] if len(ls) == 1 else None
    keys = [(o["id"], o["language"]) for o in gen[(tags[0], name)]]
    for combo in (("ft1", "ft2"), ("ft1", "ft3"), ("ft2", "ft3"), ("ft1", "ft2", "ft3")):
        agree = Counter()
        first = Counter()
        for key in keys:
            ls = [letter[(t, key)] for t in combo]
            votes = Counter(x for x in ls if x)
            label = allrows[key]["label"]
            if votes and votes.most_common(1)[0][1] >= 2:
                agree["right" if votes.most_common(1)[0][0] == label else "wrong"] += 1
            pick = next((x for x in ls if x), None)   # the first model that matches something
            if votes and votes.most_common(1)[0][1] >= 2:
                pick = votes.most_common(1)[0][0]
            first["right" if pick == label else "none" if pick is None else "wrong"] += 1
        print(f"  {name:4s} {'+'.join(combo):12s}: two agree -> RIGHT {agree['right']}, wrong {agree['wrong']} | "
              f"agreement first, else the first that matches -> RIGHT {first['right']}, wrong {first['wrong']}")
print()
for (stag, name), srows in sorted(samples.items()):
    if (stag, name) not in gen:
        continue
    allrows = problems[name]
    by_key = {}
    for o in srows:
        by_key.setdefault((o["id"], o["language"]), {})[o["seed"]] = o
    greedy = {(o["id"], o["language"]): o for o in gen[(stag, name)]}
    name = f"{stag} {name}"
    res = {k: Counter() for k in ("greedy", "pass3", "pass3_no_filter", "vote", "vote2", "vote3")}
    times = {"pass3": [], "vote": []}
    for key, g in greedy.items():
        prob = allrows[key]
        label = g["label"]

        def one(o):
            letters, printed, _ = letters_of("smart", o["code"], prob)
            return (letters[0] if len(letters) == 1 else None), printed

        g_letter, g_printed = one(g)
        extra = [by_key.get(key, {}).get(seed) for seed in (43, 44)]
        if any(e is None for e in extra):
            continue
        ex = [one(e) for e in extra]
        read_s = g["read_n"] / 11

        def tally(kind, letter):
            res[kind]["right" if letter == label else ("none" if letter is None else "wrong")] += 1

        tally("greedy", g_letter)
        # conditional Pass@3 as in copilot/llm.py and the notebook
        for kind in ("pass3", "pass3_no_filter"):
            answer, pending, spent = g_letter, None, read_s + g["predicted_n"] / 5.5
            for i in (0, 1):
                if answer is not None or spent > 35:   # llm.py: a retry needs 22 s left of 57
                    break
                letter, printed = ex[i]
                spent += extra[i]["predicted_n"] / 5.5
                if letter is None:
                    continue
                if kind == "pass3" and (printed or "").strip() == "1" and g_printed in (None, "", "0"):
                    if pending == letter:
                        answer = letter
                    pending = letter
                    continue
                answer = letter
            tally(kind, answer)
            if kind == "pass3":
                times["pass3"].append(spent)
        # voting over the greedy program and four samples
        votes = Counter(x for x in [g_letter] + [e[0] for e in ex] if x is not None)
        top = votes.most_common()
        if top:
            best = [x for x, c in top if c == top[0][1]]
            pick = g_letter if g_letter in best else best[0]
        else:
            pick = None
        tally("vote", pick)
        tally("vote2", pick if top and top[0][1] >= 2 else None)
        tally("vote3", pick if top and top[0][1] >= 3 else None)
        times["vote"].append(read_s + (g["predicted_n"] + sum(e["predicted_n"] for e in extra)) / 5.5)
    for kind, c in res.items():
        n = sum(c.values())
        if not n:
            continue
        expected = c["right"] + 0.2 * (len(allrows) - c["right"] - c["wrong"])
        t = times.get("pass3" if kind.startswith("pass3") else "vote" if kind.startswith("vote") else "", [])
        extra_txt = f" | board est. mean {sum(t) / len(t):.0f} s, worst {max(t):.0f} s, over 55 s: {sum(x > 55 for x in t)}" if t else ""
        print(f"  {name:8s} {kind:16s}: RIGHT {c['right']}, wrong {c['wrong']}, none {c['none']}, precision "
              f"{100 * c['right'] / max(c['right'] + c['wrong'], 1):.0f}% | all rows with a 20% fallback "
              f"{100 * expected / len(allrows):.1f}%{extra_txt}")

print("\n" + "=" * 100)
print("ROWS THE COPILOT SKIPS TODAY (FT1 greedy, smart sandbox): would asking them beat the 20% of a blind letter?")
import ft_gen  # noqa: E402
for name in SETS:
    for gate in ("figure", "units"):
        path = os.path.join(HERE, f"ft-ft1-{name}-{gate}.jsonl")
        if not os.path.exists(path):
            continue
        rows = load(path)
        for o in rows:
            o["code"] = pot.extract_code(o["reply"])
        run_all([("smart", o["code"]) for o in rows])
        c = Counter()
        for o in rows:
            prob = problems[name][(o["id"], o["language"])]
            if gate == "figure":
                letters, _, _ = letters_of("smart", o["code"], prob)
            else:
                printed, _ = CACHE["smart\x00" + o["code"]]
                nums = ft_gen.loose_options(prob)
                letters = []
                for part in (printed or "").split("\t")[::-1]:
                    letters = pot._match_one(nums, part)
                    if len(letters) == 1:
                        break
            c["right" if letters == [o["label"]] else "wrong" if len(letters) == 1 else "none"] += 1
        n = sum(c.values())
        gain = c["right"] - 0.2 * (c["right"] + c["wrong"])
        print(f"  {name:4s} {gate:6s}: asked {n}, RIGHT {c['right']}, wrong {c['wrong']}, none {c['none']}, precision "
              f"{100 * c['right'] / max(c['right'] + c['wrong'], 1):.0f}% | against a blind letter on the same rows: "
              f"{gain:+.1f} rows")

print("\n" + "=" * 100)
print("WHAT TO ANSWER WHEN NOTHING MATCHES (FT2 greedy, smart sandbox)")
for name in SETS:
    if ("ft2", name, "smart") not in verdict:
        continue
    allrows = problems[name]
    v = verdict[("ft2", name, "smart")]
    labels = Counter(allrows[k]["label"] for k, x in v.items() if x == "none")
    not_asked = Counter(r["label"] for k, r in allrows.items() if k not in v)
    near = Counter()
    for o in gen[("ft2", name)]:
        key = (o["id"], o["language"])
        if v[key] != "none":
            continue
        _, printed, _ = letters_of("smart", o["code"], allrows[key])
        value = pot.to_value(printed)
        if value is None:
            near["no number"] += 1
            continue
        nums = pot.numeric_options(allrows[key])
        pick = min(nums, key=lambda k: abs(nums[k] - value))
        near["nearest option right" if pick == o["label"] else "nearest option wrong"] += 1
    print(f"  {name}: label of the asked rows with no match {dict(sorted(labels.items()))}; of the rows not asked "
          f"{dict(sorted(not_asked.items()))}; printed a number that matches nothing -> {dict(near)}")
