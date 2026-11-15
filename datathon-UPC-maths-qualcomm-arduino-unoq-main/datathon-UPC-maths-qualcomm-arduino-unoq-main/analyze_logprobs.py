#!/usr/bin/env python3
"""Does the model know anything, once its letter bias is removed? Reads dump_logprobs.py files.

    python3 analyze_logprobs.py results/logprobs-val-eng.jsonl
    python3 analyze_logprobs.py results/logprobs-val-eng.jsonl results/logprobs-train-eng.jsonl

No model and no board needed: standard library only, a few seconds.

It compares ways to answer, all against the floor of the README, always D:

    baseline                 the letter the model picks today
    bias removed             the model, minus its taste for some letters
    model + frequencies      the same, weighed against how often each letter is right
    model + priors           the same, weighed against how often each letter is right
                             at this level and with this kind of options
    priors, no model         those priors alone

The weight of the model (beta) is chosen from the data. Beta 0 means that the
labels are better explained without the model, and the answer is then always D.

Everything that is fitted (the letter bias, the priors, beta) is fitted on four
fifths of the problems and scored on the other fifth, five times. A problem and
its translations stay in the same fifth. The 95% intervals resample problems,
not rows, for the same reason. The priors also use data/train.jsonl when it's
there (labels and option kinds only), which makes them steadier.
"""
import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

LETTERS = "ABCDE"
LETTER_SET = set(LETTERS)
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)  # weight of the model next to the priors
SHRINK = 20.0  # pseudo-counts that pull the prior of a small group towards the overall one
MIN_ROWS = 40  # below this, cross-validation says nothing

BASELINE = "baseline: the model's letter"
DEBIASED = "model, letter bias removed"
WITH_FREQUENCIES = "model + letter frequencies"
WITH_PRIORS = "model + priors by level, options"
PRIORS_ALONE = "priors by level, options; no model"


# --- Reading the rows -------------------------------------------------------

def load(paths):
    """One row per (id, language). A later or better row replaces an earlier one."""
    rank = {"ok": 2, "no_options": 1}
    rows = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = f"{row['id']}|{row.get('language', '')}"
                old = rows.get(key)
                if old is None or rank.get(row.get("status"), 0) >= rank.get(old.get("status"), 0):
                    row["key"] = key
                    rows[key] = row
    return list(rows.values())


def load_features(path, skip):
    """The label, level and kind of options of every problem in a data file that isn't
    in the dump. Enough to fit the answer priors, and it needs no model."""
    from dump_logprobs import option_kind  # imported here: without --train the analysis stands alone
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            problem = json.loads(line)
            key = f"{problem['id']}|{problem.get('language', '')}"
            if key in skip or problem.get("label") not in LETTER_SET:
                continue
            options = problem.get("options") or {}
            rows.append({"id": problem["id"], "language": problem.get("language", ""), "key": key,
                         "level": problem.get("level"), "label": problem["label"],
                         "has_options": bool(options), "option_kind": option_kind(options)})
    return rows


def normalise(logprobs):
    """Log probabilities over the five letters only. A letter the dump didn't find
    was below the model's top tokens: it gets a value under the lowest one found."""
    found = [v for v in logprobs.values() if v is not None]
    floor = min(found) - 2.0 if found else 0.0
    raw = [max(logprobs[l] if logprobs.get(l) is not None else floor, -100.0) for l in LETTERS]
    top = max(raw)
    total = top + math.log(sum(math.exp(v - top) for v in raw))
    return [v - total for v in raw]


def band(row):
    """Levels 0 and 1 (grades 5 and 6) have their own answer habits: A is rarely right."""
    try:
        return "low" if int(row.get("level")) <= 1 else "rest"
    except (TypeError, ValueError):
        return "rest"


def fold_of(problem_id, k):
    return int(hashlib.sha256(problem_id.encode()).hexdigest()[:8], 16) % k


# --- The three things that are fitted ---------------------------------------

def fit_bias(rows):
    """The mean log probability of each letter, per language: what the model says
    whatever the problem is. It needs no labels."""
    def mean(zs):
        return [sum(z[i] for z in zs) / len(zs) for i in range(5)]
    by_language = defaultdict(list)
    for r in rows:
        by_language[r["language"]].append(r["z"])
    bias = {"*": mean([r["z"] for r in rows])}
    for language, zs in by_language.items():
        if len(zs) >= 30:
            bias[language] = mean(zs)
    return bias


def fit_priors(rows):
    """How often each letter is right: overall, and for each (kind of options, level band).
    A group's figures are pulled towards the overall ones, so small groups don't invent habits."""
    overall = Counter(r["label"] for r in rows)
    base = [(overall[l] + 1) / (sum(overall.values()) + 5) for l in LETTERS]
    groups = defaultdict(Counter)
    for r in rows:
        groups[(r["option_kind"], band(r))][r["label"]] += 1
    cells = {g: [(c[l] + SHRINK * base[i]) / (sum(c.values()) + SHRINK) for i, l in enumerate(LETTERS)]
             for g, c in groups.items()}
    return {"base": base, "cells": cells}


def log_prior(priors, row, by_group):
    p = priors["cells"].get((row["option_kind"], band(row)), priors["base"]) if by_group else priors["base"]
    return [math.log(v) for v in p]


def scores(row, bias, prior, beta):
    """beta times the model's opinion with its bias removed, plus the log prior (if any)."""
    b = bias.get(row["language"], bias["*"])
    return [beta * (row["z"][i] - b[i]) + (prior[i] if prior else 0.0) for i in range(5)]


def log_softmax(values):
    top = max(values)
    total = top + math.log(sum(math.exp(v - top) for v in values))
    return [v - total for v in values]


def fit_beta(rows, bias, priors, by_group):
    """The weight of the model that makes the true letters most likely. 0 means:
    ignore the model, the priors alone explain the labels better."""
    def likelihood(beta):
        return sum(log_softmax(scores(r, bias, log_prior(priors, r, by_group), beta))[LETTERS.index(r["label"])]
                   for r in rows)
    return max(BETAS, key=likelihood)


def pick(values):
    return LETTERS[max(range(5), key=lambda i: values[i])]


def cross_validate(asked, empty, extra, k):
    """Out-of-fold answers for every fitted variant, and the beta chosen in each fold.
    extra are labelled problems outside the dump: they only help to fit the priors."""
    answers = defaultdict(dict)
    betas = []
    for fold in range(k):
        train = [r for r in asked if r["fold"] != fold]
        if not train:
            continue
        bias = fit_bias(train)
        priors = fit_priors(train + [r for r in empty + extra if r["fold"] != fold])
        beta = fit_beta(train, bias, priors, by_group=False)
        beta_group = fit_beta(train, bias, priors, by_group=True)
        betas.append(beta)
        for r in asked:
            if r["fold"] == fold:
                plain, group = log_prior(priors, r, False), log_prior(priors, r, True)
                answers[DEBIASED][r["key"]] = pick(scores(r, bias, None, 1.0))
                answers[WITH_FREQUENCIES][r["key"]] = pick(scores(r, bias, plain, beta))
                answers[WITH_PRIORS][r["key"]] = pick(scores(r, bias, group, beta_group))
                answers[PRIORS_ALONE][r["key"]] = pick(group)
        for r in empty:
            if r["fold"] == fold:  # nothing to ask the model: the priors answer
                answers[WITH_FREQUENCIES][r["key"]] = pick(log_prior(priors, r, False))
                answers[WITH_PRIORS][r["key"]] = answers[PRIORS_ALONE][r["key"]] = pick(log_prior(priors, r, True))
    return answers, betas


# --- Intervals and tests ----------------------------------------------------

def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q / 100
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def bootstrap(rows, answers, reference, n_boot):
    """95% intervals for each variant's accuracy and for its gain over the reference,
    resampling problem ids so that translations of a problem move together."""
    ids = sorted({r["id"] for r in rows})
    position = {pid: i for i, pid in enumerate(ids)}
    size = [0] * len(ids)
    hits = {name: [0] * len(ids) for name in answers}
    for r in rows:
        i = position[r["id"]]
        size[i] += 1
        for name, given in answers.items():
            hits[name][i] += given[r["key"]] == r["label"]
    rng = random.Random(42)
    accuracy = {name: [] for name in answers}
    gain = {name: [] for name in answers}
    for _ in range(n_boot):
        sample = [rng.randrange(len(ids)) for _ in ids]
        n = sum(size[i] for i in sample)
        got = {name: sum(h[i] for i in sample) / n for name, h in hits.items()}
        for name in answers:
            accuracy[name].append(got[name])
            gain[name].append(got[name] - got[reference])
    interval = lambda xs: (percentile(xs, 2.5), percentile(xs, 97.5))
    return {n: interval(v) for n, v in accuracy.items()}, {n: interval(v) for n, v in gain.items()}


def shuffle_test(rows, n_perm):
    """Is the model's answer related to the right one at all? Shuffle the labels among
    problems of the same kind and level band, so the priors stay intact, and see how
    often the bias-removed model does as well on shuffled labels as on the real ones.
    One language at a time: within a language every problem appears once."""
    bias = fit_bias(rows)
    values = [scores(r, bias, None, 1.0) for r in rows]
    picked = [pick(v) for v in values]
    logq = [log_softmax(v) for v in values]
    labels = [r["label"] for r in rows]
    strata = defaultdict(list)
    for i, r in enumerate(rows):
        strata[(r["option_kind"], band(r))].append(i)

    def measure(given):
        accuracy = sum(p == y for p, y in zip(picked, given)) / len(given)
        likelihood = sum(q[LETTERS.index(y)] for q, y in zip(logq, given)) / len(given)
        return accuracy, likelihood

    real = measure(labels)
    rng = random.Random(42)
    chance, as_good_accuracy, as_good_likelihood = [], 0, 0
    for _ in range(n_perm):
        shuffled = labels[:]
        for members in strata.values():
            pool = [labels[i] for i in members]
            rng.shuffle(pool)
            for i, y in zip(members, pool):
                shuffled[i] = y
        accuracy, likelihood = measure(shuffled)
        chance.append(accuracy)
        as_good_accuracy += accuracy >= real[0] - 1e-12
        as_good_likelihood += likelihood >= real[1] - 1e-12
    return {"accuracy": real[0], "chance": statistics.mean(chance), "chance_95": percentile(chance, 95),
            "p_accuracy": (1 + as_good_accuracy) / (1 + n_perm),
            "p_likelihood": (1 + as_good_likelihood) / (1 + n_perm)}


def confidence_halves(rows):
    """Accuracy of the bias-removed model on its surest half and on its least sure half.
    A model that knows something is right more often when it is sure."""
    bias = fit_bias(rows)
    scored = []
    for r in rows:
        values = scores(r, bias, None, 1.0)
        first, second = sorted(values, reverse=True)[:2]
        scored.append((first - second, pick(values) == r["label"]))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    half = len(scored) // 2
    sure, unsure = scored[:half], scored[half:]
    return sum(c for _, c in sure) / len(sure), sum(c for _, c in unsure) / len(unsure)


# --- The report -------------------------------------------------------------

def pct(x):
    return f"{100 * x:5.1f}%"


def table(title, rows, answers, n_boot):
    print(f"\n{title} (n={len(rows)}, {len({r['id'] for r in rows})} different problems)")
    print(f"  {'':36} accuracy   95% interval     gain over always D")
    intervals, gains = bootstrap(rows, answers, "always D", n_boot)
    floor = sum(r["label"] == "D" for r in rows) / len(rows)
    for name, given in answers.items():
        accuracy = sum(given[r["key"]] == r["label"] for r in rows) / len(rows)
        low, high = intervals[name]
        line = f"  {name:36} {pct(accuracy)}   {pct(low)} to {pct(high)}"
        if name != "always D":
            g_low, g_high = gains[name]
            line += f"   {100 * (accuracy - floor):+5.1f}  ({100 * g_low:+.1f} to {100 * g_high:+.1f})"
        print(line)
    return gains


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="results/logprobs-*.jsonl from dump_logprobs.py")
    p.add_argument("--train", default="data/train.jsonl",
                   help="labelled problems that also feed the answer priors; skipped if the file is missing")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--boot", type=int, default=1000, help="resamples for the 95%% intervals")
    p.add_argument("--perm", type=int, default=2000, help="label shuffles for the test")
    args = p.parse_args()

    rows = [r for r in load(args.files) if r.get("label") in LETTER_SET]
    extra = load_features(args.train, {r["key"] for r in rows}) if os.path.exists(args.train) else []
    for r in rows + extra:
        r["fold"] = fold_of(r["id"], args.folds)
    asked = [r for r in rows if r.get("status") == "ok" and r.get("has_options") and r.get("logprobs")]
    empty = [r for r in rows if not r.get("has_options")]
    failed = len(rows) - len(asked) - len(empty)
    for r in asked:
        r["z"] = normalise(r["logprobs"])
    languages = Counter(r["language"] for r in rows)
    print(f"Rows: {len(asked)} with options and a model answer, {len(empty)} without options, "
          f"{failed} failed   ({', '.join(f'{l} {n}' for l, n in sorted(languages.items()))})")
    if not asked:
        sys.exit("No rows with a model answer. Run dump_logprobs.py first.")
    print("Answer priors: fitted on the dumped problems outside each fold"
          + (f", plus {len(extra)} problems from {args.train}" if extra else
             f" only ({args.train} not found: they will be noisy)"))

    # 1. Is the dump sound?
    complete = sum(r.get("letters_found") == 5 for r in asked)
    mass = statistics.mean(r.get("letter_mass") or 0 for r in asked)
    print(f"All five letters among the model's top tokens: {complete}/{len(asked)} rows. "
          f"Share of the model's probability on A to E: {pct(mass).strip()} on average")
    if complete < 0.95 * len(asked):
        print("  Many rows miss a letter: their values are floors, not measurements. Dump again with a higher --top.")
    differ = sum(r.get("answer") != pick(r["z"]) for r in asked)
    if differ:
        print(f"  {differ} rows where the answer isn't the likeliest letter of the dump (ties or missing letters).")
    seconds = [r["seconds"] for r in asked if r.get("seconds") is not None]
    read = [r["prompt_n"] for r in asked if r.get("prompt_n") is not None]
    cached = [r["cache_n"] for r in asked if r.get("cache_n") is not None]
    if seconds:
        line = (f"Latency: mean {statistics.mean(seconds):.1f} s, median {statistics.median(seconds):.1f} s, "
                f"worst {max(seconds):.1f} s")
        if read:
            line += f"; prompt tokens read: mean {statistics.mean(read):.0f}"
        if cached:
            line += f", reused from the cache: mean {statistics.mean(cached):.0f}"
        print(line)

    if len(asked) < MIN_ROWS:
        hits = sum(r["answer"] == r["label"] for r in asked)
        print(f"\nBaseline: {hits}/{len(asked)} right. Too few rows to fit and test anything "
              f"(under {MIN_ROWS}): dump more problems.")
        return

    # 2. The ways to answer, cross-validated
    fitted, betas = cross_validate(asked, empty, extra, args.folds)
    only = lambda name: {r["key"]: fitted[name][r["key"]] for r in asked}
    with_options = {
        "always D": {r["key"]: "D" for r in asked},
        BASELINE: {r["key"]: r["answer"] for r in asked},
        DEBIASED: fitted[DEBIASED],
        WITH_FREQUENCIES: only(WITH_FREQUENCIES),
        WITH_PRIORS: only(WITH_PRIORS),
        PRIORS_ALONE: only(PRIORS_ALONE),
    }
    gains = table("Problems with options", asked, with_options, args.boot)

    everything = asked + empty
    if empty:
        overall = {"always D": {r["key"]: "D" for r in everything}}
        if all(r.get("answer") for r in empty):  # dumped with --include-empty
            overall[BASELINE] = {r["key"]: r["answer"] for r in everything}
        for name in (WITH_FREQUENCIES, WITH_PRIORS, PRIORS_ALONE):
            overall[name] = fitted[name]
        table("All problems (without options, only the priors answer)", everything, overall, args.boot)
        asked_empty = [r for r in empty if r.get("answer")]
        if asked_empty:
            right = sum(r["answer"] == r["label"] for r in asked_empty)
            d = sum(r["label"] == "D" for r in asked_empty)
            print(f"  Without options the model's letter is right in {pct(right / len(asked_empty)).strip()} "
                  f"of {len(asked_empty)} problems; always D gets {pct(d / len(asked_empty)).strip()}")

    # 3. Does the model know anything?
    print("\nDoes the model's answer have anything to do with the right one?")
    signal = []
    for language in sorted({r["language"] for r in asked}):
        group = [r for r in asked if r["language"] == language]
        if len(group) < MIN_ROWS:
            continue
        test = shuffle_test(group, args.perm)
        sure, unsure = confidence_halves(group)
        signal.append(test["p_likelihood"])
        print(f"  {language or '?'}: bias removed, the model is right in {pct(test['accuracy']).strip()}. "
              f"With shuffled labels: {pct(test['chance']).strip()} on average, "
              f"under {pct(test['chance_95']).strip()} 19 times in 20.")
        print(f"      p = {test['p_accuracy']:.3f} on accuracy, {test['p_likelihood']:.3f} on the probability "
              f"it gives the right letter (small p: the model knows something)")
        print(f"      its surest half: {pct(sure).strip()} right; its least sure half: {pct(unsure).strip()}")
    print(f"  Weight the data gives the model next to the letter frequencies (beta, one per fold): "
          f"{', '.join(f'{b:g}' for b in betas)}   (0 = ignore the model)")

    # 4. Which letters each variant answers
    print(f"\nLetters, problems with options        {'   '.join(f'{l:>4}' for l in LETTERS)}")
    truth = Counter(r["label"] for r in asked)
    print(f"  {'truth':36} " + "   ".join(f"{truth[l]:4}" for l in LETTERS))
    for name in (BASELINE, DEBIASED, WITH_FREQUENCIES, WITH_PRIORS, PRIORS_ALONE):
        counts = Counter(with_options[name][r["key"]] for r in asked)
        print(f"  {name:36} " + "   ".join(f"{counts[l]:4}" for l in LETTERS))

    # 5. In one sentence
    low, high = gains[WITH_FREQUENCIES]
    floor = sum(r["label"] == "D" for r in asked) / len(asked)
    gain = sum(with_options[WITH_FREQUENCIES][r["key"]] == r["label"] for r in asked) / len(asked) - floor
    print("\nVerdict")
    if low > 0:
        print(f"  Calibrated, the model beats always D by {100 * gain:+.1f} points on problems with options "
              f"(95% interval {100 * low:+.1f} to {100 * high:+.1f}). It knows something: keep it, calibrated.")
    else:
        print(f"  No evidence that the calibrated model beats always D on problems with options: "
              f"{100 * gain:+.1f} points (95% interval {100 * low:+.1f} to {100 * high:+.1f}).")
        if signal and min(signal) < 0.01:
            print("  The shuffle test does find a link between its answers and the right ones (p under 0.01), "
                  "so the signal is there but weak: more problems would say how many points it is worth.")
        else:
            print("  With this model, answering D every time is as good, and takes no time.")


if __name__ == "__main__":
    main()
