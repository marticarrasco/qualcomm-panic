#!/usr/bin/env python3
"""Model-free answer priors for Kangaroo multiple-choice problems.

Question: can a guess that uses no LLM at all beat the fixed fallback letter "D"?

Run from the project root (stdlib only, no network, no LLM):

    python3 experiments/priors_analysis.py

Protocol
- One row per problem id (English rows only), so translations are not counted twice.
- Everything fitted is fitted on train only. Reported on val, and with 5-fold
  cross-validation on train. Folds are grouped by normalised question text, so the
  same problem printed at two levels (same text, options reordered) never sits on
  both sides of a fold.
- Every accuracy comes with n and a binomial standard error; every comparison
  against always-D is paired (per-problem difference, its standard error and z).
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import math
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LETTERS = "ABCDE"
FOLDS = 5
SEEDS = (0, 1, 2, 3, 4)  # fold assignments; CV numbers are averaged over them


# ----------------------------------------------------------------- loading
def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    eng = [r for r in rows if r.get("language") == "eng" and r.get("label") in LETTERS]
    seen, out = set(), []
    for r in eng:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def qnum(row: dict) -> int | None:
    m = re.search(r"_(\d+)$", str(row.get("id", "")))
    return int(m.group(1)) if m else None


def group_key(row: dict) -> str:
    text = re.sub(r"^\s*\d+\s*[.)]\s*", "", row.get("text") or "").lower()
    key = re.sub(r"[^a-z0-9]+", "", text)[:80]
    return key or row["id"]


def fold_of(row: dict, seed: int) -> int:
    return zlib.crc32(f"{seed}|{group_key(row)}".encode()) % FOLDS


# ------------------------------------------------------------ option features
NUM_RE = re.compile(
    r"^\s*([-−–]?\s*\d+(?:[.,]\d+)?)(?:\s*/\s*(\d+))?\s*(?:%|°|[a-zA-Z€$²³/]{1,6}\.?)?\s*$"
)


def to_number(text: str) -> float | None:
    m = NUM_RE.match(text or "")
    if not m:
        return None
    value = float(re.sub(r"\s", "", m.group(1)).replace("−", "-").replace("–", "-").replace(",", "."))
    if m.group(2):
        den = float(m.group(2))
        if den == 0:
            return None
        value /= den
    return value


def option_values(row: dict) -> dict[str, float] | None:
    """Numeric value of each option, or None unless all five parse and are distinct."""
    opts = row.get("options") or {}
    if sorted(opts) != list(LETTERS):
        return None
    values = {k: to_number(str(opts[k])) for k in LETTERS}
    if any(v is None for v in values.values()) or len(set(values.values())) != 5:
        return None
    return values


def has_options(row: dict) -> bool:
    return sorted(row.get("options") or {}) == list(LETTERS)


def ranks(values: dict[str, float]) -> dict[str, int]:
    order = sorted(LETTERS, key=lambda k: values[k])
    return {k: i + 1 for i, k in enumerate(order)}


def ordering(row: dict) -> str:
    values = option_values(row)
    if values is None:
        return "nonnumeric"
    seq = [values[k] for k in LETTERS]
    if seq == sorted(seq):
        return "ascending"
    if seq == sorted(seq, reverse=True):
        return "descending"
    return "shuffled"


def question_numbers(row: dict) -> list[float]:
    text = re.sub(r"^\s*\d+\s*[.)]\s*", "", row.get("text") or "")
    out = []
    for tok in re.findall(r"\d+(?:[.,]\d+)?", text):
        try:
            out.append(float(tok.replace(",", ".")))
        except ValueError:
            pass
    return out


def unique_pick(scores: dict[str, float], largest: bool = True) -> str | None:
    """Letter with the strictly largest (or smallest) score, else None."""
    best = max(scores.values()) if largest else min(scores.values())
    winners = [k for k in LETTERS if scores[k] == best]
    return winners[0] if len(winners) == 1 else None


# Rules: row -> letter, or None when the rule does not apply to this problem.
def rule_rank(n: int):
    def rule(row):
        values = option_values(row)
        if values is None:
            return None
        return next(k for k, r in ranks(values).items() if r == n)
    return rule


def rule_isolated(row):
    """Numeric odd-one-out: value farthest from its nearest neighbour."""
    values = option_values(row)
    if values is None:
        return None
    gap = {k: min(abs(values[k] - values[j]) for j in LETTERS if j != k) for k in LETTERS}
    return unique_pick(gap)


def rule_crowded(row):
    """Value closest to its nearest neighbour is ambiguous (pairs), so use the
    smallest summed distance to all other values (most central value)."""
    values = option_values(row)
    if values is None:
        return None
    total = {k: sum(abs(values[k] - values[j]) for j in LETTERS) for k in LETTERS}
    return unique_pick(total, largest=False)


def rule_longest(row):
    if not has_options(row):
        return None
    return unique_pick({k: len(str(row["options"][k]).strip()) for k in LETTERS})


def rule_shortest(row):
    if not has_options(row):
        return None
    return unique_pick({k: len(str(row["options"][k]).strip()) for k in LETTERS}, largest=False)


def similarity(row) -> dict[str, float] | None:
    if not has_options(row):
        return None
    texts = {k: str(row["options"][k]).strip().lower() for k in LETTERS}
    return {
        k: sum(difflib.SequenceMatcher(None, texts[k], texts[j]).ratio() for j in LETTERS if j != k)
        for k in LETTERS
    }


def rule_most_similar(row):
    """The option that differs least from the others (distractors are built around the key)."""
    sim = similarity(row)
    return unique_pick(sim) if sim else None


def rule_least_similar(row):
    sim = similarity(row)
    return unique_pick(sim, largest=False) if sim else None


def in_question(row) -> dict[str, bool] | None:
    values = option_values(row)
    if values is None:
        return None
    nums = set(question_numbers(row))
    return {k: values[k] in nums for k in LETTERS}


def rule_only_in_question(row):
    flags = in_question(row)
    if not flags or sum(flags.values()) != 1:
        return None
    return next(k for k in LETTERS if flags[k])


def rule_only_not_in_question(row):
    flags = in_question(row)
    if not flags or sum(flags.values()) != 4:
        return None
    return next(k for k in LETTERS if not flags[k])


def derived(row) -> dict[str, bool] | None:
    """Option equals a sum, difference, product or quotient of two numbers in the question."""
    values = option_values(row)
    if values is None:
        return None
    nums = question_numbers(row)[:12]
    made = set()
    for i, a in enumerate(nums):
        for j, b in enumerate(nums):
            if i == j:
                continue
            made.update((a + b, a - b, a * b))
            if b:
                made.add(a / b)
    made = {round(v, 6) for v in made}
    return {k: round(values[k], 6) in made for k in LETTERS}


def rule_only_derived(row):
    flags = derived(row)
    if not flags or sum(flags.values()) != 1:
        return None
    return next(k for k in LETTERS if flags[k])


def rule_round_odd(row):
    """Typical-looking: the only multiple of 5 among the options, when exactly one is."""
    values = option_values(row)
    if values is None:
        return None
    flags = {k: values[k] % 5 == 0 for k in LETTERS}
    if sum(flags.values()) != 1:
        return None
    return next(k for k in LETTERS if flags[k])


def rule_non_round_odd(row):
    values = option_values(row)
    if values is None:
        return None
    flags = {k: values[k] % 5 == 0 for k in LETTERS}
    if sum(flags.values()) != 4:
        return None
    return next(k for k in LETTERS if not flags[k])


def rule_arith_break(row):
    """Four options in arithmetic progression plus one that breaks it: pick the breaker."""
    values = option_values(row)
    if values is None:
        return None
    hits = []
    for k in LETTERS:
        rest = sorted(values[j] for j in LETTERS if j != k)
        steps = {round(rest[i + 1] - rest[i], 6) for i in range(3)}
        if len(steps) == 1:
            hits.append(k)
    return hits[0] if len(hits) == 1 else None


RULES = [
    ("numeric: smallest value", rule_rank(1)),
    ("numeric: 2nd smallest", rule_rank(2)),
    ("numeric: median value", rule_rank(3)),
    ("numeric: 2nd largest", rule_rank(4)),
    ("numeric: largest value", rule_rank(5)),
    ("numeric: most isolated value (odd one out)", rule_isolated),
    ("numeric: most central value", rule_crowded),
    ("numeric: breaks an arithmetic progression", rule_arith_break),
    ("numeric: only multiple of 5", rule_round_odd),
    ("numeric: only non-multiple of 5", rule_non_round_odd),
    ("numeric: only option that appears in question", rule_only_in_question),
    ("numeric: only option NOT in question", rule_only_not_in_question),
    ("numeric: only option = a op b of question numbers", rule_only_derived),
    ("text: longest option (unique)", rule_longest),
    ("text: shortest option (unique)", rule_shortest),
    ("text: most similar to the other options", rule_most_similar),
    ("text: least similar to the other options", rule_least_similar),
]


# ------------------------------------------------------------------ statistics
def pct(hits: int, n: int) -> str:
    if not n:
        return "   -   (n=0)"
    p = hits / n
    return f"{hits:4d}/{n:<4d} {100 * p:5.1f}% ±{100 * math.sqrt(p * (1 - p) / n):4.1f}"


def paired(pred: list[str], base: list[str], gold: list[str]) -> str:
    n = len(gold)
    if n < 2:
        return "-"
    diffs = [(p == g) - (b == g) for p, b, g in zip(pred, base, gold)]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    z = mean / se if se else 0.0
    return f"{100 * mean:+5.1f}pp ±{100 * se:4.1f} (z={z:+.2f})"


def best_letter(rows: list[dict]) -> str:
    counts = Counter(r["label"] for r in rows)
    return max(LETTERS, key=lambda k: (counts[k], k == "D"))


def dist(rows: list[dict]) -> str:
    counts = Counter(r["label"] for r in rows)
    n = len(rows) or 1
    return "  ".join(f"{k}:{counts[k]:4d} ({100 * counts[k] / n:4.1f}%)" for k in LETTERS)


def chi2_uniform(rows: list[dict]) -> float:
    counts = Counter(r["label"] for r in rows)
    expected = len(rows) / 5
    return sum((counts[k] - expected) ** 2 / expected for k in LETTERS) if rows else 0.0


# ------------------------------------------------------- fitted predictors
def fit_always_d(rows):
    return lambda row: "D"


def fit_best_letter(rows):
    letter = best_letter(rows)
    return lambda row: letter


def fit_conditional(key_fn, min_count: int = 40):
    """Most frequent train letter within each key; backs off to the overall best letter."""
    def fit(rows):
        overall = best_letter(rows)
        groups = defaultdict(list)
        for r in rows:
            groups[key_fn(r)].append(r)
        table = {k: best_letter(v) for k, v in groups.items() if len(v) >= min_count}
        return lambda row: table.get(key_fn(row), overall)
    return fit


def third(row):
    q = qnum(row)
    level = row.get("level")
    if q is None:
        return None
    size = 8 if level in (0, 1) else 10  # 24-question papers at the lower levels
    return min(2, (q - 1) // size)


CONDITIONALS = [
    ("by level", lambda r: r.get("level")),
    ("by level band (0-1 / 2-3 / 4-5 / 6-7)", lambda r: (r.get("level") or 0) // 2),
    ("by year", lambda r: r.get("year")),
    ("by year band (5 years)", lambda r: (r.get("year") or 0) // 5),
    ("by text_only", lambda r: r.get("text_only")),
    ("by question number", qnum),
    ("by third of the paper (3/4/5 points)", third),
    ("by level x text_only", lambda r: (r.get("level"), r.get("text_only"))),
    ("by third x text_only", lambda r: (third(r), r.get("text_only"))),
    ("by level band x third", lambda r: ((r.get("level") or 0) // 2, third(r))),
]


# ------------------------------------------------- additive log-odds scorer
def feat_letter(row, k):
    return k


def feat_rank(row, k):
    values = option_values(row)
    return None if values is None else ranks(values)[k]


def feat_rank_shuffled(row, k):
    """Value rank, but only when the options are not printed in sorted order
    (in sorted order the rank is just the letter)."""
    if ordering(row) != "shuffled":
        return None
    return ranks(option_values(row))[k]


def feat_length(row, k):
    if not has_options(row):
        return None
    lengths = {j: len(str(row["options"][j]).strip()) for j in LETTERS}
    if unique_pick(lengths) == k:
        return "longest"
    if unique_pick(lengths, largest=False) == k:
        return "shortest"
    return "other"


def feat_similarity(row, k):
    sim = similarity(row)
    if sim is None:
        return None
    if unique_pick(sim) == k:
        return "most"
    if unique_pick(sim, largest=False) == k:
        return "least"
    return "other"


def feat_in_question(row, k):
    flags = in_question(row)
    return None if flags is None else flags[k]


def feat_isolated(row, k):
    if option_values(row) is None:
        return None
    return rule_isolated(row) == k


FEATURES = {
    "letter": feat_letter,
    "rank": feat_rank,
    "rank_shuffled": feat_rank_shuffled,
    "length": feat_length,
    "similarity": feat_similarity,
    "in_question": feat_in_question,
    "isolated": feat_isolated,
}
ALPHA = 20.0  # pseudo-count pulling every rate toward its feature-value base rate


def fit_scorer(names: tuple[str, ...]):
    def fit(rows):
        weights = {}
        for name in names:
            fn = FEATURES[name]
            correct, total = Counter(), Counter()
            for r in rows:
                for k in LETTERS:
                    v = fn(r, k)
                    if v is None:
                        continue
                    total[v] += 1
                    correct[v] += k == r["label"]
            # Base rate of "is the key" is 1/5 for every option.
            weights[name] = {
                v: math.log((correct[v] + ALPHA * 0.2) / (total[v] + ALPHA)) - math.log(0.2)
                for v in total
            }
        letter_order = sorted(LETTERS, key=lambda k: -Counter(r["label"] for r in rows)[k])

        def predict(row):
            score = {}
            for k in LETTERS:
                s = 0.0
                for name in names:
                    v = FEATURES[name](row, k)
                    if v is not None:
                        s += weights[name].get(v, 0.0)
                score[k] = s
            return max(letter_order, key=lambda k: score[k])  # ties -> more frequent letter
        predict.weights = weights
        return predict
    return fit


SCORERS = [
    ("letter", ("letter",)),
    ("letter + rank", ("letter", "rank")),
    ("letter + rank_shuffled", ("letter", "rank_shuffled")),
    ("letter + similarity", ("letter", "similarity")),
    ("letter + length", ("letter", "length")),
    ("letter + in_question", ("letter", "in_question")),
    ("letter + rank_shuffled + similarity", ("letter", "rank_shuffled", "similarity")),
    ("letter + rank_shuffled + similarity + length", ("letter", "rank_shuffled", "similarity", "length")),
    ("all seven features", tuple(FEATURES)),
]


# ------------------------------------------------------------- evaluation
def cross_validate(fit, rows, subset=None):
    """Mean CV hits over SEEDS, plus the paired comparison against always-D (seed 0)."""
    subset = subset or (lambda r: True)
    hits_per_seed, first = [], None
    for seed in SEEDS:
        preds, gold = [], []
        for fold in range(FOLDS):
            train = [r for r in rows if fold_of(r, seed) != fold]
            test = [r for r in rows if fold_of(r, seed) == fold and subset(r)]
            predict = fit(train)
            preds += [predict(r) for r in test]
            gold += [r["label"] for r in test]
        hits_per_seed.append(sum(p == g for p, g in zip(preds, gold)))
        if first is None:
            first = (preds, gold)
    n = len(first[1])
    mean_hits = sum(hits_per_seed) / len(hits_per_seed)
    return mean_hits, n, first


def report(name, fit, train, val, subset=None, width=46):
    subset = subset or (lambda r: True)
    mean_hits, n, (preds, gold) = cross_validate(fit, train, subset)
    p = mean_hits / n if n else 0.0
    cv = f"{mean_hits:6.1f}/{n:<4d} {100 * p:5.1f}% ±{100 * math.sqrt(p * (1 - p) / n):4.1f}" if n else "-"
    cv_vs_d = paired(preds, ["D"] * n, gold)
    predict = fit(train)
    vrows = [r for r in val if subset(r)]
    vpred = [predict(r) for r in vrows]
    vgold = [r["label"] for r in vrows]
    vhits = sum(a == b for a, b in zip(vpred, vgold))
    print(f"  {name:<{width}} CV {cv}  vs D {cv_vs_d:<26} | val {pct(vhits, len(vrows))}  vs D {paired(vpred, ['D'] * len(vrows), vgold)}")


def header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def breakdown(title, rows, key_fn):
    print(f"\n  {title}")
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    for key in sorted(groups, key=lambda k: (k is None, str(k) if not isinstance(k, (int, float)) else "", k if isinstance(k, (int, float)) else 0)):
        g = groups[key]
        counts = Counter(r["label"] for r in g)
        top = best_letter(g)
        print(
            f"    {str(key):<14} n={len(g):4d}  "
            + " ".join(f"{k}:{100 * counts[k] / len(g):4.1f}" for k in LETTERS)
            + f"   best={top} {100 * counts[top] / len(g):4.1f}%  D={100 * counts['D'] / len(g):4.1f}%  chi2={chi2_uniform(g):5.1f}"
        )


def main() -> None:
    train = load(DATA / "train.jsonl")
    val = load(DATA / "val.jsonl")
    full_train = load(DATA / "full_noocr" / "train.jsonl")
    full_val = load(DATA / "full_noocr" / "val.jsonl")

    header("0. Data")
    for name, rows in (("train (OCR set)", train), ("val (OCR set)", val),
                       ("full_noocr train", full_train), ("full_noocr val", full_val)):
        with_opts = sum(has_options(r) for r in rows)
        numeric = sum(option_values(r) is not None for r in rows)
        print(f"  {name:<18} ids={len(rows):4d}  with options={with_opts:4d}  all-numeric options={numeric:4d}")
    ids_train = {r["id"] for r in train} | {r["id"] for r in full_train}
    ids_val = {r["id"] for r in val} | {r["id"] for r in full_val}
    keys_train = {group_key(r) for r in train + full_train}
    print(f"  id overlap train/val (both sets pooled): {len(ids_train & ids_val)}")
    twins = sum(group_key(r) in keys_train for r in full_val)
    print(f"  full_noocr val problems whose text also occurs in train (other level): {twins}/{len(full_val)}")
    print("  (label statistics cannot leak through such twins unless the key letter is kept; see section 1c)")

    # ------------------------------------------------------------ 1. priors
    header("1. Label priors (full_noocr: all problems, labels only)")
    print("  chi2 is against a uniform 20% split; 4 d.o.f.: 9.5 = p 0.05, 13.3 = p 0.01, 18.5 = p 0.001")
    print(f"  train  n={len(full_train):4d}  {dist(full_train)}  chi2={chi2_uniform(full_train):.1f}")
    print(f"  val    n={len(full_val):4d}  {dist(full_val)}  chi2={chi2_uniform(full_val):.1f}")
    both = full_train + full_val
    print(f"  pooled n={len(both):4d}  {dist(both)}  chi2={chi2_uniform(both):.1f}")
    breakdown("by level (train)", full_train, lambda r: r.get("level"))
    breakdown("by year band (train)", full_train, lambda r: f"{(r.get('year') or 0) // 5 * 5}-{(r.get('year') or 0) // 5 * 5 + 4}")
    breakdown("by text_only (train)", full_train, lambda r: r.get("text_only"))
    breakdown("by third of the paper (train; 0 = easiest)", full_train, third)
    breakdown("by question number (train)", full_train, qnum)

    print("\n  1b. Conditional priors as predictors (fit on train folds; key groups under 40 back off)")
    print("      CV = 5-fold on full_noocr train, averaged over 5 fold assignments; ± is the binomial s.e.")
    report("always D", fit_always_d, full_train, full_val)
    report("best single letter (fitted)", fit_best_letter, full_train, full_val)
    for name, key_fn in CONDITIONALS:
        report(name, fit_conditional(key_fn), full_train, full_val)

    print("\n  1c. Same problem printed at two levels: is the key letter the same?")
    by_key = defaultdict(list)
    for r in both:
        by_key[(group_key(r), r.get("year"))].append(r)
    pairs = [(a, b) for g in by_key.values() if len(g) > 1 for i, a in enumerate(g) for b in g[i + 1:]]
    same = sum(a["label"] == b["label"] for a, b in pairs)
    print(f"      twin pairs: {len(pairs)}, same letter in both: {pct(same, len(pairs))} (20% would be chance)")

    # ------------------------------------------------- 2. option heuristics
    header("2. Option-content heuristics (OCR set, problems with five options)")
    otrain = [r for r in train if has_options(r)]
    oval = [r for r in val if has_options(r)]
    print(f"  with options   train n={len(otrain)}  {dist(otrain)}")
    print(f"                 val   n={len(oval)}  {dist(oval)}")
    ntrain = [r for r in train if not has_options(r)]
    nval = [r for r in val if not has_options(r)]
    print(f"  without        train n={len(ntrain)}  {dist(ntrain)}")
    print(f"                 val   n={len(nval)}  {dist(nval)}")
    print("\n  option order (all-numeric problems):")
    for name, rows in (("train", otrain), ("val", oval)):
        counts = Counter(ordering(r) for r in rows)
        print(f"    {name}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    print("\n  rank of the correct value among the five options (1 = smallest), train:")
    for label, keep in (("all numeric", lambda r: True),
                        ("printed ascending", lambda r: ordering(r) == "ascending"),
                        ("printed descending", lambda r: ordering(r) == "descending"),
                        ("printed shuffled", lambda r: ordering(r) == "shuffled")):
        rows = [r for r in otrain if option_values(r) is not None and keep(r)]
        counts = Counter(ranks(option_values(r))[r["label"]] for r in rows)
        n = len(rows) or 1
        expected = len(rows) / 5
        chi = sum((counts[i] - expected) ** 2 / expected for i in range(1, 6)) if rows else 0
        print(f"    {label:<19} n={len(rows):4d}  " + " ".join(f"r{i}:{counts[i]:3d} ({100 * counts[i] / n:4.1f}%)" for i in range(1, 6)) + f"  chi2={chi:.1f}")
        if rows:
            print(f"    {'':<19} letters  {dist(rows)}")

    print("\n  rules without any fitting. 'applies' = problems where the rule names exactly one option.")
    print("  Accuracy is on those problems only; D = always-D on the same problems.")
    print(f"  {'rule':<50} {'train':<24} {'D on same':<12} {'paired vs D':<27} | {'val':<24} {'D on same':<10} paired vs D")
    for name, rule in RULES:
        cells = []
        for rows in (otrain, oval):
            picks = [(rule(r), r["label"]) for r in rows]
            picks = [(p, g) for p, g in picks if p is not None]
            n = len(picks)
            hits = sum(p == g for p, g in picks)
            d_hits = sum(g == "D" for _, g in picks)
            cells.append((pct(hits, n), f"{100 * d_hits / n:4.1f}%" if n else "-",
                          paired([p for p, _ in picks], ["D"] * n, [g for _, g in picks])))
        print(f"  {name:<50} {cells[0][0]:<24} {cells[0][1]:<12} {cells[0][2]:<27} | {cells[1][0]:<24} {cells[1][1]:<10} {cells[1][2]}")

    # ------------------------------------------------------ 3. combined scorer
    header("3. Additive log-odds scorer (naive Bayes over per-option features)")
    print("  Fitted on train folds of the OCR set (all problems, with and without options).")
    print("  3a. evaluated on problems WITH options")
    report("always D", fit_always_d, train, val, has_options)
    report("best single letter (fitted)", fit_best_letter, train, val, has_options)
    for name, names in SCORERS:
        report(name, fit_scorer(names), train, val, has_options)
    print("\n  3b. evaluated on problems WITHOUT options (only the letter prior can act)")
    report("always D", fit_always_d, train, val, lambda r: not has_options(r))
    report("best single letter (fitted)", fit_best_letter, train, val, lambda r: not has_options(r))
    print("\n  3c. evaluated on ALL problems of the OCR set")
    report("always D", fit_always_d, train, val)
    report("best single letter (fitted)", fit_best_letter, train, val)
    for name, names in SCORERS[-3:]:
        report(name, fit_scorer(names), train, val)

    print("\n  3d. weights of the full scorer fitted on all of train (log-odds vs the 20% base rate)")
    model = fit_scorer(tuple(FEATURES))(train)
    for name, table in model.weights.items():
        print(f"    {name:<14} " + "  ".join(f"{v}:{w:+.2f}" for v, w in sorted(table.items(), key=lambda kv: str(kv[0]))))

    # --------------------------------------------------- 4. the shipped rule
    header("4. Shipped rule: copilot/priors.py guess()  (median value if five numeric options are out of order, else D)")
    path = ROOT / "copilot" / "priors.py"
    if not path.exists():
        print("  copilot/priors.py not present; skipped")
    else:
        spec = importlib.util.spec_from_file_location("priors_guess", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("  The rule has no fitted numbers, so train is not a fit set here, but the rule was chosen")
        print("  after looking at train: val is the clean number, and section 3 'letter + rank_shuffled'")
        print("  is the cross-validated version of the same idea.")
        subsets = [
            ("all problems", lambda r: True),
            ("with options", has_options),
            ("  numeric, out of order (rule fires)", lambda r: ordering(r) == "shuffled"),
            ("  numeric, sorted (rule = D)", lambda r: ordering(r) in ("ascending", "descending")),
            ("  non-numeric options (rule = D)", lambda r: has_options(r) and ordering(r) == "nonnumeric"),
            ("without options (rule = D)", lambda r: not has_options(r)),
        ]
        for split, rows in (("train", train), ("val", val), ("train+val", train + val)):
            print(f"\n  {split}")
            print(f"    {'subset':<38} {'guess()':<24} {'always D':<24} paired guess - D")
            for name, keep in subsets:
                sub = [r for r in rows if keep(r)]
                pred = [module.guess(r) for r in sub]
                gold = [r["label"] for r in sub]
                hits = sum(p == g for p, g in zip(pred, gold))
                d_hits = sum(g == "D" for g in gold)
                print(f"    {name:<38} {pct(hits, len(sub)):<24} {pct(d_hits, len(sub)):<24} {paired(pred, ['D'] * len(sub), gold)}")
        print("\n  robustness: guess() on degenerate inputs ->",
              [module.guess(x) for x in ({}, {"options": {}}, {"options": None}, {"options": {"A": "1"}}, None,
                                         {"options": {"A": 3, "B": 1, "C": 2, "D": 5, "E": 4}})])

    print("\nNoise floor: with n≈800 a binomial s.e. is ≈1.5pp, with n≈150 it is ≈3.5pp.")
    print("Treat any paired gain under 2 s.e. (|z| < 2) as indistinguishable from always-D,")
    print("and remember that sections 2-3 try many rules, so one |z| near 2 is expected by chance.")


if __name__ == "__main__":
    main()
