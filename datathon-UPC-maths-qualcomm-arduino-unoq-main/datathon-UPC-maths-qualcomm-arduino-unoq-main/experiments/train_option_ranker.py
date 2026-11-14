#!/usr/bin/env python3
"""Fit a choice ranker on complete text-only train data; inference is stdlib-only."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from copilot.arithmetic import LETTERS, complete_options
from copilot.option_ranker import VERSION, features
from copilot.priors import guess


def load(path):
    with path.open(encoding="utf-8-sig") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [row for row in rows if row.get("text_only") is True
            and complete_options(row) and row.get("label") in tuple(LETTERS)]


def row_features(row, arithmetic):
    base = features(row)
    if not arithmetic:
        return base
    from copilot.arithmetic import numeric_options
    from copilot.option_ranker import normalize
    from experiments.number_features import evidence
    choices = numeric_options(row)
    if choices is not None and len(set(choices)) == 5:
        from copilot.train_match import stem
        for f, weak_evidence in zip(base, evidence(stem(normalize(row["text"])), choices)):
            f.update(weak_evidence)
    return base


def tensor(rows, keys, arithmetic=False):
    return np.array([[[f.get(key, 0.0) for key in keys] for f in row_features(row, arithmetic)]
                     for row in rows])


def probs(x, w):
    scores = (x @ w)
    values = np.exp(scores - scores.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def fit(x, y, penalty):
    n, _, d = x.shape
    w = np.zeros(d)
    expected = x[np.arange(n), y].mean(axis=0)
    for _ in range(240):
        p = probs(x, w)
        grad = np.einsum("ij,ijk->k", p, x) / n - expected + penalty * w
        w -= 0.8 * grad
    return w


def score(predictions, rows):
    hits = Counter()
    by_id = {}
    for row, p in zip(rows, predictions):
        letter = LETTERS[int(p.argmax())]
        hits["ranker"] += letter == row["label"]
        hits["prior"] += guess(row) == row["label"]
        hits["always_d"] += row["label"] == "D"
        series = by_id.setdefault(row["id"], [])
        series.append((letter == row["label"], guess(row) == row["label"]))
    macro = np.mean([np.mean(v, axis=0) for v in by_id.values()], axis=0)
    return f"{dict(hits)} / {len(rows)}; ID macro ranker/prior {macro.round(3).tolist()}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("copilot/option_weights.json"))
    parser.add_argument("--eval", nargs="*", type=Path, default=[])
    parser.add_argument("--arithmetic", action="store_true")
    args = parser.parse_args()
    usable = load(args.train)
    # One vote per problem; prefer the source English OCR, keep alternate languages
    # for evaluation only. Year blocks keep cross-level variants together in CV.
    rows = {}
    for row in sorted(usable, key=lambda row: (row["language"] != "eng", row["language"])):
        rows.setdefault(row["id"], row)
    train_ids = set(rows)
    rows = list(rows.values())
    if args.arithmetic and args.output == Path("copilot/option_weights.json"):
        parser.error("use --output to keep experimental weights separate")
    keys = sorted({key for row in rows for f in row_features(row, args.arithmetic) for key in f})
    x = tensor(rows, keys, args.arithmetic)
    y = np.array([LETTERS.index(row["label"]) for row in rows])
    folds = np.array([int(row["year"]) % 5 for row in rows])
    best = None
    print(f"{len(usable)} valid train rows, {len(rows)} distinct problems")
    for penalty in (0.003, 0.01, 0.03, 0.1, 0.3):
        cv = np.zeros((len(rows), 5))
        for fold in range(5):
            keep = folds != fold
            cv[~keep] = probs(x[~keep], fit(x[keep], y[keep], penalty))
        loss = -np.log(cv[np.arange(len(rows)), y]).mean()
        print(f"CV penalty={penalty}: log loss {loss:.4f};", score(cv, rows))
        if best is None or loss < best[0]:
            best = (loss, penalty, cv)
    _, penalty, cv = best
    for threshold in (0.25, 0.30, 0.35, 0.40):
        selected = cv.max(axis=1) >= threshold
        print(f"OOF confidence >= {threshold}:",
              int((cv[selected].argmax(1) == y[selected]).sum()), "/", int(selected.sum()))
    w = fit(x, y, penalty)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "version": VERSION, "training_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "training_ids": len(rows), "l2": penalty, "weights": dict(zip(keys, w.round(7).tolist()))
    }, indent=2) + "\n", encoding="utf-8")
    for path in args.eval:
        validation = load(path)
        if train_ids.intersection(row["id"] for row in validation):
            raise ValueError(f"training and evaluation IDs overlap: {path}")
        print(path, score(probs(tensor(validation, keys, args.arithmetic), w), validation))


if __name__ == "__main__":
    main()
