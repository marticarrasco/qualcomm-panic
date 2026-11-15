"""Compare lexical and BGE neighbors, transferring only matching option values."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from copilot.arithmetic import complete_options
from copilot.embed import embed
from copilot.priors import guess
from copilot.train_match import cosine, grams, option_key, stem


def rows(path):
    with path.open(encoding="utf-8-sig") as source:
        return [row for row in (json.loads(line) for line in source if line.strip())
                if row.get("text_only") is True and complete_options(row)
                and row.get("label") in tuple("ABCDE")]


def vectorize(texts, query=False):
    vectors = []
    for start in range(0, len(texts), 16):
        vectors.extend(embed(texts[start:start+16], query=query, timeout=60))
    return vectors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--input", type=Path, default=Path("data/eval_sets/val_300.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lexical-only", action="store_true")
    args = p.parse_args()
    train = rows(args.train)
    validation = rows(args.input)
    if {row["id"] for row in train} & {row["id"] for row in validation}:
        raise ValueError("train/eval IDs overlap")
    buckets = defaultdict(list)
    for row in train:
        options = [option_key(row["options"][letter]) for letter in "ABCDE"]
        if len(set(options)) == 5:
            buckets[tuple(sorted(options))].append((row, options["ABCDE".index(row["label"])]))
    pairs = []
    candidate_texts, query_texts = set(), set()
    for row in validation:
        options = {option_key(value): key for key, value in row["options"].items()}
        if len(options) != 5:
            continue
        candidates = buckets[tuple(sorted(options))]
        if not candidates:
            continue
        # Cross-language embedding matches are unreliable with bge-small-en. Prefer
        # a same-language candidate, otherwise look at the English source.
        same = [c for c in candidates if c[0]["language"] == row["language"]]
        candidates = same or [c for c in candidates if c[0]["language"] == "eng"]
        if candidates:
            question = stem(row["text"])
            query_texts.add(question)
            candidate_texts.update(stem(candidate["text"]) for candidate, _ in candidates)
            pairs.append((row, options, candidates))
    print(f"{len(pairs)}/{len(validation)} rows share the exact option set with train;"
          f" embedding {len(candidate_texts)} candidates and {len(query_texts)} queries", flush=True)
    if not args.lexical_only:
        candidates_sorted, queries_sorted = sorted(candidate_texts), sorted(query_texts)
        documents = dict(zip(candidates_sorted, vectorize(candidates_sorted)))
        queries = dict(zip(queries_sorted, vectorize(queries_sorted, query=True)))
    else:
        documents, queries = {}, {}
    char = {text: grams(text) for text in candidate_texts | query_texts}
    results = []
    for row, options, candidates in pairs:
        question = stem(row["text"])
        scored = []
        for candidate, value in candidates:
            text = stem(candidate["text"])
            lex = cosine(char[question], char[text])
            bge = sum(a * b for a, b in zip(queries.get(question, ()), documents.get(text, ())))
            scored.append((lex, bge, candidate["id"], candidate["language"], options[value]))
        for method, order in (("lexical", 0), ("bge", 1)):
            if method == "bge" and args.lexical_only:
                continue
            lexical, bge, source, lang, answer = max(scored, key=lambda x: x[order])
            results.append(dict(method=method, id=row["id"], language=row["language"], label=row["label"],
                                answer=answer, prior=guess(row), source=source, source_language=lang,
                                lexical=round(lexical, 5), bge=round(bge, 5)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "id", "language", "label", "answer",
                                             "prior", "source", "source_language", "lexical", "bge"])
        writer.writeheader()
        writer.writerows(results)
    for method in sorted({r["method"] for r in results}):
        for threshold in (0.6, 0.75, 0.85, 0.9, 0.95):
            matched = [r for r in results if r["method"] == method and r[method] >= threshold]
            print(method, threshold, len(matched), "rows", len({r["id"] for r in matched}), "IDs",
                  "correct", sum(r["answer"] == r["label"] for r in matched),
                  "vs prior", sum(r["prior"] == r["label"] for r in matched))


if __name__ == "__main__":
    main()
