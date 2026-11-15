#!/usr/bin/env python3
"""Build the tutor's problem bank: data/tutor.db.

    python3 tutor/build_bank.py
    python3 tutor/build_bank.py --check 30     # also print 30 random problems with their topic

Reads every problem in data/full_noocr/{train,val}.jsonl (English and Catalan),
gives each a topic with the keyword classifier in tutor/topics.py, embeds the
English text with the embedding server (port 8082) and stores, for each
problem, its closest easier / same-level / harder problems of the same topic.
Standard library only, so it can be re-run on the board.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
import time
import urllib.request
from array import array

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tutor.topics import NAMES, TOPICS, classify, clean, explain  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The same problem is reused at neighbouring levels; treat these as one problem
DUPLICATE = 0.97
TOP = 3
MAX_CHARS = 1400  # about 512 tokens, the embedding model's limit

SCHEMA = """
CREATE TABLE problems (
    id TEXT PRIMARY KEY, level INTEGER, year INTEGER, number INTEGER,
    text_only INTEGER, label TEXT, topic TEXT);
CREATE TABLE texts (id TEXT, language TEXT, text TEXT, PRIMARY KEY (id, language));
CREATE TABLE options (id TEXT, language TEXT, options TEXT, PRIMARY KEY (id, language));  -- OCR, optional
CREATE TABLE vectors (id TEXT PRIMARY KEY, vec BLOB);                                     -- float32, unit length
CREATE TABLE neighbours (
    id TEXT, kind TEXT, rank INTEGER, other_id TEXT, score REAL, same_topic INTEGER,
    PRIMARY KEY (id, kind, rank));                                -- kind: easier | same_level | harder
CREATE TABLE duplicates (id TEXT, other_id TEXT, score REAL, PRIMARY KEY (id, other_id));
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="folder with train.jsonl and val.jsonl")
    p.add_argument("--categories", default="tutor/train_topics.csv", help="CSV or JSONL topic assignments by ID")
    p.add_argument("--lexical", action="store_true", help="build offline with lexical similarity")
    p.add_argument("--ocr", default="data", help="folder with train.jsonl and val.jsonl that have OCR options (optional)")
    p.add_argument("--out", default="data/tutor.db")
    p.add_argument("--embed-url", default="http://127.0.0.1:8082/v1/embeddings")
    p.add_argument("--check", type=int, default=0, help="print this many random problems with their topic")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_jsonl(folder):
    rows = []
    for name in ("train.jsonl", "val.jsonl"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                rows += [json.loads(line) for line in f if line.strip()]
    return rows


def embed(url, texts):
    request = urllib.request.Request(
        url, data=json.dumps({"input": [t[:MAX_CHARS] for t in texts]}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read())["data"]
    vectors = []
    for item in sorted(data, key=lambda d: d["index"]):
        norm = math.sqrt(sum(x * x for x in item["embedding"])) or 1.0
        vectors.append([x / norm for x in item["embedding"]])
    return vectors


def dot(a, b):
    return math.sumprod(a, b) if hasattr(math, "sumprod") else sum(x * y for x, y in zip(a, b))


def category_assignments(path):
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source) if path.endswith(".csv") else (json.loads(line) for line in source if line.strip())
        return {row["id"]: row["topic"] for row in rows if row.get("topic") in TOPICS}


def lexical_vectors(texts):
    """Lightweight TF-IDF vectors for installations without an embedding server."""
    tokens = {pid: collections.Counter(re.findall(r"[^\W\d_]{3,}", text.casefold())) for pid, text in texts.items()}
    frequency = collections.Counter(word for counts in tokens.values() for word in counts)
    features = {
        word: (int.from_bytes(hashlib.blake2s(word.encode(), digest_size=4).digest(), "big"),
               math.log((1 + len(texts)) / (1 + count)))
        for word, count in frequency.items()
    }
    vectors = {}
    for pid, counts in tokens.items():
        values = [0.0] * 128
        for word, count in counts.items():
            code, weight = features[word]
            values[code % len(values)] += (1 if code & 128 else -1) * (1 + math.log(count)) * weight
        norm = math.sqrt(dot(values, values)) or 1.0
        vectors[pid] = [value / norm for value in values]
    return vectors


def main():
    args = parse_args()
    os.chdir(ROOT)
    started = time.time()
    folder = args.data or ("data/full_noocr" if os.path.isfile("data/full_noocr/train.jsonl") else "data")
    if not args.data and any(not os.path.isfile(os.path.join(folder, f"{split}.jsonl")) for split in ("train", "val")):
        sys.exit(f"Prepare both train.jsonl and val.jsonl in {folder} before building the full bank")
    rows = [r for r in read_jsonl(folder) if r["language"] in ("eng", "cat")]
    by_language = {lang: {r["id"]: r for r in rows if r["language"] == lang} for lang in ("eng", "cat")}
    english, catalan = by_language["eng"], by_language["cat"]
    ids = sorted(english)
    if not ids:
        sys.exit(f"No English problems found in {folder}")
    if english.keys() != catalan.keys():
        sys.exit(f"Incomplete bilingual bank: {len(english.keys() - catalan.keys())} missing Catalan and "
                 f"{len(catalan.keys() - english.keys())} missing English")
    for pid in ids:
        label = english[pid]["label"]
        if label not in ("A", "B", "C", "D", "E") or label != catalan[pid]["label"]:
            sys.exit(f"Invalid or mismatching answer keys for {pid}")
    print(f"{len(ids)} problems in both Catalan and English")
    assigned = category_assignments(args.categories)
    absent = assigned.keys() - english.keys()
    if absent and not args.data:
        sys.exit(f"{len(absent)} classified train problems are missing from {folder}")
    topic = {i: assigned.get(i) or classify(english[i]["text"], english[i]["text_only"]) for i in ids}
    print(f"{sum(i in assigned for i in ids)} problems use the supplied topic assignments")
    level = {i: english[i]["level"] for i in ids}

    texts = {i: clean(english[i]["text"]) for i in ids}
    print(f"Vectorising the English texts ({'lexical' if args.lexical else 'embeddings'}) ...", flush=True)
    if args.lexical:
        vectors = lexical_vectors(texts)
    else:
        vectors = {}
        for start in range(0, len(ids), 32):
            chunk = ids[start:start + 32]
            batch = embed(args.embed_url, [texts[i] for i in chunk])
            if len(batch) != len(chunk):
                sys.exit("The embedding server returned an incomplete batch")
            vectors.update(zip(chunk, batch))
    print(f"  {len(vectors)} vectors of {len(vectors[ids[0]])} numbers, {time.time() - started:.1f} s")
    normalized = {i: text.casefold() for i, text in texts.items()}

    print("Finding neighbours ...", flush=True)
    neighbours, duplicates = [], []
    filled_other_topic = 0
    for i in ids:
        scored = sorted(((dot(vectors[i], vectors[j]), j) for j in ids if j != i), reverse=True)
        same_text = normalized[i]
        usable = []
        for score, j in scored:
            if normalized[j] == same_text or (not args.lexical and score >= DUPLICATE):
                duplicates.append((i, j, score))
            else:
                usable.append((score, j))
        for kind, keep in (("easier", lambda j: level[j] < level[i]),
                           ("same_level", lambda j: level[j] == level[i]),
                           ("harder", lambda j: level[j] > level[i])):
            pool = [(s, j) for s, j in usable if keep(j)]
            chosen = [(s, j) for s, j in pool if topic[j] == topic[i]][:TOP]
            if len(chosen) < TOP:  # too few in this topic: fill with the closest of any topic
                extra = [(s, j) for s, j in pool if topic[j] != topic[i]][:TOP - len(chosen)]
                filled_other_topic += len(extra)
                chosen += extra
            for rank, (score, j) in enumerate(chosen):
                neighbours.append((i, kind, rank, j, round(score, 4), int(topic[j] == topic[i])))

    if os.path.exists(args.out):
        os.remove(args.out)
    db = sqlite3.connect(args.out)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO problems VALUES (?, ?, ?, ?, ?, ?, ?)", [
        (i, level[i], english[i]["year"], int(i.rsplit("_", 1)[1]), int(english[i]["text_only"]),
         english[i]["label"], topic[i]) for i in ids])
    db.executemany("INSERT OR REPLACE INTO texts VALUES (?, ?, ?)",
                   [(pid, lang, r["text"]) for lang, problems in by_language.items() for pid, r in problems.items()])
    ocr_rows = rows if os.path.abspath(args.ocr) == os.path.abspath(folder) else read_jsonl(args.ocr)
    db.executemany("INSERT OR REPLACE INTO options VALUES (?, ?, ?)", [
        (r["id"], r["language"], json.dumps(r["options"], ensure_ascii=False))
        for r in ocr_rows if r.get("options") and r["id"] in english and r["language"] in ("eng", "cat")])
    db.executemany("INSERT INTO vectors VALUES (?, ?)", [(i, array("f", vectors[i]).tobytes()) for i in ids])
    db.executemany("INSERT INTO neighbours VALUES (?, ?, ?, ?, ?, ?)", neighbours)
    db.executemany("INSERT OR REPLACE INTO duplicates VALUES (?, ?, ?)", duplicates)
    db.executemany("INSERT INTO meta VALUES (?, ?)", [
        ("built", time.strftime("%Y-%m-%d %H:%M:%S")), ("duplicate_threshold", str(DUPLICATE)),
        ("similarity", "lexical" if args.lexical else "embeddings")])
    db.commit()

    print(f"\nTopic distribution ({len(ids)} problems):")
    counts = collections.Counter(topic.values())
    grid = collections.Counter((topic[i], level[i]) for i in ids)
    print(f"  {'topic':<36}{'n':>5}{'%':>6}   per level 0..7")
    for t in TOPICS:
        per_level = " ".join(f"{grid[(t, lvl)]:>3}" for lvl in range(8))
        print(f"  {NAMES[t]['eng']:<36}{counts[t]:>5}{100 * counts[t] / len(ids):>5.0f}%   {per_level}")
    kinds = collections.Counter(n[1] for n in neighbours)
    options = db.execute("SELECT COUNT(*) FROM options").fetchone()[0]
    print(f"\nNeighbours: {dict(kinds)}; {filled_other_topic} filled from another topic "
          f"({100 * filled_other_topic / max(1, len(neighbours)):.1f}%)")
    duplicate_rule = "same text" if args.lexical else f"cosine >= {DUPLICATE} or same text"
    print(f"Near-duplicate pairs ({duplicate_rule}): {len(duplicates) // 2}")
    print(f"OCR options stored for {options} (problem, language) pairs")
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) in {time.time() - started:.1f} s")
    db.close()

    if args.check:
        print(f"\n{args.check} random problems to check by hand (seed {args.seed}):")
        for i in random.Random(args.seed).sample(ids, min(args.check, len(ids))):
            fired = {t: [h[0] for h in hits] for t, hits in explain(english[i]["text"]).items()}
            print(f"\n[{topic[i]}] {i} {'text' if english[i]['text_only'] else 'picture'}"
                  f"\n  {clean(english[i]['text'])[:300]}\n  fired: {fired}")


if __name__ == "__main__":
    main()
