"""Describe the organisers' practice set and test whether M3Kang answer priors transfer to it.
Usage: python -I verify_demo.py <demo.jsonl> <m3kang_train.jsonl> <zip> <parquet> [--show]"""
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict

LETTERS = "ABCDE"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_num(val):
    s = str(val).strip().replace(",", ".").rstrip("%").strip()
    s = s.replace("−", "-").replace("–", "-")
    if re.match(r"^[-+]?\d+(\.\d+)?$", s):
        return float(s)
    m = re.match(r"^([-+]?\d+)\s*/\s*(\d+)$", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    return None


def kind(options):
    if not options:
        return "none"
    if len(options) < 5:
        return "non_num"
    nums = [parse_num(options.get(k, "")) for k in LETTERS]
    if all(v is not None for v in nums) and len(set(nums)) == 5:
        if nums == sorted(nums):
            return "inc_num"
        if nums == sorted(nums, reverse=True):
            return "dec_num"
        return "unord_num"
    return "non_num"


def dist(rows):
    c = Counter(r["label"] for r in rows)
    return "  ".join(f"{l}:{c[l]:3}" for l in LETTERS)


def main():
    demo, train = load(sys.argv[1]), load(sys.argv[2])
    zip_path, parquet_path = sys.argv[3], sys.argv[4]
    show = "--show" in sys.argv

    # The zip, without extracting it
    with zipfile.ZipFile(zip_path) as z:
        print("=== zip contents")
        for info in z.infolist():
            print(f"  {info.filename}  {info.file_size} bytes  crc={info.CRC:08x}")
        names = [i.filename for i in z.infolist() if i.filename.endswith(".parquet")]
        if names:
            with z.open(names[0]) as f:
                zh = hashlib.sha256(f.read()).hexdigest()
            with open(parquet_path, "rb") as f:
                ph = hashlib.sha256(f.read()).hexdigest()
            print(f"  parquet in zip == data/demo-00000-of-00001.parquet: {zh == ph}")

    import pyarrow.parquet as pq
    schema = pq.read_schema(parquet_path)
    meta = pq.read_metadata(parquet_path)
    print(f"=== parquet: {meta.num_rows} rows; columns: {[f'{f.name}:{f.type}' for f in schema]}")
    table = pq.read_table(parquet_path, columns=[c for c in schema.names if c != "image"])
    cols = table.to_pydict()
    for name in cols:
        if name in ("text",):
            continue
        values = cols[name]
        uniq = Counter(map(str, values))
        if len(uniq) <= 12:
            print(f"  {name}: {dict(uniq)}")
        else:
            print(f"  {name}: {len(uniq)} distinct values, e.g. {list(uniq)[:4]}")

    print(f"\n=== demo jsonl: {len(demo)} rows, {len({r['id'] for r in demo})} ids")
    print(f"  languages: {dict(Counter(r['language'] for r in demo))}")
    print(f"  levels (per id): {dict(sorted(Counter(r['level'] for r in demo if r['language'] == 'eng').items()))}")
    print(f"  years: {dict(Counter(r['year'] for r in demo))}")
    print(f"  text_only (per id): {dict(Counter(r['text_only'] for r in demo if r['language'] == 'eng'))}")
    print(f"  rows with 5 options: {sum(1 for r in demo if len(r.get('options') or {}) == 5)}/{len(demo)}")
    by_id = defaultdict(dict)
    for r in demo:
        by_id[r["id"]][r["language"]] = r
    print(f"  ids with both languages: {sum(1 for d in by_id.values() if len(d) == 2)}; "
          f"label mismatch eng/cat: {sum(1 for d in by_id.values() if len(d) == 2 and d['eng']['label'] != d['cat']['label'])}")
    mism = []
    for i, d in by_id.items():
        if len(d) == 2:
            e = [parse_num(d["eng"]["options"].get(k, "")) for k in LETTERS]
            c = [parse_num(d["cat"]["options"].get(k, "")) for k in LETTERS]
            if all(v is not None for v in e) and all(v is not None for v in c) and e != c:
                mism.append(i)
            elif (all(v is not None for v in e)) != (all(v is not None for v in c)):
                mism.append(i + " (numeric in one language only)")
    print(f"  numeric options that differ between eng and cat: {mism}")
    dups = [f"{r['id']}|{r['language']}" for r in demo if len(set(str(v).strip() for v in r['options'].values())) < 5]
    print(f"  rows with duplicated option values: {dups}")

    eng = [r for r in demo if r["language"] == "eng"]
    cat = [r for r in demo if r["language"] == "cat"]
    print("\n=== demo labels (per id, eng rows)")
    print(f"  all            n={len(eng):3}  {dist(eng)}")
    for k in ("inc_num", "dec_num", "unord_num", "non_num", "none"):
        rows = [r for r in eng if kind(r["options"]) == k]
        if rows:
            print(f"  {k:14} n={len(rows):3}  {dist(rows)}")
    for flag in (True, False):
        rows = [r for r in eng if r["text_only"] == flag]
        print(f"  text_only={flag!s:5} n={len(rows):3}  {dist(rows)}")
    for lv in sorted({r["level"] for r in eng}):
        rows = [r for r in eng if r["level"] == lv]
        print(f"  level {lv}        n={len(rows):3}  {dist(rows)}   text_only={sum(r['text_only'] for r in rows)}")
    print("  kinds, cat rows:", dict(Counter(kind(r["options"]) for r in cat)))

    # Does the M3Kang structure prior transfer? Log-likelihood of the demo labels under each prior.
    tr_eng = [r for r in train if r["language"] == "eng"]
    priors, groups = {}, defaultdict(list)
    for r in tr_eng:
        groups[kind(r.get("options"))].append(r)
    for k, rows in groups.items():
        c = Counter(r["label"] for r in rows)
        priors[k] = {l: (c[l] + 1) / (len(rows) + 5) for l in LETTERS}
    overall = Counter(r["label"] for r in tr_eng)
    base = {l: overall[l] / len(tr_eng) for l in LETTERS}
    ll_uniform = len(eng) * math.log(0.2)
    ll_base = sum(math.log(base[r["label"]]) for r in eng)
    ll_struct = sum(math.log(priors[kind(r["options"])][r["label"]]) for r in eng)
    print("\n=== Do M3Kang answer priors explain the demo labels? (log-likelihood, higher is better)")
    print(f"  uniform 20%:                 {ll_uniform:8.2f}")
    print(f"  M3Kang letter frequencies:   {ll_base:8.2f}   ({ll_base - ll_uniform:+.2f} vs uniform)")
    print(f"  M3Kang prior by structure:   {ll_struct:8.2f}   ({ll_struct - ll_uniform:+.2f} vs uniform)")
    arg = {k: max(LETTERS, key=lambda l: p[l]) for k, p in priors.items()}
    hit = sum(arg[kind(r["options"])] == r["label"] for r in eng)
    print(f"  accuracy of 'always D': {sum(r['label'] == 'D' for r in eng)}/{len(eng)};  "
          f"of the structure-prior argmax: {hit}/{len(eng)}")
    inc = [r for r in eng if kind(r["options"]) == "inc_num"]
    c = Counter(r["label"] for r in inc)
    ends = c["A"] + c["E"]
    # exact binomial tail: P(X >= ends) if P(A or E) were M3Kang's 20.2% (72/357)
    p_m3 = 72 / 357
    tail = sum(math.comb(len(inc), k) * p_m3 ** k * (1 - p_m3) ** (len(inc) - k) for k in range(ends, len(inc) + 1))
    print(f"  inc_num in demo: answer at an end (A or E) in {ends}/{len(inc)} = {100 * ends / len(inc):.1f}% "
          f"(M3Kang train: 20.2%; uniform: 40%). P(this many or more | M3Kang rate) = {tail:.4f}")

    # How much does a prior of this strength distort a model on a balanced key? Position of the value.
    print("\n=== Text length of the demo (chars), eng / cat")
    for name, rows in (("eng", eng), ("cat", cat)):
        lens = sorted(len(r["text"]) for r in rows)
        print(f"  {name}: min {lens[0]}, median {lens[len(lens) // 2]}, p90 {lens[int(0.9 * len(lens))]}, max {lens[-1]}")

    if show:
        print("\n=== The problems (eng), for reading")
        for r in sorted(eng, key=lambda r: (r["level"], int(r["id"].rsplit("_", 1)[1]))):
            o = r["options"]
            c_o = by_id[r["id"]].get("cat", {}).get("options", {})
            same = "" if [o.get(k) for k in LETTERS] == [c_o.get(k) for k in LETTERS] else f"   [cat options: {' | '.join(str(c_o.get(k)) for k in LETTERS)}]"
            print(f"\n[{r['id']}] level {r['level']} text_only={r['text_only']} label={r['label']} kind={kind(o)}")
            print(f"  {r['text']}")
            print("  " + " | ".join(f"{k}) {o.get(k)}" for k in LETTERS) + same)


if __name__ == "__main__":
    main()
