"""Check the investigation's statistical claims against data/train.jsonl and data/val.jsonl.
Usage: python -I verify_m3kang.py <train.jsonl> <val.jsonl>
Prints only counts and label statistics, never problem text."""
import json
import math
import re
import sys
from collections import Counter, defaultdict

LETTERS = "ABCDE"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_num(val):  # same rule as the investigation's _parse_num
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


def chi2_p_uniform(counts):
    """Chi-square test against the uniform distribution over 5 letters (df = 4)."""
    n = sum(counts)
    if n == 0:
        return float("nan"), float("nan")
    e = n / 5
    x = sum((c - e) ** 2 / e for c in counts)
    return x, math.exp(-x / 2) * (1 + x / 2)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def dist_line(name, rows):
    c = Counter(r["label"] for r in rows)
    n = len(rows)
    counts = [c[l] for l in LETTERS]
    x, p = chi2_p_uniform(counts)
    pcts = "  ".join(f"{l}:{100 * c[l] / n:5.1f}%" for l in LETTERS) if n else ""
    mid = sum(c[l] for l in "BCD")
    print(f"  {name:38} n={n:5}  {pcts}   BCD={100 * mid / n if n else 0:5.1f}%   chi2={x:6.2f} p={p:.4f}")


def main():
    train, val = load(sys.argv[1]), load(sys.argv[2])
    for name, rows in (("train", train), ("val", val)):
        langs = Counter(r["language"] for r in rows)
        ids = {r["id"] for r in rows}
        empty = Counter(r["language"] for r in rows if not r.get("options"))
        print(f"\n=== {name}: {len(rows)} rows, {len(ids)} ids, languages {dict(langs)}")
        for lang in sorted(langs):
            print(f"  options empty, {lang}: {empty[lang]}/{langs[lang]} = {100 * empty[lang] / langs[lang]:.1f}%")
        print(f"  options empty, all: {sum(empty.values())}/{len(rows)} = {100 * sum(empty.values()) / len(rows):.1f}%")
        to = [r for r in rows if r.get("text_only")]
        print(f"  text_only rows: {len(to)}/{len(rows)} = {100 * len(to) / len(rows):.1f}%; "
              f"of which with options: {sum(1 for r in to if r.get('options'))}")
        fig = [r for r in rows if not r.get("text_only")]
        print(f"  figure rows: {len(fig)}; of which with options: {sum(1 for r in fig if r.get('options'))} "
              f"= {100 * sum(1 for r in fig if r.get('options')) / max(len(fig), 1):.1f}%")

    print("\n=== Label distribution, overall")
    for name, rows in (("train eng+cat", train), ("train eng", [r for r in train if r["language"] == "eng"]),
                       ("val eng+cat", val), ("val eng", [r for r in val if r["language"] == "eng"])):
        dist_line(name, rows)

    print("\n=== Label distribution when options == {} (claim 2.1)")
    for name, rows in (("train eng+cat", train), ("train eng", [r for r in train if r["language"] == "eng"]),
                       ("val eng", [r for r in val if r["language"] == "eng"]), ("val eng+cat", val)):
        dist_line(name + ", no options", [r for r in rows if not r.get("options")])
    print("  (and when options are present)")
    for name, rows in (("train eng", [r for r in train if r["language"] == "eng"]),
                       ("val eng", [r for r in val if r["language"] == "eng"])):
        dist_line(name + ", with options", [r for r in rows if r.get("options")])

    print("\n=== Label distribution by option structure (claim 2.2)")
    for name, rows in (("train eng", [r for r in train if r["language"] == "eng"]),
                       ("train cat", [r for r in train if r["language"] == "cat"]),
                       ("val eng", [r for r in val if r["language"] == "eng"]),
                       ("val cat", [r for r in val if r["language"] == "cat"])):
        groups = defaultdict(list)
        for r in rows:
            groups[kind(r.get("options"))].append(r)
        print(f" {name}: with options {sum(len(v) for k, v in groups.items() if k != 'none')}")
        for k in ("inc_num", "dec_num", "unord_num", "non_num", "none"):
            dist_line(f"{name} {k}", groups.get(k, []))

    # Wilson intervals for the headline numbers
    eng = [r for r in train if r["language"] == "eng"]
    inc = [r for r in eng if kind(r.get("options")) == "inc_num"]
    c = Counter(r["label"] for r in inc)
    print("\n=== Wilson 95% intervals, train eng inc_num")
    for l in LETTERS:
        lo, hi = wilson(c[l], len(inc))
        print(f"  {l}: {c[l]}/{len(inc)} = {100 * c[l] / len(inc):.1f}%  [{100 * lo:.1f}, {100 * hi:.1f}]")

    # Does the inc_num prior depend on the level or the year? (stability = transferability)
    print("\n=== inc_num (train eng) by level")
    by_level = defaultdict(list)
    for r in inc:
        by_level[r["level"]].append(r)
    for lv in sorted(by_level):
        dist_line(f"level {lv}", by_level[lv])
    print("\n=== inc_num (train+val eng) by year band")
    all_eng = [r for r in train + val if r["language"] == "eng" and kind(r.get("options")) == "inc_num"]
    bands = defaultdict(list)
    for r in all_eng:
        y = r["year"]
        bands["2007-2012" if y <= 2012 else "2013-2018" if y <= 2018 else "2019-2024"].append(r)
    for b in sorted(bands):
        dist_line(b, bands[b])

    # Where the answer sits when options are increasing numbers: position of the label's value
    print("\n=== Policy check on val eng (priors fitted on train eng only)")
    priors = {}
    groups = defaultdict(list)
    for r in eng:
        groups[kind(r.get("options"))].append(r)
    for k, rows in groups.items():
        cc = Counter(r["label"] for r in rows)
        priors[k] = max(LETTERS, key=lambda l: cc[l])
    print(f"  argmax letter per structure (train eng): {priors}")
    for lang in ("eng", "cat"):
        v = [r for r in val if r["language"] == lang]
        hit_prior = sum(priors.get(kind(r.get("options")), "D") == r["label"] for r in v)
        hit_d = sum(r["label"] == "D" for r in v)
        print(f"  val {lang}: always D {hit_d}/{len(v)} = {100 * hit_d / len(v):.1f}%;  "
              f"structure-prior argmax {hit_prior}/{len(v)} = {100 * hit_prior / len(v):.1f}%")

    # Cross-language option leaks (claim 2.3)
    print("\n=== Cross-language option coverage (claim 2.3)")
    for name, rows in (("train", train), ("val", val)):
        by_id = defaultdict(dict)
        for r in rows:
            by_id[r["id"]][r["language"]] = r
        one_sided = [i for i, d in by_id.items() if len(d) == 2 and bool(d["eng"].get("options")) != bool(d["cat"].get("options"))]
        both = [i for i, d in by_id.items() if len(d) == 2 and d["eng"].get("options") and d["cat"].get("options")]
        differ_label = [i for i, d in by_id.items() if len(d) == 2 and d["eng"]["label"] != d["cat"]["label"]]
        dup = [f"{r['id']}|{r['language']}" for r in rows if r.get("options") and len(set(str(v).strip() for v in r["options"].values())) < 5]
        dup_ids = {d.split("|")[0] for d in dup}
        num_mismatch = 0
        for i in both:
            e, c2 = by_id[i]["eng"]["options"], by_id[i]["cat"]["options"]
            en = [parse_num(e.get(k, "")) for k in LETTERS]
            cn = [parse_num(c2.get(k, "")) for k in LETTERS]
            if all(v is not None for v in en) and all(v is not None for v in cn) and en != cn:
                num_mismatch += 1
        print(f"  {name}: ids with options in exactly one language: {len(one_sided)}; "
              f"rows with duplicated option values: {len(dup)} ({len(dup_ids)} ids); "
              f"numeric options that differ between eng and cat: {num_mismatch}; label mismatch: {len(differ_label)}")

    # Text length (prompt cost) in M3Kang, for comparison with the demo set
    print("\n=== Text length, eng rows with options (chars)")
    for name, rows in (("train", train), ("val", val)):
        lens = sorted(len(r["text"]) for r in rows if r["language"] == "eng" and r.get("options"))
        print(f"  {name}: min {lens[0]}, median {lens[len(lens) // 2]}, p90 {lens[int(0.9 * len(lens))]}, max {lens[-1]}")


if __name__ == "__main__":
    main()
