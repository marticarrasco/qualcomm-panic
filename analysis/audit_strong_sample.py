"""Compare my manual verdicts on the 60-strong sample with the keyword topics.

'mine' is my category; 'alt' are categories I consider defensible alternates.
A keyword topic outside {mine}+alt is an error to override."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
mine = {
    "lvl-6_2013_8": ("arithmetic", ["numbers"]),
    "lvl-7_2007_28": ("counting", []),
    "lvl-6_2015_15": ("geometry_measure", []),
    "lvl-6_2020_17": ("geometry_measure", []),
    "lvl-7_2009_17": ("algebra", []),
    "lvl-4_2015_14": ("geometry_spatial", []),
    "lvl-4_2013_25": ("numbers", ["counting"]),
    "lvl-6_2020_24": ("geometry_measure", []),
    "lvl-6_2016_15": ("geometry_measure", []),
    "lvl-7_2013_30": ("counting", []),
    "lvl-7_2012_18": ("geometry_measure", []),
    "lvl-4_2014_22": ("geometry_measure", []),
    "lvl-3_2016_29": ("arithmetic", []),
    "lvl-6_2013_2": ("geometry_measure", []),
    "lvl-5_2013_14": ("geometry_spatial", []),
    "lvl-3_2024_14": ("geometry_measure", []),
    "lvl-2_2024_22": ("geometry_measure", []),
    "lvl-6_2023_15": ("geometry_measure", []),
    "lvl-7_2020_21": ("algebra", []),
    "lvl-7_2013_9": ("geometry_spatial", []),
    "lvl-1_2020_15": ("geometry_spatial", []),
    "lvl-7_2010_15": ("geometry_measure", []),
    "lvl-5_2024_15": ("geometry_measure", []),
    "lvl-7_2015_7": ("geometry_spatial", []),
    "lvl-7_2012_24": ("geometry_measure", []),
    "lvl-7_2015_18": ("geometry_measure", []),
    "lvl-4_2009_14": ("geometry_measure", []),
    "lvl-7_2013_16": ("counting", []),
    "lvl-0_2019_10": ("geometry_spatial", []),
    "lvl-6_2022_2": ("arithmetic", ["geometry_measure"]),
    "lvl-2_2019_23": ("geometry_measure", []),
    "lvl-2_2016_24": ("geometry_measure", []),
    "lvl-1_2016_15": ("counting", []),
    "lvl-4_2023_2": ("geometry_spatial", []),
    "lvl-7_2011_11": ("algebra", ["arithmetic", "geometry_measure"]),
    "lvl-0_2023_5": ("geometry_spatial", []),
    "lvl-6_2014_6": ("geometry_measure", []),
    "lvl-5_2015_20": ("logic", []),
    "lvl-6_2012_9": ("geometry_measure", []),
    "lvl-7_2009_8": ("counting", []),
    "lvl-4_2016_20": ("geometry_measure", []),
    "lvl-6_2021_28": ("counting", ["numbers"]),
    "lvl-4_2022_16": ("arithmetic", []),
    "lvl-7_2014_20": ("algebra", []),
    "lvl-5_2012_22": ("geometry_measure", []),
    "lvl-6_2019_15": ("geometry_spatial", ["counting"]),
    "lvl-0_2015_8": ("geometry_spatial", []),
    "lvl-7_2017_11": ("geometry_measure", []),
    "lvl-2_2023_23": ("counting", []),
    "lvl-6_2014_21": ("counting", []),
    "lvl-7_2016_12": ("numbers", []),
    "lvl-5_2011_2": ("arithmetic", ["geometry_spatial"]),
    "lvl-6_2008_24": ("geometry_measure", []),
    "lvl-6_2024_7": ("geometry_spatial", ["counting"]),
    "lvl-2_2023_22": ("logic", ["geometry_spatial"]),
    "lvl-7_2021_5": ("geometry_measure", []),
    "lvl-5_2007_8": ("numbers", []),
    "lvl-5_2014_23": ("geometry_measure", []),
    "lvl-4_2021_22": ("fractions", []),
    "lvl-5_2012_14": ("geometry_measure", []),
}

recs = {json.loads(l)["id"]: json.loads(l)
        for l in open(ROOT / "data" / "train_problem_categories.jsonl", encoding="utf-8")}
errors, border, ok = [], [], 0
for pid, (m, alts) in mine.items():
    kw = recs[pid]["topic"]
    if kw == m:
        ok += 1
    elif kw in alts:
        border.append(f"{pid}: kw={kw} mine={m}")
    else:
        errors.append(f"{pid}: kw={kw} mine={m} | {' '.join(recs[pid]['text'].split())[:90]}")
print(f"agree: {ok}/{len(mine)}  borderline-agree: {len(border)}  errors: {len(errors)}")
for e in errors:
    print("ERR ", e)
for b in border:
    print("bord", b)
