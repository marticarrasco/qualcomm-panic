# Problem types in `data/train.jsonl`

One entry per unique problem (1466 problems; the 33,843 train rows repeat them across languages).
Vocabulary matches `tutor/topics.py`, so a category here is also a tutor topic.

## How it was produced

1. **Keyword pass** (`laptop/classify_problems.py`): each problem's English row is scored with the `tutor/topics.py` keyword rules. Problems with a high score *and* a wide margin over the runner-up are kept as-is (`source: keyword`).
2. **Manual review**: every ambiguous or weakly-scored problem (~75% of the set) plus a 60-problem audit sample of the strong ones was classified by hand from the English text (`source: review` / `source: audit`). Manual verdicts override keywords.

Sources: audit 60, keyword 317, review 1089.

## Distribution

| topic | problems | % |
|---|---:|---:|
| fractions (Fraccions i percentatges) | 60 | 4.1% |
| counting (Comptar i probabilitat) | 165 | 11.3% |
| numbers (Propietats dels nombres) | 168 | 11.5% |
| geometry_measure (Geometria: àrea, perímetre, angles) | 327 | 22.3% |
| geometry_spatial (Figures, cossos i visió espacial) | 224 | 15.3% |
| logic (Lògica i estratègia) | 155 | 10.6% |
| measures (Temps, diners i mesures) | 40 | 2.7% |
| patterns (Patrons i successions) | 28 | 1.9% |
| algebra (Àlgebra i funcions) | 40 | 2.7% |
| arithmetic (Aritmètica i problemes) | 259 | 17.7% |
| **total** | **1466** | 100% |

## By level

| level | fractions | counting | numbers | geometry_measure | geometry_spatial | logic | measures | patterns | algebra | arithmetic |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 11 | 3 | 6 | 49 | 11 | 4 | 3 | 0 | 35 |
| 1 | 1 | 6 | 6 | 5 | 30 | 12 | 1 | 1 | 0 | 27 |
| 2 | 0 | 11 | 14 | 16 | 31 | 20 | 1 | 3 | 0 | 21 |
| 3 | 4 | 9 | 12 | 22 | 23 | 11 | 2 | 4 | 0 | 24 |
| 4 | 17 | 28 | 29 | 56 | 43 | 35 | 10 | 6 | 0 | 53 |
| 5 | 18 | 25 | 25 | 60 | 21 | 28 | 11 | 4 | 3 | 48 |
| 6 | 11 | 36 | 41 | 82 | 12 | 23 | 8 | 5 | 4 | 34 |
| 7 | 9 | 39 | 38 | 80 | 15 | 15 | 3 | 2 | 33 | 17 |
