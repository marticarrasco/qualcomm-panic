# The organizers' practice set: what it tells us about the final test

Written 15 November, about 03:00. For the team, before choosing the configuration for the final run.

## What the final test is

From `datathon-qualcomm-m3kang-format/README.md`: 20 new Kangaroo-style problems written for the datathon (not M3Kang), one language each (English or Catalan), levels 0 to 7, 4 with a figure, run with `predict.py`, 60 s per problem. A practice set of 50 problems in both languages (100 rows) has the answers.

One problem is worth 5 points of the Reasoning score, so the result will be noisy whatever we do.

## What the practice set looks like

Prepared with `laptop/prepare_m3kang.py --parquet … --out data/demo` (OCR read the options of all 100 rows).

| Property | Practice set | M3Kang |
| --- | --- | --- |
| Correct letters | exactly 10 each of A to E | D is 25% |
| Numeric options | always in increasing order (43 of 43) | 39% out of order |
| Rank of the correct value, smallest to largest | 8, 10, 9, 9, 7: flat | 9%, 22%, 31%, 27%, 12% |

So the regularities our ranker and the median rule learned from M3Kang are not there. Always answering one letter scores exactly 20%.

## What we measured on the 100 rows

Same model (Qwen3.5 0.8B Q8_0), laptop, settings changed with environment variables only.

| Route | Right of 100 | English | Catalan | Notes |
| --- | --- | --- | --- | --- |
| Current default (`TEXT_ROUTE=uncertain`) | 21 | 12 | 9 | the ranker answers 66 rows and gets 12; it answers D 79 times |
| Model on every problem (`TEXT_ROUTE=model`) | 21 | 11 | 10 | 47 answers do not finish in 192 tokens |
| Earlier chain (`ANSWER_MODE=budget`) | 22 | 13 | 9 | 30 answers do not finish in 400 tokens |
| One request (`ANSWER_MODE=direct`) | 15 | 8 | 7 | 72 unparseable answers |
| Always D | 20 | 10 | 10 | |

No route is distinguishable from chance on 100 rows. The differences inside the table are:

- **The ranker has no signal here** (12 of 66, 18%).
- **The model has some when it finishes, in English**: 12 of 34 (35%) in budget mode, 8 of 25 in model-only mode. In Catalan 7 of 36 (19%).
- **Unfinished answers are the loss**: they fall back to D, which is right 20% of the time by construction. Typical failures on easy problems are endless enumeration ("3 + 1 = 4, 4 + 3 = 7, …" for 400 tokens) and algebra that never concludes.
- **Levels 0 and 1 are not easier for the model** (8 of 32 in budget mode). These are trick questions, not arithmetic.

## What we suggest

1. **Do not let the ranker answer on the final test.** Use a model-first route (`TEXT_ROUTE=model`, or `ANSWER_MODE=budget`) with the full 60 s budget. Expected gain is small (about one problem in 20) but it is the only route with any signal on this kind of problem.
2. **Balance the fallback letters.** If the final 20 are balanced like the practice set (4 of each letter), an unfinished answer should get the letter we have used least so far, not always D. Simulated on 4,000 random balanced 20-problem tests drawn from the practice set: English 5.2 right with D against 6.1 with balancing (budget mode), Catalan 3.6 against 3.8. If the final is not balanced, this costs nothing. It makes the answers depend on the order of the file, which is still deterministic.
3. **Stop long enumerations early.** A repeated-line detector that cuts the stream and falls back would free the time; whether a second attempt then helps is untested.
4. **Do not tune prompts on these 50 problems.** They are the only data that looks like the test, and 50 is too few to tune on without fooling ourselves.
5. **Time on the board** is unmeasured for any of these routes on this set. Do one timed run before the last call.

The M3Kang numbers in the README (29.5% against 23.1%) stay true for M3Kang; they should not be presented as what we expect on the final test.

## Files

- `results/eval-20261115-*-demo-model-only.csv`, `…-demo-budget.csv`, `…-demo-direct.csv`: per-problem outputs.
- `results/datathon-qualcomm-demo/`: the default-route run.
