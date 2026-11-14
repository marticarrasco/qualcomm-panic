# Text-only solver for the UNO Q

Use the existing [Qwen3.5 0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
Q8_0 service. The default text solver runs offline, uses
only Python's standard library, and keeps the official `predict.py` CSV format.
No changes are required to the llama-server service or its model files.

From the project folder **on the board**:

```bash
# Probe real hardware with the full 100-row text cohort.
ANSWER_MODE=text python3 eval.py --input data/eval_sets/val_100.jsonl --limit 0 --name unoq-text-100
ANSWER_MODE=text python3 predict.py --input problems.jsonl --output predictions.csv

# Fast mode: arithmetic, near-duplicate retrieval and a train-fitted option ranker.
# No language model calls for complete text-only problems.
ANSWER_MODE=text TEXT_ROUTE=ranker TEXT_MEDIAN_BYPASS=0 \
  python3 predict.py --input problems.jsonl --output fast.csv

# Route just the uncertain items to Qwen.
ANSWER_MODE=text TEXT_ROUTE=uncertain TEXT_MEDIAN_BYPASS=0 \
  python3 predict.py --input problems.jsonl --output routed.csv

# When an input batch contains translations of the same problem:
ANSWER_MODE=text TEXT_ROUTE=uncertain TEXT_MEDIAN_BYPASS=0 TEXT_TRANSLATIONS=1 \
  python3 predict.py --input problems.jsonl --output batch.csv

# Old policy for a matched comparison or rollback:
ANSWER_MODE=budget python3 predict.py --input problems.jsonl --output baseline.csv
```

Copy `copilot/option_weights.json` with the code and keep the existing
`data/train.jsonl` on the board for lexical retrieval. If train is absent the
solver skips retrieval. The weights are a few dozen coefficients; training needs
NumPy on the laptop, but prediction does not. The retrieval index uses train
only, and labels on the prediction input are not consulted. Translation reuse
only reads other rows in the same input and requires identical IDs and unique,
matching option values. It is off by default for a per-question benchmark.

The text path tries a small set of exact calculations and high-confidence lexical
matches first. It transfers the answer **value** only when all five options
agree, the stems in the same language are highly similar, and the candidates
agree. It excludes the current ID across languages. It then uses the configured
model routing and fallback. Incomplete OCR options are excluded from train
indices and fitting. At inference an incomplete problem still gets an A–E
answer to satisfy the contest interface.

The `shots` profile, used when the model is enabled, keeps the original three
worked examples as a stable cacheable prefix. It generates at most **192 tokens**. The client uses actual prefill misses and
generated tokens to stop around a **54 s estimate** (11 read tokens/s, 5.6
write tokens/s, both based on prior board measurements) and has an independent
**57 s wall-clock deadline**. Model timeouts and incomplete answers return the
chosen prior. A test with a continuously streaming server verifies cancellation;
the new pipeline has yet to be timed on a physical UNO Q.

On the board, check maximum wall time with `eval.py` and keep the default
time limit. If running alongside other apps slows the model, the deadline
still returns the chosen fallback. Setting `TEXT_MAX_TOKENS=160` is a faster
option, with two fewer correct decisions in the 100-row token replay when
using the old prior.

## Results

The 100-row cohort contains 100 different problems. The 300-, 1,000- and
2,961-row cohorts each contain 121 distinct problems. They share problems and
are not independent test sets. Compare the row score and the mean over IDs,
which weights each problem equally across its available translations.

The ranker-first default is based on the 300-row cohort. The model profiles and
the selective and batch modes are independently selectable because their gains
depend on model runtime and how many translations the input contains.

| Policy | Cohort | Correct | Mean over IDs |
|---|---:|---:|---:|
| Existing budget policy (pinned llama.cpp, laptop) | 100 | 37/100 | 37.0% |
| Default hybrid (same pinned llama.cpp and questions) | 100 | 40/100 | 40.0% |
| Qwen only on uncertain items (same questions; rows from the pinned 300 run) | 100 | 38/100 | 38.0% |
| Fast, no generation | 300 | 113/300 | 32.6% |
| Qwen, legacy numeric bypass + ranker fallback (Vulkan) | 300 | 97/300 | 30.4% |
| Qwen only on uncertain items (Vulkan) | 300 | 113/300 | 33.1% |
| Qwen only on uncertain items (pinned llama.cpp) | 300 | 118/300 | 33.5% |
| Qwen only on uncertain items, with translation reuse (pinned llama.cpp) | 300 | 118/300 | 33.9% |
| Fast, no generation | 1,000 | 359/1,000 | 32.5% |
| Fast with translation reuse | 1,000 | 370/1,000 | 33.1% |
| Qwen only on uncertain items, with translation reuse (Vulkan) | 1,000 | 402/1,000 | 34.4% |
| Existing model-free prior | 2,961 | 903/2,961 | 28.1% |
| Fast, no generation | 2,961 | 1,066/2,961 | 32.4% |
| Fast with translation reuse | 2,961 | 1,098/2,961 | 33.1% |
| Qwen only on uncertain items, with translation reuse (Vulkan) | 2,961 | 1,176/2,961 | 34.4% |

Eight regression tests cover exact arithmetic, avoiding ambiguous choices,
lexical and semantic retrieval, the median and numeric rules, translation reuse,
and cancellation. The default predictor
also passed its five-row smoke test with the real local model in Linux and
returned valid CSV rows when the model was unavailable under Windows.

On the matched 100, the default's projected UNO Q time fell from **29.9 s to
25.4 s mean** and from **57.0 s to 50.9 s p95**. The accuracy difference is
small: a paired bootstrap over IDs puts the 95% interval at −2 to +8 points.
Selective routing scored 38 on those same 100 rows at 12.4 s projected mean.
It scored 118/300 at **14.8 s projected mean, 48.0 s p95** on the pinned
runtime. Reusing translations in that run kept 118/300 with 42 rather than
118 model calls, at 5.1 s projected mean over the batch.
On the 300-row Vulkan cohort, the narrower routing used 118 model calls and
estimated **14.7 s mean, 47.9 s p95, 54.1 s max**, including rows that used no
model. The broader routing averaged 25.1 s. Cold-start Python index loading is outside
the per-question timer. These are projections, not hardware timings. GPU
generation used llama.cpp b12387; the pinned CPU run uses the same version as
the board. The engines may choose different tokens. Use pinned runs for paired
comparisons.

With translation reuse, the 1,000-row run made **54 model calls** and reused
866 answers. The estimate was **1.9 s averaged across the batch** and
**52.7 s at worst**. The full 2,961-row run made 77 calls, reused 2,803 answers,
and estimated 0.97 s per row on average, 50.6 s at worst. Only use these means
for batches with repeats; independent questions cannot share a solve.

As a sanity check on Python overhead, the fast process in WSL loaded the
1,000 inputs and train index in 1.23 s, used about 43 MiB peak RSS, and then
answered the entire cohort in 0.39 s before CSV I/O. This is a laptop measure.

## Retrieved few-shot experiment

`TEXT_PROFILE=retrieved` replaces the fixed examples with up to three same-language
examples found by BGE in the text-only, complete-option train set. The search runs
locally, but retrieved question text is sent to Qwen and costs input tokens. Build
the index with `python build_index.py --text-only --db data/index-text.db`; the
profile defaults to `TEXT_RAG_K=3` and `TEXT_RAG_INDEX=data/index-text.db`.
`build_index.py --batch` can increase the embedding batch size on a laptop.

On the same 100 distinct validation problems, retrieved few-shot scored **31/100**
against **40/100** for `shots`. It won 4 rows, lost 13 and tied 83. Its mean projected
UNO Q time was **34.7 s**, p95 **54.3 s**, and worst **65.6 s**; 53 rows exceeded the
54-second target and one exceeded the 60-second contest limit. The prompt budget
was hit on 55 rows. Retrieved prompting used 29,666 input tokens across the cohort,
compared with 8,046 for `shots`. The default therefore remains `shots`; the retrieved
profile stays available for further prompt and retrieval experiments.

## Ranker-first default with uncertain model routing

The default now uses the train-fitted option ranker for every complete
multiple-choice problem (`TEXT_MEDIAN_BYPASS=0`) and escalates to Qwen only when
the ranker's best probability is below **0.29** (`TEXT_ROUTE=uncertain`). This is
the measured best policy on the 300-row cohort:

| Policy | Correct | Mean over IDs | Time |
|---|---:|---:|---:|
| Always D | 78/300 (26.0%) | 26.0% | 0 s |
| Model-free prior (`priors.guess`) | 98/300 (32.7%) | — | 0 s |
| Ranker, median bypass on | 104/300 (34.7%) | 31.0% | 0 s |
| Ranker only, median bypass off | 113/300 (37.7%) | 32.6% | 0 s |
| **Ranker + model on uncertain items (default)** | **118/300 (39.3%)** | **33.5%** | **14.8 s projected mean, 48.0 s p95** |
| Previous text hybrid (`shots`, model, bypass on) | 97/300 (32.3%) | — | — |

The uncertain row made 118 model calls and reused the ranker for the rest. The
ranker-only policy remains available with `TEXT_ROUTE=ranker` for a fully
model-free run.

The median rule is worse than the ranker exactly where it fires: **25/85
(29.4%)** against **35/85 (41.2%)**. That is why `TEXT_MEDIAN_BYPASS=0` is now the
default; the bypass remains available for comparison.

On the same 50 English text-only problems, every model variant was at or below
the ranker: `shots` 14/50, `retrieved` without grammar 15/50, retrieved with the
one-line grammar 11/50, retrieved plus RAG k=3 11/50, `calc` 13/50, ranker 16/50.
A 50-row hybrid (ranker for numeric options, model for text options) scored 18/50,
but on the 300-row cohort it was 25/105 (23.8%) when stopped, so the 50-row gain
did not transfer. The model remains available for `TEXT_ROUTE=all`/`uncertain`
and for the tutor; the predictor defaults to the ranker.

## What was tried

- **Model and prompt.** Qwen3.5 0.8B with a terse scratchpad prompt scored 22/100
  versus 28/100 for the prior; the original three examples remain the strongest
  measured starting point on the pinned setup. Replaying the original 100
  prompts at a 54 s modeled deadline with the old prior scored 33, 35, 37 and 37
  at caps of 128, 160, 192 and 224 respectively. These are replays of emitted
  answers, not distinct generation runs at all four caps.
- **Other models.** [LFM2.5 1.2B Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF) with an optional calculation scored
  11/36 vs 9/36 for the prior; its constrained one-letter and expression modes
  scored 6/36 and 7/36 respectively. Qwen2.5-Math 1.5B did not finish any of the
  first 23 with the required format at 140 tokens. [LFM Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF) also ran out of
  budget on most pilot rows. The pilots do not justify replacing Qwen3.5 0.8B.
- **Retrieval.** Exact stem and option equality produced zero hits. Matching the
  option values first and using BGE on the question increased coverage to
  12/300 (12 correct, seven distinct IDs) at cosine 0.75 in this cohort. At the
  lexical threshold of 0.80 the solver hit nine rows, all correct; on train with
  the query ID excluded, it hit 220 rows, all correct. The sample counts are
  warnings against interpreting either threshold as a guarantee. Adding a
  BGE neighbor to the prompt was tested separately: 6/16 at the 192-token
  estimate vs 4/16 for the prior, but 45.2 s mean projected raw cost. BGE and
  prompt retrieval stay opt-in.
- **Train fit.** 16,297 text-only rows have complete options; one representative
  per ID yields 684 training examples. Year-grouped five-fold CV picked
  regularization by log loss, without looking at validation. The ranker scores
  184/684 on CV versus 203/684 for the median prior, so the validation uplift
  may not transfer unchanged. Extra weak arithmetic features gave 342/1,000
  versus 353/1,000 for the simple ranker and were left out. No generative fine
  tuning was needed for this result.

Raw details and manifests are in `results/text-only/` locally. Each row records
the status, token counts and elapsed time. Reproduce the ranker with
`python -m experiments.train_option_ranker`; compare exact cohort keys with
`python -m experiments.compare_text_runs --help`; run regressions with
`python -m unittest experiments.test_text_solver`.
