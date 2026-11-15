# MathQ

Development notes preserved from the earlier project README. The [current project overview](../../README.md) and [documentation index](../README.md) describe the submission layout and evaluation.

Projecte de la datathon UPC–Qualcomm per explorar la resolució de problemes matemàtics amb models de llenguatge a l'Arduino UNO Q.

Aquest repositori parteix del material base del challenge. La idea de treball inicial és desenvolupar i provar el model i el pipeline a l'ordinador, i després integrar-los i validar-los a la placa.

## Novetats d'aquesta branca (`jordi/pot-finetuning-runner`)

**Documentació completa pas a pas:** [**`POT_FINETUNING_I_RUNNER.md`**](POT_FINETUNING_I_RUNNER.md)

 A `main` ([`docs/FINAL_TEST_FINDINGS.md`](FINAL_TEST_FINDINGS.md)), vam comprovar que al set de pràctica 2026 de l'organització (`data/demo/train.jsonl`, 100 files: 50 `eng` + 50 `cat`, lletres balancejades 20% cadascuna i opcions numèriques sempre ordenades) totes les rutes de text i el *ranker* quedaven a nivell d'atzar (**21 o 22 de 100**).

En aquesta branca documentem i integrem el pipeline complet de **Program-of-Thought (PoT) + Fine-Tuning LoRA (v1 i v2) + `SmartRunner`**, que puja l'encert al mateix set de pràctica 2026 (`demo`) fins al **41,0% en una sola passada (`Pass@1`, 25 `eng` + 16 `cat`, 11 s de mitjana a la placa)** i **42,0% amb reintent condicional (`Pass@3`, 26 `eng` + 16 `cat`, ~18,6 s de mitjana a la placa)**:

1. **Generació i verificació de dades sintètiques amb l'API de Helmcode (`deepseek-v4-flash`)**:
   - Scripts: [`finetune/build_synthetic_dataset.py`](../../finetune/build_synthetic_dataset.py), [`finetune/prepare_sft_and_rag.py`](../../finetune/prepare_sft_and_rag.py) i [`finetune/build_pairs.py`](../../finetune/build_pairs.py).
   - Generació de programes Python plans (2–6 línies) per als problemes `text_only` d'entrenament de M3Kang, verificats executant-los dins del nostre sandbox AST (amb filtre anti-trampa `is_nontrivial_code` i comprovació de coincidència única amb l'opció correcta).
   - Resultat: **585 problemes verificats** i **929 parells bilingües `(enunciat, codi Python)`** en anglès i català (`pot_pairs.jsonl`, 0 solapament amb `val` ni `demo`).

2. **Fine-Tuning LoRA de `Qwen3.5-0.8B` (Versió 1 i Versió 2) i conversió GGUF `Q8_0`**:
   - Notebooks reproduïbles a Kaggle T4: [`finetune/00_gate_conversion.ipynb`](../../finetune/00_gate_conversion.ipynb) i [`finetune/01_train_pot_lora.ipynb`](../../finetune/01_train_pot_lora.ipynb) (generats per [`finetune/build_notebooks.py`](../../finetune/build_notebooks.py)).
   - **Versió 1 (`v1`)**: 3 èpoques, `LR=2e-4`, màscara de `loss` només sobre el codi usant `/apply-template` de `llama-server` (`--jinja --reasoning off`). Puja a **34,0% (`Pass@1`) / 36,0% (`Pass@3`)** a `demo`, multiplicant $\times 4$ els encerts en català i baixant el temps mitjà a la placa a **12 s** (0-shot).
   - **Versió 2 (`v2`)**: `LoRALinear` en PyTorch pur amb `Dropout(0.05)`, exclusió de les portes recurrents `in_proj_a/b/ba` de *Gated DeltaNet* per protegir la memòria d'estat recurrent, `LR=1e-4` i *Early Stopping* sub-època cada 70 passos (`step 217`, `dev loss = 0.673`). Puja a **37,0% (`Pass@1`) / 39,0% (`Pass@3`)** amb **70,4% de precisió** quan el codi casa amb una opció.

3. **Millores al `SmartRunner` i filtre anti-`'1'` ([`copilot/pot.py`](../../copilot/pot.py) i [`copilot/llm.py`](../../copilot/llm.py))**:
   - `_PermTuple(tuple)` i `smart_int` per permetre comparar tuples de dígits de `permutations(...)` directament amb enters (`1000 <= n <= 9999`, `+1,0%`).
   - `_SmartStr(str)` per suportar `7 in str(n)` amb enters i resta de strings numèrics (`+1,0%`).
   - `smart_product` / `_SmartIntProd` per suportar `product(...)` tant com a `math.prod` (multiplicació numèrica) com `itertools.product` (`+1,0%`).
   - Captura de `printed[0]` i `printed[-1]` quan un bucle `for` troba el mínim però oblida posar `break` (`+1,0%`).
   - Filtre de consens anti-`'1'` en els reintents amb temperatura (`Pass@3`).
   - **Resultat final a `demo` (100 problemes)**: **41,0% en `Pass@1` (11 s de mitjana a la placa)** i **42,0% en `Pass@3` (~18,6 s de mitjana a la placa)**.

---

## Pista: a maths tutor and solver on the Arduino UNO Q

Datathon FME 2026, Qualcomm challenge. Everything runs on an Arduino UNO Q (4 GB), offline.

- **Teaching.** A student practises real Kangaroo problems on a web page served by the board. The local model generates progressive hints and responds to wrong choices. The official key is shown after solving or requesting the answer. The board's LED matrix shows a tick, a cross, a hint or the level.
- **Reasoning.** `predict.py` answers a file of problems with one letter each, A to E, in under 60 seconds per problem.

The code is in [`datathon-UPC-maths-qualcomm-arduino-unoq-main/…`](../..). *[Photo or short video of the board and the page goes here.]*

### The idea in one paragraph

We measured first, and the small on-board model cannot solve these problems when asked for a letter or free-form prose: asked for a letter it is at chance, with or without bias correction, a bigger model, or the picture ([EXPERIMENTS.md](EXPERIMENTS.md)). So the model is never trusted to do arithmetic in its head. Pista grades against M3Kang’s official labels. The local model writes the teaching hints from the current problem and attempt. In the solver (`ANSWER_MODE=pot`), the fine-tuned `Qwen3.5-0.8B-M3Kang-PoT-v2` model translates the problem statement into a 2–6 line flat Python program (without seeing the options), Python computes the exact answer in an isolated AST-checked sandbox (`copilot/pot.py`), and we select the option matching the printed value (`74.2%` precision when matched).

## Architecture

| Where | What runs |
| --- | --- |
| Linux side of the board | `llama-server` with Qwen3.5 0.8B (port 8081) and bge-small embeddings (port 8082); `tutor/server.py` (port 8000); `predict.py` |
| Microcontroller | `applab/math-matrix`: the sketch draws the frame it gets over the Bridge on the 8x13 LED matrix |
| Laptop / Kaggle, ahead of time only | data preparation and OCR of the options (`laptop/`), synthetic PoT generation & verification (`finetune/build_synthetic_dataset.py`), LoRA fine-tuning & GGUF export (`finetune/01_train_pot_lora.ipynb`), fitting the ranker (`experiments/train_option_ranker.py`) |

How we use M3Kang:

1. **Problem bank with its original images** (tutor): the image holds the question, the figure and the options, so the full bilingual bank is usable with its original figures.
2. **A map**: topic, level and nearest easier or harder problem for every problem (embeddings and keyword rules). It drives adaptive practice and follow-up problems.
3. **Training data** for the execution-verified Program-of-Thought LoRA fine-tune (929 bilingual `(statement, Python code)` pairs from M3Kang `train`), the option ranker, and near-identical problem matching.
4. **Benchmark**: validation problems held out by problem id, plus the organizers' 2026 practice set (`data/demo/train.jsonl`, 100 rows).

**The solver** (`copilot/`, settings in `config.env`):

- **`ANSWER_MODE=pot` (default on this branch, for the 2026 final test)**:
  1. If the problem requires a figure (`text_only == False`) or has non-numeric options, return `FALLBACK` (`D`) immediately in `0.00 s` (on the 2026 practice set, `D` is right on `10/20 = 50%` of figure problems).
  2. Otherwise, ask the fine-tuned `Qwen3.5-0.8B-M3Kang-PoT-v2` (`POT_STYLE=finetuned`, or 4-shot cached `POT_STYLE=fewshot` for the stock model) for a 2–6 line flat Python script, run it in `SmartRunner` (`copilot/pot.py`), and if the output uniquely matches one option `A–E`, return it; else retry with diversity sampling if wall-clock budget remains (`> 22 s`), or fall back to `D`.
- **`ANSWER_MODE=text` (M3Kang option-ranker + budgeted prose solver)**:
  1. If a training problem is near-identical, copy its answer value.
  2. Otherwise score the five options with the train-fitted ranker (`copilot/option_ranker.py`); if it is confident, answer.
  3. If it is not confident, ask the model for short reasoning with worked examples in the cached system prompt, streamed and stopped at a wall-clock deadline (`copilot/text_solver.py`).
  4. If the reasoning does not finish, use the ranker's answer.

**The tutor** ([tutor/README.md](../../tutor/README.md)): The full Catalan and English bank has 1,747 problems. The local model offers three progressive hints, feedback on a wrong choice and conversation. Follow-up results distinguish first-try solves without help.

## How to replicate

Start with the [board setup](../setup/BOARD.md). Run the project commands from the [repository root](../../README.md).

1. On the laptop, [prepare M3Kang locally](../../data/README.md).
2. `python3 tutor/build_bank.py --lexical` builds the full bilingual bank offline. Omit `--lexical` when the embedding service is running.
3. With the board on USB, run `bash laptop/deploy_to_board.sh`. It copies the bank and images, starts the tutor and the LED app, and forwards port 8000. The tutor uses the board’s model service on port 8081.
4. Open http://127.0.0.1:8000. On the board, `python3 predict.py --input data/val.jsonl --output results/predictions.csv` is the official solver run.

The demo and pitch script is in [docs/DEMO.md](../tutor-demo-draft.md); the Project Hub draft in [docs/PROJECT_HUB.md](../project-hub-draft.md).

## Results

### Solver on the Organizers' 2026 Practice Set (`data/demo/train.jsonl`, 100 rows: 50 `eng` + 50 `cat`)

On the 2026 practice set (balanced letters A–E at 20% each, numeric options always in increasing order), M3Kang option priors and prose reasoning stay at chance (`21–22%`), whereas **Program-of-Thought + LoRA `v2` + `SmartRunner` (`ANSWER_MODE=pot`) reaches `41.0%` (`Pass@1`) and `42.0%` (`Pass@3`)**:

| Route / Model | Total Right (of 100) | English (of 50) | Catalan (of 50) | Mean est. board time (worst) |
| --- | --- | --- | --- | --- |
| Always D / 1-token letter (`ANSWER_MODE=direct`) | 20 = 20.0% | 10 (20%) | 10 (20%) | 14.5 s (22 s) |
| Option ranker + text solver (`TEXT_ROUTE=uncertain`) | 21 = 21.0% | 12 (24%) | 9 (18%) | — |
| Budgeted prose chain (`ANSWER_MODE=budget`) | 22 = 22.0% | 13 (26%) | 9 (18%) | — |
| Stock `Qwen3.5-0.8B` PoT 4-shot cached (`< 512` tok) | 33 = 33.0% | 16 (32%) | 17 (34%) | 19.5 s (44 s) |
| **LoRA `v1` PoT (`Pass@1` / `Pass@3`)** | **34.0% / 36.0%** | 18 / 19 | 16 / 17 | 12.0 s / 19.2 s |
| **LoRA `v2` PoT (`Pass@1` / `Pass@3`)** | **37.0% / 39.0%** | 19 / 20 | 18 / 19 | 11.0 s / 18.6 s |
| **LoRA `v2` PoT + `SmartRunner` (`Pass@1`)** | **41 = 41.0%** | **21 (42%)** | **20 (40%)** | **11.0 s (29 s)** |
| **LoRA `v2` PoT + `SmartRunner` (`Pass@3` shipped)** | **42 = 42.0%** | **22 (44%)** | **20 (40%)** | **18.6 s (43 s)** |

See [POT_FINETUNING_I_RUNNER.md](POT_FINETUNING_I_RUNNER.md) and [docs/FINAL_TEST_FINDINGS.md](FINAL_TEST_FINDINGS.md) for the full breakdown.

### Solver on M3Kang Validation (`ANSWER_MODE=text`)

Validation problems are held out by problem id, so none of them was used to fit anything. Run on a laptop with the same model file; board time is estimated from token counts with the speeds we measured on the board (11 tokens/s read, 5.6 written).

| Set | Problems | Always D | Ours (`ANSWER_MODE=text`) |
| --- | --- | --- | --- |
| English validation, every problem | 281 | 65 = 23.1% | **83 = 29.5%** (95% interval 24.5 to 35.1%) |
| of which text-only | 123 | 19.5% | 43 = 35.0% |
| of which need the picture | 158 | 25.9% | 40 = 25.3% |
| Text-only with options, all languages (rows) | 2,961 | 682 = 23.0% | 1,176 = 39.7% |

*(On the 562 English + Catalan M3Kang validation subset `data/val.jsonl`, 0-shot `LoRA v1 PoT` without any option ranker scores `147/562 = 26.2%` vs `21.6%` for the 1-token baseline, with a mean estimated board time of `16 s`).*

### Tutor

| Check | Result |
| --- | --- |
| "Rule two out" never removes the right letter | 0 failures in 19,217 cases over all 1,747 problems |
| The answer never reaches the browser early | 0 leaks in 8,245 replies |
| Time per step on the board | 5 to 12 ms |
| Simulated students settle at their true level | within 1 to 3 problems, no repeated problem |
| Topic classifier, hand check | 21 of 30 right |
| AI guiding questions, hand check of 30 | about 17 useful, about 6 misleading; shown as the first rung only and labelled as AI-written |

## Error analysis (M3Kang `ANSWER_MODE=text`)

The English validation run above (281 problems), broken down with `experiments/error_analysis.py`. "Always D" is the same rows scored with a constant answer, so a group that is easy to guess does not look like skill. Groups under about 30 problems are too small to rank.

**By how the answer was produced**

| Group | Problems | Right | Accuracy | 95% interval | Always D |
| --- | --- | --- | --- | --- | --- |
| prior | 95 | 28 | 29.5% | 21 to 39% | 27.4% |
| ok | 83 | 18 | 21.7% | 14 to 32% | 19.3% |
| ranker | 75 | 27 | 36.0% | 26 to 47% | 20.0% |
| fallback | 23 | 5 | 21.7% | 10 to 42% | 30.4% |
| train_match | 5 | 5 | 100.0% | 57 to 100% | 20.0% |

`ranker` and `prior` use no model. `ok` is the model finishing its reasoning; `fallback` is the model running out of time. `train_match` is a near-identical training problem whose answer value is copied.

**By level** (0 is the youngest)

| Group | Problems | Right | Accuracy | 95% interval | Always D |
| --- | --- | --- | --- | --- | --- |
| 0 | 20 | 4 | 20.0% | 8 to 42% | 45.0% |
| 1 | 17 | 8 | 47.1% | 26 to 69% | 35.3% |
| 2 | 29 | 8 | 27.6% | 15 to 46% | 31.0% |
| 3 | 12 | 2 | 16.7% | 5 to 45% | 25.0% |
| 4 | 62 | 22 | 35.5% | 25 to 48% | 19.4% |
| 5 | 53 | 14 | 26.4% | 16 to 40% | 18.9% |
| 6 | 49 | 15 | 30.6% | 20 to 45% | 18.4% |
| 7 | 39 | 10 | 25.6% | 15 to 41% | 17.9% |

**By topic** (keyword rules, about 70% accurate on a hand check of 30)

| Group | Problems | Right | Accuracy | 95% interval | Always D |
| --- | --- | --- | --- | --- | --- |
| Geometry: area, perimeter, angles | 83 | 15 | 18.1% | 11 to 28% | 24.1% |
| Shapes, solids & spatial puzzles | 56 | 21 | 37.5% | 26 to 51% | 28.6% |
| Arithmetic & word problems | 49 | 12 | 24.5% | 15 to 38% | 20.4% |
| Number properties | 28 | 13 | 46.4% | 30 to 64% | 25.0% |
| Counting & probability | 15 | 4 | 26.7% | 11 to 52% | 20.0% |
| Logic & strategy | 14 | 7 | 50.0% | 27 to 73% | 21.4% |
| Time, money & measures | 13 | 5 | 38.5% | 18 to 64% | 23.1% |
| Fractions & percentages | 11 | 2 | 18.2% | 5 to 48% | 0.0% |
| Algebra & functions | 7 | 3 | 42.9% | 16 to 75% | 28.6% |
| Patterns & sequences | 5 | 1 | 20.0% | 4 to 62% | 20.0% |

**By whether the problem needs its picture**

| Group | Problems | Right | Accuracy | 95% interval | Always D |
| --- | --- | --- | --- | --- | --- |
| needs the picture | 158 | 40 | 25.3% | 19 to 33% | 25.9% |
| text only | 123 | 43 | 35.0% | 27 to 44% | 19.5% |

**By kind of options**

| Group | Problems | Right | Accuracy | 95% interval | Always D |
| --- | --- | --- | --- | --- | --- |
| numbers, in order | 82 | 23 | 28.0% | 19 to 39% | 29.3% |
| no options | 72 | 20 | 27.8% | 19 to 39% | 27.8% |
| numbers, out of order | 64 | 27 | 42.2% | 31 to 54% | 15.6% |
| text options | 63 | 13 | 20.6% | 12 to 32% | 17.5% |

## What did not work

Details and counts for each are in [EXPERIMENTS.md](EXPERIMENTS.md).

- One letter straight from the model: chance, with a strong letter bias.
- Cancelling the bias over five option orders: the bias goes, the accuracy stays at chance.
- "Reason in two short lines": the model ignores the limit and is cut off.
- A 2B model: three times slower on the board for a few points. Seven other small models: none better in the time budget.
- The vision model, on the whole image or on the cropped figure: below always-D.
- Matching the last number of unfinished reasoning to an option: worse than the prior.
- Letter priors by level, year or question number: none beats D in cross-validation.

## Still to do

- A timed run of the final code on the board, and the per-language table.
- Check the LED matrix by eye and rehearse the demo.
- M3Kang data and images are committed in this repository's history. The licence is research-only, so the history must be cleaned (or a fresh repository made) before it goes public.
- Photo or video, licence file for our code, team names below.

## Team

*[names]*

## Credits

M3Kang: Torres-Camps et al., 2026, [arXiv:2601.16218](https://arxiv.org/abs/2601.16218), Qualcomm, research use only ([dataset card](https://huggingface.co/datasets/qualcomm/M3Kang)). [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT). Qwen3.5 by the Qwen team, GGUF by bartowski. bge-small-en-v1.5 by BAAI. RapidOCR (Apache 2.0). Fonts: Bricolage Grotesque and Instrument Sans (SIL OFL).
