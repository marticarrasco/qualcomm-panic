# MathQ

Projecte de la datathon UPC–Qualcomm per explorar la resolució de problemes matemàtics amb models de llenguatge a l'Arduino UNO Q.

Aquest repositori parteix del material base del challenge. La idea de treball inicial és desenvolupar i provar el model i el pipeline a l'ordinador, i després integrar-los i validar-los a la placa.

## Pista: a maths tutor and solver on the Arduino UNO Q

Datathon FME 2026, Qualcomm challenge. Everything runs on an Arduino UNO Q (4 GB), offline.

- **Teaching.** A student practises real Kangaroo problems on a web page served by the board. When stuck, they climb a ladder of hints that each give a little more, and the answer only appears if they ask for it. The board's LED matrix shows a tick, a cross, a hint or the level.
- **Reasoning.** `predict.py` answers a file of problems with one letter each, A to E, in under 60 seconds per problem.

The code is in [`datathon-UPC-maths-qualcomm-arduino-unoq-main/…`](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main). *[Photo or short video of the board and the page goes here.]*

### The idea in one paragraph

We measured first, and the small on-board model cannot solve these problems: asked for a letter it is at chance, with or without bias correction, a bigger model, or the picture ([EXPERIMENTS.md](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/EXPERIMENTS.md)). So the model is never the source of truth. In the tutor, correct answers come from M3Kang's labels and hints come from retrieval and fixed rules. In the solver, a ranker fitted on M3Kang's training problems reads the options, and the model is asked only when the ranker is unsure, inside a hard time budget.

## Architecture

| Where | What runs |
| --- | --- |
| Linux side of the board | `llama-server` with Qwen3.5 0.8B (port 8081) and bge-small embeddings (port 8082); `tutor/server.py` (port 8000); `predict.py` |
| Microcontroller | `applab/math-matrix`: the sketch draws the frame it gets over the Bridge on the 8x13 LED matrix |
| Laptop, ahead of time only | data preparation and OCR of the options (`laptop/`), the guiding questions (`tutor/write_questions.py`), fitting the ranker (`experiments/train_option_ranker.py`) |

How we use M3Kang:

1. **Problem bank with its original images** (tutor): the image holds the question, the figure and the options, so no OCR or vision model is needed to teach.
2. **A map**: topic, level and nearest easier or harder problem for every problem (embeddings and keyword rules). It drives adaptive practice and the "similar, easier problem" hint.
3. **Training data** for the option ranker and for near-identical problem matching.
4. **Benchmark**: validation problems held out by problem id.

**The solver** (`copilot/`, settings in `config.env`):

1. If a training problem is near-identical, copy its answer value.
2. Otherwise score the five options with the train-fitted ranker (`copilot/option_ranker.py`); if it is confident, answer.
3. If it is not confident, ask the model for short reasoning with worked examples in the cached system prompt, streamed and stopped at a wall-clock deadline (`copilot/text_solver.py`).
4. If the reasoning does not finish, use the ranker's answer.

**The tutor** ([tutor/README.md](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/tutor/README.md)): adaptive level per topic with no model in the loop, and a five-rung hint ladder: a guiding question written by the AI plus a method tip, a similar easier solved problem, two wrong options ruled out, an optional first step from the model, and the answer only on request.

## How to replicate

Start from a board set up with steps 0 to 12 of the [organizers' repository](https://github.com/mpous/datathon-UPC-maths-qualcomm-arduino-unoq). Then, from [`datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main`](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main):

1. On the laptop, prepare M3Kang as the README in that folder explains (the data is not meant to be in a public repository).
2. `python3 tutor/build_bank.py` builds the tutor's problem bank (about a minute, needs the embedding server).
3. Optional, about 2 hours: `python3 tutor/write_questions.py --llm http://127.0.0.1:8083/v1/chat/completions` writes the guiding questions with Qwen3.5 2B.
4. With the board on USB: `bash laptop/deploy_to_board.sh`. It copies the code, data, bank and images to `~/qp`, runs `predict.py` on five sample problems, starts the tutor and the LED app, and forwards port 8000.
5. Open http://127.0.0.1:8000. On the board, `python3 predict.py --input data/val.jsonl --output results/predictions.csv` is the official run.

The demo and pitch script is in [docs/DEMO.md](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/docs/DEMO.md); the Project Hub draft in [docs/PROJECT_HUB.md](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/docs/PROJECT_HUB.md).

## Results

### Solver

Validation problems are held out by problem id, so none of them was used to fit anything. Run on a laptop with the same model file; board time is estimated from token counts with the speeds we measured on the board (11 tokens/s read, 5.6 written).

| Set | Problems | Always D | Ours |
| --- | --- | --- | --- |
| English validation, every problem | 281 | 65 = 23.1% | **83 = 29.5%** (95% interval 24.5 to 35.1%) |
| of which text-only | 123 | 19.5% | 43 = 35.0% |
| of which need the picture | 158 | 25.9% | 40 = 25.3% |
| Text-only with options, all languages (rows) | 2,961 | 682 = 23.0% | 1,176 = 39.7% |

The organizers' baseline is 22.5% on 40 English text-only problems.

**These are M3Kang numbers.** On the organizers' practice set for the final test (new problems, balanced letters, options always in order) every route we tried is at chance: 21 or 22 of 100. See [docs/FINAL_TEST_FINDINGS.md](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/docs/FINAL_TEST_FINDINGS.md).

Estimated board time on the English run: mean 13.4 s, 95th percentile 48.9 s. On the laptop 11 answers ran past what the board could write in 60 s; on the board the wall-clock deadline stops those and the ranker answers instead, so the board number can differ by a few problems. A timed run on the board is the last check before submission.

### Tutor

| Check | Result |
| --- | --- |
| "Rule two out" never removes the right letter | 0 failures in 19,217 cases over all 1,747 problems |
| The answer never reaches the browser early | 0 leaks in 8,245 replies |
| Time per step on the board | 5 to 12 ms |
| Simulated students settle at their true level | within 1 to 3 problems, no repeated problem |
| Topic classifier, hand check | 21 of 30 right |
| AI guiding questions, hand check of 30 | about 17 useful, about 6 misleading; shown as the first rung only and labelled as AI-written |

## Error analysis

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

What the tables say:

- The gain comes from the options, not from the model. Where the model finishes its reasoning it is right 21.7% of the time (18 of 83), close to always-D on the same rows; the ranker is right 36%.
- Text-only problems gain 15 points over always-D. Problems that need the picture gain nothing: the solver cannot see the figure, and the vision model made things worse.
- Numbers printed out of order are the best case (42% against 16%): there the position of the letter says nothing and the value says a lot.
- Geometry with measurements is the weakest topic and is below always-D: those problems depend on the figure.
- Levels 0 to 3 are below always-D on this sample. They have the most picture problems, and D happens to be common there.
- By language we only have the text-only cohort so far (39.7% over 2,961 rows in many languages); a per-language table is still to do.

## What did not work

Details and counts for each are in [EXPERIMENTS.md](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/EXPERIMENTS.md).

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
