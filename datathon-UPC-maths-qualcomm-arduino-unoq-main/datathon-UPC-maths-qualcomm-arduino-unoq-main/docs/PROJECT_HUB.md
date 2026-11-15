# Pista: a maths tutor that runs on the Arduino UNO Q and never gives the answer away

Draft for Arduino Project Hub. Paste section by section; add the photos and the video where marked. Do not upload M3Kang problems or images: the dataset is research-only. Screenshots must show a made-up problem (`data/sample_problems.jsonl`) or have the problem blurred.

## 1. Introduction

Maths results in Catalonia are falling in assessments such as PISA, and teachers have little time for each student. An AI tutor could help, but only if it guides the student's reasoning and does not do the homework, and only if the student's data stays private.

Pista is a maths tutor that runs entirely on an Arduino UNO Q (4 GB), with no internet. A student practises real Kangaroo competition problems on a web page served by the board. The board's LED matrix shows a tick, a cross, a light bulb for a hint, and the student's level. When the student is stuck, Pista climbs a ladder of hints that each give a little more, and shows the answer only if the student asks for it.

The same board also answers Kangaroo problems on its own (one letter, A to E, in under 60 seconds), which the datathon measures on a hidden test set.

*[Photo: the board with a tick on the LED matrix next to the page.] [Video: 60 seconds of the demo.]*

## 2. Components and apps

| Part | Role |
| --- | --- |
| Arduino UNO Q, 4 GB | Runs everything. Linux side: models, tutor, solver. Microcontroller side: the LED matrix. |
| Arduino App Lab | Runs the small app that draws on the LED matrix |
| llama.cpp (commit `2cbe0ec`) | Runs the models on the CPU |
| Qwen3.5 0.8B, Q8_0 (835 MB) | The on-board language model: reasons about a problem when asked, inside a time budget |
| bge-small-en-v1.5 (37 MB) | Embeddings, to find similar problems |
| M3Kang | 1,747 Kangaroo problems, 8 levels, 108 languages: problem bank, map, training data and benchmark |

No external modules: only the board, its LED matrix, and a web page.

## 3. Architecture

```
            Student's phone or laptop  (web page, A to E, hints)
                         |  http, port 8000
+------------------------|---------------------------------------------+
|  Linux side (Qualcomm QRB2210)                                       |
|                                                                      |
|  tutor/server.py --- tutor/tutor.py --- data/tutor.db (problem bank, |
|       |                   |             topics, neighbours)          |
|       |                   +----------- data/tutor_students.db        |
|       |  /api/matrix                                                 |
|  App Lab app (python/main.py) --Bridge--> sketch.ino --> LED matrix  |
|                                                                      |
|  predict.py -- copilot/pipeline.py -- option ranker (no model)       |
|                       |                                              |
|                       +-- llama-server :8081  Qwen3.5 0.8B           |
|                           llama-server :8082  bge-small embeddings   |
+----------------------------------------------------------------------+
```

How M3Kang is used:

1. **As the problem bank, with its original images.** Each image already holds the question, the figure and the five options, so the student sees the real problem and nothing has to read the picture.
2. **As a map.** The 1,466 train assignments supply the ten topics. Each problem also has a level and neighbouring exercises to guide adaptive practice and follow-ups.
3. **As training data.** A small ranker learns from the training problems which options tend to be right (for example, rarely the smallest or the largest value).
4. **As the benchmark.** Every number below is on validation problems held out by problem id.

The key design decision: the language model is never the source of truth. We measured that it cannot solve these problems (see Results), so correct answers come from the dataset labels, and the model generates teaching hints for each attempt. The tutor filters these drafts before showing them.

## 4. Step by step

Start from a board set up with steps 0 to 12 of the [datathon repository](https://github.com/mpous/datathon-UPC-maths-qualcomm-arduino-unoq) (llama.cpp built, both model services running).

**4.1 Get the code (laptop).** Clone our repository and prepare M3Kang as its README explains. The data is not in the repository.

**4.2 Build the tutor's bank (laptop or board).**

```bash
python3 tutor/build_bank.py --lexical
```

The bank should report 1,747 problems in both Catalan and English.

**4.3 Copy everything to the board (laptop, board on USB).**

```bash
bash laptop/deploy_to_board.sh
```

You should see five sample answers, each one letter and under 60 seconds, then the tutor's address.

**4.4 Open the tutor.** On the laptop: http://127.0.0.1:8000 (it travels over the USB cable, so it works with Wi-Fi off). On a phone on the same network: `http://<board-ip>:8000`.

**4.6 The LED matrix.** The deploy script installs and starts the App Lab app `math-matrix`. It polls the tutor twice a second and draws the last event.

**4.7 Run the solver the way the jury does (board).**

```bash
python3 predict.py --input data/val.jsonl --output results/predictions.csv
```

## 5. Results

### The solver

Validation problems are held out by problem id, so none of them was used to fit anything. Run on a laptop with the same model file; board time is estimated from token counts with the speeds we measured on the board (11 tokens/s read, 5.6 written).

| Set | Problems | Always D | Ours |
| --- | --- | --- | --- |
| English validation, every problem | 281 | 65 = 23.1% | **83 = 29.5%** (95% interval 24.5 to 35.1%) |
| of which text-only | 123 | 19.5% | 43 = 35.0% |
| of which need the picture | 158 | 25.9% | 40 = 25.3% |
| Text-only with options, all languages (rows) | 2,961 | 682 = 23.0% | 1,176 = 39.7% |

The organizers' baseline is 22.5% on 40 English text-only problems.

Estimated board time on the English run: mean 13.4 s, 95th percentile 48.9 s. On the laptop 11 answers ran past what the board could write in 60 s; on the board the wall-clock deadline stops those and the ranker answers instead, so the board number can differ by a few problems. A timed run on the board is the last check before submission.

### What we tried that did not work

| Idea | Result |
| --- | --- |
| One letter straight from the model | Chance (24%), it answers D 69 times out of 100 |
| Cancelling the letter bias over five option orders | Bias gone, still chance (26%) |
| A 2B model | 31% on one-letter answers, three times slower on the board |
| Seven other small models | None beat Qwen3.5 0.8B inside the time budget |
| The vision model on the whole image | 15.3%, below always-D (25.3%) |
| The vision model on the cropped figure | 5.6 to 12.2% on 90 problems; it reads numbers but not what they mean |
| Copying the answer of a near-identical training problem | Right 5 of 5 times, but applies to 2% of problems |

### The tutor

| Check | Result |
| --- | --- |
| Hint "rule two out" never removes the right letter | 0 failures in 19,217 cases over all 1,747 problems |
| The answer never reaches the browser early | 0 leaks in 8,245 replies |
| Time per step (next problem, hint, answer) | 5 to 12 ms on the board |
| Simulated students settle at their true level | within 1 to 3 problems |
| Topic classifier, hand check | 21 of 30 right |
| AI guiding questions, hand check of 30 | prototype measurement for the earlier offline question generator |

## 6. What we learned

- Measure before you build. A day of measurements showed that prompt tricks, bias correction, bigger models and vision do not move a 0.8B model off chance on competition problems. That changed the whole design.
- On a board this size, the data is the intelligence. Labels, neighbours and the statistics of the options did more than the model.
- A cached system prompt is free. Worked examples in it cost reading time once per run, not once per problem.
- A hard wall-clock deadline is better than a token estimate: stream the answer and stop at 54 seconds.
- For teaching, the student should look at the picture, not the model.

Next: teachers judging the guiding questions, more problems in the thin topics (patterns, algebra, fractions), and spoken hints.

## Credits

M3Kang: Torres-Camps et al., 2026, arXiv:2601.16218 (Qualcomm; research use only). llama.cpp (MIT). Qwen3.5 by the Qwen team, GGUF by bartowski. bge-small-en-v1.5 by BAAI. Fonts: Bricolage Grotesque and Instrument Sans (SIL OFL).
