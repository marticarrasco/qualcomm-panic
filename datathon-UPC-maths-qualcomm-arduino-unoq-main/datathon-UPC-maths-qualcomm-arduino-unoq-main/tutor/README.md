# Maths tutor (Teaching)

A tutor that runs entirely on the Arduino UNO Q and guides a student through
M3Kang (Kangaroo) problems **without giving the answer away**. Python standard
library only, offline, English and Catalan.

The small on-board language model cannot solve these problems (about chance), so
it is never the source of truth. The right answer comes from the dataset label,
similar problems come from an embedding search done once when the bank is built,
and every teaching decision is a plain rule you can read in `tutor.py`.

## Run it

For a self-contained bilingual demo with one fictional profile, run
`python3 tutor/server.py --demo`. [Architecture and demo script in Catalan](DEMO.md).

From the project folder:

```bash
python3 tutor/build_bank.py            # once: builds data/tutor.db (needs the embedding server on 8082)
python3 tutor/write_questions.py --llm http://127.0.0.1:8083/v1/chat/completions
                                       # once: the AI's rung-1 questions, data/tutor_questions.db (about 2 h on a laptop)
python3 tutor/server.py                # http://127.0.0.1:8000
python3 tutor/server.py --host 0.0.0.0 # reachable from phones and from the LED app
python3 tutor/simulate.py              # simulated students and safety checks
```

- `build_bank.py --check 30` also prints 30 random problems with their topic and the keywords that fired.
- `server.py --no-llm` uses fallback hints for the chat and the first hint button.
- `server.py --students PATH` chooses where student profiles are kept (default `data/tutor_students.db`).
- A link such as `/?name=Aina&lang=cat&topic=counting&level=2` starts a session directly.
- Keyboard: `A`–`E` answer, `H` hint, `N` next problem, `Esc` closes the enlarged picture.

The page shows the problem image (which already contains the question, the five
options and any figure) and five A–E buttons, so picture problems work too and no
OCR or vision model is needed.

## The page

The look is a squared exercise book: blue pen for actions, a yellow highlighter for
hints, the teacher's green tick when a problem is solved. Fonts are Bricolage
Grotesque (headings) and Instrument Sans (text), both under the SIL Open Font
Licence, shipped in `static/fonts` so the page works offline on the board.

- **Problem first.** The problem picture spans the full width so its text stays
  readable. On a phone, tapping it opens it full screen, turned sideways so the
  wide strip uses the long side of the screen.
- **The hint ladder** shows help in order. The first hint also launches a new
  model-written hint in the chat panel.
- **The warm-up problem** (rung 2) appears full width under the student's problem,
  not squeezed into the ladder.
- **The board's LED matrix, mirrored.** A live 8x13 copy of what the LED app draws
  (same pictures and level staircase as `applab/math-matrix/python/main.py`), read
  from `/api/matrix`. It shows "board not connected" when the server is unreachable.
- **Your levels**: one 8-step bar per topic practised.
- **Teacher view**: totals, what to practise next, levels by topic, latest problems.
- Light only, on purpose (paper). Works from 320 px wide; keyboard focus is visible
  and animations stop when the device asks for reduced motion.

Design references: the education rows of
[UI UX Pro Max](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill) (playful
but clear hierarchy, progress always visible, soft press feedback, 44 px touch
targets) and Anthropic's `frontend-design` skill (one bold idea, no template look).

## Files

| File | What it does |
| --- | --- |
| `build_bank.py` | Builds `data/tutor.db`: problems, topic, texts, vectors, neighbours, near-duplicates. |
| `write_questions.py` | Asks the language model for one guiding question per problem and language, checks each one, writes `data/tutor_questions.db`. |
| `topics.py` | The keyword topic classifier, topic names and the method tips (English and Catalan). |
| `tutor.py` | The logic: student profile, next problem, hint ladder, summary. No web code. |
| `server.py` | `http.server` JSON API, the page and the images. |
| `index.html` | The whole page in one file. No CDN, no external scripts. |
| `static/fonts/` | The two page fonts and their OFL licences, served at `/static/`. |
| `matrix.py` | The hook for the 8x13 LED matrix. |
| `simulate.py` | Simulated students and the safety checks. |

## How the next problem is chosen

Each student has one level (0–7) per topic. No model is involved.

| What happened | Next problem |
| --- | --- |
| Solved on the first try with at most one hint | Same topic, one level up, the most similar problem there |
| Solved after one wrong try, or with two or more hints | Same topic, same level |
| Not solved, answer shown, or two or more wrong tries | Same topic, one level down, the most similar problem to the one missed |

- A problem is never repeated to the same student. Problems shown as a rung-2 example and near-duplicates (cosine ≥ 0.97; the same problem is reused at neighbouring levels) count as seen.
- If the topic has no unseen problem within one level of the target, the next problem comes from another topic at the student's level, rather than a repeat or a problem that is far too hard.
- "Mixed" practice stays four problems on a topic and then moves to the least practised one.
- A topic never tried starts at the student's average level in the topics already tried.
- A problem closes without showing the answer once only two letters are left (for example after three wrong tries), because one more guess would give it away.

## The hint ladder

Each rung reveals more than the one before. Only the last one shows the answer.

1. **A question to start.** The tutor shows a method tip immediately and asks
   the local model to draft a short hint for the current exercise. The chat also
   accepts follow-up questions. The model receives the exercise with numeric
   values masked and the previous hint when available. The tutor checks the
   generated text for answer markers, values and repetitions, and shows a
   labelled fallback when the model call or the checks fail. A bank may also
   include a guiding question prepared with `write_questions.py`.
2. **A similar problem.** The closest easier problem of the same topic from the bank, shown as an image, with "solve this one first; the same idea works". Its answer appears only on a click. (At level 0 there is no easier one, so it is a similar problem of the same level.)
3. **Rule two out.** Two letters that are not correct, chosen from the label by a fixed rule. Never the right letter, never one already tried, and at least two letters always stay open.
4. **The answer.** Only after "Show me the answer" and a confirmation. Counted as not solved, and the next problem is easier.

After a wrong answer the page says it is not right, crosses out that letter and offers the next hint. It never shows the answer.

## API and the LED matrix

`server.py` lists every route in its docstring. The label of the current problem
is only sent after the student solved it or asked for it.

After each step the server calls `tutor.matrix.show(event, **info)`. Events:
`thinking`, `correct`, `wrong`, `hint`, `level`, `idle`. The LED app can poll

```
GET /api/matrix  ->  {"event": "correct", "level": 3, "seq": 17, "time": 1794680000.0}
```

`seq` goes up by one on every new event and `level` is the last level seen (0–7).
A hint adds `"rung": 1..5`. In-process code can use `matrix.on_event(function)` instead.

## Measured

On the development Mac (Python 3.14), 14 November 2026. Not yet measured on the board.

**Bank.** 1,747 problems, 3,494 images. Build time 54 s, of which about 43 s is embedding. 52 near-duplicate pairs. 14,427 neighbour links, 37 (0.3%) filled from another topic.

| Topic | Problems |
| --- | --- |
| Geometry: area, perimeter, angles | 447 |
| Shapes, solids & spatial puzzles | 320 |
| Arithmetic & word problems | 302 |
| Number properties | 192 |
| Time, money & measures | 119 |
| Counting & probability | 108 |
| Logic & strategy | 102 |
| Fractions & percentages | 74 |
| Algebra & functions | 49 |
| Patterns & sequences | 34 |

**Topic classifier, checked by hand.** 30 random problems (`--check 30`, seed 42): 21 clearly right (70%), 3 debatable, 6 wrong. Typical mistakes: a logic problem with no logic keyword falls back to another topic, and one word decides too much ("area" in a doubling story, "circles" in a number puzzle). Two rules were adjusted after that check and the sample was not re-drawn, so treat 70% as the figure.

**Simulation** (`python3 tutor/simulate.py`, 30 problems per student, starting level 2):

| Student | Topic | Target | Estimate | Mean level, last 10 | Settled at | On topic | Repeats | ms/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact L=1 | geometry: measure | 1 | 1 | 1.0 | step 1 | 100% | 0 | 0.25 |
| exact L=3 | geometry: measure | 3 | 3 | 3.5 | step 1 | 100% | 0 | 0.56 |
| exact L=5 | geometry: measure | 5 | 5 | 5.5 | step 3 | 100% | 0 | 0.73 |
| exact L=1 | arithmetic | 1 | 1 | 1.5 | step 1 | 100% | 0 | 0.28 |
| exact L=3 | arithmetic | 3 | 3 | 3.5 | step 1 | 100% | 0 | 0.43 |
| exact L=5 | arithmetic | 5 | 5 | 5.5 | step 3 | 100% | 0 | 0.48 |
| exact L=1 | counting | 1 | 1 | 0.0 | step 1 | 83% | 0 | 0.17 |
| exact L=3 | counting | 3 | 3 | 3.0 | step 1 | 100% | 0 | 0.17 |
| exact L=5 | counting | 5 | 5 | 5.5 | step 3 | 100% | 0 | 0.22 |
| exact L=1 | numbers | 1 | 1 | 1.0 | step 1 | 77% | 0 | 0.17 |
| exact L=3 | numbers | 3 | 3 | 3.0 | step 1 | 100% | 0 | 0.30 |
| exact L=5 | numbers | 5 | 5 | 5.5 | step 3 | 100% | 0 | 0.38 |
| noisy L=2 | geometry: measure | 2 | 2 | 2.7 | – | 100% | 0 | 0.36 |
| noisy L=4 | geometry: measure | 4 | 4 | 4.7 | – | 100% | 0 | 0.65 |
| noisy L=2 | arithmetic | 2 | 3 | 3.1 | – | 100% | 0 | 0.33 |
| noisy L=4 | arithmetic | 4 | 5 | 4.5 | – | 100% | 0 | 0.54 |
| always 2 hints, right | geometry: measure | 2 | – | 2.6 | step 1 | 100% | 0 | 0.35 |

- "Exact" students are always right at or below the target level and always wrong above. "Estimate" is the highest level solved cleanly and not failed since. "Settled at" is the first step from which every problem served is within one level of the target (the staircase keeps testing one level above, so the mean sits about half a level above the target).
- "Noisy" students are right with a probability that falls with the level (50% half a level above the target). Their estimate lands within one level of the target; the mean of the last ten is the steadier number.
- The slowest single step was 2.5 ms, against a target of 50 ms. The board is slower, but there is a wide margin.
- "On topic" below 100% is the bank running dry: Counting has 25 problems at levels 0–2 and Number properties 24, fewer than a 30-problem session.
- Rung 3: 19,217 cases over all 1,747 problems, the right letter was never crossed out.
- API: one simulated student walked through 1,221 different problems over HTTP (the others were used as rung-2 examples or are near-duplicates), 8,245 replies, none contained the answer letter before the problem was solved or the answer was asked for.
- The earlier prototype's optional model sentence passed the filter on five of six text problems at about 1 s each on the Mac.

**Rung-1 questions, checked by hand** (30 random problems, both languages, seed 7):

| Model | Passed the check | Useful (English) | Repeats the question | Points the wrong way |
| --- | --- | --- | --- | --- |
| Qwen3.5 0.8B | 52 / 60 | about 8 / 30 | about 14 | about 6; Catalan mostly broken |
| Qwen3.5 2B (used) | 59 / 60 | about 17 / 30 (11 good, 6 fair) | about 6 | about 6; Catalan readable |

Good examples: "Which of the points A, B, C, D, or E has the same y-coordinate as another
point?", "If you call the length of the shortest piece x, how can you express the length of
the second piece in terms of x?". A misleading one: for an equilateral triangle overlapping
a circle, "If you add the side length of the triangle and the diameter of the circle, what is
the total length around the outside?".

## Limitations

- **Not run on the board yet.** The timings above are from a laptop. Live hints are generated in the background and fall back if the model call times out.
- **About one AI question in five points the wrong way**, and some only repeat the problem.
  The filter stops leaks, not bad advice. The page says so next to every question, and the
  hand-written method tip is always shown below it.
- **The questions are written on a laptop**, with the same model file as the board
  (`Qwen_Qwen3.5-2B-Q4_K_M.gguf`), because doing all 3,494 on the board would take about a day.
- **Topics are keyword guesses**, about 70% right on a hand check. A wrong topic means the method tip may not fit and the "similar problem" stays within the wrong topic.
- **Small topics run dry.** Patterns (34 problems), Algebra (49, none below level 4) and Fractions (74) cannot fill a long session at one level; the tutor then serves another topic.
- **The similar problem is similar in wording**, found with text embeddings. For picture problems with a short question ("Which piece is missing?") it may share little with the figure.
- **A generated hint can be unhelpful or wrong.** Text checks catch common ways of revealing an answer. Use `--no-llm` to use the fallback hints.
- **Rung 3 does not explain why** the two letters are wrong; it only names them.
- **The level estimate is a simple staircase**: one lucky or unlucky answer moves it by one level.
- **Students are identified by name only**, with no password. Two students who type the same name share a profile.
- **The OCR options help check generated hints.** The original image is the source for the options shown to students in the M3Kang bank.
- **Licence.** M3Kang is for research use only. Problem text exists only in `data/tutor.db`, never in the source files.
