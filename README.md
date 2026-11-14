# MathQ

Projecte de la datathon UPC–Qualcomm per explorar la resolució de problemes matemàtics amb models de llenguatge a l'Arduino UNO Q.

Aquest repositori parteix del material base del challenge. La idea de treball inicial és desenvolupar i provar el model i el pipeline a l'ordinador, i després integrar-los i validar-los a la placa.

## Progress (branch `abdullah/reasoning-experiments`)

Updated 14 Nov, evening. Work on this branch is not merged into `main` yet: merging changes how `solve()` answers (see below), so the team should agree first.

**Reasoning (`predict.py`)**

- The 0.8B model is at chance when asked for a letter, with or without bias correction, with a 2B model, or with the picture. Full measurements: [EXPERIMENTS.md](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/EXPERIMENTS.md).
- New answer mode, `ANSWER_MODE=budget` in `config.env` (`copilot/reasoner.py`, `copilot/priors.py`):
  1. no options, or five numeric options printed out of order: a model-free prior (median value, else D), no model call;
  2. otherwise terse reasoning, streamed and stopped at a 57 s wall-clock deadline;
  3. the prior again if the reasoning does not reach `Answer: X`.
- On 147 English validation problems with options: 50 right (34%), against 32 (22%) for always-D. On the board, problems took 10 to 57 s and none went over 60 s.
- Set `ANSWER_MODE=direct` to get the previous behaviour back.

**Teaching (in progress)**

- `tutor/`: a tutor that never relies on the model for the truth. It shows the problem image, takes A to E, adapts the level, and gives hints that are correct by construction (method tip, a similar easier solved problem, eliminating two wrong options, the answer only on request).
- `applab/math-matrix/`: App Lab app that shows the tutor's feedback on the LED matrix (tick, cross, hint, level).

**On the board (`Gary`)**

- The starter repository is untouched in `~/datathon-UPC-maths-qualcomm-arduino-unoq`. This branch's pipeline is in `~/qp`.

**Open questions for the team**

- Merge `ANSWER_MODE=budget` into `main`?
- M3Kang data is committed in this repository. The licence is research-only, so the history must be cleaned before the repository goes public.
