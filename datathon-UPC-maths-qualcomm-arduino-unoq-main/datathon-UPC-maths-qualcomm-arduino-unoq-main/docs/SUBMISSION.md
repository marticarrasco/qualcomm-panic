# Final submission: what runs, and the checklist

Branch `abdullah/final` = `jordi/pot-finetuning-runner` (which already contains `main`) plus three small
safety changes. Nothing here is on `main` yet: merging is a team decision.

## What was picked, and from whom

| Part | Choice | Why |
| --- | --- | --- |
| Solver | Jordi's program-of-thought route (`ANSWER_MODE=pot`): the model writes a short Python program, the board runs it in a sandbox and matches the printed number to an option | The only route above chance on the organizers' practice set. Rerun independently on a second laptop with the stock model: 38 of 100 (always D: 20). Jordi reports 41 with his fine-tuned model |
| Model | Jordi's fine-tuned `Qwen3.5-0.8B-M3Kang-PoT-v4-e1-Q8_0.gguf` if it is on the board, else the stock `Qwen3.5-0.8B-Q8_0` | Same size and speed, more programs that run. Not in git (794 MB) |
| Fallback | D, as in Jordi's branch | Two alternatives were tested on the practice set and neither helped (below) |
| Entry point, evaluation, OCR, services | Martí's `predict.py`, `eval.py`, `board/` | Unchanged |
| Tutor and LED matrix | `tutor/`, `applab/math-matrix/` | Unchanged |

The older routes (`text`, `budget`, `direct`, the option ranker) stay in the code and can be selected in
`config.env`, but all of them are at chance on the practice set.

## The three changes on top of Jordi's branch

1. `POT_STYLE=auto` (new default). The zero-shot prompt only works with the fine-tuned model and the few-shot
   prompt only with the stock one. `auto` asks `llama-server` which file it has loaded and picks the matching
   prompt, so a model swap on the board cannot leave them mismatched.
2. The model request now stops 2.5 s before the per-problem budget, so running the program still fits. On the
   board a retry that starts late could otherwise end at 59 s or later.
3. `laptop/deploy_to_board.sh` copies the fine-tuned model to the board, checks its SHA-256, switches the
   `math-llm` service to it and prints the prompt style in use.

## Checks done on the laptop (stock model, few-shot prompt)

- `predict.py` on 30 practice rows, twice: same answers both times, every answer one letter A to E,
  CSV columns `id, language, answer, seconds, status`.
- Practice set, 100 rows: 38 right. Programs that matched an option: 18 right of 29. Fallback D: 20 right of 71
  (10 of them on the 20 figure rows).
- Fallback alternatives, simulated on 4,000 random tests of 16 text + 4 figure problems:
  D everywhere 7.6 right of 20; least-used letter for text problems 7.6 to 7.8; least-used letter everywhere 6.5 to 6.6.
  The text reasoning of last night on the problems where no program matched: 5 right of 29. Neither is worth the risk.

## Not checked yet

- The fine-tuned model: the file is not on this laptop. Its numbers are Jordi's.
- Anything on the board. The laptop's time estimate does not model the retries: on the board a retry only
  starts when at least 22 s are left, so expect fewer retries there than on a laptop and a result nearer the
  single-attempt figure.

## Checklist for the board

1. Get `Qwen3.5-0.8B-M3Kang-PoT-v4-e1-Q8_0.gguf` from Jordi into `~/models/` on the laptop.
2. Plug the board in and run `bash laptop/deploy_to_board.sh`. It must print `prompt style: finetuned`
   and finish the five sample problems.
3. Timed run on the practice set (about 20 minutes):
   `adb push data/demo/train.jsonl /home/arduino/qp/data/demo.jsonl` then
   `adb shell 'cd ~/qp && python3 predict.py --input data/demo.jsonl --output results/demo-board.csv'`.
   Every `seconds` value must be under 60.
4. Look at the LED matrix while using the tutor at `http://127.0.0.1:8000`.
5. Before the repository goes public: remove the M3Kang data from the git history, add the licence file,
   team names and a photo or video to the README.
