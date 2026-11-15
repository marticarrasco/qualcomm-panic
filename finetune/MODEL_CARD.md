---
base_model: Qwen/Qwen3.5-0.8B
license: other
license_name: m3kang-research-use
license_link: https://huggingface.co/datasets/qualcomm/M3Kang/blob/main/LICENSE.pdf
datasets: [qualcomm/M3Kang]
language: [en, ca]
tags: [gguf, llama.cpp, math, program-of-thought, arduino-uno-q]
---

# Qwen3.5-0.8B-M3Kang-PoT-v4-e1 (GGUF, Q8_0)

Qwen3.5 0.8B fine-tuned to answer Kangaroo-style math problems by writing a short Python program. It is the solver
model of **MathQ**, built for the Datathon FME 2026 Qualcomm challenge to run offline on an Arduino UNO Q (4 GB).
Given a problem statement in English or Catalan, the model writes a 2–6 line program that prints the answer. The
copilot runs the program in a sandbox and compares the printed value with the options A–E.

Code, evaluation and the full pipeline: <https://github.com/marticarrasco/qualcomm-panic>

## File

| File | Size | SHA-256 |
| --- | --- | --- |
| `Qwen3.5-0.8B-M3Kang-PoT-v4-e1-Q8_0.gguf` | 833,592,288 bytes | `7c90dbbc3518241fd9b1661887abe1131a07d9d9f637c86a1a8a897b7a19cbb3` |

## How to use it

Serve it with llama.cpp the same way as the board does, and send the problem statement alone as the user message,
after this system prompt (`SYSTEM_FINETUNED` in `copilot/pot.py`):

```bash
llama-server -m Qwen3.5-0.8B-M3Kang-PoT-v4-e1-Q8_0.gguf --host 127.0.0.1 --port 8081 \
  --threads 4 --ctx-size 4096 --parallel 1 --cache-ram 0 --ctx-checkpoints 2 --jinja \
  --reasoning off --reasoning-budget 0
```

In the MathQ repository, `config.env` pins this file (`POT_REPO`, `POT_REVISION`, `POT_FILE`, `POT_SHA256`),
`board/03_download_models.sh` downloads and checks it, and `POT_STYLE=auto` picks the matching zero-shot prompt.

## Training

- **Base:** `Qwen/Qwen3.5-0.8B` (Hugging Face weights), converted and checked against
  `bartowski/Qwen_Qwen3.5-0.8B-GGUF` before training.
- **Data:** 1,454 (statement, program) pairs in English and Catalan:
  - 1,014 pairs from 507 problems of the [M3Kang](https://huggingface.co/datasets/qualcomm/M3Kang) training split.
    The programs were written by a teacher model (`deepseek-v4-flash`) and kept only if running them gives the
    labelled answer as exactly one option.
  - 440 pairs from 220 new low-level (0–3) problems written by the same teacher from a topic and a level only
    (no M3Kang or practice text sent). A problem was kept only if its program runs and prints exactly one option,
    a second independent call picks the same option, the program really computes, and the statement is not close
    to a practice-set or validation statement.
  - No problem from M3Kang `val` or the organizers' practice set is in the training data.
- **Recipe** (`finetune/02_train_ft4_epochs.ipynb`): LoRA r=16, alpha=32, learning rate 2e-4, no dropout, batch 2
  with 8 accumulation steps, 3 epochs, loss on the program only. Every epoch was merged and exported; **epoch 1**
  was chosen by executed answers on M3Kang `val` (score `right - wrong / 4`). The practice set took no part in
  the choice.
- **Export:** adapter merged into the base weights, then `convert_hf_to_gguf.py --outtype q8_0` with llama.cpp at
  commit `2cbe0ec`.

## Limits

- It only helps on problems with a numeric answer and the full statement in text. Geometry that needs the figure,
  and problems whose options are text, are outside what it was trained for.
- A program can run and match an option while solving a different reading of the problem.
- Because epoch 1 was chosen on M3Kang `val`, results on `val` are slightly optimistic.

## License and data

**Non-commercial research use only.** This model was trained on data derived from M3Kang, and under the
[M3Kang Data License Agreement – Research Use](https://huggingface.co/datasets/qualcomm/M3Kang/blob/main/LICENSE.pdf)
it is a *Result* of processing that data: it may be used only for non-profit research, and never for any
commercial use, including any product or service offered to third parties, paid or free.

The base weights are [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) by the Qwen team, under the
[Apache License 2.0](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE). This model modifies them by
fine-tuning.

Please cite M3Kang: Torres-Camps et al., 2026, [arXiv:2601.16218](https://arxiv.org/abs/2601.16218); Qualcomm AI
Research (2025), *M3Kang*; and Societat Catalana de Matemàtiques, *Concurs presencial: Cangur*,
<https://scm.iec.cat/concurs-presencial/cangur/>.

Team: Abdullah Baig, Martí Carrasco and Jordi Rigau.
