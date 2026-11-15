# MathQ

Projecte de la datathon UPC–Qualcomm per explorar la resolució de problemes matemàtics amb models de llenguatge a l'Arduino UNO Q.

Aquest repositori parteix del material base del challenge. La idea de treball inicial és desenvolupar i provar el model i el pipeline a l'ordinador, i després integrar-los i validar-los a la placa.

## Novetats d'aquesta branca (`jordi/pot-finetuning-runner`)

**Documentació completa pas a pas:** [**`POT_FINETUNING_I_RUNNER.md`**](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/POT_FINETUNING_I_RUNNER.md)

En aquesta branca documentem i integrem el pipeline complet que ens ha permès passar del **20,0% d'encert inicial** (nivell d'atzar al set oficial de pràctica 2026 `demo`, 100 problemes bilingües `eng` + `cat`) fins al **41,0% en una sola passada (`Pass@1`, 11 s de mitjana a la placa)** i **42,0% amb reintent condicional (`Pass@3`, ~18,6 s de mitjana a la placa)**:

1. **Generació i verificació de dades sintètiques amb l'API de Helmcode (`deepseek-v4-flash`)**:
   - Scripts: [`finetune/build_synthetic_dataset.py`](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/finetune/build_synthetic_dataset.py), [`finetune/prepare_sft_and_rag.py`](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/finetune/prepare_sft_and_rag.py) i [`finetune/build_pairs.py`](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/finetune/build_pairs.py).
   - Generació de programes Python plans (2–6 línies) per als problemes `text_only` d'entrenament, verificats executant-los dins del nostre sandbox AST (amb filtre anti-trampa `is_nontrivial_code` i comprovació de coincidència única amb l'opció correcta).
   - Resultat: **585 problemes verificats** i **929 parells bilingües `(enunciat, codi Python)`** en anglès i català (`pot_pairs.jsonl`).

2. **Fine-Tuning LoRA de `Qwen3.5-0.8B` (Versió 1 i Versió 2) i conversió GGUF `Q8_0`**:
   - Notebooks reproduïbles a Kaggle T4: [`finetune/00_gate_conversion.ipynb`](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/finetune/00_gate_conversion.ipynb) i [`finetune/01_train_pot_lora.ipynb`](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/finetune/01_train_pot_lora.ipynb) (generats per [`finetune/build_notebooks.py`](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/finetune/build_notebooks.py)).
   - **Versió 1 (`v1`)**: 3 èpoques, `LR=2e-4`, màscara de `loss` només sobre el codi usant `/apply-template` de `llama-server` (`--jinja --reasoning off`). Puja a **34,0% (`Pass@1`) / 36,0% (`Pass@3`)** a `demo`, multiplicant $\times 4$ els encerts en català i baixant el temps mitjà a la placa a **12 s** (0-shot).
   - **Versió 2 (`v2`)**: `LoRALinear` en PyTorch pur amb `Dropout(0.05)`, exclusió de les portes recurrents `in_proj_a/b/ba` de *Gated DeltaNet* per protegir la memòria d'estat recurrent, `LR=1e-4` i *Early Stopping* sub-època cada 70 passos (`step 217`, `dev loss = 0.673`). Puja a **37,0% (`Pass@1`) / 39,0% (`Pass@3`)** amb **70,4% de precisió** quan el codi casa amb una opció.

3. **Millores al `SmartRunner` i filtre anti-`'1'` ([`copilot/pot.py`](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/copilot/pot.py) i [`copilot/llm.py`](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/copilot/llm.py))**:
   - `_PermTuple(tuple)` i `smart_int` per permetre comparar tuples de dígits de `permutations(...)` directament amb enters (`1000 <= n <= 9999`, `+1,0%`).
   - `_SmartStr(str)` per suportar `7 in str(n)` amb enters i resta de strings numèrics (`+1,0%`).
   - `smart_product` / `_SmartIntProd` per suportar `product(...)` tant com a `math.prod` (multiplicació numèrica) com `itertools.product` (`+1,0%`).
   - Captura de `printed[0]` i `printed[-1]` quan un bucle `for` troba el mínim però oblida posar `break` (`+1,0%`).
   - Filtre de consens anti-`'1'` en els reintents amb temperatura (`Pass@3`).
   - **Resultat final a `demo` (100 problemes)**: **41,0% en `Pass@1` (11 s de mitjana a la placa)** i **42,0% en `Pass@3` (~18,6 s de mitjana a la placa)**.

---

## Progress (branch `abdullah/reasoning-experiments`)

Updated 14 Nov, evening. Work on this branch is not merged into `main` yet: merging changes how `solve()` answers (see below), so the team should agree first.

**Reasoning (`predict.py`)**

- The 0.8B model is at chance when asked for a letter, with or without bias correction, with a 2B model, or with the picture. Full measurements: [EXPERIMENTS.md](datathon-UPC-maths-qualcomm-arduino-unoq-main/datathon-UPC-maths-qualcomm-arduino-unoq-main/EXPERIMENTS.md).
- New answer mode, `ANSWER_MODE=budget` in `config.env` (`copilot/reasoner.py`, `copilot/priors.py`):
  1. no options, or five numeric options printed out of order: a model-free prior (median value, else D), no model call;
  2. otherwise terse reasoning, streamed and stopped at a 57 s wall-clock deadline;
  3. the prior again if the reasoning does not reach `Answer: X`.
- Full validation file (275 problems): 80 right (29.1%), against 57 (20.7%) for always-D. On problems with options: 33%. On the board, problems took 10 to 57 s and none went over 60 s.
- Set `ANSWER_MODE=direct` to get the previous behaviour back.

**Teaching (in progress)**

- `tutor/`: a tutor that never relies on the model for the truth. It shows the problem image, takes A to E, adapts the level, and gives hints that are correct by construction (method tip, a similar easier solved problem, eliminating two wrong options, the answer only on request).
- `applab/math-matrix/`: App Lab app that shows the tutor's feedback on the LED matrix (tick, cross, hint, level).

**On the board (`Gary`)**

- The starter repository is untouched in `~/datathon-UPC-maths-qualcomm-arduino-unoq`. This branch's pipeline is in `~/qp`.

**Open questions for the team**

- Merge `ANSWER_MODE=pot` / `ANSWER_MODE=budget` into `main`?
- M3Kang data is committed in this repository. The licence is research-only, so the history must be cleaned before the repository goes public.
