# Sources and repository map

Code and result paths are relative to the repository root. The [organizers’ practice-set specification](../benchmark/PRACTICE_SET.md) is preserved in the documentation. The charts and [aggregate JSON](assets/evidence.json) accompany this documentation. The original problems, model files, and detailed run outputs are prepared or saved locally. [M3Kang dataset card and license](https://huggingface.co/datasets/qualcomm/M3Kang)

## Code and deployment

| File | Role |
| --- | --- |
| [`predict.py`](../../predict.py) · [`eval.py`](../../eval.py) | Submission contract and evaluator. `eval.py` uses the same `batch_solver`, adds the answer key, and saves a CSV and manifest. |
| `copilot/config.py` · `config.env` | Shared deployment configuration. Recognized environment variables take precedence. |
| `copilot/pipeline.py` | Solver routing, time budget, and final result. |
| `copilot/llm.py` · `copilot/client.py` | Active prompts, local requests, retry gating, code extraction, and option matching. |
| `copilot/pot.py` | Zero-shot prompt, SmartRunner, and training-data checks. |
| `copilot/text_solver.py` · `copilot/reasoner.py` · `copilot/stream.py` | Text-generation variants and streaming. |
| `copilot/rules.py` · `copilot/arithmetic.py` · `copilot/option_ranker.py` · `copilot/priors.py` | Rules, exact arithmetic, distractor features, and fallback policies for the text routes. |
| `copilot/train_match.py` · `copilot/translations.py` | Lexical matching and reuse of a translation already in the batch. |
| `build_index.py` · `copilot/rag.py` · `copilot/embed.py` | Optional embedding-based indexing and retrieval. |
| `board/02_build_llama_cpp.sh` · `board/04_install_services.sh` · `board/systemd/math-llm.service` | Build, installation, and service settings for the UNO Q. |
| `board/05_smoke_test.sh` | Integrated service and predictor contract check. Its embedding service belongs to the full stack. |

## Data and protocol

| Local source | What it covers |
| --- | --- |
| [Practice-set specification](../benchmark/PRACTICE_SET.md) | Final test protocol, 20 new problems, and 60-second limit. Practice metadata, 50 problem IDs and 100 bilingual rows. |
| [Project overview](../../README.md) · [Board setup](../setup/BOARD.md) | Challenge brief, hardware references, and step-by-step setup. |
| `data/README.md` · `laptop/prepare_m3kang.py` · `laptop/OCR_REVIEW.md` | Data provenance, OCR, review, and dataset splitting. |
| `laptop/expand_m3kang_languages.py` · `laptop/create_val_cohorts.py` | Multilingual expansion and cohort creation with explicit filters. |
| [`analysis/PROBLEM_CATEGORIES.md`](../../analysis/PROBLEM_CATEGORIES.md) · `laptop/classify_problems.py` | Taxonomy and thresholds for the exploratory topic analysis. The reviewed labels apply to historical `train`. The practice analysis uses the automated thresholds alone. |

## Design decisions and experiments

| Local source | What it covers |
| --- | --- |
| [PoT fine-tuning and SmartRunner](../experiments/POT_FINETUNING_I_RUNNER.md) | PoT evolution, synthetic generation, V1/V2 settings, and SmartRunner ablations on the practice set. |
| `finetune/build_synthetic_dataset.py` · `finetune/build_pairs.py` | Offline preparation for inference, two-stage verification, and the construction of training pairs. |
| `finetune/00_gate_conversion.ipynb` · `finetune/01_train_pot_lora.ipynb` · `finetune/build_notebooks.py` | Reference conversion and the fine-tuning recipe with loss computed on output tokens. |
| `finetune/build_pot_v3.py` · `finetune/gen_synth_lowlevel.py` · `finetune/02_train_ft4_epochs.ipynb` · `finetune/MODEL_CARD.md` | V4 training data (v3 pairs plus synthetic level 0–3 problems), the per-epoch training and selection notebook, and the published model's card. |
| [Practice-set findings](../experiments/FINAL_TEST_FINDINGS.md) | Context for the new test, dataset shift, and historical solver results. |
| [Experiments](../experiments/EXPERIMENTS.md) · [Text-only routes](../experiments/TEXT_ONLY.md) · [Model-only benchmark](../experiments/EVAL_MODEL_ONLY.md) | Model experiments, RAG, text alternatives, figure experiments, and cohort methodology. |
| [Evaluation options](../experiments/EVALUACIO.md) · [Results viewer](../../results-viewer/README.md) | Result fields, filters, and run inspection. |

## Runs and evidence

| Result | Manifest or local source |
| --- | --- |
| V4 practice | `results/pot-v4-e1-demo/eval-20261115-105958-184892-pot-v4-e1-smart-runner-demo.json` |
| V4 practice, second run with other OCR options (41/100) | `eval-20261115-110654-819345-demo-pot-v4.json`, in the `results/` folder of the `abdullah/final` branch (before the reorganization) |
| V1 practice | `results/eval-20261115-094216-937240-pot-v1-smart-runner-demo.json` |
| Earlier hybrid practice result | `results/datathon-qualcomm-demo/eval-20261115-011634-910547-all-rows-100.json` |
| Base PoT and V2 practice | [Historical ablation table](../experiments/POT_FINETUNING_I_RUNNER.md) and [practice-set findings](../experiments/FINAL_TEST_FINDINGS.md). |
| V4 Vulkan on `val` and `train` | `results/pot-v4-e1-train-val-text-only-vulkan/summary.json` and the four manifests it lists. |
| English text-only cohort | [Model-only benchmark](../experiments/EVAL_MODEL_ONLY.md) and the `eng-text-only-strict`, `eng-text-only-hybrid-default`, and `eng-text-only-ranker-only` manifests under `results/`. |
| Multilingual hybrid route | [Aggregate summary](../../results/text-only/routed-batch-full-summary.json) and local manifest `results/text-only/eval-20261114-213953-266528-text-routed-batch-full.json`. |

Each `eval.py` manifest points to its results CSV through `details_csv`. To verify denominators and configurations, this documentation uses counts recomputed from the CSV by problem ID and language where available. It identifies historical scores that are available only in project notes. [Evidence generator](tools/build_evidence.py)

## Primary references

- Qwen Team. Base model card and Qwen3.5 family configuration. [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- Arduino. UNO Q hardware and processor specifications. [Arduino UNO Q](https://docs.arduino.cc/hardware/uno-q/)
- ggml-org. Inference engine, model format, and local server. [llama.cpp](https://github.com/ggml-org/llama.cpp) · [Build b12405](https://github.com/ggml-org/llama.cpp/releases/tag/b12405)
- Chen et al. Separating problem formulation from executable computation. [Program of Thoughts Prompting](https://arxiv.org/abs/2211.12588)
- Hu et al. Model adaptation with low-rank updates. [LoRA](https://arxiv.org/abs/2106.09685)
- Torres-Camps et al. Multilingual benchmark of Kangaroo math problems. [M3Kang paper](https://arxiv.org/abs/2601.16218) · [Dataset card](https://huggingface.co/datasets/qualcomm/M3Kang)
