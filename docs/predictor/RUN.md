# Running and reproducing the predictor

Run commands from the directory containing `predict.py`, `eval.py`, and `config.env`. PoT inference requires Python 3, a compatible `llama-server`, and the local GGUF. The examples use Debian or a Bash shell.

## Set up the board and select the weights

The original README and `board/` scripts describe the setup. `00_check_board.sh` checks the environment, `01_install_packages.sh` prepares Debian, and `02_build_llama_cpp.sh` builds the version of `llama.cpp` pinned in `config.env`. The board setup uses commit `2cbe0ec429b849a88664e0f62388fc92a5abd699`. Once set up, the model and input are read from local disk. [Deployment map](SOURCES.md#code-and-deployment)

`config.env` pins two language models. `LLM_*` is the **base** Qwen3.5-0.8B, which the tutor uses. `POT_*` is the fine-tuned **`Qwen3.5-0.8B-M3Kang-PoT-v4-e1-Q8_0.gguf`**, published on Hugging Face as [Rigau/Qwen3.5-0.8B-M3Kang-PoT-GGUF](https://huggingface.co/Rigau/Qwen3.5-0.8B-M3Kang-PoT-GGUF) (repository and revision in `POT_REPO` and `POT_REVISION`). `board/03_download_models.sh` downloads and checks both, and `board/04_install_services.sh` makes the `math-llm` service on port 8081 serve the V4 file whenever it is on disk. Check the effective unit’s `-m` argument with `systemctl cat math-llm`. `POT_STYLE=auto` picks the zero-shot prompt from the file `llama-server` has loaded.

Verify the artifact used in the V4 runs with these values.

```text
Size     833592288 bytes
SHA-256  7c90dbbc3518241fd9b1661887abe1131a07d9d9f637c86a1a8a897b7a19cbb3
```

## Start the PoT server

Set `LLAMA_SERVER` to the path of the built binary and `MODEL` to the verified file. For a local checkout, `models/` is an ignored location for model weights. The example uses test port `18081`. On the UNO Q, reserve memory for one language-model instance during the test. If your usual service already runs V4, point `LLM_PORT` to that service.

```bash
LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
MODEL="$HOME/models/Qwen3.5-0.8B-M3Kang-PoT-v4-e1-Q8_0.gguf"
sha256sum "$MODEL"
"$LLAMA_SERVER" -m "$MODEL" --host 127.0.0.1 --port 18081 \
  --threads 4 --ctx-size 4096 --parallel 1 \
  --cache-ram 0 --ctx-checkpoints 2 --jinja \
  --reasoning off --reasoning-budget 0 --alias qwen3.5-0.8b
```

Four threads and a single 4,096-token slot follow the board configuration. The service uses the Jinja template with thinking output disabled. In another terminal, use `/health` to check that the server is responding and `/props` to check `model_path`. The alias is stable across models, so verify the actual path.

```bash
curl -fsS http://127.0.0.1:18081/health
curl -fsS http://127.0.0.1:18081/props
export LLM_PORT=18081 ANSWER_MODE=pot POT_STYLE=finetuned
mkdir -p results
python3 predict.py --input data/sample_problems.jsonl --output results/predict-smoke.csv
```

The CSV should have one row per input and a letter from `A` to `E`. To verify inference, check the status on at least one text-only row with five numeric options. `pot_match` and `pot_fallback` show that the solver took the code-generation route. A CSV containing only fallback answers can still satisfy the output format.

## Prepare the organizers’ practice set

With the practice Parquet available, install `laptop/requirements.txt` on the laptop for preprocessing and prepare the JSONL files. `--val-fraction 0` puts all 100 practice rows in `train.jsonl` **inside that output folder**. The name comes from the export tool. This file is the separate practice evaluation set.

```bash
python3 laptop/prepare_m3kang.py --parquet data/demo-00000-of-00001.parquet \
  --languages eng,cat --val-fraction 0 --out data/datathon-demo-eval
```

Exact reproduction requires the same OCR options used in the evaluation snapshot. Check the input options before interpreting score changes. `predict.py` produces the submission CSV. Use `eval.py` to score an input with `label` fields and save diagnostic data.

```bash
python3 predict.py --input data/datathon-demo-eval/train.jsonl \
  --output results/predict-demo-v4.csv

python3 eval.py --input data/datathon-demo-eval/train.jsonl \
  --languages eng,cat --limit 0 --name pot-v4-demo \
  --results-dir results/repro-demo
```

To compare with the historical M3Kang benchmark, prepare `data/train.jsonl` and `data/val.jsonl` using `data/README.md`. `--limit 0` keeps every row after filtering. `--text-only` also keeps text-only rows without five numeric options, which use the predictor’s fallback.

```bash
python3 eval.py --input data/val.jsonl --languages eng,cat --text-only \
  --limit 0 --name pot-v4-val --results-dir results/repro-val

python3 eval.py --input data/train.jsonl --languages eng,cat --text-only \
  --limit 0 --name pot-v4-train --results-dir results/repro-train
```

## Recorded runtime environments

| Run | Environment and settings |
| --- | --- |
| V4 practice set, 100 rows | Windows, `llama.cpp` b12405, CPU, one 4,096-token slot, four threads, `POT_STYLE=finetuned`, `rag_k=0`. |
| V4 validation and partial `train` run | Windows, `llama.cpp` b12405 with Vulkan on an NVIDIA T600 Laptop GPU, four 4,096-token slots. Four `eval.py` processes, one per split and language, with `rag_k=0`. |
| UNO Q target | ARM Linux, single-slot service configuration. Board times in this documentation are estimates from token counts. |

Changing from CPU to Vulkan can change the answers at the same seed. Retries also depend on wall-clock time, so a slower server may make fewer attempts. Keep the binary, backend, GGUF, flags, cohort, and concurrency fixed when comparing small differences. [llama.cpp b12405 release](https://github.com/ggml-org/llama.cpp/releases/tag/b12405)

## Regenerate the evidence and charts

From this documentation directory, point the script at the checkout containing the original manifests and result files. Matplotlib is required for the charts. With `--no-figures`, the script validates and rebuilds only `assets/evidence.json`.

```bash
python3 -m pip install matplotlib
python3 tools/build_evidence.py --project /path/to/predictor
```

The script verifies counts and scores against the manifests and joins results to the input by `id` and language. This documentation contains aggregated counts. The original JSONL, OCR options, images, indexes, model weights, and row-level predictions remain in the working checkout. [Source inventory](SOURCES.md)
