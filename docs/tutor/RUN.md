# Run Pista

Run the commands from the project directory containing `tutor/`, `laptop/` and `config.env`. Use Python 3.10 or later. The tutor itself uses the standard library. Data preparation runs on a laptop. The resulting bank and images can run on the laptop or the Arduino UNO Q.

## Prepare the bilingual bank

If you have prepared bilingual data, place `train.jsonl` and `val.jsonl` in `data/full_noocr/` or `data/`, with an `images/` subdirectory alongside them. Images are expected under the name `<id>_<language>.png`. The server determines the image type from its bytes.

To prepare the data from M3Kang, use the four `high_quality` shards. This command installs the dependencies needed for the OCR-free path and prepares Catalan and English statements and original images.

```bash
python3 -m pip install huggingface_hub pyarrow
python3 -X utf8 laptop/prepare_m3kang.py --split high_quality --shards all --languages eng,cat --no-ocr --save-images --out data/full_noocr
python3 tutor/build_bank.py --lexical
```

On Windows, substitute `py` for `python3` and keep `-X utf8` on the preparation command. To repeat the submission data build from a fixed source, use the shard names and revision in `data/m3kang_expansion_manifest.json` and pass the downloaded files with `--parquet` in place of `--shards all`. The tutor uses the union of the two local splits. Topic assignment is keyed by problem ID. [M3Kang dataset and research-use terms](https://huggingface.co/datasets/qualcomm/M3Kang#dataset-license)

The builder writes `data/tutor.db` and reports the bilingual count, topic distribution, neighbours and OCR coverage. Expect 1,747 problem IDs and 1,466 supplied topic assignments for the full submission bank. By default the builder also looks for optional OCR options in `data/train.jsonl` and `data/val.jsonl`. Use `--ocr` to select a different folder. For local embeddings, omit `--lexical` with the embedding service listening on port 8082. Similarity is computed at build time in both modes.

## Run locally

Start the local coaching service on port 8081. For a manual launch, use the GGUF in `config.env` as the model path. Run the tutor in a second terminal.

```bash
llama-server -m /path/to/coaching-model.gguf --host 127.0.0.1 --port 8081 -c 4096 -t 4 -np 1 --jinja
```

```bash
python3 tutor/server.py
```

Open [Pista on localhost](http://127.0.0.1:8000). Startup checks the bank and both sets of images. On Windows the manual model launch is optional. With default arguments Pista reuses a listener on port 8081 or attempts to start an installed llama-server using the configured GGUF. Its startup log is `data/tutor_model.log`.

| Option | Use |
| --- | --- |
| `--no-llm` | Run the bilingual bank, grading, adaptive practice and teacher view without model help |
| `--llm URL` | Select a compatible local chat-completions endpoint |
| `--vision --llm URL` | Send the original exercise image for diagram problems to a local multimodal endpoint with its projector loaded |
| `--students PATH` | Use a separate SQLite profile database for a rehearsal or evaluation |
| `--bank PATH` and `--images PATH` | Open a custom bank and its corresponding images |

```bash
python3 tutor/server.py --no-llm --students data/pista-rehearsal.db
```

Profiles are named records. The teacher view can list every profile on the instance. Run the demo on localhost, over USB, or on a trusted local network with demo profiles. Keep the name-based profile database and any local logs private.

## Run on the Arduino UNO Q

1. Set up the 4 GB UNO Q with Arduino App Lab and connect it over USB. Follow the organisers' board setup through the model services, using the configured local model. Verify that `adb devices` lists the board and `math-llm` responds on board port 8081. [Organisers' setup guide](https://github.com/mpous/datathon-UPC-maths-qualcomm-arduino-unoq)
2. Prepare the bank and images as above. The tutor needs `tutor/`, `copilot/config.py` and its Python package, `config.env`, `data/tutor.db` and the matching images. In the full datathon checkout with `data/train.jsonl` and `data/val.jsonl`, the project script copies the tutor and runs the sample solver check. It also installs and starts the `math-matrix` App Lab app and forwards port 8000 over USB. For a standalone tutor checkout, use the manual deployment below.

   ```bash
   bash laptop/deploy_to_board.sh
   ```

3. Open [Pista on localhost](http://127.0.0.1:8000) on the laptop. To verify the offline demo after provisioning, disconnect Wi-Fi and keep USB attached. Try a wrong choice or request a hint with the board in view.

For a tutor-only deployment, transfer the listed files while preserving their directory layout. Copy `applab/math-matrix/` to the board's Arduino Apps folder and start it from App Lab. On the board, start Pista from the deployed project directory with `python3 tutor/server.py --host 0.0.0.0` and point it at the image directory with `--images` if needed. On the laptop, forward the port with `adb forward tcp:8000 tcp:8000`. Check `/api/info` for the bank and `/api/matrix` for the shared event feed. The App Lab app polls the tutor over the board's network interface.

## Reproduce the checks

```bash
python3 -m unittest -v tutor.test_teaching
python3 tutor/simulate.py
```

`simulate.py --quick` walks 150 problems over HTTP for a short check. The full run reaches all IDs in the prepared bank. Both use temporary profile databases. See [Evaluation](EVALUATION.md) for what each check measures.

## Source map

| Project path | Responsibility |
| --- | --- |
| `tutor/index.html` and `tutor/static/` | Browser practice, teacher view and local assets |
| `tutor/server.py` | HTTP routes, image loading and connection to board events |
| `tutor/tutor.py` | Grading, practice state, follow-up selection and reports |
| `tutor/chat.py` and `tutor/local_model.py` | Model prompts, filtering, worker lifecycle and Windows startup |
| `tutor/build_bank.py`, `tutor/topics.py` and `tutor/train_topics.csv` | Bank preparation, topic assignment and similarity |
| `tutor/matrix.py` and `applab/math-matrix/` | Practice events, App Lab rendering and MCU display |
| `tutor/simulate.py` and `tutor/test_teaching.py` | Full-bank simulation and controlled coaching tests |

For the public repository, keep the documented preparation route and use original examples in public screenshots. Keep research dataset exports, source images, generated banks containing source material and pupil databases local, subject to the dataset terms. [M3Kang dataset terms](https://huggingface.co/datasets/qualcomm/M3Kang#dataset-license)
