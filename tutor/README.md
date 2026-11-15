# Pista · Teaching with M3Kang

Pista serves the full Catalan and English M3Kang bank prepared by this project. It contains 1,747 exercises from train and validation, with 3,494 original problem images. The 1,466 train exercises use the ten topic assignments in `train_topics.csv`. Validation exercises without a supplied assignment are classified when the bank is built.

## Run it

Prepare the bilingual dataset using the [Pista run guide](../docs/tutor/RUN.md). From the repository root, run:

```bash
python3 tutor/build_bank.py --lexical
python3 tutor/server.py
```

Open [Pista](http://127.0.0.1:8000). On Windows, use `py` in place of `python3`. To use embeddings for adaptive practice, omit `--lexical` with the embedding service running on port 8082. The server finds the original images in `data/full_noocr/images` or `data/images`. Use `--images PATH` for a different location.

The tutor calls the local chat completions endpoint on port 8081. On Windows, `py tutor/server.py` starts the model if the port is free, `llama-server.exe` is installed and the base GGUF configured in `config.env` is in `%USERPROFILE%/models`. If it fails to start, see `data/tutor_model.log`. On the UNO Q, start the model using the board's service. Use `--llm URL` to point at another local model or `--no-llm` to run correction and practice without model assistance. For a multimodal llama-server with its projector loaded, start Pista with `--vision --llm http://127.0.0.1:8083/v1/chat/completions` to pass the figure to that server. The model receives the problem text and explicit instructions about the unavailable figure in text mode.

## How practice works

- The local model generates the first step, the next step and a way to check the method. It can also answer questions and discuss the option the pupil just got wrong.
- The official answer key grades responses. The server checks draft hints for answer markers, conflicting numbers and repetition before showing them. A failed generation leaves the same hint available to retry.
- Solving or requesting the answer shows the official key for the exercise.
- Pista chooses exercises using the pupil's level and attempt history by topic. After a difficult attempt it can offer a follow-up on the same topic and records whether the pupil solved it on the first try without hints. Every exercise ID can be served.
- SQLite stores completed hints, attempts and topic progress. Use `--students PATH` to choose the profile database.

Keyboard shortcuts are A–E to answer, H for the next hint, N for the next completed problem, and Escape to close the image viewer.

```bash
python3 tutor/simulate.py
python3 -m unittest tutor.test_teaching
```

The simulation walks the full bank over HTTP and checks image coverage, grading and progression. The teaching tests exercise the model gateway using controlled outputs. See the [Catalan test walkthrough](DEMO.md) for a run with the actual local model.
