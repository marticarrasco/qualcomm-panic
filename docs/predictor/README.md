# MathQ Predictor

[MathQ project](../../README.md) · [Documentation index](../README.md)

MathQ solves multiple-choice math problems on the Arduino UNO Q using a **0.8-billion-parameter model**. The model turns the problem statement into a short program, Python computes the result, and `predict.py` maps it to a letter from `A` to `E`. Once the board is set up and the model is copied over, the entire prediction pipeline runs locally.

## Main result

**V4-e1 with Program-of-Thought and SmartRunner** answered **43 of 100 rows** correctly on the Datathon FME 2026 organizers’ practice set. The set contains **50 original problems**, each in Catalan and English. Accuracy was **44% in Catalan** and **42% in English**. V1 scored **36/100** on the same rows, and always answering `D` scored **20/100**.

| Measure | V4 result |
| --- | ---: |
| Full Datathon practice set | **43/100 · 43.0%** |
| Text-only rows | **33/80 · 41.3%** |
| Accuracy when the program identifies an option | **27/40 · 67.5%** |
| Mean time on the laptop | **4.52 s** |
| Estimated mean time on the UNO Q | **18.59 s** |

These scores are from the practice set. The organizers’ final test uses 20 different problems, one language per problem, and a 60-second limit per answer. [Methods, breakdowns, and sources](EVALUATION.md)

## Three design decisions

| Decision | Implementation and rationale |
| --- | --- |
| Move the calculation outside the model | The prompt asks for a short program from the problem statement. Python performs the operations and the predictor checks the result against the options. This reduces how much the board has to generate. |
| Specialize a model that fits on the board | Qwen3.5-0.8B is the starting point. The documented LoRA recipe adapts it to generate code using verified bilingual examples. The tested V4-e1 Q8_0 GGUF is 833,592,288 bytes. |
| Bound the cost per question | Each row has a 60-second budget. The first attempt uses temperature zero, and retries run when enough time remains. A fallback lets the predictor emit a letter when the main route ends without a match. |

## Documentation

| Guide | Contents |
| --- | --- |
| [Architecture](ARCHITECTURE.md) | The `predict.py` contract, the path through the solver, SmartRunner, and alternative approaches. |
| [Model and training](MODEL.md) | Qwen3.5, verified synthetic data, LoRA, conversion, V4-e1, and model limitations. |
| [Evaluation](EVALUATION.md) | Datathon practice results, comparisons, charts, M3Kang validation, and timing. |
| [Running and reproducing the predictor](RUN.md) | Model startup, commands, board settings, a smoke test, and reproducible evidence. |
| [Sources and repository map](SOURCES.md) | Code, internal notes, result files, and primary references. |
