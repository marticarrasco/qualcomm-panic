# Model and training

## Why Qwen3.5-0.8B

Model size shapes the entire system. Linux on the 4 GB UNO Q has to hold the weights, session state, and Python process. The V4-e1 **Q8_0** GGUF is **833.6 MB** (795.0 MiB). Eight-bit quantization reduces the memory required for weights and lets `llama.cpp` load the model on the board’s CPU. The working memory also includes context, recurrent state, buffers, and the operating system. [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B) · [Board specifications](https://docs.arduino.cc/hardware/uno-q/)

The `qwen35` text backbone combines 18 Gated DeltaNet blocks and six full-attention blocks. The model card gives a hidden dimension of 1,024 and a vocabulary of 248,320 entries. The combination of recurrent and attention blocks affects prompt caching. With the build and flags used in the project, the four-example bilingual prefix fit in 496 tokens, below the 512-token microbatch, and could be reused across consecutive problems. The specialized model uses a zero-shot prompt, which also lowers prefill cost with a cold cache. These observations depend on the `llama.cpp` version, template, and checkpoint settings. [Model card](https://huggingface.co/Qwen/Qwen3.5-0.8B) · [Original cache analysis](SOURCES.md#design-decisions-and-experiments)

## Why generate a program

The limit is **60 seconds of wall-clock time** per answer. At the project’s reference rates of 11 tokens/s for prefill and 5.6 tokens/s for generation, reading 60 tokens takes about 5.5 s and writing 140 takes 25 s. A 300-token response with the same prompt already takes over 59 s before Python and other overhead. In free-form text, an enumeration or a long derivation can run out the clock before giving a letter. Experiments with larger models, option rotations, and retrieved examples motivated shorter generation and separate measurements for each change. [Experiment history](SOURCES.md#design-decisions-and-experiments)

Program-of-Thought delegates calculation to the interpreter. The model still has to understand the problem and represent its constraints correctly. A program can enumerate possibilities, calculate numeric values, and check conditions with many operations per generated token. The comparison with the answer options gives the solver a signal for accepting the result or retrying. [Chen et al., Program of Thoughts](https://arxiv.org/abs/2211.12588)

## Building the PoT corpus

1. **Select from M3Kang `train`.** Start with English text-only problems with all five options available. Join the Catalan translation by `id` when available. The historical recipe started from 684 candidates.
2. **Generate candidate solutions.** `finetune/build_synthetic_dataset.py` sends the available translations, options, and correct value to the teacher model, `deepseek-v4-flash`, through Helmcode. It asks for a short, flat program. This data preparation happens before offline inference. If verification fails, the teacher receives the diagnostic and can try again, up to three attempts in total.
3. **Verify by execution.** The generator runs each program and checks the expected value. An AST heuristic looks for signs of computation to reject cases such as printing the answer verbatim. `build_pairs.py` performs a second verification on the reference statement using the runner in `copilot/pot.py` and requires a unique match with the correct option.
4. **Create training pairs.** The documented historical corpus contains 585 verified programs and 929 statement–code pairs. The teacher has the expected result as supervision. The student receives the statement and learns to generate the program. The V1/V2 notes keep `val` and the practice set outside training and group translations by `id` for the internal train/development split.

The counts refer to the initial recipe’s corpus. The current `build_pairs.py` iterates through all translations in its input that have numeric options. Exact reproduction of the bilingual corpus requires the original training snapshot. [Recipe and scripts](SOURCES.md#design-decisions-and-experiments)

## Fine-tuning and export

LoRA freezes the base weights and learns a low-rank update to selected linear layers. This specializes the checkpoint at much lower training cost than updating all weights. The notebooks first compare the base model’s Hugging Face to GGUF conversion with the reference file, inspect the template and tokens, and run generation checks. [Hu et al., LoRA](https://arxiv.org/abs/2106.09685)

To match the inference service, the recipe tries to render prompts through `llama-server`’s `/apply-template` endpoint, with the Hugging Face template and `enable_thinking=False` as a fallback. The loss covers the code and the end-of-assistant marker. Prompt tokens are masked with `-100`. Selecting the supervised positions before cross-entropy avoids an unnecessary dense copy of the logits. The selected adapter is merged into the base weights before Q8_0 export.

| Version | Documented decisions |
| --- | --- |
| **V1** | Rank 16, `alpha=32`, learning rate `2e-4`, three epochs, and an effective batch size of 16. The adaptation included recurrent projections. Development loss rose from `0.658` to `0.917` between the first and third epochs. |
| **V2** | Keeps the rank, alpha, and batch size. Uses learning rate `1e-4`, two epochs, `dropout=0.05`, and `weight_decay=0.01`. Excludes the recurrent update and decay gates, as well as embeddings, the output head, and the vision branch. Validates every 70 minibatch steps and at epoch boundaries, then restores the adapter with the lowest development loss. The historical run recorded `0.673` at step 217. |
| **V4-e1** | The checkpoint used in the latest tests with `ANSWER_MODE=pot` and `POT_STYLE=finetuned`. Same LoRA (rank 16, `alpha=32`) at learning rate `2e-4`, no dropout, batch 2 with 8 accumulation steps, three epochs, one GGUF per epoch. Epoch 1 was chosen by executed answers on M3Kang `val` (`right - wrong / 4`); the practice set took no part in the choice. Data: 1,454 bilingual pairs, 1,014 from 507 M3Kang `train` problems (`finetune/build_pot_v3.py`: multi-line programs, canonical programs for some families, no overlap with `val` or the practice set) and 440 from 220 new level 0–3 problems written by the teacher from a topic and a level only (`finetune/gen_synth_lowlevel.py`). Recipe: [`finetune/02_train_ft4_epochs.ipynb`](../../finetune/02_train_ft4_epochs.ipynb). Model card: [`finetune/MODEL_CARD.md`](../../finetune/MODEL_CARD.md). |

## Limits that affect accuracy

| Limit | Practical effect |
| --- | --- |
| Understanding the statement | A program can run successfully and compute a result for the wrong interpretation. A matching option is a weak check for this kind of error. |
| Problem type | The PoT route focuses on numeric results. Geometry that requires a diagram, text options, and incomplete OCR reduce coverage. |
| Syntax, length, and computation | An expensive loop, truncated program, or misspelled variable can leave no usable output. Built-in primitives and the timeout bound the cost. |
| Dataset shift | Letter and numeric priors that help on M3Kang can lose their value on newly written problems. The practice set shows behavior on that format for a small collection. |
| Exact reproduction | Seeds fix input order and sampling for a configuration. Differences in backend, template, speed, and remaining time can change outputs and retries. |

[Coverage, accuracy, and timing evidence](EVALUATION.md)
