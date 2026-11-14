# Experiments and findings

What we measured, what worked, and what did not. Every number here comes from a script in `experiments/`; raw outputs go to `results_local/` (not committed).

Unless stated otherwise: Qwen3.5 0.8B Q8_0, temperature 0, seed 42, English problems, run on a laptop for accuracy. Latency is only quoted when measured on the board. On small samples a difference of a few problems is noise; we give counts so you can judge.

## Summary

| Question | Answer |
| --- | --- |
| Can the 0.8B model pick the right letter directly? | No. Chance level, with a strong letter bias. |
| Does cancelling the letter bias help? | No. Bias goes away, accuracy stays at chance. |
| Does a bigger model (2B) help? | Barely (31%), at a third of the speed. Rejected. |
| Does letting it reason help? | A little, and only if it finishes. It rarely finishes in 60 s unless the reasoning is terse. |
| Does the vision model help on picture problems? | No. Worse than always answering D. |
| Is there signal that needs no model? | Yes. The correct value is rarely the smallest or the largest option. |
| What do we ship? | Prior where it is strong, budgeted reasoning elsewhere, prior as fallback: 34% on problems with options, against 22% for always-D. |

## The board

Measured on our UNO Q (`Gary`):

| What | Qwen3.5 0.8B Q8_0 | Qwen3.5 2B Q4_K_M |
| --- | --- | --- |
| Reads | 11 tokens/s | 3.3 to 4.6 tokens/s |
| Writes | 5.6 tokens/s | 2.5 tokens/s |
| One letter for a 142-token problem | 14.5 s | 30.5 s |

The server reuses the start of the prompt between requests: with a 181-token prompt, the second request read 4 tokens and a different question with the same system prompt read 19. So a long system prompt costs its reading time once per run, not once per problem. We use that to hold worked examples.

## 1. One letter, forced by a grammar (the starter baseline)

`experiments/run_variants.py`, 100 text-only problems (levels 0 to 2), 75 with options.

| Variant | Right | Notes |
| --- | --- | --- |
| Always D | 27 | |
| One letter (starter kit) | 24 | answers D 69 times |
| "Two short lines, then Answer: X", 128 tokens | 28 | 74 answers never reach `Answer: X` and fall back to D |
| Reason 128 tokens, then force a letter | 24 | the forced letter is A 71 times |
| Reason 280 tokens, then force a letter | 28 | would take about 79 s on the board |

The model ignores "two short lines", writes a long solution, and is cut off. Forcing a letter after cut-off reasoning does not use the reasoning.

## 2. Letter bias: log-probabilities and option rotations

`experiments/letter_scores.py`. We read the log-probability of each letter, for the original option order and for the four rotations of the options, so every option visits every letter position.

Text-only problems with options, validation plus a train sample (222 problems):

| Method | 0.8B | 2B Q4_K_M |
| --- | --- | --- |
| Raw (original order) | 52 = 23% | 47 = 21% |
| Subtract each letter's average log-probability | 44 = 20% | 49 = 22% |
| Sum over 5 rotations | 58 = 26% | 69 = 31% |

After five rotations the picks are spread evenly over the letters, so the bias is gone, and the accuracy is still chance for 0.8B. The bias was hiding the fact that there is no signal underneath. Five rotations would not fit in 60 s on the board in any case.

## 3. Reasoning inside the time limit

`experiments/reasoning_budget.py`. Each problem gets as many output tokens as the board could write in the time left after reading it. A system prompt with three made-up worked examples asks for calculations only.

147 English validation problems with options, all levels:

| | Count |
| --- | --- |
| Reached `Answer: X` in budget | 107 of 147 |
| Right when finished | about 27 to 29% |
| Unfinished, last number matched to an option ("salvage") | worse than the prior; dropped |

Finishing more often did not make the answers right: an earlier 63% (12 of 19) came from short, easy problems that finished on their own. Across levels the model's finished answers are only a little better than a guess.

## 4. A prior that needs no model

`experiments/priors_analysis.py`, fitted on train, checked with 5-fold cross-validation grouped by problem and on validation.

- D is the most common key (25.0% of 1,466 train problems). No conditional prior (by level, year, question number, text-only) beats it in cross-validation.
- When the five options are numbers, the key is rarely an extreme value: smallest 8.5%, second 22.0%, median 30.6%, fourth 27.4%, largest 11.6% (614 train problems).
- If the options are printed in order, that adds nothing to the letter prior. If they are printed out of order, the median value is right 36.6% of the time (238 train problems), against 18.1% for D. On validation: 37.8% against 13.3% (45 problems).

`copilot/priors.py` implements exactly this: the median value when five distinct numeric options are out of order, otherwise D. No fitted constants. On all validation problems: 28.0% against 23.5% for always-D (243 problems).

## 5. What we ship: `ANSWER_MODE=budget`

`copilot/reasoner.py`, called from `copilot/pipeline.py`:

1. No options, or five numeric options out of order: answer the prior. No model call, 0 s.
2. Otherwise: terse reasoning, streamed, stopped at a wall-clock deadline of 57 s. If it reaches `Answer: X`, use it.
3. Otherwise: the prior.

On the 147 English validation problems with options:

| Policy | Right |
| --- | --- |
| Always D | 32 = 22% |
| Prior only | 43 = 29% |
| Model, else D | 45 = 31% |
| **Prior where the rule fires, else model, else D (shipped)** | **50 = 34%** |

The order of the steps was chosen on these same problems, so expect a little less on new data. Where the rule fires the prior beats the model (15 against 8 right of 36); elsewhere the model beats D (24 against 17 right of 84 finished).

On the board, with an earlier version of step 2 only, problems took 10 to 57 s and none went over the limit; the server cancels the generation when we close the connection at the deadline.

## 6. Pictures

54% of M3Kang problems need the picture (936 of 1,747), about 80% at levels 0 and 1. Each image is the whole problem as a wide strip: question, options, and a small figure.

`experiments/vision_eval.py`, 150 image-dependent validation problems, Qwen3.5 0.8B with its projector, `--image-max-tokens 128`:

| | Right |
| --- | --- |
| Always D | 38 = 25.3% |
| Text only, no picture | 34 = 22.7% |
| Whole image to the vision model | 23 = 15.3% (answers A 99 times) |

The model does read the problem text out of the image, but misdescribes the figures. It also costs about 30 s per problem on the board. We do not use vision in `predict.py`. `experiments/figure_crop.py` and `experiments/vision_figure_eval.py` test cropped figures (in progress).

For teaching, the picture problem solves itself: the tutor shows the image to the student, so nothing has to read it.

## 7. Retrieval

About 5% of validation problems have an almost identical problem in train at another level of the same year. The options are in a different order there, so copying the neighbour's letter is right only about half the time. Copying the answer value was right 5 of 5 times, but applies to only 5 of 243 validation problems. Not built into the pipeline: for a test set of unseen problems it would never fire.

## What did not work

- One-letter answers, with or without bias correction.
- Asking for "two short lines": the model ignores the length limit.
- Forcing a letter after cut-off reasoning (collapses to A).
- Matching the last number of unfinished reasoning to an option.
- A 2B model: three times slower for a few points on one-letter answers.
- The vision model on whole problem images.
- Conditional letter priors by level, year or question number.
