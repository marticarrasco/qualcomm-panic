# Evaluation

The tutor was evaluated on a Windows laptop with the full lexical bank, controlled model responses, and a local quantised Qwen 0.8B model through llama.cpp. The full-bank run exercises the grading and practice contract. Controlled tests exercise the teaching gateway. Small live probes show how actual coaching behaves.

## Full-bank and controlled tests

| Check | Observed result | Scope |
| --- | --- | --- |
| Practice coverage | **1,747/1,747** unique problem IDs served over HTTP without repeats | Walk through grading, moving on, failed attempts and explicit reveal |
| Premature answer-key fields | **0 in 7,863** HTTP replies checked | Structured response fields in a walk with the model disabled, with labels verified where expected |
| Bilingual images | **3,494/3,494** present on disk and **60/60** sampled image requests returned image data | Both languages checked at each sampled ID |
| Adaptive selection | **12/12** threshold profiles reached the expected mastery level and settled in **one to three** in-topic exercises | Four topics, three target levels, 30 exercises per profile. Settled means subsequent in-topic exercises remained within one level of the target. |
| Other progression cases | Zero repeated IDs across the measured simulated profiles. Mixed practice covered **eight topics in 30 exercises**. | The simulation also included four noisy profiles and one that used two hints on every exercise |
| Coaching filter fixtures | **8/8** deliberately unsuitable replies rejected and **2/2** coaching replies accepted | Fixed examples cover answer phrasing, answer values and unsupported numbers |
| Teaching gateway | **7/7** tests passed | Bilingual hint progression and persistence, wrong-choice feedback, follow-up scoring, rejected-draft retry, closed and stale requests, offline response and chat-help accounting |
| Hardware interface | The HTTP event feed returned a valid event, sequence and level | Verification covered the feed. Physical display behaviour requires the UNO Q rehearsal. |
| Fresh data preparation | **1,747** bilingual IDs and **3,494** images recreated from the pinned source shards, with a passing **150-exercise** API check | Rebuilt with lexical similarity and no OCR |

The mean time per simulated step ranged from **0.24 to 0.75 ms** across profiles on the laptop. The timer surrounds local tutor logic for answering and advancing. It excludes browser rendering, HTTP and live model inference. [Reproduction commands](RUN.md#reproduce-the-checks)

## Live model checks

An exploratory first-hint probe used ten problem IDs, one per topic, and submitted each in both languages to the same local text endpoint. Five used a diagram and five were marked text-only. The gateway returned an accepted first hint for **20/20 requests**, including optional retries. The median round-trip time was **2.1 s** on the laptop, with a range of **0.8–9.9 s**. Catalan returned **10/10** with a **3.9 s** median and English **10/10** with a **1.6 s** median. Catalan ran first on each pair, and the sample is small. These timings describe this run and include polling, with no inference about the underlying language cost or UNO Q latency.

| Topic | Problem ID | Level | Exercise format |
| --- | --- | ---: | --- |
| Fractions & percentages | `lvl-4_2008_15` | 4 | Text-only |
| Counting & probability | `lvl-3_2016_15` | 3 | Diagram |
| Number properties | `lvl-7_2016_11` | 7 | Text-only |
| Geometry, area, perimeter and angles | `lvl-2_2017_28` | 2 | Diagram |
| Shapes, solids & spatial puzzles | `lvl-2_2019_24` | 2 | Text-only |
| Logic & strategy | `lvl-5_2015_21` | 5 | Diagram |
| Time, money & measures | `lvl-0_2016_15` | 0 | Text-only |
| Patterns & sequences | `lvl-3_2022_16` | 3 | Diagram |
| Algebra & functions | `lvl-7_2017_28` | 7 | Text-only |
| Arithmetic & word problems | `lvl-1_2022_10` | 1 | Diagram |

In two additional scripted counting sessions, one per language, the live endpoint returned **5 of 6** requested coaching turns. The Catalan diagram exercise's wrong-choice feedback was rejected and the attempt remained playable. Its accepted chat reply mostly rephrased the exercise. On the English text-only exercise, the accepted wrong-choice feedback repeated the opening strategy without testing the selected option. Both scripted sessions produced a follow-up and correctly recorded its first-try solve in the teacher totals. The quality observations are examples from the two sessions. The acceptance count measures passage through the gateway, and the follow-up outcome records a scripted action.

A separate API rehearsal verified the full demo sequence with a live first hint, a wrong choice, a language switch preserving the same exercise and try count, and a follow-up recorded as an independent solve in the teacher report.

## What the results point to

| Dimension | Observed constraint | Implication |
| --- | --- | --- |
| Topic coverage | Geometry, area, perimeter and angles has **410** problems and arithmetic has **305**. Patterns has **33** and algebra has **47**, with all algebra problems at levels 5–7. | Tailor the demo to a populated topic. Expand underrepresented topic and level combinations to improve practice continuity and follow-up coverage. |
| Difficulty | The threshold simulation reached its target in all 12 cases. With noisy responses, two of four mastery estimates finished one level above the simulated threshold. | The level rule reacts to performance quickly. A real pupil's mastery estimate needs repeated observations. |
| Diagram dependence | **936/1,747** exercises are marked as requiring a figure. The ordinary text endpoint does not receive it. | The pupil sees the original figure throughout. Assess diagram coaching separately before relying on it, including whether the optional visual endpoint improves grounding. |
| Option context | The evaluated bank had OCR options for **1,322/1,747** IDs per language. | The problem image and answer key still support full practice. Missing options limit the model's wrong-choice context and the check for the correct value or wording. |
| Bilingual quality | Both languages completed the paired first-hint probe. The separate Catalan diagram turn exposed a rephrasing failure. | Review the Catalan and English content for correctness, specificity and natural phrasing, with the same problems paired across languages. |

## Next experiments

1. **Have teachers score coaching.** Sample across language, topic, difficulty and exercise format. Ask reviewers to rate mathematical validity, usefulness of the proposed next step and whether the reply reveals too much. Include the first hint, later hints, wrong-choice feedback and free-form chat.
2. **Check the follow-up matches and measure learning.** Have teachers review whether each pair has a solution method in common. Compare first-try, no-help performance on immediate and delayed unseen exercises after static and adaptive help. Record the difficulty and earlier support to interpret each outcome.
3. **Characterise the UNO Q experience.** Measure wait times by coaching mode in cold and warm runs, and measure acceptance, retry and failure rates. Compare text and visual coaching on the same diagram problems. Verify the physical matrix on each event and preserve visible results when level updates follow immediately.
4. **Prepare for shared classroom use.** Add role-based access to profiles and teacher reports, then measure concurrent coaching demand on the board. Add teacher-vetted exercises for sparse topic and level combinations.
