# Pista

**An offline maths tutor that asks pupils to try again on a new problem.**

Pista turns the Arduino UNO Q into a bilingual practice station for Kangaroo maths. A pupil chooses a topic and starting level, works from the original exercise image, and can ask for a hint or explain where they got stuck. After a supported or unsuccessful attempt, Pista can select a related exercise. The teacher view records whether the pupil solved that follow-up on the first try without help.

Built for the **Datathon FME · Qualcomm Math on the Edge challenge**. The browser, the board's Linux processor and its microcontroller each have a role. The board serves the exercises, stores progress and runs the coaching model. Arduino App Lab connects the practice session to the built-in LED matrix. A USB connection is enough for the demo once the local assets and model are installed. [Challenge brief](https://github.com/mpous/datathon-UPC-maths-qualcomm-arduino-unoq) · [Arduino UNO Q](https://docs.arduino.cc/hardware/uno-q/)

## What the jury can try

1. **Work in Catalan or English.** Choose one of ten topics or start mixed practice. Switch languages mid-attempt and keep the same exercise, tried options and progress. The bank has 1,747 distinct problems across eight levels and 3,494 original exercise images.
2. **Ask for a next step.** Request an opening move, a way to apply the method, or a check. After a wrong choice, Pista can suggest a condition to test against that choice. The chat also accepts a specific question from the pupil.
3. **Check the answer and try a related problem.** The official M3Kang key grades every choice. Accepted hints, wrong tries and the final outcome inform what comes next. A related exercise can test whether the pupil can apply the method independently.
4. **See what happened.** The teacher view shows topic-level progress, how much help pupils used and results on follow-up attempts. The browser and the UNO Q matrix share practice events for visible feedback.

## Design decisions

| Decision | How Pista implements it | What it buys |
| --- | --- | --- |
| Use the small model for coaching | Prompts are specific to the hint rung, latest mistake or pupil question. The official key stays in the tutor's grading and checking logic. | A model reply can suggest a route through the problem while grading remains consistent and immediate. |
| Keep the original exercise visible | Each problem ID links the two source images, topic, level and official answer. | Pupils see the same diagrams and options as in the competition and can change languages without starting over. |
| Look for independent application | A related follow-up must be unseen, in the same topic and of the same diagram or text-only type. Pista records a first-try solve without help separately. | Teachers can distinguish a solve achieved with support from an independent solve on a new exercise. |
| Make local operation part of the design | The board serves a small browser interface, reads a precomputed exercise map and writes progress to SQLite. Model calls go through a local endpoint. | Prepared installations can run offline, and grading and practice remain available if the model is unavailable. |

## Verified results

The submission check served **1,747 of 1,747 unique exercises** over HTTP and found **zero premature answer-key fields in 7,863 responses**. All **3,494** expected image files were present. In the progression simulation, all **12** threshold profiles reached the expected mastery level, settled within one level of their target in **one to three** topic exercises, and received no repeated IDs. All **seven** teaching tests passed. [Evaluation and limitations](EVALUATION.md)

## Documentation

| Guide | What it covers |
| --- | --- |
| [Design and architecture](DESIGN.md) | Teaching flow, the exercise map, adaptation, model integration, classroom view and hardware |
| [Evaluation](EVALUATION.md) | Test methods, results, model behaviour and next experiments |
| [Run Pista](RUN.md) | Data preparation, laptop setup, board deployment and reproducibility |
| [Live demo](DEMO.md) | A three-minute jury walkthrough and rehearsal checks |

## Credits

Pista uses the M3Kang research dataset by Torres-Camps et al., with exercise material from the Societat Catalana de Matemàtiques. Coaching uses a compact Qwen model through llama.cpp. The interface bundles Bricolage Grotesque and Instrument Sans under the SIL Open Font License. [M3Kang dataset card and terms](https://huggingface.co/datasets/qualcomm/M3Kang#dataset-license) · [M3Kang paper](https://arxiv.org/abs/2601.16218) · [llama.cpp](https://github.com/ggml-org/llama.cpp)
