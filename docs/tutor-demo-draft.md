# Demo and pitch: 3 minutes

One person talks, one person drives the laptop. The board sits where the jury can see its LED matrix.

## Before you walk up

| Check | How |
| --- | --- |
| Board is powered and on USB | `adb devices` shows one device |
| Latest code and data are on the board | `bash laptop/deploy_to_board.sh --no-images` (2 minutes) |
| Tutor answers | open http://127.0.0.1:8000, the on-screen LED copy is not "board not connected" |
| Wi-Fi is off on the laptop | the page still loads: it goes over the USB cable |
| Demo student is fresh | use a new name each rehearsal, for example `Aina1`, `Aina2` |
| Backup | a 60-second screen recording of the demo below, ready to play |

Open these links in two tabs before you start (replace the name each time):

- Student: `http://127.0.0.1:8000/?name=Aina&lang=cat&topic=counting&level=2`
- Teacher view: the "teacher view" link on the same page

## The script

| Time | Who | What is said and done |
| --- | --- | --- |
| 0:00 to 0:20 | Speaker | "Maths results in Catalonia are falling, and a tutor that just gives the answer does not help. Ours runs entirely on this board, offline, and it never gives the answer away. Student data never leaves the device." |
| 0:20 to 0:35 | Driver | Show the page: a real Kangaroo problem, as a picture, in Catalan. "The student sees the original problem with its figure. No OCR, no vision model: nothing can misread it." |
| 0:35 to 0:50 | Driver | Answer wrong on purpose. The LED matrix shows a cross, the page crosses out the letter. "It says it is not right. It does not say what is." |
| 0:50 to 1:20 | Driver | Click Hint three times, one beat each. Speaker names the rungs: "The local model suggests where to start, what to try next and how to check your work." |
| 1:20 to 1:35 | Driver | Answer right. The board shows a tick from the official key. Open a follow-up on the same topic and show how the tutor records a first-try solve without help. |
| 1:35 to 1:50 | Driver | Open the teacher view. "The teacher sees levels by topic and what to practise next. It is a SQLite file on the board." |
| 1:50 to 2:30 | Speaker | The numbers (see below). |
| 2:30 to 3:00 | Speaker | What we learned (see below). |

## The numbers to say

From the laptop run of 15 November on the 281 English validation problems. Replace with the board run if it differs.

- "The organizers' baseline is 22.5%, below always answering D."
- "We measured first. The small model is at chance on these problems: one letter, bias correction, a bigger model, the picture. None of it helps. We tested eight models."
- "So our solver does not trust it. A ranker fitted on M3Kang's training problems reads the options themselves, for example the right answer is rarely the smallest or the largest value, and the model is only asked when the ranker is unsure, with a hard 54-second budget."
- "On validation problems the model never saw: **29.5%** against **23.1%** for always-D on every problem, and **35%** against **19.5%** on the text-only ones."
- "In the tutor: we exercised all 1,747 problems over HTTP and checked that the official key was withheld until each problem was solved or revealed."

## What we learned (pick two)

- The honest finding: on this hardware the model cannot be the source of truth. Everything that works comes from M3Kang itself: labels, neighbours, and the statistics of the options.
- Reasoning only helps if it finishes. Worked examples in the cached system prompt cost nothing per problem and raised the share of finished answers from about half to three quarters.
- Vision made answers worse (15% against 25% for always-D) and costs 30 seconds. For teaching it is not needed: the student is the one who looks at the picture.
- Next: have teachers judge the guiding questions, and add spoken hints for students with reading difficulties.

## Questions the jury may ask

| Question | Answer |
| --- | --- |
| Does the AI ever give a wrong hint? | The model can make mistakes. The tutor checks for answer markers, numerical leaks and repeated questions. The official M3Kang key grades each answer. |
| Can you use another model? | Pista accepts a local chat completions endpoint via `--llm`. |
| How do you use M3Kang? | Four ways: as the problem bank with its original images, as a map (topic, level, nearest easier problem), as training data for the option ranker, and as the validation set for every number we quote. |
| Does it work offline? | Yes. Fonts, page, images and models are all on the board. The demo runs with Wi-Fi off. |
| What about picture problems? | In the tutor the student sees the picture. In the solver we answer from the text and the options; adding the image lowered accuracy. |
