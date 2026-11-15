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
| 0:50 to 1:20 | Driver | Click Hint three times, one beat each. Speaker names the rungs: "First a guiding question written by the on-board AI, with a method tip. Then a similar, easier problem from M3Kang, found with embeddings. Then it rules two options out. Each rung gives a little more, and the answer only comes if the student asks for it." |
| 1:20 to 1:35 | Driver | Answer right. Tick on the matrix, level bar goes up, next problem is one level harder on the same topic. "Right answer: same topic, one level up. Wrong: one level down. No model in that loop, so each step takes milliseconds." |
| 1:35 to 1:50 | Driver | Open the teacher view. "The teacher sees levels by topic and what to practise next. It is a SQLite file on the board." |
| 1:50 to 2:30 | Speaker | The numbers (see below). |
| 2:30 to 3:00 | Speaker | What we learned (see below). |

## The numbers to say

Fill in the two marked values from the latest run before the pitch (`results/` and `EXPERIMENTS.md`).

- "The organizers' baseline is 22.5%, below always answering D."
- "We measured first. The small model is at chance on these problems: one letter, bias correction, a bigger model, the picture. None of it helps. We tested eight models."
- "So our solver does not trust it. A ranker fitted on M3Kang's training problems reads the options themselves, for example the right answer is rarely the smallest or the largest value, and the model is only asked when the ranker is unsure, with a hard 54-second budget."
- "On validation problems the model never saw: **[X]%** against **[Y]%** for always-D, and no problem over 60 seconds."
- "In the tutor: across all 1,747 problems, a hint never eliminates the right answer and the answer never reaches the browser early. We checked every one in code."

## What we learned (pick two)

- The honest finding: on this hardware the model cannot be the source of truth. Everything that works comes from M3Kang itself: labels, neighbours, and the statistics of the options.
- Reasoning only helps if it finishes. Worked examples in the cached system prompt cost nothing per problem and raised the share of finished answers from about half to three quarters.
- Vision made answers worse (15% against 25% for always-D) and costs 30 seconds. For teaching it is not needed: the student is the one who looks at the picture.
- Next: have teachers judge the guiding questions, and add spoken hints for students with reading difficulties.

## Questions the jury may ask

| Question | Answer |
| --- | --- |
| Does the AI ever give a wrong hint? | The hints that matter cannot be wrong: they come from the labels and from retrieval. The AI's guiding question is filtered in code (no answer, no option letter, no number that is not in the problem) and labelled as AI-written. In a hand check of 30, about 17 were useful and about 1 in 5 was misleading, which is why it is only the first rung and never the only help. |
| Why not a bigger model? | We tried: the 2B model reads at a third of the speed and gains almost nothing in 60 seconds. We use it only ahead of time, to write the guiding questions. |
| How do you use M3Kang? | Four ways: as the problem bank with its original images, as a map (topic, level, nearest easier problem), as training data for the option ranker, and as the validation set for every number we quote. |
| Does it work offline? | Yes. Fonts, page, images and models are all on the board. The demo runs with Wi-Fi off. |
| What about picture problems? | In the tutor the student sees the picture. In the solver we answer from the text and the options; adding the image lowered accuracy. |
