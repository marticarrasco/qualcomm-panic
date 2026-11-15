# Three-minute jury demo

## 0:00–0:25 · Introduce the pupil's task

> A pupil can get a correct answer today and still be stuck when the numbers change. We built Pista to give one useful step at a time, then let the pupil try a related problem independently. The tutor runs on this Arduino UNO Q in Catalan and English.

Show the UNO Q and open Pista over USB. Start a fresh Catalan profile in a rehearsed topic and level. Open the image viewer briefly so the jury sees the original exercise and options.

## 0:25–1:20 · Show help and grading

Ask for the first hint. While it loads, explain that the local model receives the exercise and the requested coaching step. The official M3Kang key stays on the server for grading. Use the hint as a next action the pupil could take.

Choose the rehearsed wrong option. Show that it is graded immediately and that the tutor can suggest a check for that choice. If a draft is rejected, show that the pupil can continue and retry help. Point to the browser's event display and the board's matrix during the hint or wrong-choice event.

## 1:20–2:25 · Show independent practice

Switch the same attempt to English and point out that the exercise and tried choice are preserved. Choose the correct answer. Advance to the related exercise and solve it on the first try with zero hints or chat help.

Open the teacher view. Show the topic level, earlier support use and the follow-up result. The **1/1** counter demonstrates how Pista records an independent follow-up. The jury can see the difference between the earlier supported attempt and the new independent one.

## 2:25–3:00 · Close with the engineering evidence

> The same board serves 1,747 exercises in both languages. We walked the entire bank over HTTP and found zero early answer-key fields across 7,863 responses. In the simulation, all 12 threshold profiles reached their expected mastery level. The next experiment is to ask teachers to rate coaching quality and test whether pupils solve more new problems independently after using it.

## Rehearsal

- Start from a dedicated demo database with `--students data/pista-demo.db`. Use a fresh profile for each rehearsal that needs the opening exercise again.
- Rehearse the exact topic, first exercise, wrong choice, correct choice and follow-up using the board's active model. Keep the correct letters available to the presenter. Check that the selected follow-up can be solved independently in the time available.
- Verify the local model endpoint, USB port forwarding, bilingual images and App Lab matrix before presenting. Practise with Wi-Fi disconnected after setup.
- If a live draft is unavailable, complete the first exercise after a wrong choice, or use the explicit reveal. A suitable follow-up can still be offered. Bring a short recording of a successful local coaching turn to show the generation path if needed.

[Run Pista](RUN.md) · [Evaluation](EVALUATION.md)
