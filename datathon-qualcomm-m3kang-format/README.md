---
license: other
task_categories:
- visual-question-answering
- multiple-choice
language:
- en
- ca
pretty_name: Datathon FME 2026 · Qualcomm challenge (M3Kang format)
configs:
- config_name: default
  data_files:
  - split: demo
    path: data/demo-*
dataset_info:
  features:
  - name: id
    dtype: string
  - name: level
    dtype: int32
  - name: language
    dtype: string
  - name: text_only
    dtype: bool
  - name: image
    dtype: image
  - name: text
    dtype: string
  - name: label
    dtype: string
  splits:
  - name: demo
    num_examples: 100
---

# Datathon FME 2026 · Qualcomm challenge (M3Kang format)

Kangaroo-style problems written for the Datathon FME 2026, in the same format as [qualcomm/M3Kang](https://huggingface.co/datasets/qualcomm/M3Kang). They are not M3Kang problems.

- `demo`: 50 problems for practice, each in English (`eng`) and Catalan (`cat`): 100 rows.
- The final evaluation uses 20 different problems (one language each, levels 0–7, 4 with a figure), run with your `predict.py` after the 12:00 last call: at most 60 s per problem, no network.

Levels 0–7 (level 0 is grade 5), 40 text-only problems and 10 with a figure, every letter A–E correct 10 times. Translations share the same `id`.

| Column | Type | Content |
| --- | --- | --- |
| `id` | string | `lvl-<level>_2026_<number>` |
| `level` | int32 | 0 to 7 |
| `language` | string | `eng` or `cat` |
| `text_only` | bool | `false` if the problem needs its figure |
| `image` | image | The full problem: statement, figure if any, and the options A) to E) |
| `text` | string | `"<number>. <statement>"`, without the options |
| `label` | string | The correct letter, A to E |

With the challenge repository:

```bash
python3 laptop/prepare_m3kang.py --parquet data/demo-00000-of-00001.parquet --languages eng,cat --val-fraction 0 --out data/demo --save-images
```
