# M3Kang data (local, research use)

`train.jsonl` and `val.jsonl` contain the `high_quality` M3Kang split across all four Parquet shards: 40,617 rows for 1,747 problems in 108 languages. Each problem is available in 2–100 languages; not every problem has all 108 translations. There is one row per `(id, language)`. Train has 33,843 rows / 1,466 problems; validation has 6,774 rows / 281 problems. The train/validation assignment is inherited by problem ID from the previous company set, so translations of the same problem stay in one split.

OCR options are shared across translations only when the English/Catalan option sets agree after whitespace normalization. This applies to 1,115 problem IDs; it fills 24,963 new-language rows and 60 previously blank English/Catalan rows. Where the source languages disagree (207 IDs), the original English/Catalan options are preserved and new-language rows are left empty; those cases are listed in `ocr_option_conflicts.jsonl`. Another 425 IDs have no recognized source options. In total, 27,607 rows have all five options. No OCR was run on the added-language images.

`m3kang_expansion_manifest.json` records the source revision and aggregate row counts. The source Parquet shards are cached by Hugging Face outside this directory. The `images/` folder contains the company source images already present before expansion (English and Catalan only).

To create fixed, nested evaluation cohorts containing only `text_only=true` rows with complete A–E options, run from the project root:

```powershell
python laptop\create_val_cohorts.py
python eval.py --input data/eval_sets/val_100.jsonl --name val100
python eval.py --input data/eval_sets/val_text_only_all.jsonl --name val-text-only-all
```

The generator writes 100, 300, 500, and 1,000-row cohorts, the full 2,961-row text-only subset with options, and a manifest with the source hash and subgroup counts. It balances language and level, selects each problem ID once before adding its other language rows, and uses a fixed seed. The 100-row cohort has 100 distinct problems; the larger cohorts cover every problem ID and language available in this filtered subset. If `val.jsonl` changes, the generator stops unless `--force` is passed to replace the cohorts. This filter applies to `eval_sets`; the full `train.jsonl` and `val.jsonl` remain unchanged.

To rebuild from the pinned M3Kang revision, run from the project root:

```powershell
.\.venv\Scripts\python.exe laptop\expand_m3kang_languages.py
```

M3Kang is licensed for research use. This repository includes derived train/validation JSONL data, evaluation cohorts, and the OCR conflict report for research evaluation. Source images, the local OCR cache, and the embedding index remain local. See the [dataset card](https://huggingface.co/datasets/qualcomm/M3Kang) for the source terms.
