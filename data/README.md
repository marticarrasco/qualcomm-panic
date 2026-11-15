# Local datasets and sample input

[Project overview](../README.md) · [Documentation index](../docs/README.md)

The repository includes the synthetic [`sample_problems.jsonl`](sample_problems.jsonl) for a predictor smoke test and the [expansion manifest](m3kang_expansion_manifest.json) for the historical M3Kang snapshot. Prepare the source problems, images, OCR options, and tutor bank in your local checkout under the dataset terms. [M3Kang dataset card](https://huggingface.co/datasets/qualcomm/M3Kang) · [Dataset license](https://huggingface.co/datasets/qualcomm/M3Kang/blob/main/LICENSE.pdf)

## Prepare local inputs

For bilingual predictor data with OCR options, run from the repository root.

```bash
python3 -m pip install -r laptop/requirements.txt
python3 laptop/prepare_m3kang.py --split high_quality --shards all --languages eng,cat --out data
```

Use the [Pista run guide](../docs/tutor/RUN.md) to prepare original exercise images and build the tutor bank. If you have the organizers’ practice Parquet, place it at `data/demo-00000-of-00001.parquet` and follow the [predictor run guide](../docs/predictor/RUN.md#prepare-the-organizers-practice-set). The organizers’ practice set is separate from M3Kang.

## Historical multilingual snapshot

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

If you have the original bilingual, OCR-reviewed `train.jsonl` and `val.jsonl`, the expansion script can preserve their assignments and extend them using the pinned M3Kang revision. Its input directory defaults to `data`.

```powershell
.\.venv\Scripts\python.exe laptop\expand_m3kang_languages.py
```

Dataset exports, OCR reports, images, indexes, and pupil databases are local artifacts. The [results guide](../results/README.md) explains how to save and inspect detailed runs locally.
