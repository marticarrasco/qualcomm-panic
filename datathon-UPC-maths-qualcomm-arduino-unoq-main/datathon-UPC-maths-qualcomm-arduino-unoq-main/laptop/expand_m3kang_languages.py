#!/usr/bin/env python3
"""Expand the active M3Kang train/val files to all high_quality languages.

The split is inherited by problem ID from the existing data/train.jsonl and
data/val.jsonl. OCR options are copied across translations only when all
available source option sets agree (apart from harmless whitespace). Conflicts
are written to data/ocr_option_conflicts.jsonl for local review; they are not
propagated to additional languages.

This reads metadata and text from the Parquet shards, not their images, and it
does not run OCR. The source shard files are cached by huggingface_hub outside
the project directory.
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATASET = "qualcomm/M3Kang"
REVISION = "e772e611d45679cf48f1c92618b2963a40d1d584"
SHARD_COUNT = 4
SHARD_TEMPLATE = "data/high_quality-{index:05d}-of-00004.parquet"
LETTERS = "ABCDE"
FIELDS = ("id", "language", "level", "text_only", "label", "text")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=REVISION,
                        help="pinned Hugging Face dataset revision")
    parser.add_argument("--parquet", nargs="*",
                        help="four local high_quality parquet files; default downloads/caches them")
    parser.add_argument("--source-dir", default=str(DATA),
                        help="existing train/val JSONL split and OCR options")
    parser.add_argument("--out", default=str(DATA),
                        help="output directory for expanded train/val and audit files")
    return parser.parse_args()


def get_shards(parquet_paths, revision):
    if parquet_paths:
        paths = [Path(value).expanduser().resolve() for value in parquet_paths]
        if len(paths) != SHARD_COUNT or not all(path.is_file() for path in paths):
            raise SystemExit("Pass exactly four existing high_quality Parquet files.")
        return paths

    paths = []
    for index in range(SHARD_COUNT):
        filename = SHARD_TEMPLATE.format(index=index)
        print(f"M3Kang high_quality shard {index + 1}/{SHARD_COUNT}: {filename}", flush=True)
        paths.append(Path(hf_hub_download(
            repo_id=DATASET,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )))
    return paths


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def clean_options(value):
    if not value:
        return {}
    if not isinstance(value, dict) or set(value) != set(LETTERS):
        raise ValueError("Options must be empty or contain exactly A-E.")
    cleaned = {letter: str(value[letter]).strip() for letter in LETTERS}
    if not all(cleaned.values()):
        raise ValueError("Options contain an empty A-E value.")
    return cleaned


def option_count(options):
    return sum(bool(options.get(letter)) for letter in LETTERS)


def comparison_text(value):
    """Collapse OCR spacing without joining distinct numbers or symbols."""
    value = re.sub(r"\s+", " ", value.strip())
    return re.sub(r"\s+([,.;:%°])", r"\1", value)


def option_signature(options):
    return tuple(comparison_text(options[letter]) for letter in LETTERS)


def load_existing(source_dir):
    source_dir = Path(source_dir).resolve()
    source_rows = {}
    split_by_id = {}
    ids_by_split = defaultdict(set)
    for split in ("train", "val"):
        for row in read_jsonl(source_dir / f"{split}.jsonl"):
            problem_id, language = row.get("id"), row.get("language")
            if not problem_id or not language:
                raise ValueError(f"Missing id/language in {split}.jsonl")
            key = (problem_id, language)
            if key in source_rows:
                raise ValueError(f"Duplicate existing key: {key}")
            old_split = split_by_id.setdefault(problem_id, split)
            if old_split != split:
                raise ValueError(f"Problem {problem_id} appears in both train and val.")
            options = clean_options(row.get("options"))
            if option_count(options) not in (0, 5):
                raise ValueError(f"Partial options for {key}.")
            source_rows[key] = {**row, "options": options}
            ids_by_split[split].add(problem_id)
    if not source_rows:
        raise ValueError("No existing source train/val rows found.")
    return source_rows, split_by_id, ids_by_split


def load_m3kang_rows(paths):
    rows_by_key = {}
    rows_by_id = defaultdict(list)
    for path in paths:
        table = pq.read_table(path, columns=list(FIELDS))
        for raw in table.to_pylist():
            row = {
                "id": raw["id"],
                "language": raw["language"],
                "level": int(raw["level"]),
                "text_only": bool(raw["text_only"]),
                "label": raw["label"],
                "text": (raw["text"] or "").strip(),
            }
            key = (row["id"], row["language"])
            previous = rows_by_key.get(key)
            if previous is not None:
                if previous != row:
                    raise ValueError(f"Conflicting duplicate M3Kang row: {key}")
                continue
            rows_by_key[key] = row
            rows_by_id[row["id"]].append(row)
    if not rows_by_key:
        raise ValueError("No rows found in the supplied Parquet shards.")
    return rows_by_key, rows_by_id


def resolve_option_sets(source_rows, rows_by_id, split_by_id):
    source_options = defaultdict(dict)
    for (problem_id, language), row in source_rows.items():
        options = row["options"]
        if options:
            source_options[problem_id][language] = options

    canonical = {}
    conflicts = []
    both_empty = 0
    for problem_id in rows_by_id:
        options_by_language = source_options.get(problem_id, {})
        signatures = {option_signature(options) for options in options_by_language.values()}
        if not signatures:
            both_empty += 1
        elif len(signatures) == 1:
            # Prefer English for stable formatting when both source languages agree.
            preferred_language = "eng" if "eng" in options_by_language else sorted(options_by_language)[0]
            chosen = options_by_language[preferred_language]
            canonical[problem_id] = {
                letter: comparison_text(chosen[letter]) for letter in LETTERS
            }
        else:
            conflicts.append({
                "id": problem_id,
                "split": split_by_id[problem_id],
                "options_by_language": {
                    language: options_by_language[language]
                    for language in sorted(options_by_language)
                },
            })
    return canonical, conflicts, both_empty, source_options


def atomic_jsonl(path, rows):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def main():
    args = parse_args()
    source_rows, split_by_id, original_ids_by_split = load_existing(args.source_dir)
    parquet_paths = get_shards(args.parquet, args.revision)
    rows_by_key, rows_by_id = load_m3kang_rows(parquet_paths)

    m3kang_ids = set(rows_by_id)
    source_ids = set(split_by_id)
    missing_split_ids = m3kang_ids - source_ids
    stale_source_ids = source_ids - m3kang_ids
    if missing_split_ids or stale_source_ids:
        raise ValueError(
            f"Source and M3Kang IDs differ: {len(missing_split_ids)} missing splits, "
            f"{len(stale_source_ids)} stale source IDs. No files were written."
        )

    # M3Kang translations must share the same answer letter and text_only flag.
    for problem_id, translations in rows_by_id.items():
        if len({row["label"] for row in translations}) != 1:
            raise ValueError(f"Label differs across translations for {problem_id}.")
        if len({row["text_only"] for row in translations}) != 1:
            raise ValueError(f"text_only differs across translations for {problem_id}.")

    canonical, conflicts, no_source_count, source_options = resolve_option_sets(
        source_rows, rows_by_id, split_by_id
    )
    conflict_ids = {row["id"] for row in conflicts}
    output = {"train": [], "val": []}
    languages_by_split = {"train": Counter(), "val": Counter()}
    rows_with_options = Counter()
    filled_existing = 0
    filled_new = 0

    for key in sorted(rows_by_key):
        problem_id, language = key
        row = rows_by_key[key]
        split = split_by_id[problem_id]
        if problem_id in conflict_ids:
            # Preserve observed English/Catalan OCR; do not guess for new languages.
            options = source_rows.get(key, {}).get("options", {})
        else:
            options = canonical.get(problem_id, {})
            if options and not source_rows.get(key, {}).get("options"):
                if key in source_rows:
                    filled_existing += 1
                else:
                    filled_new += 1

        record = {
            "id": problem_id,
            "language": language,
            "level": row["level"],
            "year": int(problem_id.split("_")[1]),
            "text_only": row["text_only"],
            "text": row["text"],
            "options": options,
            "label": row["label"],
        }
        output[split].append(record)
        languages_by_split[split][language] += 1
        if options:
            rows_with_options[split] += 1

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        atomic_jsonl(out_dir / f"{split}.jsonl", output[split])
    atomic_jsonl(out_dir / "ocr_option_conflicts.jsonl", conflicts)

    manifest = {
        "dataset": DATASET,
        "revision": args.revision,
        "source_split": "high_quality",
        "shards": [path.name for path in parquet_paths],
        "unique_problems": len(m3kang_ids),
        "languages": sorted({row["language"] for row in rows_by_key.values()}),
        "rows_total": len(rows_by_key),
        "rows_by_split": {split: len(output[split]) for split in ("train", "val")},
        "problem_ids_by_split": {split: len(original_ids_by_split[split]) for split in ("train", "val")},
        "rows_with_options_by_split": dict(rows_with_options),
        "problem_ids_with_consensus_options": len(canonical),
        "problem_ids_with_option_conflicts": len(conflicts),
        "problem_ids_without_source_options": no_source_count,
        "existing_rows_filled": filled_existing,
        "new_language_rows_filled": filled_new,
        "languages_by_split": {split: dict(sorted(counts.items())) for split, counts in languages_by_split.items()},
    }
    manifest_path = out_dir / "m3kang_expansion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"M3Kang high_quality rows: {len(rows_by_key):,} across {len(manifest['languages'])} languages")
    print(f"Unique problems: {len(m3kang_ids):,}; train IDs: {len(original_ids_by_split['train']):,}; val IDs: {len(original_ids_by_split['val']):,}")
    print(f"Rows: train={len(output['train']):,} ({rows_with_options['train']:,} with options), "
          f"val={len(output['val']):,} ({rows_with_options['val']:,} with options)")
    print(f"Option sets propagated by consensus: {len(canonical):,} problem IDs; "
          f"filled existing rows={filled_existing:,}, new-language rows={filled_new:,}")
    print(f"Unresolved option conflicts: {len(conflicts):,}; no-source option IDs: {no_source_count:,}")
    print(f"Wrote {out_dir / 'train.jsonl'} and {out_dir / 'val.jsonl'}")
    print(f"Conflict review: {out_dir / 'ocr_option_conflicts.jsonl'}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
