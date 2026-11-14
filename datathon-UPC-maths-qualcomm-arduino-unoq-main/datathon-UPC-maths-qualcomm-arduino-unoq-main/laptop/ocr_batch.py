#!/usr/bin/env python3
"""Run one resumable M3Kang OCR batch for a chosen language.

By default, process all problems for that language in the first cached
high_quality shard, regardless of text_only. Use --text-only to filter.
Example: python laptop/ocr_batch.py --size 300 --language eng
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from prepare_m3kang import parse_options
from export_ocr_dataset import export_dataset

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE_PATH = DATA / "ocr" / ".ocr_cache.jsonl"


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, required=True, help="maximum problems in this batch")
    parser.add_argument("--language", required=True, help="M3Kang language code, e.g. eng, cat, esp")
    parser.add_argument("--parquet", help="local parquet file; defaults to the cached high_quality shard 0")
    parser.add_argument("--text-only", action="store_true",
                        help="process only rows where text_only is true (default: all problems)")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="share of completed problem IDs assigned to validation (default 0.15)")
    return parser.parse_args()


def find_parquet(value):
    if value:
        path = Path(value).resolve()
        if not path.is_file():
            raise SystemExit(f"No existeix el parquet: {path}")
        return path
    cache = Path.home() / ".cache/huggingface/hub/datasets--qualcomm--M3Kang/snapshots"
    paths = sorted(cache.rglob("high_quality-00000-of-00004.parquet"))
    if not paths:
        raise SystemExit("No trobo el shard local. Descarrega M3Kang primer o passa --parquet <fitxer>.")
    return paths[-1]


def read_cache(path):
    results = {}
    if path.exists():
        # A prior Windows run may have used the console's legacy code page;
        # read each JSONL row independently so an older row won't block resume.
        for raw_line in path.read_bytes().splitlines():
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                line = raw_line.decode("cp1252")
            row = json.loads(line)
            results[row["key"]] = row["options"]
    return results


def read_saved_options(language):
    """Treat problems already written to JSONL as completed even if the OCR cache is absent."""
    paths = [
        DATA / "train.jsonl", DATA / "val.jsonl",
        DATA / "train_reviewed.jsonl", DATA / "val_reviewed.jsonl",
    ]
    batch_dir = DATA / "ocr_batches" / language
    if batch_dir.exists():
        paths.extend(sorted(batch_dir.glob("batch_*/*.jsonl")))
    saved = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("language") == language and row.get("id"):
                    saved[f"{row['id']}|{language}"] = row.get("options") or {}
    return saved


def next_batch_number(folder):
    existing = []
    if folder.exists():
        for path in folder.glob("batch_*"):
            if path.is_dir() and path.name[6:].isdigit():
                existing.append(int(path.name[6:]))
    return max(existing, default=0) + 1


def main():
    args = args_parser()
    if args.size < 1:
        raise SystemExit("--size ha de ser com a mínim 1.")
    if not 0 < args.val_fraction < 1:
        raise SystemExit("--val-fraction ha d'estar entre 0 i 1.")
    language = args.language.strip()
    if not language or "," in language:
        raise SystemExit("Indica un sol idioma amb --language, per exemple eng.")

    parquet_path = find_parquet(args.parquet)
    meta = pq.read_table(parquet_path, columns=["id", "language", "text_only"])
    ids = meta.column("id").to_pylist()
    languages = meta.column("language").to_pylist()
    text_only = meta.column("text_only").to_pylist()
    eligible = [
        i for i, (problem_id, lang, is_text_only) in enumerate(zip(ids, languages, text_only))
        if lang == language and (not args.text_only or is_text_only is True)
    ]
    if not eligible:
        raise SystemExit(f"No hi ha problemes per a l'idioma {language!r} amb aquests filtres.")

    all_rows = pq.read_table(parquet_path)
    unique_ids = {ids[i] for i in eligible}
    cache = read_cache(CACHE_PATH)
    # The review/preparation JSONL is also a durable record of completed OCR.
    # This matters if .ocr_cache.jsonl was removed after exporting the files.
    cache.update(read_saved_options(language))
    done_ids = {problem_id for problem_id in unique_ids if f"{problem_id}|{language}" in cache}
    remaining_indices = [i for i in eligible if f"{ids[i]}|{language}" not in cache]
    total = len(unique_ids)
    done_before = len(done_ids)
    pct_before = 100 * done_before / total
    print(f"Idioma: {language} | Filtre: {'text_only=true' if args.text_only else 'tots els problemes'}")
    print(f"Parquet: {parquet_path}")
    print(f"Abans del lot: {done_before}/{total} fets ({pct_before:.1f}%) | {total-done_before} pendents")
    if not remaining_indices:
        print("No queden problemes per processar.")
        print("Actualitzo train/val amb els resultats OCR de la memòria cau:")
        export_dataset(parquet_path=parquet_path, val_fraction=args.val_fraction, out_dir=DATA)
        return

    chosen = remaining_indices[:args.size]
    if len(chosen) < args.size:
        print(f"Només queden {len(chosen)}; processaré tots els que falten.")

    batch_root = DATA / "ocr_batches" / language
    batch_dir = batch_root / f"batch_{next_batch_number(batch_root):04d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = batch_dir / "train.jsonl", batch_dir / "val.jsonl"

    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()
    table = all_rows.take(chosen)
    n_train = n_val = 0
    started = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a", encoding="utf-8", newline="\n") as cache_file, \
            train_path.open("w", encoding="utf-8", newline="\n") as train_file, \
            val_path.open("w", encoding="utf-8", newline="\n") as val_file:
        for completed, index in enumerate(range(table.num_rows), 1):
            row = {name: table.column(name)[index].as_py()
                   for name in ("id", "language", "level", "text_only", "label", "text")}
            key = f"{row['id']}|{row['language']}"
            image = table.column("image")[index].as_py()["bytes"]
            boxes, _ = ocr(image)
            options = parse_options(boxes or [])

            year = int(row["id"].split("_")[1])
            bucket = int(hashlib.sha256(row["id"].encode()).hexdigest(), 16) % 1000
            split = "val" if bucket < args.val_fraction * 1000 else "train"
            output = {
                "id": row["id"], "language": row["language"], "level": row["level"],
                "year": year, "text_only": row["text_only"], "text": row["text"].strip(),
                "options": options, "label": row["label"],
            }
            target = val_file if split == "val" else train_file
            target.write(json.dumps(output, ensure_ascii=False) + "\n")
            target.flush()
            n_val += split == "val"
            n_train += split == "train"

            # Commit the result to the cache only after the batch JSONL has it.
            cache_file.write(json.dumps({"key": key, "options": options}, ensure_ascii=False) + "\n")
            cache_file.flush()
            cache[key] = options

            done_now = done_before + completed
            elapsed = max(time.time() - started, 0.001)
            rate = completed / elapsed
            eta = (len(chosen) - completed) / rate if rate else 0
            options_count = len(options)
            print(
                f"\rOCR {language}: {completed}/{len(chosen)} del lot | "
                f"opcions {options_count}/5 | total {done_now}/{total} ({100*done_now/total:.1f}%) | "
                f"{rate:.2f} problemes/s | ETA {eta/60:.1f} min",
                end="\n" if completed == len(chosen) else "", flush=True,
            )

    done_after = len({problem_id for problem_id in unique_ids if f"{problem_id}|{language}" in cache})
    remaining_after = total - done_after
    ocr_successes = sum(bool(cache.get(f"{ids[i]}|{language}")) for i in chosen)
    print(f"\nLot desat a: {batch_dir}")
    print(f"JSONL: train={n_train}, val={n_val}; opcions OCR trobades en {ocr_successes}/{len(chosen)}")
    print(f"Acumulat OCR: {done_after}/{total} fets ({100*done_after/total:.1f}%) | {remaining_after} pendents")
    print("\nActualitzo data/train.jsonl i data/val.jsonl amb tots els resultats OCR de la memòria cau:")
    export_dataset(parquet_path=parquet_path, val_fraction=args.val_fraction, out_dir=DATA)


if __name__ == "__main__":
    main()
