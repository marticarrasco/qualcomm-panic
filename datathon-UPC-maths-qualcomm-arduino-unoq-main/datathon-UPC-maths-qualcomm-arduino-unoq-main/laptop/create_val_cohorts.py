#!/usr/bin/env python3
"""Create reproducible, nested, stratified evaluation cohorts from val.jsonl."""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "val.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "eval_sets"
DEFAULT_SIZES = (100, 300, 500, 1000)
SEED = "m3kang-val-cohorts-v1-2026-11-14"
STRATEGY_VERSION = "text_only_options_nested_language_coverage_v3"
FILTER = {"text_only": True, "options_complete": True}
FEATURES = ("language", "level")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true",
                        help="replace existing cohorts if the source val file changed")
    return parser.parse_args()


def stable_hash(value):
    return hashlib.sha256(f"{SEED}|{value}".encode("utf-8")).hexdigest()


def feature_values(row):
    return {
        "language": str(row["language"]),
        "level": str(row["level"]),
    }


def complete_options(row):
    options = row.get("options")
    return (isinstance(options, dict) and set(options) == set("ABCDE")
            and all(isinstance(options[key], str) and options[key].strip() for key in "ABCDE"))


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]
    keys = [(row.get("id"), row.get("language")) for row in rows]
    if any(not problem_id or not language for problem_id, language in keys):
        raise ValueError("Every validation row must have an id and language.")
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate (id, language) key in validation source.")
    if any(row.get("label") not in set("ABCDE") for row in rows):
        raise ValueError("Every validation row must have an A-E label.")
    return rows


def select_order(rows, max_size):
    """Greedily balance marginal strata; cover each problem ID before repeats."""
    if max_size > len(rows):
        raise ValueError(f"Requested {max_size} rows, but source has only {len(rows)}.")

    unique_ids = {row["id"] for row in rows}
    if max_size < len(unique_ids):
        raise ValueError("Largest cohort must be at least the number of unique problem IDs.")

    totals = {name: Counter() for name in FEATURES}
    groups = defaultdict(list)
    row_features = {}
    for row in rows:
        key = (row["id"], row["language"])
        values = feature_values(row)
        row_features[key] = values
        for name in FEATURES:
            totals[name][values[name]] += 1
        stratum = tuple(values[name] for name in FEATURES)
        groups[stratum].append(row)

    for stratum, candidates in groups.items():
        candidates.sort(key=lambda row: stable_hash(f"{row['id']}|{row['language']}"))
    strata = sorted(groups, key=lambda stratum: stable_hash("|".join(stratum)))
    selected_counts = {name: Counter() for name in FEATURES}
    selected_ids = set()
    order = []

    for step in range(max_size):
        sample_size = step + 1
        prioritize_new_id = len(selected_ids) < len(unique_ids)
        best_stratum = None
        best_score = float("inf")
        best_row = None
        best_index = None

        for stratum in strata:
            candidates = groups[stratum]
            candidate_index = next((i for i, row in enumerate(candidates)
                                    if not prioritize_new_id or row["id"] not in selected_ids), None)
            if candidate_index is None:
                continue

            values = dict(zip(FEATURES, stratum))
            score = 0.0
            for name in FEATURES:
                category = values[name]
                target = sample_size * totals[name][category] / len(rows)
                before = selected_counts[name][category]
                # Normalize by the expected count so common and rare categories
                # both contribute to the proportional-balance objective.
                denominator = max(1.0, target)
                score += ((before + 1 - target) ** 2 - (before - target) ** 2) / denominator

            if score < best_score - 1e-12:
                best_stratum = stratum
                best_score = score
                best_row = groups[stratum][candidate_index]
                best_index = candidate_index

        if best_row is None:
            raise RuntimeError(f"Could not select row {sample_size} of {max_size}.")

        groups[best_stratum].pop(best_index)
        order.append(best_row)
        selected_ids.add(best_row["id"])
        values = row_features[(best_row["id"], best_row["language"])]
        for name in FEATURES:
            selected_counts[name][values[name]] += 1

    return order, totals, len(unique_ids)


def ensure_language_coverage(order, rows, full_counts, size=300, preserve_prefix=100):
    """Include every available language by the 300-row cohort."""
    source_by_language = defaultdict(list)
    for row in rows:
        source_by_language[row["language"]].append(row)
    for language, candidates in source_by_language.items():
        candidates.sort(key=lambda row: stable_hash(f"{row['id']}|{row['language']}"))

    for missing_language in sorted(set(source_by_language) - {row["language"] for row in order[:size]}):
        prefix = order[:size]
        selected_keys = {(row["id"], row["language"]) for row in prefix}
        id_counts = Counter(row["id"] for row in prefix)
        category_counts = {
            name: Counter(feature_values(row)[name] for row in prefix)
            for name in FEATURES
        }
        best = None

        for new_row in source_by_language[missing_language]:
            if (new_row["id"], new_row["language"]) in selected_keys:
                continue
            new_values = feature_values(new_row)
            for index in range(preserve_prefix, size):
                old_row = order[index]
                old_values = feature_values(old_row)

                # Preserve problem-ID coverage and all currently represented strata.
                if old_row["id"] != new_row["id"] and id_counts[old_row["id"]] <= 1:
                    continue
                if (old_values["language"] != new_values["language"]
                        and category_counts["language"][old_values["language"]] <= 1):
                    continue
                if (old_values["level"] != new_values["level"]
                        and category_counts["level"][old_values["level"]] <= 1):
                    continue
                score = 0.0
                for name in FEATURES:
                    old_category = old_values[name]
                    new_category = new_values[name]
                    if old_category == new_category:
                        continue
                    old_target = size * full_counts[name][old_category] / len(rows)
                    new_target = size * full_counts[name][new_category] / len(rows)
                    old_count = category_counts[name][old_category]
                    new_count = category_counts[name][new_category]
                    score += (((old_count - 1 - old_target) ** 2 - (old_count - old_target) ** 2)
                              / max(1.0, old_target))
                    score += (((new_count + 1 - new_target) ** 2 - (new_count - new_target) ** 2)
                              / max(1.0, new_target))

                tie_break = stable_hash(
                    f"{missing_language}|{new_row['id']}|{old_row['id']}|{old_row['language']}"
                )
                choice = (score, tie_break, index, new_row)
                if best is None or choice[:2] < best[:2]:
                    best = choice

        if best is None:
            raise RuntimeError(f"Could not include validation language {missing_language} by row {size}.")

        _, _, index, new_row = best
        old_row = order[index]
        new_key = (new_row["id"], new_row["language"])
        later_index = next((i for i in range(size, len(order))
                            if (order[i]["id"], order[i]["language"]) == new_key), None)
        if later_index is not None:
            order[later_index] = old_row
        order[index] = new_row

    return order


def dimension_counts(rows):
    counts = {name: Counter() for name in FEATURES}
    for row in rows:
        values = feature_values(row)
        for name in FEATURES:
            counts[name][values[name]] += 1
    return {name: dict(sorted(values.items())) for name, values in counts.items()}


def write_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temp_path, path)


def main():
    args = parse_args()
    source = args.input.resolve()
    output = args.out.resolve()
    rows = read_rows(source)
    sizes = tuple(size for size in DEFAULT_SIZES if size <= len(rows))
    if not sizes or max(sizes) < len({row["id"] for row in rows}):
        raise SystemExit("Validation data must contain at least 1,000 rows and enough rows to cover every problem ID.")

    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file() and not args.force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_hash = previous.get("source", {}).get("sha256")
        previous_filter = previous.get("selection", {}).get("filter")
        if ((previous_hash and previous_hash != source_sha256)
                or previous_filter != FILTER
                or previous.get("selection", {}).get("strategy_version") != STRATEGY_VERSION):
            raise SystemExit("The source or cohort filter changed; pass --force to replace these cohorts.")

    source_rows = rows
    rows = [row for row in source_rows
            if row.get("text_only") is True and complete_options(row)]
    sizes = tuple(size for size in DEFAULT_SIZES if size <= len(rows))
    if not sizes or max(sizes) < 1000:
        raise SystemExit(f"The text_only=true subset has {len(rows)} rows; 1,000 are needed for all cohorts.")

    order, full_counts, unique_ids = select_order(rows, max(sizes))
    if 300 in sizes:
        order = ensure_language_coverage(order, rows, full_counts, size=300, preserve_prefix=100)
    cohort_manifest = {
        "schema_version": 1,
        "source": {
            "path": str(source),
            "sha256": source_sha256,
            "rows": len(source_rows),
            "filtered_rows": len(rows),
            "filter": FILTER,
            "unique_problem_ids": unique_ids,
        },
        "selection": {
            "strategy_version": STRATEGY_VERSION,
            "seed": SEED,
            "sizes": list(sizes),
            "dimensions": list(FEATURES),
            "filter": FILTER,
            "method": "Nested greedy stratification over language and level; each problem ID is selected once before repeats; all available languages are represented by the 300-row cohort.",
        },
        "cohorts": {},
    }

    for size in sizes:
        cohort = order[:size]
        path = output / f"val_{size}.jsonl"
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cohort)
        write_atomic(path, content)
        cohort_manifest["cohorts"][str(size)] = {
            "path": str(path),
            "rows": len(cohort),
            "unique_problem_ids": len({row["id"] for row in cohort}),
            "distinct_languages": len({row["language"] for row in cohort}),
            "counts": dimension_counts(cohort),
            "full_validation_counts": {name: dict(sorted(counts.items()))
                                       for name, counts in full_counts.items()},
        }
        print(f"{path.name}: {len(cohort)} rows, "
              f"{cohort_manifest['cohorts'][str(size)]['unique_problem_ids']} unique problems, "
              f"{cohort_manifest['cohorts'][str(size)]['distinct_languages']} languages")

    all_text_only_path = output / "val_text_only_all.jsonl"
    all_text_only_content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    write_atomic(all_text_only_path, all_text_only_content)
    cohort_manifest["all_text_only"] = {
        "path": str(all_text_only_path),
        "rows": len(rows),
        "unique_problem_ids": len({row["id"] for row in rows}),
        "distinct_languages": len({row["language"] for row in rows}),
        "counts": dimension_counts(rows),
    }
    print(f"{all_text_only_path.name}: {len(rows)} rows, "
          f"{cohort_manifest['all_text_only']['unique_problem_ids']} unique problems, "
          f"{cohort_manifest['all_text_only']['distinct_languages']} languages")

    write_atomic(manifest_path, json.dumps(cohort_manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
