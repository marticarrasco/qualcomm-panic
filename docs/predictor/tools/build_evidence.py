"""Recompute publishable aggregates from the local predictor checkout.

Usage: python tools/build_evidence.py --project PATH_TO_PREDICT_PY
Matplotlib is only needed for the figures. Input data and raw outputs stay local.
"""

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


DOCS = Path(__file__).resolve().parents[1]
DEMO_INPUT = "data/datathon-demo-eval/train.jsonl"
DEMO_V4 = "results/pot-v4-e1-demo/eval-20261115-105958-184892-pot-v4-e1-smart-runner-demo.json"
DEMO_V1 = "results/eval-20261115-094216-937240-pot-v1-smart-runner-demo.json"
DEMO_TEXT = "results/datathon-qualcomm-demo/eval-20261115-011634-910547-all-rows-100.json"
BILINGUAL = "results/pot-v4-e1-train-val-text-only-vulkan/summary.json"


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def data_rows(path):
    with path.open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def scored_rows(project, manifest_path):
    path = project / manifest_path
    manifest = json_file(path)
    with path.with_name(manifest["details_csv"]).open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    predictions = []
    for row in rows:
        correct = int(row["is_correct"])
        assert row["selected_answer"] in tuple("ABCDE")
        assert correct == (row["selected_answer"] == row["correct_answer"])
        predictions.append({**row, "correct": correct})
    assert len(predictions) == manifest["completed_problems"], manifest_path
    assert sum(row["correct"] for row in predictions) == manifest["correct_count"], manifest_path
    return predictions, manifest


def key(row):
    return row["problem_id"], row["language"]


def score(rows):
    return {
        "n": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "always_d": sum(row["correct_answer"] == "D" for row in rows),
    }


def grouped(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return {str(name): score(group) for name, group in sorted(groups.items())}


def matched_truth(rows, inputs, *, complete):
    reference = {(row["id"], row["language"]): row for row in inputs}
    assert len(reference) == len(inputs)
    assert len({key(row) for row in rows}) == len(rows)
    if complete:
        assert set(reference) == {key(row) for row in rows}
    for row in rows:
        original = reference[key(row)]
        assert original["label"] == row["correct_answer"]
        assert original["level"] == int(row["level"])
        row["requires_figure"] = not original["text_only"]


def pair_statistics(v4, v1):
    v1_by_key = {key(row): row for row in v1}
    by_problem = defaultdict(list)
    for row in v4:
        by_problem[row["problem_id"]].append(row)
    assert all(len(group) == 2 for group in by_problem.values())
    successes = [sum(row["correct"] for row in group) for group in by_problem.values()]
    delta = [sum(row["correct"] - v1_by_key[key(row)]["correct"] for row in group)
             for group in by_problem.values()]
    rng = random.Random(42)
    draws = [rng.choices(range(len(successes)), k=len(successes)) for _ in range(10000)]

    def interval(values):
        rates = [100 * sum(values[i] for i in sample) / len(v4) for sample in draws]
        quantiles = statistics.quantiles(rates, n=40, method="inclusive")
        return [round(quantiles[0], 2), round(quantiles[-1], 2)]

    return {
        "unique_problems": len(by_problem),
        "neither_language_correct": successes.count(0),
        "one_language_correct": successes.count(1),
        "both_languages_correct": successes.count(2),
        "same_predicted_letter": sum(group[0]["selected_answer"] == group[1]["selected_answer"]
                                     for group in by_problem.values()),
        "v4_wins_vs_v1": sum(row["correct"] > v1_by_key[key(row)]["correct"] for row in v4),
        "v4_losses_vs_v1": sum(row["correct"] < v1_by_key[key(row)]["correct"] for row in v4),
        "v4_percent_cluster_bootstrap_95": interval(successes),
        "v4_minus_v1_pp_cluster_bootstrap_95": interval(delta),
    }


def topic_statistics(project, inputs, rows):
    sys.path.insert(0, str(project))
    from laptop.classify_problems import MIN_MARGIN, MIN_SCORE, TOPICS, scores

    topics = {}
    for item in inputs:
        if item["language"] != "eng":
            continue
        scored = scores(item["text"])
        first, second = sorted(TOPICS, key=lambda topic: -scored[topic])[:2]
        confident = scored[first] >= MIN_SCORE and scored[first] - scored[second] >= MIN_MARGIN
        topics[item["id"]] = first if confident else "unclassified"
    return grouped([{**row, "topic": topics[row["problem_id"]]} for row in rows], "topic")


def option_order(inputs):
    from copilot.pot import numeric_options

    english_options = [numeric_options(row) for row in inputs if row["language"] == "eng"]
    numeric_sets = [list(options.values()) for options in english_options if options]
    return {"numeric_sets": len(numeric_sets),
            "in_increasing_order": sum(values == sorted(values) for values in numeric_sets)}


def latency(rows, field):
    seconds = [float(row[field]) for row in rows if row[field] != ""]
    return {
        "n": len(seconds),
        "mean": round(statistics.mean(seconds), 2),
        "median": round(statistics.median(seconds), 2),
        "p95": round(statistics.quantiles(seconds, n=20, method="inclusive")[-1], 2),
        "max": max(seconds),
        "over_60s": sum(value > 60 for value in seconds),
    }


def figures(v4, v1, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "#182d39", "axes.labelcolor": "#182d39",
                         "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False,
                         "axes.spines.bottom": False, "axes.axisbelow": True,
                         "xtick.color": "#325260", "ytick.color": "#325260",
                         "figure.facecolor": "#f7faf9", "axes.facecolor": "#f7faf9"})
    teal, gray, navy = "#007d73", "#b1c8d0", "#182d39"
    output.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(output / name, dpi=160, bbox_inches="tight", pad_inches=0.28)
        plt.close(fig)

    first, latest = grouped(v1, "level"), grouped(v4, "level")
    positions = list(range(len(latest)))
    fig, ax = plt.subplots(figsize=(9.2, 4.25), layout="constrained")
    for series, offset, color, label in ((first, -0.18, gray, "V1"), (latest, 0.18, teal, "V4")):
        counts = list(series.values())
        ax.bar([x + offset for x in positions], [row["correct"] / row["n"] for row in counts],
               width=0.34, color=color, label=label)
        for x, row in zip(positions, counts):
            ax.text(x + offset, row["correct"] / row["n"] + 0.017, str(row["correct"]),
                    ha="center", va="bottom", color=navy, fontsize=10)
    ax.set_title("Accuracy by level on the Datathon practice set", loc="left", pad=16)
    ax.set_xticks(positions, [f"{k}\n(n={v['n']})" for k, v in latest.items()])
    ax.set_xlabel("Level and number of rows per model", labelpad=8)
    ax.set_ylim(0, 0.76)
    ax.set_ylabel("Correct answers")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#dae5e5")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    save(fig, "practice-by-level.png")

    routes = grouped(v4, "status")
    keys = ("pot_match", "pot_fallback", "prior")
    labels = ("Code matched an option", "Fallback after code generation", "Direct fallback")
    correct = [routes[k]["correct"] for k in keys]
    totals = [routes[k]["n"] for k in keys]
    fig, ax = plt.subplots(figsize=(8.8, 3.55), layout="constrained")
    ax.barh(labels, correct, color=teal, height=0.5, label="Correct")
    ax.barh(labels, [n - c for n, c in zip(totals, correct)], left=correct,
            color=gray, height=0.5, label="Incorrect")
    for i, (count, total) in enumerate(zip(correct, totals)):
        ax.text(total + 0.8, i, f"{count}/{total}", va="center", color=navy)
    ax.invert_yaxis()
    ax.set_xlim(0, 48)
    ax.set_xlabel("Rows in the V4 practice run")
    ax.set_title("V4 accuracy by answer route", loc="left", pad=15)
    ax.grid(axis="x", color="#dae5e5")
    ax.legend(loc="lower right", frameon=False)
    save(fig, "practice-by-route.png")

    fig, ax = plt.subplots(figsize=(8.8, 3.7), layout="constrained")
    for field, label, color in (("elapsed_seconds", "Laptop · measured", teal),
                                ("arduino_estimated_seconds", "UNO Q · estimated", "#5251a0")):
        seconds = sorted(float(row[field]) for row in v4)
        ax.step([0, *seconds, 60], [0, *(i / len(seconds) for i in range(1, len(seconds) + 1)), 1],
                where="post", color=color, linewidth=2.3, label=label)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Seconds per row")
    ax.set_ylabel("Cumulative share of rows")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(color="#dae5e5")
    ax.set_title("Response times on the V4 practice run", loc="left", pad=15)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "practice-latency.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    v4, _ = scored_rows(project, DEMO_V4)
    v1, _ = scored_rows(project, DEMO_V1)
    text, _ = scored_rows(project, DEMO_TEXT)
    inputs = data_rows(project / DEMO_INPUT)
    for rows in (v4, v1, text):
        matched_truth(rows, inputs, complete=True)

    bilingual = json_file(project / BILINGUAL)
    m3kang = {}
    for split, totals in bilingual["by_split"].items():
        runs = [scored_rows(project, bilingual["runs"][f"{split}-{lang}"]["manifest"])
                for lang in ("cat", "eng")]
        rows = [row for chunk, _ in runs for row in chunk]
        source = [row for row in data_rows(project / f"data/{split}.jsonl")
                  if row["language"] in ("cat", "eng") and row["text_only"] is True]
        assert len(source) == totals["planned_rows"]
        matched_truth(rows, source, complete=totals["status"] == "completed")
        assert len(rows) == totals["rows"] and sum(row["correct"] for row in rows) == totals["correct"]
        m3kang[split] = {
            "status": totals["status"], "planned": len(source), **score(rows),
            "by_level": grouped(rows, "level"), "by_language": grouped(rows, "language"),
            "by_route": grouped(rows, "status"),
            "latency_seconds": latency(rows, "elapsed_seconds"),
            "uno_q_estimated_seconds": latency(rows, "arduino_estimated_seconds"),
        }

    metadata = {
        "demo": {
            "source": DEMO_INPUT,
            "manifests": [DEMO_V4, DEMO_V1, DEMO_TEXT],
            "v4": score(v4), "v1": score(v1), "text_hybrid": score(text),
            "by_level_v4": grouped(v4, "level"), "by_level_v1": grouped(v1, "level"),
            "by_language": grouped(v4, "language"), "by_figure": grouped(v4, "requires_figure"),
            "by_route": grouped(v4, "status"), "by_topic": topic_statistics(project, inputs, v4),
            "english_numeric_option_order": option_order(inputs),
            "predicted_letters": dict(sorted(Counter(row["selected_answer"] for row in v4).items())),
            "correct_letters": dict(sorted(Counter(row["correct_answer"] for row in v4).items())),
            "paired": pair_statistics(v4, v1),
            "latency_seconds": latency(v4, "elapsed_seconds"),
            "uno_q_estimated_seconds": latency(v4, "arduino_estimated_seconds"),
        },
        "m3kang": m3kang,
        "m3kang_manifest": BILINGUAL,
    }
    (DOCS / "assets").mkdir(parents=True, exist_ok=True)
    (DOCS / "assets" / "evidence.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not args.no_figures:
        figures(v4, v1, DOCS / "assets")


if __name__ == "__main__":
    main()
