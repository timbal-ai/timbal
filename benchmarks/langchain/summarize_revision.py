"""Summarize the saved PR #222 fresh-process repeats (all times in microseconds)."""

import argparse
import json
import statistics
from pathlib import Path


def summarize_group(directory, group, count):
    runs = {
        label: [json.loads((directory / f"{label}_{group}_{i}.json").read_text()) for i in range(1, count + 1)]
        for label in ("base", "pr")
    }
    result = {}
    for key in runs["base"][0]["results"]:
        row = {}
        for label in ("base", "pr"):
            entries = [run["results"][key] for run in runs[label]]
            assert all(len(entry["samples_us"]) == 200 for entry in entries)
            medians = [entry["p50_us"] for entry in entries]
            tails = [entry["p95_us"] for entry in entries]
            row[label] = {
                "p50_us": statistics.median(medians),
                "p50_range_us": [min(medians), max(medians)],
                "p95_us": statistics.median(tails),
                "p95_range_us": [min(tails), max(tails)],
            }
        row["change_percent"] = 100 * (row["pr"]["p50_us"] / row["base"]["p50_us"] - 1)
        row["p95_change_percent"] = 100 * (row["pr"]["p95_us"] / row["base"]["p95_us"] - 1)
        result[key] = row
    return result, runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    regression, runs = summarize_group(args.directory, "repeat", 3)
    summary = {
        "method": "Median of fresh-process p50/p95s; 200 Timbal runs per case, 10 warmups. Ranges are process ranges, not confidence intervals.",
        "regression": regression,
        "agent_comparison": {},
    }
    for scenario in (1, 2, 3):
        key = f"agent_{scenario}"
        timbal = regression[key]["pr"]["p50_us"]
        row = {"timbal_p50_us": timbal}
        for api in ("legacy", "current"):
            entries = [run["langgraph"][f"{api}_{key}"] for run in runs["pr"]]
            assert all(len(entry["samples_us"]) == 100 for entry in entries)
            medians = [entry["p50_us"] for entry in entries]
            row[api] = {
                "p50_us": statistics.median(medians),
                "p50_range_us": [min(medians), max(medians)],
                "ratio_vs_timbal": statistics.median(medians) / timbal,
            }
        summary["agent_comparison"][key] = row
    for group, output in (("targeted", "focused_regression"), ("tail", "tail_regression")):
        if (args.directory / f"base_{group}_1.json").exists():
            summary[output], _ = summarize_group(args.directory, group, 4)
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
