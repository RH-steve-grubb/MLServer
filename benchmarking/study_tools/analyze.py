"""Summarize whole paired-trial ratios; never convert missing trials into wins."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def interval(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values)
    rng = np.random.default_rng(20261027)
    medians = np.median(rng.choice(array, (20000, len(array)), replace=True), axis=1)
    return {
        "pairs": len(values),
        "ratios": values,
        "median": float(np.median(array)),
        "ci95": list(map(float, np.percentile(medians, [2.5, 97.5]))),
    }


def analyze(summaries: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    grouped: dict[tuple, dict[str, dict]] = {}
    invalid = []
    for summary in summaries:
        for row in summary["results"]:
            key = (row["session"], row["round"], row["concurrency"])
            if row["variant"] in grouped.setdefault(key, {}):
                raise ValueError(f"Duplicate trial identity: {key}, {row['variant']}")
            grouped[key][row["variant"]] = row
            if not row["valid"]:
                invalid.append(
                    {"pair": key, "variant": row["variant"], "case": row["case"]}
                )
    ratios: dict[int, dict[str, list[float]]] = {}
    missing = []
    for key, rows in sorted(grouped.items()):
        if any(name not in rows or not rows[name]["valid"] for name in names):
            missing.append(key)
            continue
        control, candidate = (rows[name] for name in names)
        metrics = ratios.setdefault(
            key[2], {"rps": [], "p95": [], "p99": [], "cpu": []}
        )
        metrics["rps"].append(candidate["rps"] / control["rps"])
        for metric in ("p95", "p99"):
            metrics[metric].append(
                candidate["latency_ms"][metric] / control["latency_ms"][metric]
            )
        if control.get("cpu_ms_per_request_envelope") and candidate.get(
            "cpu_ms_per_request_envelope"
        ):
            metrics["cpu"].append(
                candidate["cpu_ms_per_request_envelope"]
                / control["cpu_ms_per_request_envelope"]
            )
    return {
        "ratio_direction": "candidate/control",
        "names": names,
        "invalid_trials": invalid,
        "missing_or_invalid_pairs": missing,
        "metrics": {
            str(c): {
                name: interval(values) for name, values in metrics.items() if values
            }
            for c, metrics in ratios.items()
        },
        "acceptance": (
            "NOT AUTOMATED: audit provenance, counts, sessions, errors, "
            "CPU windows and all predeclared criteria."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        [json.loads(path.read_text()) for path in args.summary],
        [args.control, args.candidate],
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
