"""Compute paired accuracy changes and exact McNemar tests from archived predictions."""

import argparse
import csv
import json
from math import comb
from pathlib import Path

import numpy as np


def model_rows(path, *models):
    result = {model: {} for model in models}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            if model not in result:
                continue
            key = row["sha256"]
            if key in result[model]:
                raise ValueError(f"Duplicate image ID for {model}: {key}")
            result[model][key] = (row["true_label"], row["prediction"])
    if any(not rows for rows in result.values()):
        raise ValueError(f"Missing predictions in {path}")
    return tuple(result[model] for model in models)


def compare(before, after, *, rng, resamples):
    if before.keys() != after.keys():
        raise ValueError("Prediction files must cover the same image identities")
    fixed = broken = correct_before = correct_after = 0
    for key, (truth, previous) in before.items():
        later_truth, following = after[key]
        if truth != later_truth:
            raise ValueError(f"True labels differ for {key}")
        was_correct, is_correct = previous == truth, following == truth
        fixed += is_correct and not was_correct
        broken += was_correct and not is_correct
        correct_before += was_correct
        correct_after += is_correct
    size = len(before)
    different = fixed + broken
    probability = min(1.0, 2 * sum(comb(different, i) for i in range(min(fixed, broken) + 1)) / 2**different)
    # Resampling the empirical paired gains (-1, 0, +1) is equivalent to drawing multinomial counts.
    draws = rng.multinomial(size, [broken / size, (size - different) / size, fixed / size], size=resamples)
    deltas = 100 * (draws[:, 2] - draws[:, 0]) / size
    return {"images": size, "before_correct": correct_before, "after_correct": correct_after,
            "corrected": fixed, "regressed": broken, "discordant": different, "delta_pp": 100 * (fixed - broken) / size,
            "paired_image_bootstrap_95_ci_pp": np.percentile(deltas, [2.5, 97.5]).tolist(), "mcnemar_exact_p": probability}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Directory with *_predictions.csv outputs")
    parser.add_argument("--audit", nargs="+", required=True, help="Cohorts to report separately")
    parser.add_argument("--historical", nargs="+", required=True, help="Cohorts to pool as the historical comparison")
    parser.add_argument("--before", default="incumbent")
    parser.add_argument("--after", default="directional_majority")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=5_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.resamples <= 0 or args.before == args.after:
        parser.error("Specify distinct models and a positive bootstrap resample count")
    rng = np.random.default_rng(args.seed)
    historical_before, historical_after = {}, {}
    for name in args.historical:
        before, after = model_rows(args.archive / f"{name}_predictions.csv", args.before, args.after)
        historical_before.update({(name, key): value for key, value in before.items()})
        historical_after.update({(name, key): value for key, value in after.items()})
    historical = compare(historical_before, historical_after, rng=rng, resamples=args.resamples)
    audits = {}
    for name in args.audit:
        before, after = model_rows(args.archive / f"{name}_predictions.csv", args.before, args.after)
        audits[name] = compare(before, after, rng=rng, resamples=args.resamples)
    result = {"before": args.before, "after": args.after, "seed": args.seed, "resamples": args.resamples,
              "interval_scope": "Image-level paired 95% percentile intervals conditional on each archived cohort. "
                                "They do not estimate capture-group or training-seed variation.",
              "historical_pool": historical, "audits": audits}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
