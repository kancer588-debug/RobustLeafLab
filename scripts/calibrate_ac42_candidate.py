"""Fit class score offsets on the existing development sets and replay a locked rule."""

import csv
import itertools
import math

import numpy as np
import torch
from torch import nn

from run_backbone_trial import FIRST_RUN, development, evaluate_comparison
from run_hierarchical_trial import experiment_policy, new_model
from train_online_candidate import WORKSPACE, checked_checkpoint, read_json, write_json


RUN = WORKSPACE / "outputs" / "ac42_score_calibration"
OFFSETS = (-0.75, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 0.75)


class CalibratedClassifier(nn.Module):
    def __init__(self, base, offsets):
        super().__init__()
        self.base = base
        self.register_buffer("offsets", torch.as_tensor(offsets, dtype=torch.float32))

    def forward(self, images):
        return self.base(images) + self.offsets


def metrics_for(logits, actual, offsets, malik_count):
    predictions = (logits + np.asarray(offsets, dtype=logits.dtype)).argmax(axis=1)
    cells = np.bincount(3 * actual + predictions, minlength=9).reshape(3, 3)
    totals = cells.sum(axis=1) + cells.sum(axis=0)
    f1 = 2 * cells.diagonal() / totals
    malik_true = actual[-malik_count:]
    malik_prediction = predictions[-malik_count:]
    return {
        "early_f1": float(f1[1]), "macro_f1": float(f1.mean()),
        "late_to_healthy": int(cells[2, 0]),
        "healthy_false_alarms": int(cells[0, 1:].sum()),
        "healthy_n": int(cells[0].sum()),
        "malik_early_correct": int(((malik_true == 1) & (malik_prediction == 1)).sum()),
        "malik_late_to_healthy": int(((malik_true == 2) & (malik_prediction == 0)).sum()),
    }


def main():
    frozen, partitions, _ = experiment_policy()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent_sha = read_json(FIRST_RUN / "selection.json")["candidate_sha256"]
    parent = checked_checkpoint(FIRST_RUN / "candidate.pt", parent_sha)
    base = new_model("flat", parent).to(device, memory_format=torch.channels_last).eval()
    batches = development(frozen, device)
    with torch.inference_mode():
        scores = torch.cat([base(images).cpu() for images, _ in batches]).numpy()
    actual = torch.cat([labels for _, labels in batches]).numpy()
    del batches
    malik_count = sum(frozen["counts"]["development_malik"].values())
    reference = metrics_for(scores, actual, (0.0, 0.0, 0.0), malik_count)
    healthy_limit = reference["healthy_false_alarms"] + math.floor(0.01 * reference["healthy_n"])
    grid = []
    for early, late in itertools.product(OFFSETS, repeat=2):
        offsets = (0.0, early, late)
        values = metrics_for(scores, actual, offsets, malik_count)
        eligible = (
            values["late_to_healthy"] <= reference["late_to_healthy"]
            and values["healthy_false_alarms"] <= healthy_limit
            and values["malik_early_correct"] >= reference["malik_early_correct"]
            and values["malik_late_to_healthy"] <= reference["malik_late_to_healthy"]
            and values["macro_f1"] >= reference["macro_f1"]
            and values["early_f1"] > reference["early_f1"]
        )
        grid.append({"early_offset": early, "late_offset": late, **values, "eligible": eligible})
    best = max((row for row in grid if row["eligible"]), key=lambda row: (
        row["early_f1"], row["macro_f1"], -abs(row["early_offset"]) - abs(row["late_offset"]),
    ), default=None)
    RUN.mkdir(parents=True, exist_ok=False)
    with (RUN / "development_search.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=grid[0])
        writer.writeheader()
        writer.writerows(grid)
    decision = {
        "parent_sha256": parent_sha, "tested_offsets": OFFSETS, "starting_development": reference,
        "healthy_false_alarm_limit": healthy_limit, "selected_on_development": best,
        "selection": "Highest early F1 among candidates that improve early F1, keep macro F1, "
                     "and meet pooled and Malik error limits",
    }
    write_json(RUN / "calibration.json", decision)
    print("Calibration decision:", decision, flush=True)
    if best is not None:
        calibrated = CalibratedClassifier(
            base, [0.0, best["early_offset"], best["late_offset"]]
        ).to(device)
        entries = {"starting_candidate": base, "calibrated_candidate": calibrated}
        for name in ("plantwild_replay", "plantcity_replay"):
            evaluate_comparison(partitions[name], RUN, name, entries, device)


if __name__ == "__main__":
    main()
