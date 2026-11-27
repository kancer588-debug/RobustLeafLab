"""Compare training-set sizes at a matched optimization budget and evaluate locked weights."""

import argparse
import csv
import json
from itertools import chain, islice, repeat

import torch
from torch.utils.data import ConcatDataset, DataLoader

from prepare_expanded_field_trial import TRIAL, excluded_evaluation
from run_backbone_trial import BATCH_SIZE, FIRST_RUN, evaluate_comparison
from run_early_source_trial import cached_batches, development_metrics, meets_limits, new_flat
from train_online_candidate import (
    CLASSES, ManifestDataset, WORKSPACE, checked_checkpoint, original_checkpoint,
    read_json, seed_everything, sha256, train_epoch, write_json,
)


OUTPUT = WORKSPACE / "outputs" / "expanded_field_trial"
ARMS = ("control", "expanded")
SEEDS = (42, 43, 44)
ROUNDS = 4
HEAD_LR = 5e-5
BLOCK_LR = 5e-6
LATE_RECALL_TOLERANCE = 0.02


def verified(path, frozen_hash):
    if sha256(path) != frozen_hash:
        raise ValueError(f"Frozen data have changed: {path}")
    return path


def score(metrics):
    return (metrics["macro_f1"] + metrics["malik"]["macro_f1"]) / 2


def eligible(metrics, limits):
    return meets_limits(metrics, limits) and metrics["confusion"][2][2] >= limits["minimum_late_correct"]


def set_limits(start):
    healthy = start["class_counts"]["healthy"]
    late = start["class_counts"]["late_blight"]
    return {
        "late_to_healthy": start["late_to_healthy"],
        "healthy_false_alarm_count": start["healthy_false_alarm_count"] + healthy // 100,
        "minimum_late_correct": start["confusion"][2][2] - int(late * LATE_RECALL_TOLERANCE),
        "malik_early_correct": start["malik"]["confusion"][1][1],
        "malik_late_to_healthy": start["malik"]["late_to_healthy"],
    }


def snapshots(model, averaged, parent):
    current = model.state_dict()
    starting = parent["model_state"]
    interpolated = {
        name: starting[name].to(value.device).lerp(value, 0.5) if name.startswith(("fc.", "layer4.1."))
        and value.is_floating_point() else value
        for name, value in current.items()
    }
    averaged.load_state_dict(interpolated)
    return (("direct", model), ("midpoint", averaged))


def train_arm(arm, seed, dataset, total_steps, parent, pooled, malik, limits, device):
    seed_everything(seed)
    model = new_flat(parent, device)
    average = new_flat(parent, device)
    model.requires_grad_(False)
    block = model.layer4[-1]
    block.requires_grad_(True)
    model.fc.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {"params": model.fc.parameters(), "lr": HEAD_LR},
        {"params": block.parameters(), "lr": BLOCK_LR},
    ], weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    directory = OUTPUT / f"{arm}_s{seed}"
    directory.mkdir()
    history_fields = ("round", "weights", "train_loss", "development_score", "macro_f1", "early_f1",
                      "late_correct", "late_to_healthy", "healthy_false_alarms", "malik_macro_f1", "eligible")
    selection = None
    best_rank = None
    with (directory / "history.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history_fields)
        writer.writeheader()
        for round_no in range(1, ROUNDS + 1):
            fit_batches = islice(chain.from_iterable(repeat(loader)), total_steps)
            loss = train_epoch(model, fit_batches, optimizer, device)
            for weights, candidate in snapshots(model, average, parent):
                metrics = development_metrics(candidate, pooled, malik, device)
                record = {
                    "round": round_no, "weights": weights, "train_loss": loss,
                    "development_score": score(metrics), "macro_f1": metrics["macro_f1"],
                    "early_f1": metrics["class_f1"]["early_blight"], "late_correct": metrics["confusion"][2][2],
                    "late_to_healthy": metrics["late_to_healthy"],
                    "healthy_false_alarms": metrics["healthy_false_alarm_count"],
                    "malik_macro_f1": metrics["malik"]["macro_f1"], "eligible": eligible(metrics, limits),
                }
                writer.writerow(record)
                stream.flush()
                print(arm, seed, json.dumps(record), flush=True)
                rank = record["eligible"], record["development_score"], record["early_f1"]
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    temp = directory / "best.tmp"
                    torch.save({
                        "model_state": {key: value.detach().cpu() for key, value in candidate.state_dict().items()},
                        "model_name": "resnet18", "classes": CLASSES,
                        "round": round_no, "seed": seed, "weights": weights,
                    }, temp)
                    temp.replace(directory / "best.pt")
                    selection = {"record": record, "metrics": metrics}
    selection["checkpoint_sha256"] = sha256(directory / "best.pt")
    write_json(directory / "selection.json", selection)
    return selection


def train(parent, parent_hash, training, evaluation, device):
    freeze = training["training_sha256"]
    sets = {arm: ManifestDataset(verified(TRIAL / f"{arm}_train.csv", freeze[f"{arm}_train"]), train=True)
            for arm in ARMS}
    references = excluded_evaluation()
    manifests = {name: verified(references[name], training["excluded_manifest_sha256"][str(references[name])])
                 for name in ("development_field", "development_malik")}
    groups = [ManifestDataset(path) for path in manifests.values()]
    pooled = cached_batches(ConcatDataset(groups), device)
    malik = cached_batches(groups[1], device)
    start = development_metrics(new_flat(parent, device), pooled, malik, device)
    limits = set_limits(start)
    steps = len(sets["expanded"]) // BATCH_SIZE
    OUTPUT.mkdir(exist_ok=False)
    write_json(OUTPUT / "protocol.json", {
        "parent_sha256": parent_hash, "device": str(device), "seeds": SEEDS, "rounds": ROUNDS,
        "batch_size": BATCH_SIZE, "steps_per_round": steps, "learning_rates": [HEAD_LR, BLOCK_LR],
        "training_freeze_sha256": sha256(TRIAL / "training_freeze.json"),
        "evaluation_freeze_sha256": sha256(TRIAL / "evaluation_freeze.json"),
        "training": training, "evaluation": evaluation, "starting_development": start, "limits": limits,
        "checkpoint_order": "Guardrails, mean of pooled and Malik Macro-F1, then pooled early F1",
        "promotion": "An expanded model must beat its matched control and starting development score, "
                     "preserve starting pooled Macro-F1, and meet every guardrail. Seed 42 and at least "
                     "one confirmatory seed must pass. New and replay evaluation follows checkpoint lock.",
        "target": "80% accuracy on the frozen new-image evaluation, reported with recalls and Macro-F1.",
        "audit_policy": "New field and PlantDoc are evaluated only after the decision is written.",
    })
    print("Guardrails:", json.dumps(limits), "matched updates per round:", steps, flush=True)
    selections = {}
    for seed in SEEDS:
        for arm in ARMS:
            selections[f"{arm}_s{seed}"] = train_arm(
                arm, seed, sets[arm], steps, parent, pooled, malik, limits, device
            )
    checks = {}
    for seed in SEEDS:
        expanded = selections[f"expanded_s{seed}"]
        control = selections[f"control_s{seed}"]
        metrics = expanded["metrics"]
        checks[seed] = bool(expanded["record"]["eligible"] and
                            score(metrics) > max(score(start), score(control["metrics"])) and
                            metrics["macro_f1"] >= start["macro_f1"])
    preferred = "expanded_s42" if checks[42] and any(checks[seed] for seed in SEEDS[1:]) \
        else "starting_candidate"
    write_json(OUTPUT / "decision.json", {
        "preferred_by_development": preferred, "promotion_passes": checks, "selections": selections,
    })


def evaluate(parent, frozen, device):
    protocol = read_json(OUTPUT / "protocol.json")
    verified(TRIAL / "training_freeze.json", protocol["training_freeze_sha256"])
    verified(TRIAL / "evaluation_freeze.json", protocol["evaluation_freeze_sha256"])
    decision = read_json(OUTPUT / "decision.json")
    entries = {"original_AC42": new_flat(original_checkpoint("AC42"), device),
               "starting_candidate": new_flat(parent, device)}
    for name, selection in decision["selections"].items():
        entries[name] = new_flat(checked_checkpoint(
            OUTPUT / name / "best.pt", selection["checkpoint_sha256"]), device)
    cohort = verified(TRIAL / "rachin_evaluation.csv", frozen["sha256"])
    evaluate_comparison(cohort, OUTPUT, "rachin_new_images", entries, device)
    main_entries = {name: entries[name] for name in
                    ("original_AC42", "starting_candidate", "control_s42", "expanded_s42")}
    for name, item in (frozen["historical_replays"] | frozen["external_audits"]).items():
        path = verified(WORKSPACE / item["path"], item["sha256"])
        evaluate_comparison(path, OUTPUT, name, main_entries, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("all", "train", "evaluate"), nargs="?", default="all")
    args = parser.parse_args()
    training = read_json(TRIAL / "training_freeze.json")
    evaluation = read_json(TRIAL / "evaluation_freeze.json")
    verified(TRIAL / "rachin_evaluation.csv", evaluation["sha256"])
    parent_hash = read_json(FIRST_RUN / "selection.json")["candidate_sha256"]
    parent = checked_checkpoint(FIRST_RUN / "candidate.pt", parent_hash)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    if args.phase in ("all", "train"):
        train(parent, parent_hash, training, evaluation, device)
    if args.phase in ("all", "evaluate"):
        evaluate(parent, evaluation, device)


if __name__ == "__main__":
    main()
