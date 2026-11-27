"""Run the matched AC42 early-blight source experiment with a locked field test."""

import argparse
import csv
import json
from collections import Counter

import torch
from torch.utils.data import ConcatDataset, DataLoader

from run_backbone_trial import (
    BATCH_SIZE, FIRST_RUN, TRIAL_SPLITS, add_class_f1, evaluate_comparison,
    loader, manifest, move_images, read_rows,
)
from run_hierarchical_trial import BLOCK_LR, EPOCHS, HEAD_LR, allowed, new_model, run_epoch
from train_online_candidate import (
    CLASSES, WORKSPACE, ManifestDataset, checked_checkpoint, measure, original_checkpoint,
    read_json, seed_everything, sha256, write_json,
)


SPLITS = WORKSPACE / "datasets" / "early_source_trial"
OUTPUT = WORKSPACE / "outputs" / "early_source_trial"
ARMS = ("control", "replacement")
SEEDS = (42, 43, 44)
EARLY = "early_blight"


def verified_manifests(frozen):
    paths = {name: SPLITS / f"{name}.csv" for name in (*ARMS, "fresh_test")}
    for name, path in paths.items():
        if sha256(path) != frozen["sha256"][name]:
            raise ValueError(f"Manifest changed: {path}")
    for name in ARMS:
        rows = read_rows(paths[name])
        if Counter(row["label"] for row in rows) != frozen["training_counts"]:
            raise ValueError(f"Training class counts changed: {name}")
    return paths


def new_flat(parent, device):
    return new_model("flat", parent).to(device, memory_format=torch.channels_last)


def cached_batches(dataset, device):
    return [(move_images(images, device), labels) for images, labels in loader(dataset)]


def development_metrics(model, pooled, malik, device):
    metrics = add_class_f1(measure(model, pooled, device)[0])
    metrics["malik"] = add_class_f1(measure(model, malik, device)[0])
    return metrics


def meets_limits(metrics, limits):
    return (
        allowed(metrics, limits)
        and metrics["malik"]["confusion"][1][1] >= limits["malik_early_correct"]
        and metrics["malik"]["late_to_healthy"] <= limits["malik_late_to_healthy"]
    )


def train_arm(name, seed, parent, fit_manifest, pooled, malik, limits, device):
    seed_everything(seed)
    model = new_flat(parent, device)
    model.requires_grad_(False)
    block = model.layer4[-1]
    block.requires_grad_(True)
    model.fc.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {"params": model.fc.parameters(), "lr": HEAD_LR},
        {"params": block.parameters(), "lr": BLOCK_LR},
    ], weight_decay=1e-4)
    training = DataLoader(
        ManifestDataset(fit_manifest, train=True), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, generator=torch.Generator().manual_seed(seed),
    )
    location = OUTPUT / f"{name}_s{seed}"
    location.mkdir()
    selected = None
    best_rank = None
    history = ("epoch", "train_loss", "early_f1", "macro_f1", "late_recall", "late_to_healthy",
               "healthy_false_alarms", "malik_early_correct", "malik_late_to_healthy", "eligible")
    with (location / "history.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history)
        writer.writeheader()
        for epoch in range(1, EPOCHS + 1):
            loss = run_epoch(model, training, optimizer, device, None)
            metrics = development_metrics(model, pooled, malik, device)
            eligible = meets_limits(metrics, limits)
            early_f1 = metrics["class_f1"][EARLY]
            record = dict(
                epoch=epoch, train_loss=loss, early_f1=early_f1, macro_f1=metrics["macro_f1"],
                late_recall=metrics["late_recall"], late_to_healthy=metrics["late_to_healthy"],
                healthy_false_alarms=metrics["healthy_false_alarm_count"],
                malik_early_correct=metrics["malik"]["confusion"][1][1],
                malik_late_to_healthy=metrics["malik"]["late_to_healthy"], eligible=eligible,
            )
            writer.writerow(record)
            stream.flush()
            print(name, seed, json.dumps(record), flush=True)
            rank = eligible, early_f1, metrics["macro_f1"]
            if best_rank is None or rank > best_rank:
                best_rank = rank
                temporary = location / "best.tmp"
                torch.save({
                    "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "model_name": "resnet18", "classes": CLASSES, "epoch": epoch, "seed": seed,
                }, temporary)
                temporary.replace(location / "best.pt")
                selected = {"epoch": epoch, "metrics": metrics, "eligible": eligible}
    selected["checkpoint_sha256"] = sha256(location / "best.pt")
    write_json(location / "selection.json", selected)
    return selected


def train(parent, paths, frozen, device):
    development = read_json(TRIAL_SPLITS / "freeze.json")
    manifests = {key: manifest(key, development) for key in ("development_field", "development_malik")}
    groups = [ManifestDataset(path) for path in manifests.values()]
    pooled = cached_batches(ConcatDataset(groups), device)
    malik = cached_batches(groups[1], device)
    start = development_metrics(new_flat(parent, device), pooled, malik, device)
    limits = {
        "late_to_healthy": start["late_to_healthy"],
        "healthy_false_alarm_count": start["healthy_false_alarm_count"]
        + start["class_counts"]["healthy"] // 100,
        "malik_early_correct": start["malik"]["confusion"][1][1],
        "malik_late_to_healthy": start["malik"]["late_to_healthy"],
    }
    OUTPUT.mkdir(exist_ok=False)
    write_json(OUTPUT / "protocol.json", {
        "parent_sha256": parent_digest(), "seeds": SEEDS, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "optimizer": {"name": "AdamW", "head_lr": HEAD_LR, "block_lr": BLOCK_LR, "weight_decay": 1e-4},
        "trainable": ["layer4.1", "fc"], "frozen_batchnorm_running_stats": True,
        "frozen_manifests": frozen, "development_sha256": {k: sha256(p) for k, p in manifests.items()},
        "starting_development": start, "guardrails": limits,
        "selection": "Eligibility, then early F1, then macro F1; earliest epoch on ties",
        "promotion": "Seed 42 is primary. Require replacement to beat both starting and matched control "
                     "on development early F1, preserve macro F1 versus starting, and pass all limits "
                     "for seed 42 and at least one of seeds 43 or 44.",
        "fresh_test_policy": "Evaluate all locked checkpoints once after saving the development decision.",
    })
    print("Development guardrails", json.dumps(limits), flush=True)
    selections = {}
    for seed in SEEDS:
        for name in ARMS:
            selections[f"{name}_s{seed}"] = train_arm(
                name, seed, parent, paths[name], pooled, malik, limits, device
            )
    passes = {}
    for seed in SEEDS:
        control = selections[f"control_s{seed}"]["metrics"]
        replacement = selections[f"replacement_s{seed}"]
        metrics = replacement["metrics"]
        passes[seed] = (
            replacement["eligible"]
            and metrics["class_f1"][EARLY] > max(start["class_f1"][EARLY], control["class_f1"][EARLY])
            and metrics["macro_f1"] >= start["macro_f1"]
        )
    preferred = "replacement_s42" if passes[42] and (passes[43] or passes[44]) else "starting_candidate"
    write_json(OUTPUT / "decision.json", {
        "preferred_by_development": preferred, "promotion_passes": passes, "selections": selections,
    })


def parent_digest():
    return read_json(FIRST_RUN / "selection.json")["candidate_sha256"]


def test(parent, paths, device):
    decision = read_json(OUTPUT / "decision.json")
    protocol = read_json(OUTPUT / "protocol.json")
    if sha256(paths["fresh_test"]) != protocol["frozen_manifests"]["sha256"]["fresh_test"]:
        raise ValueError("Fresh test manifest differs from the pre-training protocol")
    entries = {"original_AC42": new_flat(original_checkpoint("AC42"), device),
               "starting_candidate": new_flat(parent, device)}
    for key, selection in decision["selections"].items():
        artifact = OUTPUT / key / "best.pt"
        entries[key] = new_flat(checked_checkpoint(artifact, selection["checkpoint_sha256"]), device)
    evaluate_comparison(paths["fresh_test"], OUTPUT, "imtiaz_fresh", entries, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "test", "all"), default="all", nargs="?")
    options = parser.parse_args()
    frozen = read_json(SPLITS / "freeze.json")
    paths = verified_manifests(frozen)
    parent = checked_checkpoint(FIRST_RUN / "candidate.pt", parent_digest())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    if options.phase in ("train", "all"):
        train(parent, paths, frozen, device)
    if options.phase in ("test", "all"):
        test(parent, paths, device)


if __name__ == "__main__":
    main()
