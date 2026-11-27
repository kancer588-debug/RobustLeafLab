"""Train an AC42 derivative on public data and evaluate it after checkpoint selection."""

import argparse
import os
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader


WORKSPACE = Path(__file__).resolve().parents[1]
REFERENCE = Path(os.environ.get("ROBUSTLEAFLAB_REFERENCE", WORKSPACE / "reference"))
SPLITS = WORKSPACE / "datasets" / "independent_test"
RUN = WORKSPACE / "outputs" / "independent_test" / "ac42_public_s42"
SEED = 42
EPOCHS = 4
BATCH_SIZE = 32
LR_HEAD = 1e-4
LR_BLOCK = 1e-5

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/reference_source/src"))
from common import CLASSES, ManifestDataset, create_model, predict, seed_everything

ORIGINAL_CHECKPOINTS = {
    "AC42": ("unified_ac_s42/best.pt",
             "9755d64cc689ffa2f8bd0d8b0c99ec935fbcd6f5112f474be64d10b1261c3400"),
    "C42": ("pad_aspect/bg_replace_pad_s42/best.pt",
            "66c0164a1c36228646352497213634ed9fc4f9ec640c4b5f27a930a8ee2b2a70"),
}


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    with path.open("x", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def checked_checkpoint(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["model_name"], tuple(checkpoint["classes"])) != ("resnet18", CLASSES):
        raise ValueError(f"Unexpected model architecture or classes: {path}")
    return checkpoint


def original_checkpoint(name):
    path, digest = ORIGINAL_CHECKPOINTS[name]
    return checked_checkpoint(REFERENCE / "outputs/train_improve_late" / path, digest)


def manifest(name, freeze):
    path = SPLITS / f"{name}.csv"
    if sha256(path) != freeze["manifest_sha256"][name]:
        raise ValueError(f"Manifest changed after freeze: {path}")
    return path


def measure(model, loader, device):
    truth, predictions, probabilities, loss = predict(model, loader, device)
    matrix = confusion_matrix(truth, predictions, labels=range(len(CLASSES)))
    healthy, early, late = (int(value) for value in matrix.sum(axis=1))
    metrics = {
        "n": len(truth),
        "accuracy": float((truth == predictions).mean()),
        "macro_f1": float(f1_score(truth, predictions, labels=range(len(CLASSES)), average="macro", zero_division=0)),
        "cross_entropy": loss,
        "healthy_recall": int(matrix[0, 0]) / healthy,
        "early_recall": int(matrix[1, 1]) / early,
        "late_recall": int(matrix[2, 2]) / late,
        "healthy_false_alarm_count": int(matrix[0, 1:].sum()),
        "healthy_false_alarm_rate": int(matrix[0, 1:].sum()) / healthy,
        "late_to_healthy": int(matrix[2, 0]),
        "late_to_early": int(matrix[2, 1]),
        "class_counts": dict(zip(CLASSES, (healthy, early, late))),
        "confusion": matrix.tolist(),
    }
    return metrics, truth, predictions, probabilities


def train_epoch(model, loader, optimizer, device):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    loss_total = 0.0
    images_seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(images), labels)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise ValueError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        loss_total += loss_value * labels.numel()
        images_seen += labels.numel()
    return loss_total / images_seen


def make_loader(dataset, *, shuffle=False):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def train(freeze, device):
    fit_path = manifest("train", freeze)
    validation_path = manifest("validation", freeze)
    parent = original_checkpoint("AC42")
    seed_everything(SEED)
    model = create_model("resnet18")
    model.load_state_dict(parent["model_state"])
    model.requires_grad_(False)
    model.layer4[-1].requires_grad_(True)
    model.fc.requires_grad_(True)
    model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.fc.parameters(), "lr": LR_HEAD},
        {"params": model.layer4[-1].parameters(), "lr": LR_BLOCK},
    ], weight_decay=1e-4)
    training = make_loader(ManifestDataset(fit_path, train=True), shuffle=True)
    validation = make_loader(ManifestDataset(validation_path))
    RUN.mkdir(parents=True, exist_ok=False)
    config = {
        "train_manifest_sha256": freeze["manifest_sha256"]["train"],
        "validation_manifest_sha256": freeze["manifest_sha256"]["validation"],
        "parent_sha256": ORIGINAL_CHECKPOINTS["AC42"][1],
        "device": str(device), "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "trainable_modules": ["layer4.1", "fc"], "batchnorm_running_stats_frozen": True,
        "lr_head": LR_HEAD, "lr_block": LR_BLOCK, "eval_view": "center",
        "selection": "Highest validation macro-F1; ties retain the earliest epoch",
    }
    write_json(RUN / "config.json", config)
    baseline, *_ = measure(model, validation, device)
    write_json(RUN / "validation_ac42.json", baseline)
    print("Original AC42 validation macro-F1:", baseline["macro_f1"], flush=True)
    best_score = -1.0
    best_epoch = best_state = best_metrics = None
    with (RUN / "history.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("epoch", "train_loss", "validation_macro_f1",
                                                  "validation_accuracy", "validation_late_recall"))
        writer.writeheader()
        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, training, optimizer, device)
            metrics, *_ = measure(model, validation, device)
            score = metrics["macro_f1"]
            record = {"epoch": epoch, "train_loss": loss, "validation_macro_f1": score,
                      "validation_accuracy": metrics["accuracy"], "validation_late_recall": metrics["late_recall"]}
            writer.writerow(record)
            file.flush()
            print(json.dumps(record), flush=True)
            if score > best_score:
                best_score, best_epoch, best_metrics = score, epoch, metrics
                best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    trainable = {key for key, tensor in model.named_parameters() if tensor.requires_grad}
    if any(not torch.equal(tensor, parent["model_state"][key])
           for key, tensor in best_state.items() if key not in trainable):
        raise ValueError("A frozen AC42 tensor changed during fine-tuning")
    artifact = RUN / "candidate.pt"
    with artifact.open("xb") as file:
        torch.save({"model_name": "resnet18", "classes": CLASSES, "model_state": best_state,
                    "seed": SEED, "epoch": best_epoch, "pretrained": True, "bg_prob": 0.0,
                    "init_checkpoint_sha256": config["parent_sha256"], "eval_view": "center"}, file)
    selection = {"epoch": best_epoch, "candidate_sha256": sha256(artifact),
                 "validation_metrics": best_metrics, "selection_rule": config["selection"]}
    write_json(RUN / "selection.json", selection)
    print("Selected checkpoint:", best_epoch, selection["candidate_sha256"], flush=True)


def test(freeze, device):
    selection = read_json(RUN / "selection.json")
    candidate = checked_checkpoint(RUN / "candidate.pt", selection["candidate_sha256"])
    checkpoints = {"AC42": original_checkpoint("AC42"), "C42": original_checkpoint("C42"),
                   "candidate": candidate}
    dataset = ManifestDataset(manifest("test", freeze))
    loader = make_loader(dataset)
    model = create_model("resnet18").to(device)
    predictions = []
    results = {"manifest_sha256": freeze["manifest_sha256"]["test"],
               "candidate_sha256": selection["candidate_sha256"], "models": {}}
    for name, checkpoint in checkpoints.items():
        model.load_state_dict(checkpoint["model_state"])
        metrics, truth, guesses, probabilities = measure(model, loader, device)
        results["models"][name] = metrics
        for row, actual, guess, probability in zip(dataset.rows, truth, guesses, probabilities):
            predictions.append({"model": name, "sha256": row["sha256"], "origin": row["origin"],
                                "true_label": CLASSES[actual], "prediction": CLASSES[guess],
                                "confidence": float(probability[guess])})
        print(name, json.dumps(metrics), flush=True)
    with (RUN / "test_predictions.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=predictions[0])
        writer.writeheader()
        writer.writerows(predictions)
    write_json(RUN / "test_results.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "test"))
    args = parser.parse_args()
    freeze = read_json(SPLITS / "freeze.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.phase == "train":
        train(freeze, device)
    else:
        test(freeze, device)


if __name__ == "__main__":
    main()
