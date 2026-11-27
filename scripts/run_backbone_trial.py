"""Compare ImageNet ResNet18 and ConvNeXt-Tiny on one frozen public-data protocol."""

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from itertools import combinations

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import models

from train_online_candidate import (
    CLASSES, REFERENCE, WORKSPACE, ManifestDataset, checked_checkpoint, create_model,
    measure, original_checkpoint, read_json, seed_everything, sha256, write_json,
)


FIRST_SPLITS = WORKSPACE / "datasets" / "independent_test"
FIRST_RUN = WORKSPACE / "outputs" / "independent_test" / "ac42_public_s42"
TRIAL_SPLITS = WORKSPACE / "datasets" / "backbone_trial"
RUN = WORKSPACE / "outputs" / "backbone_trial"
CONTINUATION = "convnext_continue_s42"
WEIGHTS = WORKSPACE / "datasets" / "backbone_weights"
FOLD = REFERENCE / "outputs" / "train_improve_late" / "folds" / "field_extra_merged"
ARCHITECTURES = {
    "resnet18": ("resnet18-f37072fd.pth", "f37072fd"),
    "convnext_tiny": ("convnext_tiny-983f1562.pth", "983f1562"),
}
SEED = 42
EPOCHS = 4
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
HEAD_LR = 1e-3
BACKBONE_LR = 1e-4
CONTINUATION_LR_FACTOR = 0.25


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def freeze():
    TRIAL_SPLITS.mkdir(parents=True, exist_ok=True)
    frozen = TRIAL_SPLITS / "freeze.json"
    if frozen.exists():
        raise FileExistsError(f"Trial is already frozen: {frozen}")
    paths = {
        "train": FOLD / "train.csv",
        "development_field": FOLD / "heldout.csv",
        "development_malik": FIRST_SPLITS / "validation.csv",
        "plantcity_historical": FIRST_SPLITS / "test.csv",
    }
    ref_hashes = read_json(FOLD / "fold.json")["fold_manifests_sha256"]
    previous_hashes = read_json(FIRST_SPLITS / "freeze.json")["manifest_sha256"]
    expected = dict(train=ref_hashes["train.csv"], development_field=ref_hashes["heldout.csv"],
                    development_malik=previous_hashes["validation"],
                    plantcity_historical=previous_hashes["test"])
    partitions = {}
    for name, path in paths.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"Archival manifest changed: {path}")
        partitions[name] = read_rows(path)
    hashes = {name: {row["sha256"] for row in rows} for name, rows in partitions.items()}
    counts = {name: dict(Counter(row["label"] for row in rows)) for name, rows in partitions.items()}
    reserved_paths = (REFERENCE / "data_v5/new_field_test.csv",
                      REFERENCE / "data/external_plantdoc.csv")
    reserved = {row["sha256"] for path in reserved_paths for row in read_rows(path)}
    for name, rows in partitions.items():
        if (len(hashes[name]) != len(rows) or set(counts[name]) != set(CLASSES)
                or any(not Path(row["path"]).is_file() for row in rows)
                or hashes[name] & reserved):
            raise ValueError(f"Invalid image identities or coverage: {name}")
    for name, other in combinations(paths, 2):
        if hashes[name] & hashes[other]:
            raise ValueError(f"Overlapping image files: {name}, {other}")
    for name, path in paths.items():
        target = TRIAL_SPLITS / f"{name}.csv"
        with path.open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination)
    policy = {
        "sha256": expected, "counts": counts, "seed": SEED, "epochs": EPOCHS,
        "selection": "Highest macro-F1 on pooled development data; earliest epoch breaks ties",
        "roles": {
            "development_field": "Existing source-held-out archived field fold, explicitly used for development in this trial",
            "development_malik": "Existing Malik v2 source-disjoint validation",
            "plantcity_historical": "Previously evaluated PlantCity v4 fold; paired retrospective evaluation only",
            "new_field_plantdoc": "Reserved external audits; archived hashes used solely to check identity overlap",
        },
    }
    write_json(frozen, policy)
    print(json.dumps(policy, indent=2), flush=True)


def manifest(name, policy):
    path = TRIAL_SPLITS / f"{name}.csv"
    if sha256(path) != policy["sha256"][name]:
        raise ValueError(f"Frozen manifest changed: {path}")
    return path


def model_for(architecture, *, pretrained=False):
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        feature_stage = model.layer4
        head = model.fc
    else:
        model = models.convnext_tiny(weights=None)
        feature_stage = model.features[-1]
        head = model.classifier[-1]
    if pretrained:
        filename, expected = ARCHITECTURES[architecture]
        path = WEIGHTS / filename
        digest = sha256(path)
        if not digest.startswith(expected):
            raise ValueError(f"Unexpected pretrained weights: {path}")
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    else:
        digest = None
    new_head = nn.Linear(head.in_features, len(CLASSES))
    if architecture == "resnet18":
        model.fc = new_head
    else:
        model.classifier[-1] = new_head
    return model, feature_stage, new_head, digest


def loader(dataset, *, shuffle=False):
    generator = torch.Generator().manual_seed(SEED) if shuffle else None
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, generator=generator)


def move_images(images, device):
    return images.to(device, non_blocking=True, memory_format=torch.channels_last)


def development(policy, device):
    sets = [ManifestDataset(manifest(name, policy))
            for name in ("development_field", "development_malik")]
    batches = [(move_images(images, device), labels)
               for images, labels in loader(ConcatDataset(sets))]
    print("Cached development batches:", len(batches), flush=True)
    return batches


def add_class_f1(metrics):
    matrix = metrics["confusion"]
    metrics["class_f1"] = {
        label: 2 * matrix[i][i] / (sum(matrix[i]) + sum(row[i] for row in matrix))
        for i, label in enumerate(CLASSES)
    }
    return metrics


def train_epoch(model, stage, head, fit_loader, optimizer, scaler, weights, device):
    model.eval()
    stage.train()
    head.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    accumulated = 0.0
    seen = 0
    for images, labels in fit_loader:
        images, labels = move_images(images, device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = nn.functional.cross_entropy(model(images), labels, weight=weights)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        accumulated += loss.item() * len(labels)
        seen += len(labels)
    if not math.isfinite(accumulated):
        raise RuntimeError("Training loss is non-finite")
    return accumulated / seen


def train(architecture, policy, device, val_batches, *, continue_training=False):
    seed_everything(SEED)
    destination = RUN / (CONTINUATION if continue_training else architecture)
    parent = None
    if continue_training:
        parent_dir = RUN / architecture
        parent = read_json(parent_dir / "selection.json")
        parent_config = read_json(parent_dir / "config.json")
        if parent_config["fit_manifest_sha256"] != policy["sha256"]["train"]:
            raise ValueError("Continuation training data differs from the parent run")
        model, stage, head, _ = model_for(architecture)
        model.load_state_dict(load_tuned_checkpoint(architecture, parent_dir, parent)["model_state"])
        pretrained_sha = parent_config["pretrained_sha256"]
    else:
        model, stage, head, pretrained_sha = model_for(architecture, pretrained=True)
    lr_factor = CONTINUATION_LR_FACTOR if continue_training else 1.0
    head_lr, backbone_lr = HEAD_LR * lr_factor, BACKBONE_LR * lr_factor
    model.requires_grad_(False)
    stage.requires_grad_(True)
    head.requires_grad_(True)
    model.to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW([
        {"params": head.parameters(), "lr": head_lr},
        {"params": stage.parameters(), "lr": backbone_lr},
    ], weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    fit_manifest = manifest("train", policy)
    training = loader(ManifestDataset(fit_manifest, train=True), shuffle=True)
    torch.manual_seed(SEED)
    counts = torch.tensor([policy["counts"]["train"][label] for label in CLASSES],
                          dtype=torch.float32, device=device)
    weights = counts.rsqrt()
    weights /= weights.mean()
    config = {
        "architecture": architecture, "pretrained_sha256": pretrained_sha,
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "head_lr": head_lr, "backbone_lr": backbone_lr, "device": str(device),
        "trainable": "final backbone stage and three-class head",
        "fit_manifest_sha256": sha256(fit_manifest), "validation_rule": policy["selection"],
        "training_class_weights": dict(zip(CLASSES, weights.cpu().tolist())),
    }
    if parent is not None:
        config["continuation"] = {
            "parent_epoch": parent["epoch"], "parent_checkpoint_sha256": parent["checkpoint_sha256"],
            "optimizer": "Fresh AdamW state initialized from the selected model weights",
            "parent_development_macro_f1": parent["development"]["macro_f1"],
        }
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "config.json", config)
    best_score = -1.0
    selection = None
    fields = ("epoch", "train_loss", "development_macro_f1", "development_early_f1",
              "development_late_recall", "development_healthy_false_alarm")
    with (destination / "history.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, stage, head, training, optimizer, scaler, weights, device)
            metrics, *_ = measure(model, val_batches, device)
            metrics = add_class_f1(metrics)
            record = dict(epoch=epoch, train_loss=loss, development_macro_f1=metrics["macro_f1"],
                          development_early_f1=metrics["class_f1"]["early_blight"],
                          development_late_recall=metrics["late_recall"],
                          development_healthy_false_alarm=metrics["healthy_false_alarm_rate"])
            writer.writerow(record)
            file.flush()
            print(destination.name, json.dumps(record), flush=True)
            if metrics["macro_f1"] > best_score:
                best_score = metrics["macro_f1"]
                checkpoint_path = destination / "best.pt"
                temporary = checkpoint_path.with_suffix(".tmp")
                torch.save({"model_name": architecture, "classes": CLASSES, "epoch": epoch,
                            "model_state": {key: tensor.detach().cpu()
                                            for key, tensor in model.state_dict().items()}}, temporary)
                temporary.replace(checkpoint_path)
                selection = {"epoch": epoch, "development": metrics,
                             "checkpoint_sha256": sha256(checkpoint_path)}
    if parent is not None:
        parent_score = parent["development"]["macro_f1"]
        selection["parent_development_macro_f1"] = parent_score
        selection["development_improved"] = best_score > parent_score
    write_json(destination / "selection.json", selection)
    print(destination.name, "selected epoch", selection["epoch"], flush=True)


def load_tuned_checkpoint(architecture, destination, selection):
    path = destination / "best.pt"
    if sha256(path) != selection["checkpoint_sha256"]:
        raise ValueError(f"Selected checkpoint changed: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != architecture or tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError(f"Unexpected checkpoint format: {path}")
    return checkpoint


def tuned_model(architecture, device, *, run_name=None):
    destination = RUN / (run_name or architecture)
    selection = read_json(destination / "selection.json")
    checkpoint = load_tuned_checkpoint(architecture, destination, selection)
    model, *_ = model_for(architecture)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device, memory_format=torch.channels_last)


def compare(test_path, result_name, device, *, include_reference):
    entries = {name: tuned_model(name, device) for name in ARCHITECTURES}
    if include_reference:
        selection = read_json(FIRST_RUN / "selection.json")
        checkpoints = {
            "AC42": original_checkpoint("AC42"),
            "candidate_513": checked_checkpoint(FIRST_RUN / "candidate.pt", selection["candidate_sha256"]),
        }
        for name, checkpoint in checkpoints.items():
            model = create_model("resnet18")
            model.load_state_dict(checkpoint["model_state"])
            entries[name] = model.to(device, memory_format=torch.channels_last)
    evaluate_comparison(test_path, RUN, result_name, entries, device)


def evaluate_comparison(test_path, destination, result_name, entries, device):
    dataset = ManifestDataset(test_path)
    batches = [(move_images(images, device), labels) for images, labels in loader(dataset)]
    results = {}
    predictions_path = destination / f"{result_name}_predictions.csv"
    with predictions_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("model", "sha256", "origin", "true_label",
                                                  "prediction", "confidence"))
        writer.writeheader()
        for name, model in entries.items():
            metrics, truth, prediction, probability = measure(model, batches, device)
            results[name] = add_class_f1(metrics)
            for row, actual, guess, scores in zip(dataset.rows, truth, prediction, probability):
                writer.writerow(dict(model=name, sha256=row["sha256"], origin=row.get("origin", ""),
                                     true_label=CLASSES[actual], prediction=CLASSES[guess],
                                     confidence=float(scores[guess])))
            print(result_name, name, json.dumps(results[name]), flush=True)
    write_json(destination / f"{result_name}_results.json",
               {"manifest_sha256": sha256(test_path), "models": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "train", "continue-convnext", "history-test",
                                            "fresh-test"))
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
        return
    policy = read_json(TRIAL_SPLITS / "freeze.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    if args.command == "train":
        manifest("fresh_test", read_json(TRIAL_SPLITS / "fresh_freeze.json"))
        val_batches = development(policy, device)
        architectures = [args.architecture] if args.architecture else ARCHITECTURES
        for architecture in architectures:
            train(architecture, policy, device, val_batches)
    elif args.command == "continue-convnext":
        fresh = read_json(TRIAL_SPLITS / "fresh_freeze.json")
        val_batches = development(policy, device)
        train("convnext_tiny", policy, device, val_batches, continue_training=True)
        del val_batches
        entries = {"starting_convnext": tuned_model("convnext_tiny", device),
                   "continued_convnext": tuned_model("convnext_tiny", device, run_name=CONTINUATION)}
        for name, frozen in (("fresh_test", fresh), ("plantcity_historical", policy)):
            evaluate_comparison(manifest(name, frozen), RUN / CONTINUATION,
                                name, entries, device)
    elif args.command == "history-test":
        compare(manifest("plantcity_historical", policy), "plantcity_historical",
                device, include_reference=True)
    else:
        fresh = read_json(TRIAL_SPLITS / "fresh_freeze.json")
        path = manifest("fresh_test", fresh)
        compare(path, "fresh_test", device, include_reference=True)


if __name__ == "__main__":
    main()
