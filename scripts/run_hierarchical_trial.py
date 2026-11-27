"""Compare flat and hierarchical fine-tuning from the same AC42-derived checkpoint."""

import argparse
import csv
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from run_backbone_trial import (
    FIRST_RUN, FIRST_SPLITS, TRIAL_SPLITS, add_class_f1, development,
    evaluate_comparison, loader, manifest, move_images, read_rows,
)
from train_online_candidate import (
    CLASSES, WORKSPACE, ManifestDataset, checked_checkpoint, create_model, measure,
    read_json, seed_everything, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "hierarchical_trial"
SEED = 42
EPOCHS = 4
HEAD_LR = 5e-5
BLOCK_LR = 5e-6
SUBTYPE_LOSS_WEIGHT = 0.5
MAX_HEALTHY_FALSE_ALARM_INCREASE = 0.01


class HierarchicalResNet18(nn.Module):
    """Add separate log-odds corrections while preserving the starting classifier."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.prior_head = backbone.fc
        backbone.fc = nn.Identity()
        self.disease_head = nn.Linear(self.prior_head.in_features, 1)
        self.early_given_disease_head = nn.Linear(self.prior_head.in_features, 1)
        for layer in (self.disease_head, self.early_given_disease_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def scores(self, images):
        features = self.backbone(images)
        prior = self.prior_head(features)
        healthy, early, late = prior.unbind(dim=1)
        disease = torch.logsumexp(prior[:, 1:], dim=1) - healthy
        disease = disease + self.disease_head(features).squeeze(1)
        early_given_disease = early - late + self.early_given_disease_head(features).squeeze(1)
        log_disease = F.logsigmoid(disease)
        log_probs = torch.stack((
            F.logsigmoid(-disease),
            log_disease + F.logsigmoid(early_given_disease),
            log_disease + F.logsigmoid(-early_given_disease),
        ), dim=1)
        return log_probs, early_given_disease

    def forward(self, images):
        return self.scores(images)[0]


def new_model(arm, parent):
    model = create_model("resnet18")
    model.load_state_dict(parent["model_state"])
    if arm == "hierarchical":
        model = HierarchicalResNet18(model)
    return model


def experiment_policy():
    first = read_json(FIRST_SPLITS / "freeze.json")
    backbone = read_json(TRIAL_SPLITS / "freeze.json")
    fresh = read_json(TRIAL_SPLITS / "fresh_freeze.json")
    fit = FIRST_SPLITS / "train.csv"
    if sha256(fit) != first["manifest_sha256"]["train"]:
        raise ValueError("The first fine-tuning manifest has changed")
    partitions = {
        "train": fit,
        "development_field": manifest("development_field", backbone),
        "development_malik": manifest("development_malik", backbone),
        "plantwild_replay": manifest("fresh_test", fresh),
        "plantcity_replay": manifest("plantcity_historical", backbone),
    }
    used = set()
    for name, path in partitions.items():
        records = read_rows(path)
        ids = {row["sha256"] for row in records}
        if len(ids) != len(records) or ids & used:
            raise ValueError(f"Duplicate or overlapping image identities in {name}")
        used.update(ids)
    return backbone, partitions, first["class_counts"]["train"]


def run_epoch(model, train_loader, optimizer, device, subtype_weights):
    model.train()
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()
    total_loss = 0.0
    images_seen = 0
    for images, labels in train_loader:
        images, labels = move_images(images, device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, HierarchicalResNet18):
            scores, subtype = model.scores(images)
            class_loss = F.nll_loss(scores, labels)
            is_early = (labels == 1).to(subtype.dtype)
            subtype_terms = F.binary_cross_entropy_with_logits(subtype, is_early, reduction="none")
            sample_weights = subtype_weights[labels]
            conditional_loss = (subtype_terms * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
            loss = class_loss + SUBTYPE_LOSS_WEIGHT * conditional_loss
        else:
            loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError("Non-finite training loss")
        total_loss += loss_value * labels.numel()
        images_seen += labels.numel()
    return total_loss / images_seen


def allowed(metrics, limits):
    return (metrics["late_to_healthy"] <= limits["late_to_healthy"]
            and metrics["healthy_false_alarm_count"] <= limits["healthy_false_alarm_count"])


def train_arm(arm, parent, fit_set, development_batches, counts, limits, device):
    seed_everything(SEED)
    model = new_model(arm, parent).to(device, memory_format=torch.channels_last)
    model.requires_grad_(False)
    if isinstance(model, HierarchicalResNet18):
        block = model.backbone.layer4[-1]
        heads = (model.prior_head, model.disease_head, model.early_given_disease_head)
    else:
        block = model.layer4[-1]
        heads = (model.fc,)
    block.requires_grad_(True)
    for head in heads:
        head.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {"params": [parameter for head in heads for parameter in head.parameters()], "lr": HEAD_LR},
        {"params": block.parameters(), "lr": BLOCK_LR},
    ], weight_decay=1e-4)
    disease_images = counts["early_blight"] + counts["late_blight"]
    subtype_weights = torch.tensor(
        [0.0, disease_images / (2 * counts["early_blight"]), disease_images / (2 * counts["late_blight"])],
        device=device,
    )
    torch.manual_seed(SEED)
    train_loader = loader(fit_set, shuffle=True)
    location = RUN / arm
    location.mkdir()
    artifact = location / "best.pt"
    best_rank = None
    selected = None
    fields = ("epoch", "train_loss", "early_f1", "macro_f1", "late_recall", "late_to_healthy",
              "healthy_false_alarms", "meets_guardrails")
    with (location / "history.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, EPOCHS + 1):
            loss = run_epoch(model, train_loader, optimizer, device, subtype_weights)
            metrics, *_ = measure(model, development_batches, device)
            metrics = add_class_f1(metrics)
            eligible = allowed(metrics, limits)
            early_f1 = metrics["class_f1"]["early_blight"]
            record = dict(epoch=epoch, train_loss=loss, early_f1=early_f1, macro_f1=metrics["macro_f1"],
                          late_recall=metrics["late_recall"], late_to_healthy=metrics["late_to_healthy"],
                          healthy_false_alarms=metrics["healthy_false_alarm_count"], meets_guardrails=eligible)
            writer.writerow(record)
            handle.flush()
            print(arm, json.dumps(record), flush=True)
            rank = eligible, early_f1, metrics["macro_f1"]
            if best_rank is None or rank > best_rank:
                best_rank = rank
                temporary = artifact.with_suffix(".tmp")
                torch.save({"model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                            "model_name": "hierarchical_resnet18" if arm == "hierarchical" else "resnet18",
                            "classes": CLASSES, "epoch": epoch, "seed": SEED}, temporary)
                temporary.replace(artifact)
                selected = {"epoch": epoch, "metrics": metrics, "meets_guardrails": eligible}
    selected["sha256"] = sha256(artifact)
    write_json(location / "selection.json", selected)
    return selected


def load_arm(arm, parent, selection, device):
    artifact = RUN / arm / "best.pt"
    if sha256(artifact) != selection["sha256"]:
        raise ValueError(f"Selected checkpoint changed after training: {arm}")
    model = new_model(arm, parent).to(device, memory_format=torch.channels_last)
    model.load_state_dict(torch.load(artifact, map_location="cpu", weights_only=True)["model_state"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    frozen, partitions, counts = experiment_policy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    parent_selected = read_json(FIRST_RUN / "selection.json")
    parent_digest = parent_selected["candidate_sha256"]
    parent = checked_checkpoint(FIRST_RUN / "candidate.pt", parent_digest)
    baseline = new_model("flat", parent).to(device, memory_format=torch.channels_last).eval()
    development_batches = development(frozen, device)
    hierarchy = new_model("hierarchical", parent).to(device, memory_format=torch.channels_last).eval()
    with torch.inference_mode():
        images, _ = development_batches[0]
        observed = hierarchy(images).exp()
        expected = baseline(images).softmax(1)
        torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-6)
    del hierarchy
    reference, *_ = measure(baseline, development_batches, device)
    reference = add_class_f1(reference)
    limits = {
        "late_to_healthy": reference["late_to_healthy"],
        "healthy_false_alarm_count": reference["healthy_false_alarm_count"] + math.floor(
            MAX_HEALTHY_FALSE_ALARM_INCREASE * reference["class_counts"]["healthy"]),
    }
    RUN.mkdir(parents=True, exist_ok=False)
    write_json(RUN / "protocol.json", {
        "parent_sha256": parent_digest, "seed": SEED, "epochs_per_arm": EPOCHS,
        "head_lr": HEAD_LR, "block_lr": BLOCK_LR, "subtype_loss_weight": SUBTYPE_LOSS_WEIGHT,
        "manifest_sha256": {name: sha256(path) for name, path in partitions.items()},
        "training_counts": counts, "device": str(device),
        "checkpoint_selection": "Prefer guardrail-eligible states, then highest development early F1, "
                                "then macro F1; exact ties retain the earlier state",
        "promotion": "Require improved early F1 and no reduction in macro F1 versus the starting model",
        "guardrails": limits, "starting_development": reference,
    })
    print("Starting development and limits", json.dumps({"metrics": reference, "limits": limits}), flush=True)
    fit_set = ManifestDataset(partitions["train"], train=True)
    selections = {arm: train_arm(arm, parent, fit_set, development_batches, counts, limits, device)
                  for arm in ("flat", "hierarchical")}
    del development_batches
    acceptable = {name: item for name, item in selections.items() if item["meets_guardrails"]
                  and item["metrics"]["class_f1"]["early_blight"] > reference["class_f1"]["early_blight"]
                  and item["metrics"]["macro_f1"] >= reference["macro_f1"]}
    preferred = max(acceptable, key=lambda name: (
        acceptable[name]["metrics"]["class_f1"]["early_blight"], acceptable[name]["metrics"]["macro_f1"]
    )) if acceptable else "starting_candidate"
    write_json(RUN / "decision.json", {"preferred_by_development": preferred, "arms": selections})
    entries = {"starting_candidate": baseline}
    entries.update((arm, load_arm(arm, parent, selection, device)) for arm, selection in selections.items())
    for name in ("plantwild_replay", "plantcity_replay"):
        evaluate_comparison(partitions[name], RUN, name, entries, device)


if __name__ == "__main__":
    main()
