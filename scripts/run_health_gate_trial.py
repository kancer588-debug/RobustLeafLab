"""Fit a healthy-versus-diseased decision using two frozen AC42 classifiers."""

import argparse
import csv
import json
from collections import Counter
from itertools import product
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from run_backbone_trial import FIRST_RUN, TRIAL_SPLITS, loader, manifest, move_images, read_rows
from run_expanded_field_trial import score, set_limits, verified
from run_local_leaf_trial import measure
from train_online_candidate import (
    CLASSES, ORIGINAL_CHECKPOINTS, WORKSPACE, ManifestDataset, checked_checkpoint, create_model,
    original_checkpoint, read_json, seed_everything, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "health_gate_trial"
EXPANDED = WORKSPACE / "datasets" / "expanded_field_trial"
TAIWAN = WORKSPACE / "datasets" / "local_leaf_trial" / "fresh_test_freeze.json"
SEEDS = (42, 43, 44)
ARMS = ("class_balanced", "source_balanced")
MIX_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
OFFSETS = (-0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5)
EPOCHS = 24
BATCH_SIZE = 64
LR = 0.015


def make_experts(device):
    parent_hash = read_json(FIRST_RUN / "selection.json")["candidate_sha256"]
    checkpoints = (original_checkpoint("AC42"), checked_checkpoint(FIRST_RUN / "candidate.pt", parent_hash))
    models = [create_model("resnet18") for _ in checkpoints]
    for model, checkpoint in zip(models, checkpoints):
        model.load_state_dict(checkpoint["model_state"])
        model.requires_grad_(False).to(device, memory_format=torch.channels_last).eval()
    return models, parent_hash


@torch.no_grad()
def collect(models, path, name, device):
    parts, labels = [], []
    for images, targets in loader(ManifestDataset(path)):
        images = move_images(images, device)
        parts.append(torch.stack([model(images).cpu() for model in models], dim=1))
        labels.append(targets)
    result = torch.cat(parts), torch.cat(labels)
    print(f"Scored {name}: {len(result[1])} images", flush=True)
    return result


def disease_margins(logits):
    return logits[:, :, 1:].amax(dim=-1) - logits[:, :, 0]


def gate_features(logits, margins):
    subtype_gap = (logits[:, :, 1] - logits[:, :, 2]).abs()
    disagreement = (margins[:, :1] - margins[:, 1:]).abs()
    return torch.cat((subtype_gap, disagreement), dim=1)


class AdaptiveGate(nn.Module):
    """Blend expert margins with a confidence-dependent weight and a shared offset."""

    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("scale", scale.clone())
        self.mix = nn.Linear(3, 1)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, logits):
        margins = disease_margins(logits)
        normalized = (gate_features(logits, margins) - self.mean) / self.scale
        original_weight = self.mix(normalized.clamp(-5.0, 5.0)).squeeze(1).sigmoid()
        return torch.lerp(margins[:, 1], margins[:, 0], original_weight) + self.offset


def route(logits, margin):
    subtype = logits[:, 1, 1:]
    relative = subtype - subtype.amax(dim=1, keepdim=True)
    return torch.cat((torch.zeros_like(margin[:, None]), relative + margin[:, None]), dim=1)


def fixed_margin(logits, original_weight, offset):
    margins = disease_margins(logits)
    return torch.lerp(margins[:, 1], margins[:, 0], original_weight) + offset


def compare(logits, labels, malik):
    pooled = measure(logits, labels)[0]
    pooled["malik"] = measure(logits[-malik:], labels[-malik:])[0]
    return pooled


def promotion_limits(start):
    limits = set_limits(start)
    limits.update(
        healthy_false_alarm_count=start["healthy_false_alarm_count"] - 1,
        minimum_early_correct=start["confusion"][1][1] - start["class_counts"]["early_blight"] * 2 // 100,
        maximum_disease_to_healthy=start["confusion"][1][0] + start["late_to_healthy"],
        malik_healthy_false_alarms=start["malik"]["healthy_false_alarm_count"],
        minimum_pooled_macro_f1=start["macro_f1"],
        minimum_malik_macro_f1=start["malik"]["macro_f1"],
    )
    return limits


def eligible(metrics, limits):
    matrix, malik = metrics["confusion"], metrics["malik"]
    return (
        metrics["healthy_false_alarm_count"] <= limits["healthy_false_alarm_count"]
        and metrics["late_to_healthy"] <= limits["late_to_healthy"]
        and matrix[1][0] + matrix[2][0] <= limits["maximum_disease_to_healthy"]
        and matrix[1][1] >= limits["minimum_early_correct"]
        and matrix[2][2] >= limits["minimum_late_correct"]
        and malik["confusion"][1][1] >= limits["malik_early_correct"]
        and malik["late_to_healthy"] <= limits["malik_late_to_healthy"]
        and malik["healthy_false_alarm_count"] <= limits["malik_healthy_false_alarms"]
        and metrics["macro_f1"] >= limits["minimum_pooled_macro_f1"]
        and malik["macro_f1"] >= limits["minimum_malik_macro_f1"]
    )


def observation(metrics, limits):
    return {
        "eligible": eligible(metrics, limits), "score": score(metrics),
        "healthy_false_alarms": metrics["healthy_false_alarm_count"],
        "early_to_healthy": metrics["confusion"][1][0], "late_to_healthy": metrics["late_to_healthy"],
        "early_correct": metrics["confusion"][1][1], "late_correct": metrics["confusion"][2][2],
        "macro_f1": metrics["macro_f1"], "malik_macro_f1": metrics["malik"]["macro_f1"],
    }


def rank(record):
    return record["eligible"], record["score"], -record["healthy_false_alarms"]


def write_rows(path, rows, *, replace=False):
    pending = path.with_suffix(".tmp") if replace else path
    with pending.open("w" if replace else "x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if replace:
        pending.replace(path)


def static_search(development, limits):
    logits, labels, malik = development
    results = []
    for weight, offset in product(MIX_WEIGHTS, OFFSETS):
        metrics = compare(route(logits, fixed_margin(logits, weight, offset)), labels, malik)
        record = {"original_weight": weight, "offset": offset, **observation(metrics, limits)}
        results.append({"record": record, "metrics": metrics})
    write_rows(RUN / "static_search.csv", [result["record"] for result in results])
    return max(results, key=lambda row: (*rank(row["record"]), -abs(row["record"]["offset"])))


def source_bucket(row):
    source = row["source"]
    return source if source in {"field_extra", "plantvillage"} else "other_public"


def fitting_weights(rows, labels, arm):
    groups = [(row["label"], source_bucket(row)) for row in rows]
    counts = Counter(groups)
    values = torch.ones(len(labels))
    if arm == "source_balanced":
        values = torch.tensor([counts[group] ** -0.5 for group in groups])
    for category in range(len(CLASSES)):
        mask = labels == category
        values[mask] *= len(labels) / (len(CLASSES) * values[mask].sum())
    return values


def train_gate(arm, seed, fit, rows, development, limits):
    seed_everything(seed)
    logits, labels = fit
    dev_logits, dev_labels, malik = development
    design = gate_features(logits, disease_margins(logits))
    gate = AdaptiveGate(design.mean(dim=0), design.std(dim=0, unbiased=False).clamp_min(1e-4))
    optimizer = torch.optim.AdamW(gate.parameters(), lr=LR, weight_decay=0.02)
    weights = fitting_weights(rows, labels, arm)
    targets = (labels != 0).float()
    history = []
    best = None
    for epoch in range(1, EPOCHS + 1):
        loss_total = 0.0
        for indices in torch.randperm(len(logits)).split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(gate(logits[indices]), targets[indices],
                                                       weight=weights[indices])
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(indices)
        with torch.no_grad():
            metrics = compare(route(dev_logits, gate(dev_logits)), dev_labels, malik)
        record = {"epoch": epoch, "train_loss": loss_total / len(logits), **observation(metrics, limits)}
        history.append(record)
        if best is None or rank(record) > rank(best["record"]):
            best = {"record": record, "metrics": metrics}
            state = {name: value.detach().clone() for name, value in gate.state_dict().items()}
    name = f"{arm}_s{seed}"
    directory = RUN / name
    directory.mkdir()
    write_rows(directory / "history.csv", history)
    weights_path = directory / "gate.pt"
    torch.save({"state": state, "seed": seed, "arm": arm}, weights_path)
    best["checkpoint_sha256"] = sha256(weights_path)
    print(f"{name} {json.dumps(best['record'])}", flush=True)
    return best


def checked_splits():
    freeze = read_json(EXPANDED / "training_freeze.json")
    dev = read_json(TRIAL_SPLITS / "freeze.json")
    paths = {
        "training": verified(EXPANDED / "expanded_train.csv", freeze["training_sha256"]["expanded_train"]),
        "development_field": manifest("development_field", dev),
        "development_malik": manifest("development_malik", dev),
    }
    seen = set()
    for path in paths.values():
        rows = read_rows(path)
        identities = {row["sha256"] for row in rows}
        if len(rows) != len(identities) or seen & identities:
            raise ValueError(f"Duplicate images in training or development: {path}")
        seen.update(identities)
    return paths


def train(models, parent_hash, device):
    paths = checked_splits()
    RUN.mkdir(exist_ok=False)
    write_json(RUN / "protocol.json", {
        "original_ac42_sha256": ORIGINAL_CHECKPOINTS["AC42"][1], "parent_sha256": parent_hash,
        "splits": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "static_original_weights": MIX_WEIGHTS, "static_offsets": OFFSETS, "seeds": SEEDS,
        "epochs": EPOCHS, "learning_rate": LR, "batch_size": BATCH_SIZE,
        "selection": "Require fewer development healthy false alarms than the starting candidate, "
                     "no increase in development disease-to-healthy errors, retain existing late and Malik "
                     "guards, cap early-correct losses at 2%, and retain pooled and Malik Macro-F1. Rank by "
                     "their mean and then lower healthy false alarms. A learned arm needs the main seed and "
                     "at least one other seed to qualify. Rank all qualifying arms by the same rule.",
        "evaluation": "After decision lock only, replay PlantCity, PlantWild, Imtiaz, Rachin and Taiwan. "
                      "New field and PlantDoc are reserved and are not loaded in this experiment.",
    })
    fit = collect(models, paths["training"], "training", device)
    fit_rows = read_rows(paths["training"])
    assert [CLASSES[label] for label in fit[1].tolist()] == [row["label"] for row in fit_rows]
    field = collect(models, paths["development_field"], "development field", device)
    malik = collect(models, paths["development_malik"], "development Malik", device)
    development = torch.cat((field[0], malik[0])), torch.cat((field[1], malik[1])), len(malik[1])
    logits, labels, malik_count = development
    torch.testing.assert_close(route(logits, fixed_margin(logits, 0.0, 0.0)).argmax(1), logits[:, 1].argmax(1))
    start = compare(logits[:, 1], labels, malik_count)
    limits = promotion_limits(start)
    original = compare(logits[:, 0], labels, malik_count)
    static = static_search(development, limits)
    learned = {f"{arm}_s{seed}": train_gate(arm, seed, fit, fit_rows, development, limits)
               for arm, seed in product(ARMS, SEEDS)}
    candidates = {"static_mix": static} if static["record"]["eligible"] else {}
    for arm in ARMS:
        main = learned[f"{arm}_s42"]
        if main["record"]["eligible"] and any(learned[f"{arm}_s{seed}"]["record"]["eligible"] for seed in SEEDS[1:]):
            candidates[f"{arm}_s42"] = main
    preferred = max(candidates, key=lambda name: rank(candidates[name]["record"]), default="starting_candidate")
    write_json(RUN / "decision.json", {
        "preferred_by_development": preferred, "qualified_candidates": list(candidates), "limits": limits,
        "original_development": original, "starting_development": start, "static_mix": static, "learned": learned,
    })
    print("Decision locked:", preferred, flush=True)


def evaluation_receipts(training_hash):
    previous = read_json(EXPANDED / "evaluation_freeze.json")
    taiwan = read_json(TAIWAN)
    if taiwan["compatible_training_sha256"] != training_hash:
        raise ValueError("Taiwan screening is for a different training manifest")
    return {
        **previous["historical_replays"],
        "rachin_replay": {"path": str(EXPANDED / "rachin_evaluation.csv"), "sha256": previous["sha256"]},
        **taiwan["tests"],
    }


def asymmetric_margin(logits, parameters):
    original, parent = disease_margins(logits).unbind(dim=1)
    subtype = logits[:, 1, 1:].argmax(dim=1) + 1
    veto = ((parent > 0) & (parent <= parameters["veto_window"])
            & (original < -parameters["veto_support"]))
    if parameters["veto_subtype"]:
        veto &= subtype == parameters["veto_subtype"]
    rescue = ((parent <= 0) & (parent > -parameters["rescue_window"])
              & (original > parameters["rescue_support"]))
    if parameters["rescue_subtype"]:
        rescue &= subtype == parameters["rescue_subtype"]
    margin = torch.where(veto, -parent, parent)
    return torch.where(rescue, -parent + torch.finfo(parent.dtype).eps, margin)


def asymmetric(models, parent_hash, device):
    """Select an asymmetric disagreement rule without opening replay images."""
    location = RUN / "asymmetric"
    prior = read_json(RUN / "protocol.json")
    decision_hash = sha256(RUN / "decision.json")
    decision = read_json(RUN / "decision.json")
    if parent_hash != prior["parent_sha256"]:
        raise ValueError("The locked starting checkpoint has changed")
    veto_options = [(0.0, 0.0, 0), *product((0.125, 0.25, 0.5), (0.25, 0.5), (0, 1, 2))]
    rescue_options = [(0.0, 0.0, 0), *product((0.25, 0.5, 1.0), (0.25, 0.5), (0, 1, 2))]
    location.mkdir(exist_ok=False)
    write_json(location / "protocol.json", {
        "parent_decision_sha256": decision_hash, "veto_options": veto_options, "rescue_options": rescue_options,
        "terms": ["parent_margin_window", "original_margin_support", "parent_subtype_or_zero_for_either"],
        "selection": "Apply the unchanged promotion limits and ranking. Prefer fewer changed development "
                     "predictions on exact ties. Do not evaluate until both decisions are locked.",
    })
    splits = prior["splits"]
    field, malik = [collect(models, verified(Path(splits[name]["path"]), splits[name]["sha256"]), name, device)
                    for name in ("development_field", "development_malik")]
    logits = torch.cat((field[0], malik[0]))
    labels = torch.cat((field[1], malik[1]))
    parent_predictions = logits[:, 1].argmax(1)
    limits = decision["limits"]
    selections = []
    for veto, rescue in product(veto_options, rescue_options):
        parameters = dict(zip(("veto_window", "veto_support", "veto_subtype",
                               "rescue_window", "rescue_support", "rescue_subtype"), (*veto, *rescue)))
        scores = route(logits, asymmetric_margin(logits, parameters))
        metrics = compare(scores, labels, len(malik[1]))
        record = {**parameters, **observation(metrics, limits),
                  "changed": int((scores.argmax(1) != parent_predictions).sum())}
        selections.append({"record": record, "metrics": metrics})
    write_rows(location / "development_search.csv", [item["record"] for item in selections])
    best = max(selections, key=lambda item: (*rank(item["record"]), -item["record"]["changed"]))
    write_json(location / "decision.json", {
        "preferred_by_development": "asymmetric_gate" if best["record"]["eligible"] else "starting_candidate",
        "qualifying_rules": sum(item["record"]["eligible"] for item in selections),
        "locked_rule": best, "parent_decision_sha256": decision_hash,
    })
    print("Asymmetric decision locked:", json.dumps(best["record"]), flush=True)


def evaluate(models, parent_hash, device):
    decision = read_json(RUN / "decision.json")
    primary_hash = sha256(RUN / "decision.json")
    asymmetric_path = RUN / "asymmetric" / "decision.json"
    asymmetric_lock = read_json(asymmetric_path)
    asymmetric_hash = sha256(asymmetric_path)
    if asymmetric_lock["parent_decision_sha256"] != primary_hash:
        raise ValueError("The asymmetric rule used a different development decision")
    protocol = read_json(RUN / "protocol.json")
    if protocol["parent_sha256"] != parent_hash:
        raise ValueError("The locked starting checkpoint has changed")
    training = protocol["splits"]["training"]
    receipts = evaluation_receipts(training["sha256"])
    forbidden = {row["sha256"] for split in protocol["splits"].values()
                 for row in read_rows(verified(Path(split["path"]), split["sha256"]))}
    gates = {}
    for arm in ARMS:
        key = f"{arm}_s42"
        path = verified(RUN / key / "gate.pt", decision["learned"][key]["checkpoint_sha256"])
        state = torch.load(path, map_location="cpu", weights_only=True)["state"]
        gates[key] = AdaptiveGate(state["mean"], state["scale"])
        gates[key].load_state_dict(state)
        gates[key].eval()
    parameters = decision["static_mix"]["record"]
    for name, receipt in receipts.items():
        path = verified(Path(receipt["path"]), receipt["sha256"])
        rows = read_rows(path)
        keys = {row["sha256"] for row in rows}
        if len(keys) != len(rows) or keys & forbidden:
            raise ValueError(f"Evaluation images overlap with training or development: {name}")
        output = RUN / f"{name}_results.json"
        if output.exists():
            values = read_json(output)
            if (values["manifest_sha256"] != receipt["sha256"] or values["decision_sha256"] != primary_hash
                    or values["asymmetric_decision_sha256"] != asymmetric_hash):
                raise ValueError(f"The existing result uses a different manifest or decision: {name}")
            continue
        logits, truth = collect(models, path, name, device)
        with torch.no_grad():
            candidates = {"original_AC42": logits[:, 0], "starting_candidate": logits[:, 1],
                          "static_mix": route(logits, fixed_margin(logits, parameters["original_weight"],
                                                                  parameters["offset"])),
                          "asymmetric_gate": route(logits, asymmetric_margin(logits,
                                                   asymmetric_lock["locked_rule"]["record"])),
                          **{key: route(logits, gate(logits)) for key, gate in gates.items()}}
        metrics = {}
        predictions = []
        for key, guesses in candidates.items():
            metrics[key], classes = measure(guesses, truth)
            subtype = guesses[:, 1:].argmax(1) + 1
            ceiling_correct = int(((truth == 0) | (subtype == truth)).sum())
            metrics[key]["fixed_subtype_ceiling_correct"] = ceiling_correct
            metrics[key]["fixed_subtype_ceiling_accuracy"] = ceiling_correct / len(truth)
            for row, label, disease in zip(rows, classes.tolist(), subtype.tolist()):
                predictions.append({"model": key, "sha256": row["sha256"], "true_label": row["label"],
                                    "prediction": CLASSES[label], "subtype_if_diseased": CLASSES[disease]})
            print(f"{name} {key} accuracy={metrics[key]['accuracy']:.5f}", flush=True)
        write_rows(RUN / f"{name}_predictions.csv", predictions, replace=True)
        write_json(output, {"manifest_sha256": receipt["sha256"], "models": metrics,
                            "decision_sha256": primary_hash, "asymmetric_decision_sha256": asymmetric_hash})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = {"train": train, "asymmetric": asymmetric, "evaluate": evaluate}
    parser.add_argument("phase", choices=phases)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, parent_hash = make_experts(device)
    phases[args.phase](models, parent_hash, device)


if __name__ == "__main__":
    main()
