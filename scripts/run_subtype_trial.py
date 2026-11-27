"""Compare frozen ResNet18 and DINOv2 probes for early-versus-late blight."""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from run_backbone_trial import FIRST_RUN, TRIAL_SPLITS, loader, manifest, move_images, read_rows
from run_expanded_field_trial import eligible, score, set_limits, verified
from run_local_leaf_trial import measure
from train_online_candidate import (
    CLASSES, ORIGINAL_CHECKPOINTS, WORKSPACE, ManifestDataset, checked_checkpoint, create_model,
    original_checkpoint, read_json, seed_everything, sha256, write_json,
)


DATA = WORKSPACE / "datasets" / "subtype_trial"
EXPANDED = WORKSPACE / "datasets" / "expanded_field_trial"
RUN = WORKSPACE / "outputs" / "subtype_trial"
DINO = WORKSPACE / "datasets" / "backbone_weights" / "dinov2"
DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_HASH = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
ARMS = {"resnet_augmented": ("resnet", "augmented"), "dino_base": ("dino", "base"),
        "dino_augmented": ("dino", "augmented")}
SEEDS = (42, 43, 44)
FUSION = (0.25, 0.5, 0.75, 1.0)
EPOCHS = 16
LR = 0.004
HEAD_BATCH = 64


class Specialists(nn.Module):
    """Share the original input transform and collect two frozen representations."""

    def __init__(self, parent, device):
        super().__init__()
        os.environ["XFORMERS_DISABLED"] = "1"
        sys.path.insert(0, str(DINO / f"dinov2-{DINO_REVISION}"))
        from dinov2.hub.backbones import dinov2_vits14

        self.resnet = create_model("resnet18")
        self.resnet.load_state_dict(parent["model_state"])
        self.dino = dinov2_vits14(pretrained=False)
        weight_path = verified(DINO / "dinov2_vits14_pretrain.pth", DINO_HASH)
        self.dino.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
        self.device = device
        self.resnet.to(device, memory_format=torch.channels_last)
        self.dino.to(device)
        self.requires_grad_(False).eval()

    def resnet_features(self, images):
        resnet = self.resnet
        features = resnet.maxpool(resnet.relu(resnet.bn1(resnet.conv1(images))))
        features = resnet.layer4(resnet.layer3(resnet.layer2(resnet.layer1(features))))
        return resnet.avgpool(features).flatten(1)

    def forward(self, images):
        resnet_features = self.resnet_features(images)
        tokens = self.dino.forward_features(images.contiguous())
        token_features = torch.cat((tokens["x_norm_clstoken"], tokens["x_norm_patchtokens"].mean(1)), dim=1)
        return resnet_features, token_features, self.resnet.fc(resnet_features)


@torch.no_grad()
def encode(model, path, name, original=None):
    buffers = []
    for images, truth in loader(ManifestDataset(path)):
        images = move_images(images, model.device)
        resnet, dino, prior = model(images)
        row = {"resnet": resnet.cpu(), "dino": dino.cpu(), "prior": prior.cpu(), "truth": truth}
        if original is not None:
            row["original"] = original(images).cpu()
        buffers.append(row)
    data = {kind: torch.cat([row[kind] for row in buffers]) for kind in buffers[0]}
    print(f"Encoded {name}: {len(data['truth'])} images", flush=True)
    return data


class SubtypeProbe(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("scale", scale.clone())
        self.linear = nn.Linear(mean.numel(), 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features):
        return self.linear(((features - self.mean) / self.scale).clamp(-6.0, 6.0)).flatten()


def combine(prior, probe_margin, fraction):
    prior_margin = prior[:, 1] - prior[:, 2]
    margin = torch.lerp(prior_margin, probe_margin, fraction)
    diseased = prior[:, 1:].amax(dim=1)
    scores = torch.stack((prior[:, 0], diseased + margin.clamp(max=0), diseased + (-margin).clamp(max=0)), dim=1)
    if not torch.equal(scores.argmax(1) == 0, prior.argmax(1) == 0):
        raise RuntimeError("The subtype probe changed healthy-versus-diseased routing")
    return scores


def measurement(logits, truth):
    metrics, guesses = measure(logits, truth)
    early, late = truth == 1, truth == 2
    subtype = logits[:, 1:].argmax(1) + 1
    early_correct = int((subtype[early] == 1).sum())
    late_correct = int((subtype[late] == 2).sum())
    metrics.update(
        early_to_late=metrics["confusion"][1][2], early_to_healthy=metrics["confusion"][1][0],
        subtype_early_correct=early_correct, subtype_late_correct=late_correct,
        subtype_balanced_accuracy=(early_correct / int(early.sum()) + late_correct / int(late.sum())) / 2,
        fixed_subtype_ceiling_correct=int((truth == 0).sum()) + early_correct + late_correct,
    )
    return metrics, guesses, subtype


def development_metrics(scores, development, malik_n):
    target = development["truth"]
    pooled = measurement(scores, target)[0]
    pooled["malik"] = measurement(scores[-malik_n:], target[-malik_n:])[0]
    return pooled


def metrics_record(epoch, fraction, loss, metrics, start, limits):
    passes = (eligible(metrics, limits) and score(metrics) > score(start)
              and metrics["macro_f1"] >= start["macro_f1"]
              and metrics["class_f1"]["early_blight"] > start["class_f1"]["early_blight"])
    return {"epoch": epoch, "probe_fraction": fraction, "loss": loss, "eligible": passes,
            "score": score(metrics), "macro_f1": metrics["macro_f1"],
            "early_f1": metrics["class_f1"]["early_blight"], "late_correct": metrics["confusion"][2][2],
            "early_to_late": metrics["early_to_late"], "late_to_early": metrics["late_to_early"],
            "subtype_balanced_accuracy": metrics["subtype_balanced_accuracy"]}


def rank(record):
    return record["eligible"], record["score"], record["macro_f1"], record["early_f1"], -record["probe_fraction"]


def training_weights(rows, truths):
    groups = [(row["label"], row.get("dataset") or row["source"]) for row in rows]
    occurrences = Counter(groups)
    weights = torch.tensor([occurrences[group] ** -0.5 for group in groups])
    for label in (1, 2):
        mask = truths == label
        weights[mask] *= len(rows) / (2 * weights[mask].sum())
    return weights


def write_rows(path, records):
    pending = path.with_suffix(".tmp")
    with pending.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    pending.replace(path)


def train_arm(arm, seed, fit, fit_rows, dev, malik_n, start, limits):
    feature, _ = ARMS[arm]
    seed_everything(seed)
    subset = fit["truth"] > 0
    x = fit[feature][subset]
    truths = fit["truth"][subset]
    targets = (truths == 1).float()
    sample_weights = training_weights([row for row, include in zip(fit_rows, subset) if include], truths)
    probe = SubtypeProbe(x.mean(0), x.std(0, unbiased=False).clamp_min(0.02))
    optimizer = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=0.05)
    selected, history = None, []
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for idx in torch.randperm(len(x)).split(HEAD_BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(probe(x[idx]), targets[idx], weight=sample_weights[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        with torch.no_grad():
            margin = probe(dev[feature])
            for fraction in FUSION:
                metrics = development_metrics(combine(dev["prior"], margin, fraction), dev, malik_n)
                record = metrics_record(epoch, fraction, total / len(x), metrics, start, limits)
                history.append(record)
                if selected is None or rank(record) > rank(selected["record"]):
                    selected = {"record": record, "metrics": metrics}
                    state = {name: tensor.detach().clone() for name, tensor in probe.state_dict().items()}
    directory = RUN / f"{arm}_s{seed}"
    directory.mkdir(exist_ok=True)
    write_rows(directory / "history.csv", history)
    weight = directory / "probe.pt"
    temporary = weight.with_suffix(".tmp")
    torch.save({"state": state, "feature": feature, "fraction": selected["record"]["probe_fraction"]}, temporary)
    temporary.replace(weight)
    selected["weight_sha256"] = sha256(weight)
    print(f"{directory.name} {json.dumps(selected['record'])}", flush=True)
    return selected


def splits():
    base = read_json(EXPANDED / "training_freeze.json")
    source = read_json(DATA / "freeze.json")
    dev = read_json(TRIAL_SPLITS / "freeze.json")
    files = source["files"]
    paths = {"base": verified(EXPANDED / "expanded_train.csv", base["training_sha256"]["expanded_train"]),
             "augmented": verified(Path(files["training"]["path"]), files["training"]["sha256"]),
             "development_field": manifest("development_field", dev), "development_malik": manifest("development_malik", dev)}
    receipts = {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}
    return paths, receipts, source


def checked_records(paths):
    records = {name: read_rows(path) for name, path in paths.items()}
    ids = {name: {item["sha256"] for item in group} for name, group in records.items()}
    if any(len(ids[name]) != len(group) for name, group in records.items()):
        raise ValueError("Duplicate image identity in a manifest")
    if not ids["base"] < ids["augmented"]:
        raise ValueError("The augmented training list must extend the fixed base list")
    fit, field, malik = (ids[name] for name in ("augmented", "development_field", "development_malik"))
    if fit & (field | malik) or field & malik:
        raise ValueError("Training and development identities overlap")
    return records


def train(model, parent_hash):
    paths, receipts, source = splits()
    records = checked_records(paths)
    RUN.mkdir(exist_ok=True)
    if (RUN / "decision.json").exists():
        raise FileExistsError("The subtype decision is already locked")
    protocol = {"parent_sha256": parent_hash, "original_ac42_sha256": ORIGINAL_CHECKPOINTS["AC42"][1],
                "dinov2_revision": DINO_REVISION, "dinov2_sha256": DINO_HASH, "manifests": receipts,
                "fresh_test": source["files"].get("test"), "cohort_freeze_sha256": sha256(DATA / "freeze.json"),
                "arms": {key: list(value) for key, value in ARMS.items()}, "seeds": list(SEEDS),
                "epochs": EPOCHS, "fractions": list(FUSION), "learning_rate": LR,
                "selection": "Use pooled and Malik development Macro-F1. Require improvements in their mean and "
                             "pooled early F1, preserve pooled Macro-F1 and existing late and Malik guardrails. "
                             "A new arm needs the main and at least one confirmatory seed. Preserve all healthy decisions.",
                "protection": "No test images enter fitting or selection. The Khagan/Charabag publisher is not used "
                              "when its test cohort fails overlap screening. New field and PlantDoc supply identities only."}
    protocol_path = RUN / "protocol.json"
    if protocol_path.exists():
        if read_json(protocol_path) != protocol:
            raise ValueError("The run directory contains a different protocol")
    else:
        write_json(protocol_path, protocol)
    augmented = encode(model, paths["augmented"], "augmented training")
    ordered = {row["sha256"]: index for index, row in enumerate(records["augmented"])}
    indices = [ordered[row["sha256"]] for row in records["base"]]
    base = {key: value[indices] for key, value in augmented.items()}
    data = {"base": base, "augmented": augmented}
    for key, split in data.items():
        if [CLASSES[item] for item in split["truth"].tolist()] != [row["label"] for row in records[key]]:
            raise ValueError("Image order changed before fitting")
    field, malik = [encode(model, paths[name], name) for name in ("development_field", "development_malik")]
    dev = {key: torch.cat((field[key], malik[key])) for key in field}
    malik_n = len(malik["truth"])
    start = development_metrics(dev["prior"], dev, malik_n)
    limits = set_limits(start)
    selection = {f"{arm}_s{seed}": train_arm(arm, seed, data[cohort], records[cohort], dev, malik_n, start, limits)
                 for (arm, (_, cohort)), seed in product(ARMS.items(), SEEDS)}
    qualifying = [f"{arm}_s42" for arm in ARMS if selection[f"{arm}_s42"]["record"]["eligible"]
                  and any(selection[f"{arm}_s{seed}"]["record"]["eligible"] for seed in SEEDS[1:])]
    preferred = max(qualifying, key=lambda name: rank(selection[name]["record"]), default="starting_candidate")
    write_json(RUN / "decision.json", {"preferred_by_development": preferred, "qualifying": qualifying,
                                       "starting_development": start, "limits": limits, "selections": selection})
    print("Decision locked:", preferred, flush=True)


def test_receipts(protocol):
    replay = read_json(EXPANDED / "evaluation_freeze.json")
    tests = {**replay["historical_replays"],
             "rachin_replay": {"path": str(EXPANDED / "rachin_evaluation.csv"), "sha256": replay["sha256"]}}
    if protocol["fresh_test"]:
        tests["khagan_charabag"] = protocol["fresh_test"]
    return tests


def evaluate(model, parent_hash, *, external=False):
    protocol = read_json(RUN / "protocol.json")
    if protocol["parent_sha256"] != parent_hash:
        raise ValueError("The locked starting checkpoint has changed")
    decision_path = RUN / "decision.json"
    decision = read_json(decision_path)
    decision_hash = sha256(decision_path)
    original = create_model("resnet18")
    original.load_state_dict(original_checkpoint("AC42")["model_state"])
    original.requires_grad_(False).to(model.device, memory_format=torch.channels_last).eval()
    forbidden = {row["sha256"] for split in protocol["manifests"].values()
                 for row in read_rows(verified(Path(split["path"]), split["sha256"]))}
    probes = {}
    keys = [f"{arm}_s42" for arm in ARMS]
    selected = decision["preferred_by_development"]
    if selected in keys:
        selected_arm = selected.rsplit("_s", 1)[0]
        keys.extend(f"{selected_arm}_s{seed}" for seed in SEEDS[1:])
    for key in keys:
        weight = verified(RUN / key / "probe.pt", decision["selections"][key]["weight_sha256"])
        state = torch.load(weight, map_location="cpu", weights_only=True)
        values = state["state"]
        probe = SubtypeProbe(values["mean"], values["scale"])
        probe.load_state_dict(values)
        probes[key] = (probe, state["feature"], state["fraction"])
    location = RUN
    tests = test_receipts(protocol)
    if external:
        location = RUN / "external_audits"
        location.mkdir(exist_ok=True)
        tests = read_json(EXPANDED / "evaluation_freeze.json")["external_audits"]
        audit_protocol = location / "protocol.json"
        if not audit_protocol.exists():
            write_json(audit_protocol, {"decision_sha256": decision_hash, "audits": tests,
                                        "role": "Inference only after primary checkpoint and fusion lock."})
        elif read_json(audit_protocol)["decision_sha256"] != decision_hash:
            raise ValueError("The external audit used a different subtype decision")
    for name, receipt in tests.items():
        path = verified(Path(receipt["path"]), receipt["sha256"])
        entries = read_rows(path)
        keys = {row["sha256"] for row in entries}
        if len(keys) != len(entries) or keys & forbidden:
            raise ValueError(f"Duplicate or overlapping test identities: {name}")
        result = location / f"{name}_results.json"
        if result.exists():
            frozen = read_json(result)
            if frozen["decision_sha256"] != decision_hash or frozen["manifest_sha256"] != receipt["sha256"]:
                raise ValueError(f"Test was evaluated with a different lock: {name}")
            continue
        data = encode(model, path, name, original)
        with torch.no_grad():
            scores = {"original_AC42": data["original"], "starting_candidate": data["prior"],
                      **{key: combine(data["prior"], probe(data[feature]), alpha)
                         for key, (probe, feature, alpha) in probes.items()}}
        metrics, predictions = {}, []
        for key, logits in scores.items():
            metrics[key], guesses, subtype = measurement(logits, data["truth"])
            for row, guessed, conditioned in zip(entries, guesses.tolist(), subtype.tolist()):
                predictions.append({"model": key, "sha256": row["sha256"], "true_label": row["label"],
                                    "prediction": CLASSES[guessed], "subtype_if_diseased": CLASSES[conditioned]})
            print(f"{name} {key} accuracy={metrics[key]['accuracy']:.5f}", flush=True)
        write_rows(location / f"{name}_predictions.csv", predictions)
        write_json(result, {"manifest_sha256": receipt["sha256"], "decision_sha256": decision_hash, "models": metrics})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate", "audit"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent_hash = read_json(FIRST_RUN / "selection.json")["candidate_sha256"]
    parent = checked_checkpoint(FIRST_RUN / "candidate.pt", parent_hash)
    model = Specialists(parent, device)
    if args.phase == "train":
        train(model, parent_hash)
    else:
        evaluate(model, parent_hash, external=args.phase == "audit")


if __name__ == "__main__":
    main()
