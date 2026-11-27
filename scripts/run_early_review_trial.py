"""Fit an early-blight review head for the healthy/early decision boundary."""

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from run_backbone_trial import move_images
from run_dino_health_trial import RUN as HEALTH_RUN, TAIWAN, after_gate, load_gate, source_name, training_splits
from run_late_protection_trial import load_heads, margins
from run_local_leaf_trial import measure
from run_subtype_trial import (
    CLASSES, DINO_HASH, EXPANDED, FIRST_RUN, RUN as SUBTYPE_RUN, SEEDS, Specialists, SubtypeProbe,
    combine, development_metrics, measurement, read_rows, test_receipts, verified, write_rows,
)
from train_online_candidate import (
    WORKSPACE, ManifestDataset, checked_checkpoint, create_model, original_checkpoint, read_json,
    seed_everything, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "early_review_trial"
VIEWS = ("standard", "high_resolution")
MODES = ("rescue_only", "two_way")
STRENGTHS = (0.5, 1.0, 2.0, 4.0)
SNAPSHOTS = (1, 2, 4, 8, 12)
HIGH_RESOLUTION = 336
MAX_HEALTHY_FP_INCREASE = 2
LR = 0.004
HEAD_BATCH = 64
IMAGE_BATCH = 16


@torch.no_grad()
def encode_views(model, path, name, original=None):
    dataset = ManifestDataset(path)
    standard = dataset.tf
    enlarged = transforms.Compose([transforms.Resize(384), transforms.CenterCrop(HIGH_RESOLUTION),
                                   *standard.transforms[2:]])
    dataset.tf = lambda image: (standard(image), enlarged(image))
    collection = []
    for (normal, large), truth in DataLoader(dataset, batch_size=IMAGE_BATCH, shuffle=False, num_workers=0):
        normal = move_images(normal, model.device)
        _, embed, prior = model(normal)
        high_tokens = model.dino.forward_features(large.to(model.device))
        high = torch.cat((high_tokens["x_norm_clstoken"], high_tokens["x_norm_patchtokens"].mean(1)), dim=1)
        data = {"dino": embed.cpu(), "standard": embed.cpu(), "high_resolution": high.cpu(),
                "prior": prior.cpu(), "truth": truth}
        if original is not None:
            data["original"] = original(normal).cpu()
        collection.append(data)
    result = {key: torch.cat([item[key] for item in collection]) for key in collection[0]}
    print(f"Encoded {name}: {len(result['truth'])} images at 224 and 336", flush=True)
    return result


@torch.no_grad()
def incumbent_scores(data, heads, gate):
    layer, strength = gate
    subtype = combine(data["prior"], margins(data, heads)[42], 1.0)
    return after_gate(subtype, layer(data["dino"]), strength)


def apply_review(incumbent, evidence, strength, mode):
    """Reconsider the healthy score on images for which early is the leading disease."""
    top = incumbent.argmax(1)
    early_leads = incumbent[:, 1] >= incumbent[:, 2]
    eligible = early_leads if mode == "two_way" else early_leads & (top == 0)
    correction = strength * (0.5 * evidence).tanh()
    if mode == "rescue_only":
        correction = correction.clamp_min(0)
    healthy = incumbent[:, 0] - torch.where(eligible, correction, 0)
    return torch.cat((healthy[:, None], incumbent[:, 1:]), dim=1)


def fitting_weights(rows, scores, labels):
    groups = [(row["label"], source_name(row)) for row in rows]
    freq = Counter(groups)
    weights = torch.tensor([max(freq[group], 8) ** -0.5 for group in groups])
    early_margin = scores[:, 1] - scores[:, 0]
    mismatch = (early_margin > 0) != (labels == 1)
    weights *= 1 + (-early_margin.abs()).exp() + 2 * mismatch
    for label, share in enumerate((0.375, 0.5, 0.125)):
        mask = labels == label
        weights[mask] *= share * len(rows) / weights[mask].sum()
    return weights


def summarize(metrics, start, epoch, mode, strength, loss):
    confusion = metrics["confusion"]
    old = start["confusion"]
    malik, old_malik = metrics["malik"], start["malik"]
    false_alarm_increase = metrics["healthy_false_alarm_count"] - start["healthy_false_alarm_count"]
    early_rescued = old[1][0] - confusion[1][0]
    guardrails = (false_alarm_increase <= MAX_HEALTHY_FP_INCREASE
                  and confusion[2][2] == old[2][2]
                  and confusion[2][1] <= old[2][1] + 1
                  and malik["healthy_false_alarm_count"] <= old_malik["healthy_false_alarm_count"]
                  and malik["macro_f1"] >= old_malik["macro_f1"]
                  and metrics["macro_f1"] >= start["macro_f1"])
    return {"epoch": epoch, "mode": mode, "strength": strength, "loss": loss,
            "eligible": guardrails and early_rescued > 0, "other_guardrails": guardrails,
            "early_rescued_net": early_rescued, "healthy_false_alarm_change": false_alarm_increase,
            "late_to_early_change": confusion[2][1] - old[2][1], "early_correct": confusion[1][1],
            "late_correct": confusion[2][2], "macro_f1": metrics["macro_f1"], "malik_macro_f1": malik["macro_f1"]}


def priority(item):
    row = item["record"]
    return (row["eligible"], row["other_guardrails"], row["early_rescued_net"],
            (row["macro_f1"] + row["malik_macro_f1"]) / 2, -row["healthy_false_alarm_change"],
            -row["strength"], -row["epoch"])


def fit_head(view, seed, fit, incumbent_fit, fit_rows, dev, incumbent_dev, start, malik_n):
    seed_everything(seed)
    x, truth = fit[view], fit["truth"]
    layer = SubtypeProbe(x.mean(0), x.std(0, unbiased=False).clamp_min(0.02))
    optimizer = torch.optim.AdamW(layer.parameters(), lr=LR, weight_decay=0.05)
    target = (truth == 1).float()
    weights = fitting_weights(fit_rows, incumbent_fit, truth)
    best, diagnostic, history = None, None, []
    for epoch in range(1, SNAPSHOTS[-1] + 1):
        total_loss = 0.0
        for ids in torch.randperm(len(x)).split(HEAD_BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(layer(x[ids]), target[ids], weight=weights[ids])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(ids)
        if epoch not in SNAPSHOTS:
            continue
        with torch.no_grad():
            evidence = layer(dev[view])
            for mode, strength in product(MODES, STRENGTHS):
                scores = apply_review(incumbent_dev, evidence, strength, mode)
                metrics = development_metrics(scores, dev, malik_n)
                record = summarize(metrics, start, epoch, mode, strength, total_loss / len(x))
                record["development_predictions_changed"] = int((scores.argmax(1) != incumbent_dev.argmax(1)).sum())
                history.append(record)
                item = {"record": record, "metrics": metrics}
                if diagnostic is None or (record["early_rescued_net"], priority(item)) > (
                        diagnostic["record"]["early_rescued_net"], priority(diagnostic)):
                    diagnostic = item
                if best is None or priority(item) > priority(best):
                    best = item
                    state = {name: value.detach().clone() for name, value in layer.state_dict().items()}
    directory = RUN / f"{view}_s{seed}"
    directory.mkdir(exist_ok=True)
    write_rows(directory / "history.csv", history)
    path = directory / "early_head.pt"
    temporary = path.with_suffix(".tmp")
    torch.save({"state": state, "view": view, "mode": best["record"]["mode"],
                "strength": best["record"]["strength"]}, temporary)
    temporary.replace(path)
    best["checkpoint_sha256"] = sha256(path)
    best["most_development_early_rescued"] = diagnostic["record"]
    print(directory.name, json.dumps(best["record"]), flush=True)
    return best


def train(model, protocol, heads, gate, prior_hash):
    splits, records, _ = training_splits(protocol)
    RUN.mkdir(exist_ok=True)
    if (RUN / "decision.json").exists():
        raise FileExistsError("The early review decision has already been locked")
    plan = {"health_decision_sha256": prior_hash, "subtype_sha256": sha256(SUBTYPE_RUN / "decision.json"),
            "dinov2_sha256": DINO_HASH, "manifests": {name: protocol["manifests"][name] for name in splits},
            "views": list(VIEWS), "modes": list(MODES), "seeds": list(SEEDS), "max_healthy_false_alarm_increase": 2,
            "strengths": list(STRENGTHS), "snapshots": list(SNAPSHOTS), "fit_lr": LR, "fit_batch": HEAD_BATCH,
            "selection": "Require fewer early-to-healthy errors, at most two added healthy false positives, "
                         "preserved overall and Malik Macro-F1 and Malik healthy false positives, fixed late recall, "
                         "and at most one extra late-to-early error. Confirm the primary seed in at least one other seed.",
            "evaluation": "Use only development for fitting decisions. Open public replays and external audits after lock."}
    config = RUN / "protocol.json"
    if config.exists():
        if read_json(config) != plan:
            raise ValueError("This directory contains a different trial protocol")
    else:
        write_json(config, plan)
    fit, field, malik = [encode_views(model, path, name) for name, path in splits.items()]
    if [CLASSES[item] for item in fit["truth"].tolist()] != [row["label"] for row in records["augmented"]]:
        raise ValueError("The fitting manifest and feature order differ")
    dev = {key: torch.cat((field[key], malik[key])) for key in field}
    fit_scores, dev_scores = (incumbent_scores(data, heads, gate) for data in (fit, dev))
    malik_n = len(malik["truth"])
    start = development_metrics(dev_scores, dev, malik_n)
    locked = read_json(HEALTH_RUN / "decision.json")
    expected = locked["selections"][locked["preferred_by_development"]]["metrics"]
    if start["confusion"] != expected["confusion"] or start["malik"]["confusion"] != expected["malik"]["confusion"]:
        raise ValueError("The frozen health gate does not reproduce its development predictions")
    results = {f"{view}_s{seed}": fit_head(view, seed, fit, fit_scores, records["augmented"], dev,
                                           dev_scores, start, malik_n) for view, seed in product(VIEWS, SEEDS)}
    eligible = [f"{view}_s42" for view in VIEWS if results[f"{view}_s42"]["record"]["eligible"]
                and any(results[f"{view}_s{seed}"]["record"]["eligible"] for seed in SEEDS[1:])]
    preferred = max(eligible, key=lambda name: priority(results[name]), default="incumbent")
    write_json(RUN / "decision.json", {"preferred_by_development": preferred, "qualifying": eligible,
                                     "incumbent": start, "selections": results, "health_decision_sha256": prior_hash})
    print(f"Decision locked: {preferred}", flush=True)


def restore(name, card):
    path = verified(RUN / name / "early_head.pt", card["checkpoint_sha256"])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["state"]
    layer = SubtypeProbe(state["mean"], state["scale"])
    layer.load_state_dict(state)
    return layer.eval(), saved["view"], saved["mode"], saved["strength"]


def evaluate(model, protocol, heads, gate, prior_hash, *, extra_audit=None):
    decision_path = RUN / "decision.json"
    decision = read_json(decision_path)
    decision_hash = sha256(decision_path)
    if decision["health_decision_sha256"] != prior_hash:
        raise ValueError("The incumbent health gate has changed")
    _, _, forbidden = training_splits(protocol)
    replays = test_receipts(protocol)
    protected = read_json(EXPANDED / "evaluation_freeze.json")["external_audits"]
    taiwan = read_json(TAIWAN)
    if taiwan["compatible_training_sha256"] != protocol["manifests"]["base"]["sha256"]:
        raise ValueError("Taiwan was screened for a different underlying training cohort")
    additions = {}
    if extra_audit is not None:
        cohort = read_json(extra_audit)
        if cohort["compatible_train_sha256"] != protocol["manifests"]["augmented"]["sha256"]:
            raise ValueError("The extra audit was screened for different training data")
        additions = cohort["tests"]
        if set(additions) & (set(replays) | set(taiwan["tests"]) | set(protected)):
            raise ValueError("An extra audit would replace an existing test name")
    names = [f"{view}_s42" for view in VIEWS]
    selected = decision["preferred_by_development"]
    if selected in names:
        family = selected.rsplit("_s", 1)[0]
        names.extend(f"{family}_s{seed}" for seed in SEEDS[1:])
    layers = {name: restore(name, decision["selections"][name]) for name in names}
    baseline = create_model("resnet18")
    baseline.load_state_dict(original_checkpoint("AC42")["model_state"])
    baseline.requires_grad_(False).to(model.device, memory_format=torch.channels_last).eval()
    for name, receipt in {**replays, **taiwan["tests"], **protected, **additions}.items():
        path = verified(Path(receipt["path"]), receipt["sha256"])
        rows = read_rows(path)
        identities = {row["sha256"] for row in rows}
        if len(identities) != len(rows) or identities & forbidden:
            raise ValueError(f"Duplicate or overlapping identities in {name}")
        path_out = RUN / f"{name}_results.json"
        if path_out.exists():
            existing = read_json(path_out)
            if existing["decision_sha256"] != decision_hash or existing["manifest_sha256"] != receipt["sha256"]:
                raise ValueError(f"Existing evaluation is for a different manifest or decision: {name}")
            continue
        data = encode_views(model, path, name, baseline)
        incumbent = incumbent_scores(data, heads, gate)
        with torch.no_grad():
            alternatives = {"original_AC42": data["original"], "incumbent": incumbent,
                            **{key: apply_review(incumbent, layer(data[view]), strength, mode)
                               for key, (layer, view, mode, strength) in layers.items()}}
        metrics, predictions = {}, []
        truth = data["truth"]
        for key, scores in alternatives.items():
            if truth.unique().numel() == len(CLASSES):
                values, labels, subtype = measurement(scores, truth)
            else:
                values, labels = measure(scores, truth)
                subtype = scores[:, 1:].argmax(1) + 1
            metrics[key] = values
            predictions.extend({"model": key, "sha256": row["sha256"], "true_label": row["label"],
                                "prediction": CLASSES[label], "subtype_if_diseased": CLASSES[disease]}
                               for row, label, disease in zip(rows, labels.tolist(), subtype.tolist()))
            matrix = values["confusion"]
            print(f"{name} {key} accuracy={values['accuracy']:.5f} early_to_healthy={matrix[1][0]} "
                  f"healthy_fp={values['healthy_false_alarm_count']}", flush=True)
        write_rows(RUN / f"{name}_predictions.csv", predictions)
        role = "external_audit" if name in protected else receipt.get("role", "historical_replay")
        write_json(path_out, {"decision_sha256": decision_hash, "manifest_sha256": receipt["sha256"],
                              "role": role, "models": metrics})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    parser.add_argument("--extra-audit", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = read_json(SUBTYPE_RUN / "protocol.json")
    health_path = HEALTH_RUN / "decision.json"
    health = read_json(health_path)
    subtype_path = verified(SUBTYPE_RUN / "decision.json", health["subtype_decision_sha256"])
    starting = read_json(FIRST_RUN / "selection.json")["candidate_sha256"]
    if protocol["parent_sha256"] != starting:
        raise ValueError("The starting model has changed")
    name = health["preferred_by_development"]
    gate = load_gate(name, health["selections"][name])
    heads = load_heads(read_json(subtype_path))
    model = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", starting), device)
    if args.phase == "train":
        if args.extra_audit is not None:
            parser.error("An extra audit may only be opened after selection")
        train(model, protocol, heads, gate, sha256(health_path))
    else:
        evaluate(model, protocol, heads, gate, sha256(health_path), extra_audit=args.extra_audit)


if __name__ == "__main__":
    main()
