"""Train a DINOv2 health gate while keeping the locked early/late decisions."""

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

import torch
from torch.nn import functional as F

from run_late_protection_trial import RUN as PROTECTION, load_heads, margins
from run_local_leaf_trial import measure
from run_subtype_trial import (
    CLASSES, DINO_HASH, EXPANDED, FIRST_RUN, RUN as SUBTYPE, SEEDS, Specialists, SubtypeProbe,
    combine, development_metrics, encode, measurement, read_rows, test_receipts, verified, write_rows,
)
from train_online_candidate import (
    WORKSPACE, checked_checkpoint, create_model, original_checkpoint, read_json, seed_everything, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "dino_health_trial"
TAIWAN = WORKSPACE / "datasets" / "local_leaf_trial" / "fresh_test_freeze.json"
ARMS = ("class_balanced", "source_balanced", "boundary_weighted")
STRENGTHS = (0.25, 0.5, 1.0, 2.0, 4.0)
SNAPSHOTS = (1, 2, 4, 8, 12)
LR = 0.004
BATCH = 64


def healthy_margin(scores):
    return scores[:, 1:].amax(1) - scores[:, 0]


def after_gate(incumbent, evidence, strength):
    """Move only the healthy/disease boundary and retain the disease subtype order."""
    subtype_scores = incumbent[:, 1:]
    relative = subtype_scores - subtype_scores.amax(1, keepdim=True)
    new_margin = healthy_margin(incumbent) + strength * (evidence * 0.5).tanh()
    return torch.cat((torch.zeros_like(new_margin[:, None]), relative + new_margin[:, None]), dim=1)


def source_name(row):
    name = (row.get("dataset") or row["source"]).strip().lower()
    if name.startswith("tom2024"):
        return "tom2024"
    return name


def sample_weights(rows, labels, base_margin, arm):
    keys = [(row["label"], source_name(row)) for row in rows]
    counts = Counter(keys)
    weights = torch.ones(len(rows))
    if arm != "class_balanced":
        weights = torch.tensor([max(counts[key], 8) ** -0.5 for key in keys])
    if arm == "boundary_weighted":
        wrong = (base_margin > 0) != (labels > 0)
        weights *= 1 + (-base_margin.abs()).exp() + 2 * wrong
    for label, share in enumerate((0.5, 0.25, 0.25)):
        chosen = labels == label
        weights[chosen] *= share * len(rows) / weights[chosen].sum()
    return weights


def capture_indices(rows, truth):
    groups = {}
    for index, row in enumerate(rows):
        group = row["capture_group"] or "unspecified"
        groups.setdefault(group, []).append(index)
    return [(indices, truth[indices] > 0) for indices in groups.values()]


def group_score(scores, groups):
    diseased = scores.argmax(1) > 0
    results = []
    for indices, truth in groups:
        hits = diseased[indices] == truth
        recalls = [hits[truth == cls].float().mean().item() for cls in (False, True) if (truth == cls).any()]
        results.append(sum(recalls) / len(recalls))
    return sum(results) / len(results)


def record(scores, development, malik_n, groups, start, epoch, strength, loss):
    measures = development_metrics(scores, development, malik_n)
    matrix, old = measures["confusion"], start["confusion"]
    malik, old_malik = measures["malik"], start["malik"]
    current_groups = group_score(scores, groups)
    passes = (
        measures["healthy_false_alarm_count"] <= start["healthy_false_alarm_count"]
        and matrix[1][0] <= old[1][0] and matrix[2][0] <= old[2][0]
        and matrix[1][1] >= old[1][1] and matrix[2][2] >= old[2][2]
        and measures["macro_f1"] >= start["macro_f1"]
        and malik["macro_f1"] >= old_malik["macro_f1"]
        and malik["healthy_false_alarm_count"] <= old_malik["healthy_false_alarm_count"]
        and malik["late_to_healthy"] <= old_malik["late_to_healthy"]
        and current_groups >= start["capture_group_binary_recall"]
        and (measures["healthy_false_alarm_count"] < start["healthy_false_alarm_count"]
             or matrix[1][0] + matrix[2][0] < old[1][0] + old[2][0])
    )
    result = {"epoch": epoch, "strength": strength, "loss": loss, "eligible": passes,
              "macro_f1": measures["macro_f1"], "malik_macro_f1": malik["macro_f1"],
              "capture_group_binary_recall": current_groups,
              "healthy_false_alarms": measures["healthy_false_alarm_count"],
              "early_to_healthy": matrix[1][0], "late_to_healthy": matrix[2][0],
              "early_correct": matrix[1][1], "late_correct": matrix[2][2]}
    measures["capture_group_binary_recall"] = current_groups
    return {"record": result, "metrics": measures}


def priority(card):
    row = card["record"]
    return (row["eligible"], row["capture_group_binary_recall"],
            (row["macro_f1"] + row["malik_macro_f1"]) / 2, -row["strength"], -row["epoch"])


def train_head(arm, seed, fit, rows, dev, incumbent, malik_n, groups, start):
    seed_everything(seed)
    features, labels = fit["dino"], fit["truth"]
    classifier = SubtypeProbe(features.mean(0), features.std(0, unbiased=False).clamp_min(0.02))
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR, weight_decay=0.05)
    weights = sample_weights(rows, labels, healthy_margin(fit["prior"]), arm)
    targets = (labels > 0).float()
    best, history = None, []
    for epoch in range(1, SNAPSHOTS[-1] + 1):
        total_loss = 0.0
        for indices in torch.randperm(len(features)).split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(classifier(features[indices]), targets[indices],
                                                       weight=weights[indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        if epoch not in SNAPSHOTS:
            continue
        with torch.no_grad():
            evidence = classifier(dev["dino"])
            for strength in STRENGTHS:
                scores = after_gate(incumbent, evidence, strength)
                card = record(scores, dev, malik_n, groups, start, epoch, strength, total_loss / len(features))
                history.append(card["record"])
                if best is None or priority(card) > priority(best):
                    best = card
                    state = {name: value.detach().clone() for name, value in classifier.state_dict().items()}
    key = f"{arm}_s{seed}"
    location = RUN / key
    location.mkdir(exist_ok=True)
    write_rows(location / "history.csv", history)
    checkpoint = location / "gate.pt"
    temporary = checkpoint.with_suffix(".tmp")
    torch.save({"state": state, "strength": best["record"]["strength"]}, temporary)
    temporary.replace(checkpoint)
    best["checkpoint_sha256"] = sha256(checkpoint)
    print(key, json.dumps(best["record"]), flush=True)
    return best


def training_splits(protocol):
    names = ("augmented", "development_field", "development_malik")
    splits, identities, records = {}, set(), {}
    for name in names:
        spec = protocol["manifests"][name]
        splits[name] = verified(Path(spec["path"]), spec["sha256"])
        data = read_rows(splits[name])
        hashes = {row["sha256"] for row in data}
        if len(hashes) != len(data) or hashes & identities:
            raise ValueError(f"Duplicate or overlapping manifest: {name}")
        records[name] = data
        identities.update(hashes)
    return splits, records, identities


def train(model, context, heads):
    protocol, prior_hash, routing_hash = context
    splits, rows, _ = training_splits(protocol)
    RUN.mkdir(exist_ok=True)
    if (RUN / "decision.json").exists():
        raise FileExistsError("The health gate selection is already locked")
    plan = {"subtype_decision_sha256": prior_hash, "late_protection_decision_sha256": routing_hash,
            "dinov2_sha256": DINO_HASH, "manifests": {name: protocol["manifests"][name] for name in splits},
            "arms": list(ARMS), "strengths": list(STRENGTHS), "snapshots": list(SNAPSHOTS), "seeds": list(SEEDS),
            "selection": "Keep the seed-42 locked subtype model fixed. Require improved healthy/diseased routing "
                         "with no increase in any pooled class error across the binary boundary, pooled and Malik "
                         "Macro-F1, subtype recalls, and mean capture-group binary recall preserved, plus Malik "
                         "healthy false positives and late false negatives preserved. Require seed 42 and at least "
                         "one confirmatory seed. Rank by group balanced binary recall, then mean pooled/Malik Macro-F1.",
            "evaluation": "Prior public cohorts and protected audits are opened only after the decision lock."}
    plan_path = RUN / "protocol.json"
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError("The existing run used a different protocol")
    else:
        write_json(plan_path, plan)
    fit, field, malik = [encode(model, path, name) for name, path in splits.items()]
    if [CLASSES[value] for value in fit["truth"].tolist()] != [row["label"] for row in rows["augmented"]]:
        raise ValueError("Training rows changed order")
    dev = {key: torch.cat((field[key], malik[key])) for key in field}
    groups = capture_indices(rows["development_field"] + rows["development_malik"], dev["truth"])
    candidate = combine(dev["prior"], margins(dev, heads)[42], 1.0)
    malik_n = len(malik["truth"])
    start = development_metrics(candidate, dev, malik_n)
    start["capture_group_binary_recall"] = group_score(candidate, groups)
    results = {f"{arm}_s{seed}": train_head(arm, seed, fit, rows["augmented"], dev, candidate,
                                          malik_n, groups, start) for arm, seed in product(ARMS, SEEDS)}
    qualified = [f"{arm}_s42" for arm in ARMS if results[f"{arm}_s42"]["record"]["eligible"]
                 and any(results[f"{arm}_s{s}"]["record"]["eligible"] for s in SEEDS[1:])]
    best = max(qualified, key=lambda name: priority(results[name]), default="dino_incumbent")
    write_json(RUN / "decision.json", {"preferred_by_development": best, "qualifying": qualified,
                                     "incumbent_development": start, "selections": results,
                                     "subtype_decision_sha256": prior_hash})
    print(f"Decision locked: {best}", flush=True)


def load_gate(name, card):
    path = verified(RUN / name / "gate.pt", card["checkpoint_sha256"])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["state"]
    layer = SubtypeProbe(state["mean"], state["scale"])
    layer.load_state_dict(state)
    return layer.eval(), saved["strength"]


def evaluate(model, context, heads):
    protocol, prior_hash, _ = context
    decision_path = RUN / "decision.json"
    decision = read_json(decision_path)
    decision_hash = sha256(decision_path)
    if decision["subtype_decision_sha256"] != prior_hash:
        raise ValueError("The locked subtype head changed after gate selection")
    _, _, forbidden = training_splits(protocol)
    receipts = test_receipts(protocol)
    taiwan = read_json(TAIWAN)
    if taiwan["compatible_training_sha256"] != protocol["manifests"]["base"]["sha256"]:
        raise ValueError("Taiwan data were screened for a different base sample")
    protected = read_json(EXPANDED / "evaluation_freeze.json")["external_audits"]
    original = create_model("resnet18")
    original.load_state_dict(original_checkpoint("AC42")["model_state"])
    original.requires_grad_(False).to(model.device, memory_format=torch.channels_last).eval()
    names = [f"{arm}_s42" for arm in ARMS]
    selected = decision["preferred_by_development"]
    if selected in names:
        family = selected.rsplit("_s", 1)[0]
        names.extend(f"{family}_s{s}" for s in SEEDS[1:])
    gates = {name: load_gate(name, decision["selections"][name]) for name in names}
    for name, receipt in {**receipts, **taiwan["tests"], **protected}.items():
        path = verified(Path(receipt["path"]), receipt["sha256"])
        rows = read_rows(path)
        hashes = {row["sha256"] for row in rows}
        if len(hashes) != len(rows) or hashes & forbidden:
            raise ValueError(f"Evaluation overlaps training or development: {name}")
        result = RUN / f"{name}_results.json"
        if result.exists():
            saved = read_json(result)
            if saved["decision_sha256"] != decision_hash or saved["manifest_sha256"] != receipt["sha256"]:
                raise ValueError(f"Test already uses another decision or manifest: {name}")
            continue
        data = encode(model, path, name, original)
        prior = data["prior"]
        incumbent = combine(prior, margins(data, heads)[42], 1.0)
        with torch.no_grad():
            experiments = {"original_AC42": data["original"], "starting_candidate": prior, "dino_incumbent": incumbent,
                           **{key: after_gate(incumbent, layer(data["dino"]), strength)
                              for key, (layer, strength) in gates.items()}}
        truth = data["truth"]
        three_classes = truth.unique().numel() == len(CLASSES)
        metrics, predictions = {}, []
        for key, scores in experiments.items():
            if three_classes:
                values, classified, subtype = measurement(scores, truth)
            else:
                values, classified = measure(scores, truth)
                subtype = scores[:, 1:].argmax(1) + 1
            metrics[key] = values
            predictions.extend({"model": key, "sha256": row["sha256"], "true_label": row["label"],
                                "prediction": CLASSES[guess], "subtype_if_diseased": CLASSES[conditioned]}
                               for row, guess, conditioned in zip(rows, classified.tolist(), subtype.tolist()))
            cells = values["confusion"]
            print(f"{name} {key} accuracy={values['accuracy']:.5f} healthy_fp={values['healthy_false_alarm_count']} "
                  f"early_to_healthy={cells[1][0]} late_to_healthy={cells[2][0]}", flush=True)
        write_rows(RUN / f"{name}_predictions.csv", predictions)
        role = "external_audit" if name in protected else "historical_replay"
        write_json(result, {"manifest_sha256": receipt["sha256"], "decision_sha256": decision_hash,
                            "role": role, "models": metrics})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent = read_json(FIRST_RUN / "selection.json")["candidate_sha256"]
    protocol = read_json(SUBTYPE / "protocol.json")
    verified(SUBTYPE / "decision.json", read_json(PROTECTION / "decision.json")["prior_decision_sha256"])
    if protocol["parent_sha256"] != parent or read_json(PROTECTION / "decision.json")["preferred_by_development"] != "dino_incumbent":
        raise ValueError("The starting configuration differs from the incumbent")
    context = protocol, sha256(SUBTYPE / "decision.json"), sha256(PROTECTION / "decision.json")
    model = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", parent), device)
    heads = load_heads(read_json(SUBTYPE / "decision.json"))
    phase = train if args.phase == "train" else evaluate
    phase(model, context, heads)


if __name__ == "__main__":
    main()
