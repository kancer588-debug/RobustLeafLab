"""Compare within-batch and cross-batch subtype ranking on a fixed image cohort."""

import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import torch
from torch.nn import functional as F

from run_dino_health_trial import RUN as HEALTH_RUN, load_gate, training_splits
from run_early_review_trial import RUN as EARLY_RUN, restore
from run_late_protection_trial import load_heads
from run_local_leaf_trial import measure
from run_patch_subtype_trial import AGRI_AUDIT, TAIWAN, card, previous_scores, ranking, source_weights
from run_subtype_trial import (
    CLASSES, EXPANDED, FIRST_RUN, RUN as SUBTYPE_RUN, SEEDS, Specialists, SubtypeProbe, combine,
    development_metrics, encode, measurement, read_rows, test_receipts, verified, write_rows,
)
from train_online_candidate import (
    WORKSPACE, checked_checkpoint, create_model, original_checkpoint, read_json, seed_everything, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "matched_capture_trial"
DATA = WORKSPACE / "datasets" / "matched_capture_trial"
ARMS = ("pointwise", "cross_batch_pairs", "within_batch_pairs")
SNAPSHOTS = (1, 2, 4, 8, 12)
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
LR = 0.004
BATCH = 64
PAIRS_PER_GROUP = 8
PAIR_WEIGHT = 0.35
PAIR_MARGIN = 1.0


def capture_groups(rows):
    batches = defaultdict(lambda: {1: [], 2: []})
    for index, row in enumerate(rows):
        if row.get("dataset") == "tomato_village" and row["capture_group"]:
            batches[row["capture_group"]][CLASSES.index(row["label"])].append(index)
    groups = {name: labels for name, labels in sorted(batches.items()) if all(labels.values())}
    if len(groups) < 2:
        raise ValueError("A cross-batch control needs two or more mixed training batches")
    return groups


def pair_batch(groups, within_batch, index, rng):
    names = list(groups)
    offset = 0 if within_batch else 1 + index % (len(names) - 1)
    early, late = [], []
    for at, source in enumerate(names):
        comparison = names[(at + offset) % len(names)]
        for dest, members in ((early, groups[source][1]), (late, groups[comparison][2])):
            picks = torch.randint(len(members), (PAIRS_PER_GROUP,), generator=rng)
            dest.extend(members[item] for item in picks.tolist())
    return torch.tensor(early), torch.tensor(late)


def fit_head(arm, seed, fit, rows, dev, incumbent, start, malik_n):
    seed_everything(seed)
    disease_mask = fit["truth"] > 0
    labels = fit["truth"][disease_mask]
    disease_rows = [row for row, is_disease in zip(rows, disease_mask) if is_disease]
    targets = (labels == 1).float()
    x = fit["dino"][disease_mask]
    weights = source_weights(disease_rows, labels)
    groups = capture_groups(disease_rows)
    classifier = SubtypeProbe(x.mean(0), x.std(0, unbiased=False).clamp_min(0.02))
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR, weight_decay=0.05)
    shuffle = torch.Generator().manual_seed(seed)
    pairing = torch.Generator().manual_seed(seed + 1000)
    history, average_pairs, best = [], [], None
    step = 0
    for epoch in range(1, SNAPSHOTS[-1] + 1):
        total, paired_losses = 0.0, []
        for indices in torch.randperm(len(targets), generator=shuffle).split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(classifier(x[indices]), targets[indices],
                                                       weight=weights[indices])
            if arm != "pointwise":
                early, late = pair_batch(groups, arm == "within_batch_pairs", step, pairing)
                delta = classifier(x[early]) - classifier(x[late])
                penalty = F.softplus(PAIR_MARGIN - delta).mean()
                loss = loss + PAIR_WEIGHT * penalty
                paired_losses.append(penalty.item())
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
            step += 1
        if paired_losses:
            average_pairs.append(sum(paired_losses) / len(paired_losses))
        if epoch not in SNAPSHOTS:
            continue
        with torch.no_grad():
            evidence = classifier(dev["dino"])
            for fraction in FRACTIONS:
                item = card(combine(incumbent, evidence, fraction), dev, malik_n, start, epoch, fraction,
                            total / len(targets))
                item["record"]["pair_loss"] = average_pairs[-1] if average_pairs else None
                history.append(item["record"])
                if best is None or ranking(item) > ranking(best):
                    best = item
                    chosen = {key: value.detach().clone() for key, value in classifier.state_dict().items()}
    location = RUN / f"{arm}_s{seed}"
    location.mkdir(exist_ok=True)
    write_rows(location / "history.csv", history)
    artifact = location / "subtype_head.pt"
    temporary = artifact.with_suffix(".tmp")
    torch.save({"state": chosen, "fraction": best["record"]["fraction"]}, temporary)
    temporary.replace(artifact)
    best["checkpoint_sha256"] = sha256(artifact)
    best["steps"] = step
    print(location.name, json.dumps(best["record"]), flush=True)
    return best


def verified_cohort(protocol):
    splits, records, forbidden = training_splits(protocol)
    receipt = read_json(DATA / "freeze.json")
    if receipt["base_sha256"] != protocol["manifests"]["augmented"]["sha256"]:
        raise ValueError("The batch metadata were recovered for a different training cohort")
    spec = receipt["files"]["training"]
    source = verified(Path(spec["path"]), spec["sha256"])
    joined = read_rows(source)
    if [(row["sha256"], row["label"], row["path"]) for row in joined] != [
            (row["sha256"], row["label"], row["path"]) for row in records["augmented"]]:
        raise ValueError("The trial is a fixed-image comparison; only the metadata may change")
    splits["augmented"], records["augmented"] = source, joined
    return splits, records, forbidden, receipt


def train(encoder, protocol, components, prior_digest):
    splits, records, _, receipt = verified_cohort(protocol)
    RUN.mkdir(exist_ok=True)
    if (RUN / "decision.json").exists():
        raise FileExistsError("The matched-capture selection is already locked")
    plan = {"incumbent_decision_sha256": prior_digest, "training_manifest_sha256": sha256(splits["augmented"]),
            "capture_freeze_sha256": sha256(DATA / "freeze.json"), "mixed_groups": receipt["mixed_group_counts"],
            "seeds": list(SEEDS), "arms": list(ARMS), "snapshots": list(SNAPSHOTS), "fractions": list(FRACTIONS),
            "pair_weight": PAIR_WEIGHT, "pair_margin": PAIR_MARGIN, "pairs_per_group": PAIRS_PER_GROUP,
            "rank": "Maintain pooled and Malik Macro-F1 and both disease recalls. Require fewer subtype errors "
                    "for seed 42 and at least one confirmatory seed. Rank on development only, then lock.",
            "controls": "All arms use the same 555 training disease images and optimization steps. Cross-batch "
                        "and within-batch use the same number of pairs from the same publisher and capture days."}
    plan_path = RUN / "protocol.json"
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError("The existing run contains another protocol")
    else:
        write_json(plan_path, plan)
    fit, field, malik = [encode(encoder, path, name) for name, path in splits.items()]
    if [CLASSES[label] for label in fit["truth"].tolist()] != [row["label"] for row in records["augmented"]]:
        raise ValueError("Training images changed order")
    dev = {key: torch.cat((field[key], malik[key])) for key in field}
    incumbent = previous_scores(dev, *components)
    malik_n = len(malik["truth"])
    start = development_metrics(incumbent, dev, malik_n)
    current = read_json(EARLY_RUN / "decision.json")
    expected = current["selections"][current["preferred_by_development"]]["metrics"]
    if start["confusion"] != expected["confusion"] or start["malik"]["confusion"] != expected["malik"]["confusion"]:
        raise ValueError("Current candidate differs from its archived development predictions")
    outcomes = {f"{arm}_s{seed}": fit_head(arm, seed, fit, records["augmented"], dev, incumbent, start, malik_n)
                for arm, seed in product(ARMS, SEEDS)}
    qualifying = [f"{arm}_s42" for arm in ARMS if outcomes[f"{arm}_s42"]["record"]["eligible"]
                  and any(outcomes[f"{arm}_s{s}"]["record"]["eligible"] for s in SEEDS[1:])]
    best = max(qualifying, key=lambda key: ranking(outcomes[key]), default="incumbent")
    write_json(RUN / "decision.json", {"preferred_by_development": best, "qualifying": qualifying,
                                     "incumbent": start, "selections": outcomes, "incumbent_decision_sha256": prior_digest})
    print(f"Decision locked: {best}", flush=True)


def restore_matched(name, item):
    path = verified(RUN / name / "subtype_head.pt", item["checkpoint_sha256"])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["state"]
    model = SubtypeProbe(state["mean"], state["scale"])
    model.load_state_dict(state)
    return model.eval(), saved["fraction"]


def evaluate(encoder, protocol, components, prior_digest):
    _, _, forbidden, _ = verified_cohort(protocol)
    decision_path = RUN / "decision.json"
    decision = read_json(decision_path)
    decision_digest = sha256(decision_path)
    if decision["incumbent_decision_sha256"] != prior_digest:
        raise ValueError("The current model changed after checkpoint selection")
    protected = read_json(EXPANDED / "evaluation_freeze.json")["external_audits"]
    tests = {**test_receipts(protocol), **read_json(TAIWAN)["tests"], **protected, **read_json(AGRI_AUDIT)["tests"]}
    names = [f"{arm}_s42" for arm in ARMS]
    best = decision["preferred_by_development"]
    if best in names:
        group = best.rsplit("_s", 1)[0]
        names.extend(f"{group}_s{s}" for s in SEEDS[1:])
    alternatives = {key: restore_matched(key, decision["selections"][key]) for key in names}
    original = create_model("resnet18")
    original.load_state_dict(original_checkpoint("AC42")["model_state"])
    original.requires_grad_(False).to(encoder.device, memory_format=torch.channels_last).eval()
    for name, spec in tests.items():
        path = verified(Path(spec["path"]), spec["sha256"])
        rows = read_rows(path)
        keys = {row["sha256"] for row in rows}
        if len(rows) != len(keys) or keys & forbidden:
            raise ValueError(f"Duplicate or overlapping evaluation images: {name}")
        result = RUN / f"{name}_results.json"
        if result.exists():
            existing = read_json(result)
            if existing["decision_sha256"] != decision_digest or existing["manifest_sha256"] != spec["sha256"]:
                raise ValueError(f"Evaluation was previously performed with a different lock: {name}")
            continue
        data = encode(encoder, path, name, original)
        incumbent = previous_scores(data, *components)
        with torch.no_grad():
            scores = {"original_AC42": data["original"], "incumbent": incumbent,
                      **{key: combine(incumbent, model(data["dino"]), fraction)
                         for key, (model, fraction) in alternatives.items()}}
        metrics, predictions = {}, []
        truth = data["truth"]
        three_class = truth.unique().numel() == len(CLASSES)
        for key, logits in scores.items():
            if three_class:
                metrics[key], predictions_for_head, subtype = measurement(logits, truth)
            else:
                metrics[key], predictions_for_head = measure(logits, truth)
                subtype = logits[:, 1:].argmax(1) + 1
            predictions.extend({"model": key, "sha256": row["sha256"], "true_label": row["label"],
                                "prediction": CLASSES[label], "subtype_if_diseased": CLASSES[specialization]}
                               for row, label, specialization in zip(rows, predictions_for_head.tolist(), subtype.tolist()))
            matrix = metrics[key]["confusion"]
            print(f"{name} {key} accuracy={metrics[key]['accuracy']:.5f} early={matrix[1][1]} late={matrix[2][2]}", flush=True)
        write_rows(RUN / f"{name}_predictions.csv", predictions)
        write_json(result, {"decision_sha256": decision_digest, "manifest_sha256": spec["sha256"],
                            "role": "external_audit" if name in protected else spec.get("role", "historical_replay"),
                            "models": metrics})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lock = EARLY_RUN / "decision.json"
    early = read_json(lock)
    health = read_json(verified(HEALTH_RUN / "decision.json", early["health_decision_sha256"]))
    subtype = read_json(verified(SUBTYPE_RUN / "decision.json", health["subtype_decision_sha256"]))
    health_key, early_key = health["preferred_by_development"], early["preferred_by_development"]
    components = (load_heads(subtype), load_gate(health_key, health["selections"][health_key]),
                  restore(early_key, early["selections"][early_key]))
    protocol = read_json(SUBTYPE_RUN / "protocol.json")
    model = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", protocol["parent_sha256"]), device)
    phase = train if args.phase == "train" else evaluate
    phase(model, protocol, components, sha256(lock))


if __name__ == "__main__":
    main()
