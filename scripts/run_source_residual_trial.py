"""Test source-debiased residuals for early-versus-late blight classification."""

import argparse
from itertools import product
from pathlib import Path

import torch
from torch.nn import functional as F

from run_dino_health_trial import RUN as HEALTH_RUN, load_gate, source_name
from run_early_review_trial import RUN as EARLY_RUN, restore
from run_late_protection_trial import load_heads
from run_local_leaf_trial import measure
from run_matched_capture_trial import verified_cohort
from run_patch_subtype_trial import AGRI_AUDIT, TAIWAN, card, previous_scores, ranking, source_weights
from run_subtype_trial import (
    CLASSES, EXPANDED, FIRST_RUN, RUN as SUBTYPE_RUN, SEEDS, Specialists, SubtypeProbe, combine,
    development_metrics, encode, measurement, read_rows, test_receipts, verified, write_rows,
)
from train_online_candidate import (
    WORKSPACE, checked_checkpoint, create_model, original_checkpoint, read_json, seed_everything, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "source_residual_trial"
ARMS = ("unprojected", "source_projected")
SNAPSHOTS = (1, 2, 4, 8, 12)
STRENGTHS = (0.25, 0.5, 1.0, 2.0)
MIN_PER_DISEASE = 12
LEARNING_RATE = 0.002
BATCH_SIZE = 64


def disease_margin(scores):
    return scores[:, 1] - scores[:, 2]


def source_basis(features, labels, rows):
    sources = [source_name(row) for row in rows]
    classes_by_source = {
        name: [features[torch.tensor([source == name for source in sources]) & (labels == label)]
               for label in (1, 2)]
        for name in sorted(set(sources))
    }
    covered = {name: pair for name, pair in classes_by_source.items()
               if min(map(len, pair)) >= MIN_PER_DISEASE}
    if len(covered) < 2:
        raise ValueError("Source projection requires two training sources covering both diseases")
    means = torch.stack([torch.stack([samples.mean(0) for samples in pair]) for pair in covered.values()])
    class_means = means.mean(0)
    disease_axis = F.normalize(class_means[0] - class_means[1], dim=0)
    offsets = (means - class_means).flatten(0, 1)
    offsets = offsets - (offsets @ disease_axis).unsqueeze(1) * disease_axis
    _, singular, basis = torch.linalg.svd(offsets, full_matrices=False)
    basis = basis[singular > singular.max() * 1e-5]
    if not len(basis) or (basis @ disease_axis).abs().max() > 1e-4:
        raise ValueError("Unexpected source basis or disease direction")
    counts = {name: dict(zip(CLASSES[1:], map(len, pair))) for name, pair in covered.items()}
    return basis, counts


def project(features, basis):
    return features - (features @ basis.T) @ basis if len(basis) else features


def corrected_margin(margin, evidence, scale, strength):
    confidence_weight = (1 + (margin / scale).square()).reciprocal()
    return margin + strength * scale * confidence_weight * evidence.tanh()


def predict(incumbent, features, head, basis, scale, strength):
    margin = disease_margin(incumbent)
    adjusted = corrected_margin(margin, head(project(features, basis)), scale, strength)
    return combine(incumbent, adjusted, 1.0)


def train_head(arm, seed, x, labels, weights, margins, dev, incumbent, start, malik_n, basis):
    seed_everything(seed)
    x = project(x, basis)
    projected_dev = project(dev["dino"], basis)
    dev_margin = disease_margin(incumbent)
    scale = margins.abs().median().clamp_min(0.5)
    targets = (labels == 1).float()
    head = SubtypeProbe(x.mean(0), x.std(0, unbiased=False).clamp_min(0.02))
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    shuffle = torch.Generator().manual_seed(seed)
    best, history = None, []
    for epoch in range(1, SNAPSHOTS[-1] + 1):
        total = 0.0
        for indices in torch.randperm(len(x), generator=shuffle).split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            prediction = corrected_margin(margins[indices], head(x[indices]), scale, STRENGTHS[-1])
            loss = F.binary_cross_entropy_with_logits(prediction / scale, targets[indices], weight=weights[indices])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        if epoch not in SNAPSHOTS:
            continue
        with torch.no_grad():
            evidence = head(projected_dev)
            for strength in STRENGTHS:
                adjusted = corrected_margin(dev_margin, evidence, scale, strength)
                item = card(combine(incumbent, adjusted, 1.0), dev, malik_n, start, epoch, strength, total / len(x))
                history.append(item["record"])
                if best is None or ranking(item) > ranking(best):
                    best = item
                    state = {key: tensor.detach().clone() for key, tensor in head.state_dict().items()}
    location = RUN / f"{arm}_s{seed}"
    location.mkdir(exist_ok=True)
    write_rows(location / "history.csv", [{"strength" if key == "fraction" else key: value
                                           for key, value in row.items()} for row in history])
    checkpoint = location / "residual.pt"
    pending = checkpoint.with_suffix(".tmp")
    torch.save({"state": state, "basis": basis, "scale": scale, "strength": best["record"]["fraction"]}, pending)
    pending.replace(checkpoint)
    best["record"]["strength"] = best["record"]["fraction"]
    best["checkpoint_sha256"] = sha256(checkpoint)
    print(location.name, best["record"], flush=True)
    return best


def train(encoder, protocol, components, prior_digest, early):
    splits, records, _, _ = verified_cohort(protocol)
    RUN.mkdir(exist_ok=True)
    if (RUN / "decision.json").exists():
        raise FileExistsError("This trial is already locked")
    plan = {"incumbent_decision_sha256": prior_digest, "manifest_sha256": {key: sha256(path) for key, path in splits.items()},
            "seeds": list(SEEDS), "arms": list(ARMS), "snapshots": list(SNAPSHOTS), "strengths": list(STRENGTHS),
            "min_per_disease_for_source_projection": MIN_PER_DISEASE, "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE, "weight_decay": 0.05, "script_sha256": sha256(Path(__file__)),
            "method": "Fit a bounded additive early-minus-late correction, downweighted at high confidence. Both arms use "
                      "identical fit weights and optimization. The projected arm removes within-class source centroid "
                      "variation orthogonal to the equally source-weighted disease mean contrast.",
            "selection": "Reduce subtype errors and preserve both recalls and Macro-F1 on pooled development and Malik. "
                         "Seed 42 and at least one confirmation seed must pass. Evaluation images open only after locking."}
    plan_path = RUN / "protocol.json"
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError("The trial protocol has changed")
    else:
        write_json(plan_path, plan)
    fit, field, malik = [encode(encoder, path, name) for name, path in splits.items()]
    dev = {name: torch.cat((field[name], malik[name])) for name in field}
    incumbent = previous_scores(dev, *components)
    malik_n = len(malik["truth"])
    start = development_metrics(incumbent, dev, malik_n)
    expected = early["selections"][early["preferred_by_development"]]["metrics"]
    if any(actual["confusion"] != known["confusion"] for actual, known in ((start, expected), (start["malik"], expected["malik"]))):
        raise ValueError("Incumbent does not match its frozen development predictions")
    disease = fit["truth"] > 0
    x, labels = fit["dino"][disease], fit["truth"][disease]
    disease_rows = [row for row, include in zip(records["augmented"], disease) if include]
    if [row["label"] for row in disease_rows] != [CLASSES[label] for label in labels]:
        raise ValueError("Encoded training labels do not match the frozen manifest")
    weights = source_weights(disease_rows, labels)
    margins = disease_margin(previous_scores(fit, *components))[disease]
    basis, counts = source_basis(x, labels, disease_rows)
    directions = {"unprojected": basis[:0], "source_projected": basis}
    outcomes = {f"{arm}_s{seed}": train_head(arm, seed, x, labels, weights, margins, dev, incumbent, start, malik_n,
                                             directions[arm]) for arm, seed in product(ARMS, SEEDS)}
    qualifying = [f"{arm}_s42" for arm in ARMS if outcomes[f"{arm}_s42"]["record"]["eligible"]
                  and any(outcomes[f"{arm}_s{seed}"]["record"]["eligible"] for seed in SEEDS[1:])]
    chosen = max(qualifying, default="incumbent", key=lambda key: ranking(outcomes[key]))
    write_json(RUN / "decision.json", {"preferred_by_development": chosen, "qualifying": qualifying,
                                      "incumbent": start, "incumbent_decision_sha256": prior_digest,
                                      "protocol_sha256": sha256(plan_path), "projection_rank": len(basis),
                                      "projection_source_counts": counts, "selections": outcomes})
    print(f"Decision locked: {chosen}", flush=True)


def restore_residual(name, item):
    path = verified(RUN / name / "residual.pt", item["checkpoint_sha256"])
    data = torch.load(path, map_location="cpu", weights_only=True)
    state = data["state"]
    model = SubtypeProbe(state["mean"], state["scale"])
    model.load_state_dict(state, strict=True)
    return model.eval(), data["basis"], data["scale"], data["strength"]


@torch.no_grad()
def evaluate(encoder, protocol, components, prior_digest):
    _, _, forbidden, _ = verified_cohort(protocol)
    lock = RUN / "decision.json"
    decision = read_json(lock)
    plan = read_json(verified(RUN / "protocol.json", decision["protocol_sha256"]))
    verified(Path(__file__), plan["script_sha256"])
    if decision["incumbent_decision_sha256"] != prior_digest:
        raise ValueError("The incumbent changed after checkpoint selection")
    audits = read_json(EXPANDED / "evaluation_freeze.json")["external_audits"]
    tests = {**test_receipts(protocol), **read_json(TAIWAN)["tests"], **read_json(AGRI_AUDIT)["tests"], **audits}
    candidate_names = [f"{arm}_s42" for arm in ARMS]
    chosen = decision["preferred_by_development"]
    if chosen in candidate_names:
        arm = chosen.rsplit("_s", 1)[0]
        candidate_names.extend(f"{arm}_s{seed}" for seed in SEEDS[1:])
    alternatives = {name: restore_residual(name, decision["selections"][name]) for name in candidate_names}
    original = create_model("resnet18")
    original.load_state_dict(original_checkpoint("AC42")["model_state"])
    original.requires_grad_(False).to(encoder.device, memory_format=torch.channels_last).eval()
    lock_sha = sha256(lock)
    for name, spec in tests.items():
        result = RUN / f"{name}_results.json"
        if result.exists():
            existing = read_json(result)
            if existing["decision_sha256"] != lock_sha or existing["manifest_sha256"] != spec["sha256"]:
                raise ValueError(f"Evaluation has a different lock: {name}")
            continue
        path = verified(Path(spec["path"]), spec["sha256"])
        rows = read_rows(path)
        keys = {row["sha256"] for row in rows}
        if len(rows) != len(keys) or keys & forbidden:
            raise ValueError(f"Duplicate or overlapping evaluation images: {name}")
        data = encode(encoder, path, name, original)
        incumbent = previous_scores(data, *components)
        scores = {"original_AC42": data["original"], "incumbent": incumbent,
                  **{key: predict(incumbent, data["dino"], *model) for key, model in alternatives.items()}}
        metrics, predictions = {}, []
        truth = data["truth"]
        for key, logits in scores.items():
            if truth.unique().numel() == len(CLASSES):
                metrics[key], labels, subtype = measurement(logits, truth)
            else:
                metrics[key], labels = measure(logits, truth)
                subtype = logits[:, 1:].argmax(1) + 1
            predictions.extend({"model": key, "sha256": row["sha256"], "true_label": row["label"],
                                "prediction": CLASSES[label], "subtype_if_diseased": CLASSES[disease]}
                               for row, label, disease in zip(rows, labels.tolist(), subtype.tolist()))
            print(f"{name} {key} accuracy={metrics[key]['accuracy']:.5f}", flush=True)
        write_rows(RUN / f"{name}_predictions.csv", predictions)
        write_json(result, {"decision_sha256": lock_sha, "manifest_sha256": spec["sha256"],
                            "role": "external_audit" if name in audits else spec.get("role", "historical_replay"),
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
    encoder = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", protocol["parent_sha256"]), device)
    if args.phase == "train":
        train(encoder, protocol, components, sha256(lock), early)
    else:
        evaluate(encoder, protocol, components, sha256(lock))


if __name__ == "__main__":
    main()
