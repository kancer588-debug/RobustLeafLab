"""Test learned DINOv2 patch pooling for early-versus-late blight classification."""

import argparse
import json
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from run_backbone_trial import loader, move_images
from run_dino_health_trial import RUN as HEALTH_RUN, TAIWAN, load_gate, source_name, training_splits
from run_early_review_trial import RUN as EARLY_RUN, apply_review, incumbent_scores, restore
from run_late_protection_trial import load_heads
from run_local_leaf_trial import measure
from run_subtype_trial import (
    CLASSES, DINO_HASH, EXPANDED, FIRST_RUN, RUN as SUBTYPE_RUN, SEEDS, Specialists, combine,
    development_metrics, measurement, read_rows, test_receipts, verified, write_rows,
)
from train_online_candidate import (
    WORKSPACE, ManifestDataset, checked_checkpoint, create_model, original_checkpoint, read_json,
    seed_everything, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "patch_subtype_trial"
AGRI_AUDIT = WORKSPACE / "datasets" / "early_review_trial" / "agrivision_health" / "freeze.json"
ARMS = ("uniform_pool", "learned_patch_pool")
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
SNAPSHOTS = (1, 2, 4, 8, 12)
LR = 0.002
BATCH = 48
EMBED = 384
HIDDEN = 48


@torch.no_grad()
def encode_patches(model, path, name, original=None):
    buffers = []
    for images, truth in loader(ManifestDataset(path)):
        images = move_images(images, model.device)
        prior = model.resnet.fc(model.resnet_features(images))
        tokens = model.dino.forward_features(images.contiguous())
        patches = tokens["x_norm_patchtokens"]
        cls = tokens["x_norm_clstoken"]
        patches_8x8 = F.avg_pool2d(patches.transpose(1, 2).reshape(len(images), EMBED, 16, 16), 2)
        data = {"dino": torch.cat((cls, patches.mean(1)), dim=1).cpu(), "cls": cls.cpu(),
                "patches": patches_8x8.flatten(2).transpose(1, 2).cpu(), "prior": prior.cpu(), "truth": truth}
        if original is not None:
            data["original"] = original(images).cpu()
        buffers.append(data)
    result = {key: torch.cat([row[key] for row in buffers]) for key in buffers[0]}
    print(f"Encoded {name}: {len(result['truth'])} images and {result['patches'].shape[1]} regions per image", flush=True)
    return result


class PatchSubtype(nn.Module):
    """Project every region, pool it, and combine with the same whole-image token."""

    def __init__(self, learned_pooling):
        super().__init__()
        self.region = nn.Linear(EMBED, HIDDEN)
        self.whole = nn.Linear(EMBED, HIDDEN)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(2 * HIDDEN, 1)
        self.pool_logit = nn.Linear(HIDDEN, 1, bias=False) if learned_pooling else None
        if self.pool_logit is not None:
            nn.init.zeros_(self.pool_logit.weight)

    def forward(self, cls, patches):
        regions = F.gelu(self.region(patches))
        context = F.gelu(self.whole(cls))
        if self.pool_logit is None:
            pooled = regions.mean(1)
        else:
            weights = self.pool_logit(regions).softmax(1)
            pooled = (regions * weights).sum(1)
        return self.output(self.dropout(torch.cat((context, pooled), dim=1))).squeeze(1)


def previous_scores(data, heads, health_gate, early_head):
    layer, view, mode, strength = early_head
    if view != "standard":
        raise ValueError("This trial requires the locked 224-pixel early reviewer")
    with torch.no_grad():
        scores = incumbent_scores(data, heads, health_gate)
        return apply_review(scores, layer(data["dino"]), strength, mode)


def source_weights(rows, labels):
    groups = [(row["label"], source_name(row)) for row in rows]
    counts = Counter(groups)
    weights = torch.tensor([max(counts[key], 8) ** -0.5 for key in groups])
    for label in (1, 2):
        chosen = labels == label
        weights[chosen] *= len(labels) / (2 * weights[chosen].sum())
    return weights


def card(scores, dev, malik_n, start, epoch, fraction, loss):
    metrics = development_metrics(scores, dev, malik_n)
    table, old = metrics["confusion"], start["confusion"]
    malik, old_malik = metrics["malik"], start["malik"]
    fewer = old[1][2] + old[2][1] - table[1][2] - table[2][1]
    guards = (table[1][1] >= old[1][1] and table[2][2] >= old[2][2]
              and malik["confusion"][1][1] >= old_malik["confusion"][1][1]
              and malik["confusion"][2][2] >= old_malik["confusion"][2][2]
              and malik["macro_f1"] >= old_malik["macro_f1"] and metrics["macro_f1"] >= start["macro_f1"])
    return {"record": {"epoch": epoch, "fraction": fraction, "loss": loss,
                       "eligible": guards and fewer > 0, "guardrails": guards, "fewer_subtype_errors": fewer,
                       "early_correct": table[1][1], "late_correct": table[2][2],
                       "early_to_late": table[1][2], "late_to_early": table[2][1],
                       "macro_f1": metrics["macro_f1"], "malik_macro_f1": malik["macro_f1"]}, "metrics": metrics}


def ranking(result):
    row = result["record"]
    return (row["eligible"], row["guardrails"], (row["macro_f1"] + row["malik_macro_f1"]) / 2,
            row["fewer_subtype_errors"], -row["fraction"], -row["epoch"])


def train_one(arm, seed, fit, fit_sources, dev, baseline, start, malik_n, device):
    seed_everything(seed)
    model = PatchSubtype(arm == "learned_patch_pool").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.03)
    diseased = fit["truth"] > 0
    fit_cls, fit_patches = (fit[key][diseased].to(device) for key in ("cls", "patches"))
    dev_cls, dev_patches = (dev[key].to(device) for key in ("cls", "patches"))
    fit_labels = fit["truth"][diseased]
    weights = source_weights([row for row, keep in zip(fit_sources, diseased) if keep], fit_labels).to(device)
    target = (fit_labels == 1).float().to(device)
    history, best = [], None
    for epoch in range(1, SNAPSHOTS[-1] + 1):
        model.train()
        total = 0.0
        for selection in torch.randperm(len(target), device=device).split(BATCH):
            evidence = model(fit_cls[selection], fit_patches[selection])
            loss = F.binary_cross_entropy_with_logits(evidence, target[selection], weight=weights[selection])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(selection)
        if epoch not in SNAPSHOTS:
            continue
        model.eval()
        with torch.no_grad():
            evidence = model(dev_cls, dev_patches).cpu()
            for fraction in FRACTIONS:
                result = card(combine(baseline, evidence, fraction), dev, malik_n, start, epoch, fraction,
                              total / len(target))
                history.append(result["record"])
                if best is None or ranking(result) > ranking(best):
                    best = result
                    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    location = RUN / f"{arm}_s{seed}"
    location.mkdir(exist_ok=True)
    write_rows(location / "history.csv", history)
    artifact = location / "patch_head.pt"
    temp = artifact.with_suffix(".tmp")
    torch.save({"state": state, "arm": arm, "fraction": best["record"]["fraction"]}, temp)
    temp.replace(artifact)
    best["checkpoint_sha256"] = sha256(artifact)
    print(location.name, json.dumps(best["record"]), flush=True)
    return best


def train(encoder, protocol, components, previous_hash):
    splits, rows, _ = training_splits(protocol)
    RUN.mkdir(exist_ok=True)
    if (RUN / "decision.json").exists():
        raise FileExistsError("Patch subtype selection has already been locked")
    plan = {"early_review_sha256": previous_hash, "dinov2_sha256": DINO_HASH,
            "manifests": {name: protocol["manifests"][name] for name in splits},
            "arms": list(ARMS), "seeds": list(SEEDS), "fractions": list(FRACTIONS),
            "snapshots": list(SNAPSHOTS), "learning_rate": LR, "head_batch": BATCH,
            "selection": "Require fewer pooled subtype errors and preserve both pooled and Malik class recalls and "
                         "both Macro-F1 scores. Require the primary seed and at least one confirmatory seed. Health "
                         "routing is held identical to the previous model. Read test images only after locking."}
    config = RUN / "protocol.json"
    if config.exists():
        if read_json(config) != plan:
            raise ValueError("This run directory has another protocol")
    else:
        write_json(config, plan)
    fit, field, malik = [encode_patches(encoder, path, name) for name, path in splits.items()]
    if [CLASSES[label] for label in fit["truth"].tolist()] != [row["label"] for row in rows["augmented"]]:
        raise ValueError("Training image order differs from the manifest")
    dev = {key: torch.cat((field[key], malik[key])) for key in field}
    scores = previous_scores(dev, *components)
    malik_n = len(malik["truth"])
    start = development_metrics(scores, dev, malik_n)
    prior = read_json(EARLY_RUN / "decision.json")
    expected = prior["selections"][prior["preferred_by_development"]]["metrics"]
    if start["confusion"] != expected["confusion"] or start["malik"]["confusion"] != expected["malik"]["confusion"]:
        raise ValueError("The incumbent scores do not match the prior validation run")
    candidates = {f"{arm}_s{seed}": train_one(arm, seed, fit, rows["augmented"], dev, scores,
                                            start, malik_n, encoder.device) for arm, seed in product(ARMS, SEEDS)}
    qualifying = [f"{arm}_s42" for arm in ARMS if candidates[f"{arm}_s42"]["record"]["eligible"]
                  and any(candidates[f"{arm}_s{seed}"]["record"]["eligible"] for seed in SEEDS[1:])]
    preferred = max(qualifying, key=lambda key: ranking(candidates[key]), default="incumbent")
    write_json(RUN / "decision.json", {"early_review_sha256": previous_hash, "incumbent": start,
                                     "preferred_by_development": preferred, "qualifying": qualifying,
                                     "selections": candidates})
    print(f"Decision locked: {preferred}", flush=True)


def restore_patch(name, card, device):
    path = verified(RUN / name / "patch_head.pt", card["checkpoint_sha256"])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = PatchSubtype(saved["arm"] == "learned_patch_pool")
    model.load_state_dict(saved["state"])
    return model.to(device).eval(), saved["fraction"]


@torch.no_grad()
def pooled_margins(data, incumbent, members, device):
    cls, patches = (data[key].to(device) for key in ("cls", "patches"))
    baseline = incumbent[:, 1] - incumbent[:, 2]
    return {key: torch.lerp(baseline, model(cls, patches).cpu(), fraction)
            for key, (model, fraction) in members.items()}


def ensemble(encoder, protocol, components, previous_hash):
    original = RUN / "decision.json"
    primary = read_json(original)
    if primary["early_review_sha256"] != previous_hash:
        raise ValueError("The incumbent changed since the primary development search")
    destination = RUN / "ensemble_decision.json"
    if destination.exists():
        raise FileExistsError("The ensemble rule is already locked")
    plan = {"primary_decision_sha256": sha256(original), "members": list(SEEDS),
            "combination": "Equal mean of the three already-selected fused margins within each arm.",
            "guardrails": "Apply the primary recall, Macro-F1 and subtype-error limits. Confirm on at least two "
                          "of the three leave-one-head-out pairs. All evaluation follows this lock."}
    settings = RUN / "ensemble_protocol.json"
    if settings.exists():
        if read_json(settings) != plan:
            raise ValueError("An existing ensemble protocol differs from this one")
    else:
        write_json(settings, plan)
    splits, _, _ = training_splits(protocol)
    field, malik = [encode_patches(encoder, splits[name], name) for name in ("development_field", "development_malik")]
    dev = {key: torch.cat((field[key], malik[key])) for key in field}
    baseline = previous_scores(dev, *components)
    malik_n = len(malik["truth"])
    start = development_metrics(baseline, dev, malik_n)
    saved = {}
    for arm in ARMS:
        keys = [f"{arm}_s{seed}" for seed in SEEDS]
        models = {key: restore_patch(key, primary["selections"][key], encoder.device) for key in keys}
        margins = pooled_margins(dev, baseline, models, encoder.device)
        full = combine(baseline, torch.stack(list(margins.values())).mean(0), 1.0)
        candidate = card(full, dev, malik_n, start, 0, 1.0, 0.0)
        checks = {}
        for first, second in combinations(keys, 2):
            scores = combine(baseline, (margins[first] + margins[second]) / 2, 1.0)
            checks[f"{first}+{second}"] = card(scores, dev, malik_n, start, 0, 1.0, 0.0)["record"]
        passed = sum(check["eligible"] for check in checks.values())
        candidate["record"]["confirmation_pairs"] = passed
        candidate["record"]["eligible"] = candidate["record"]["eligible"] and passed >= 2
        saved[f"{arm}_ensemble"] = {**candidate, "confirmations": checks, "members": keys}
        print(arm, json.dumps(candidate["record"]), flush=True)
    eligible = [key for key, choice in saved.items() if choice["record"]["eligible"]]
    winner = max(eligible, key=lambda key: ranking(saved[key]), default=primary["preferred_by_development"])
    write_json(destination, {"primary_decision_sha256": plan["primary_decision_sha256"],
                             "preferred_by_development": winner, "ensembles": saved})
    print(f"Ensemble decision locked: {winner}", flush=True)


def evaluate(encoder, protocol, components, previous_hash):
    lock = RUN / "decision.json"
    decision = read_json(lock)
    decision_hash = sha256(lock)
    if decision["early_review_sha256"] != previous_hash:
        raise ValueError("Incumbent changed after development lock")
    ensemble_lock = RUN / "ensemble_decision.json"
    ensemble_decision = read_json(ensemble_lock)
    ensemble_hash = sha256(ensemble_lock)
    if ensemble_decision["primary_decision_sha256"] != decision_hash:
        raise ValueError("The ensemble uses a different primary decision")
    _, _, forbidden = training_splits(protocol)
    protected = read_json(EXPANDED / "evaluation_freeze.json")["external_audits"]
    datasets = {**test_receipts(protocol), **read_json(TAIWAN)["tests"], **protected, **read_json(AGRI_AUDIT)["tests"]}
    keys = [f"{arm}_s42" for arm in ARMS]
    selected = decision["preferred_by_development"]
    if selected in keys:
        arm = selected.rsplit("_s", 1)[0]
        keys.extend(f"{arm}_s{seed}" for seed in SEEDS[1:])
    dependencies = {key for item in ensemble_decision["ensembles"].values() for key in item["members"]}
    heads = {name: restore_patch(name, decision["selections"][name], encoder.device)
             for name in sorted(set(keys) | dependencies)}
    original = create_model("resnet18")
    original.load_state_dict(original_checkpoint("AC42")["model_state"])
    original.requires_grad_(False).to(encoder.device, memory_format=torch.channels_last).eval()
    for name, spec in datasets.items():
        path = verified(Path(spec["path"]), spec["sha256"])
        rows = read_rows(path)
        identities = {row["sha256"] for row in rows}
        if len(identities) != len(rows) or identities & forbidden:
            raise ValueError(f"Evaluation shares training or validation images: {name}")
        result_path = RUN / f"{name}_results.json"
        if result_path.is_file():
            saved = read_json(result_path)
            if (saved["manifest_sha256"] != spec["sha256"] or saved["decision_sha256"] != decision_hash
                    or saved["ensemble_sha256"] != ensemble_hash):
                raise ValueError(f"Evaluation already exists with another decision or list: {name}")
            continue
        data = encode_patches(encoder, path, name, original=original)
        incumbent = previous_scores(data, *components)
        margins = pooled_margins(data, incumbent, heads, encoder.device)
        scores = {"original_AC42": data["original"], "incumbent": incumbent,
                  **{key: combine(incumbent, margins[key], 1.0) for key in keys}}
        for key, item in ensemble_decision["ensembles"].items():
            averaged = torch.stack([margins[member] for member in item["members"]]).mean(0)
            scores[key] = combine(incumbent, averaged, 1.0)
        truth = data["truth"]
        all_classes = truth.unique().numel() == len(CLASSES)
        metrics, predictions = {}, []
        for key, values in scores.items():
            if all_classes:
                metrics[key], labels, subtype = measurement(values, truth)
            else:
                metrics[key], labels = measure(values, truth)
                subtype = values[:, 1:].argmax(1) + 1
            predictions.extend({"model": key, "sha256": row["sha256"], "true_label": row["label"],
                                "prediction": CLASSES[label], "subtype_if_diseased": CLASSES[disease]}
                               for row, label, disease in zip(rows, labels.tolist(), subtype.tolist()))
            m = metrics[key]["confusion"]
            print(f"{name} {key} accuracy={metrics[key]['accuracy']:.5f} early={m[1][1]} late={m[2][2]}", flush=True)
        write_rows(RUN / f"{name}_predictions.csv", predictions)
        role = "external_audit" if name in protected else spec.get("role", "historical_replay")
        write_json(result_path, {"decision_sha256": decision_hash, "ensemble_sha256": ensemble_hash,
                                 "manifest_sha256": spec["sha256"], "role": role, "models": metrics})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = {"train": train, "ensemble": ensemble, "evaluate": evaluate}
    parser.add_argument("phase", choices=phases)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    previous = EARLY_RUN / "decision.json"
    early = read_json(previous)
    health = read_json(verified(HEALTH_RUN / "decision.json", early["health_decision_sha256"]))
    subtype = read_json(verified(SUBTYPE_RUN / "decision.json", health["subtype_decision_sha256"]))
    health_key, early_key = health["preferred_by_development"], early["preferred_by_development"]
    components = (load_heads(subtype), load_gate(health_key, health["selections"][health_key]),
                  restore(early_key, early["selections"][early_key]))
    protocol = read_json(SUBTYPE_RUN / "protocol.json")
    checkpoint = checked_checkpoint(FIRST_RUN / "candidate.pt", protocol["parent_sha256"])
    encoder = Specialists(checkpoint, device)
    phases[args.phase](encoder, protocol, components, sha256(previous))


if __name__ == "__main__":
    main()
