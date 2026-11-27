"""Isolate crop-only evidence and compare it with archived whole-image fusion."""

import argparse
from itertools import product
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from roi_localizer import crop, locate
from run_backbone_trial import move_images
from run_dino_health_trial import RUN as HEALTH_RUN, load_gate
from run_early_review_trial import RUN as EARLY_RUN, restore
from run_late_protection_trial import load_heads
from run_local_leaf_trial import measure
from run_matched_capture_trial import verified_cohort
from run_patch_subtype_trial import AGRI_AUDIT, TAIWAN, card, previous_scores, ranking, source_weights
from run_roi_trial import (
    FRACTIONS, HEAD_BATCH, HISTORICAL, IMAGE_BATCH, SNAPSHOTS, InputViews, global_embedding, load_probe,
    RUN as ROI_RUN,
)
from run_subtype_trial import (
    CLASSES, EXPANDED, FIRST_RUN, RUN as SUBTYPE_RUN, SEEDS, Specialists, SubtypeProbe, combine,
    development_metrics, measurement, read_rows, test_receipts, verified, write_rows,
)
from train_online_candidate import (
    WORKSPACE, checked_checkpoint, create_model, original_checkpoint, read_json, seed_everything, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "view_ablation"
PLANTSEG = WORKSPACE / "datasets" / "view_ablation" / "plantseg" / "freeze.json"
NEW = ("center_only", "roi_only", "center_agreement")
CONTROLS = {"whole": "whole", "center_fusion": "matched_center_crop", "roi_fusion": "localized_crop"}


@torch.no_grad()
def encode(encoder, locator, components, path, name, original=None):
    buffers = []
    for standard, canvas, layouts, truth in DataLoader(InputViews(path), batch_size=IMAGE_BATCH):
        standard = move_images(standard, encoder.device)
        canvas = canvas.to(encoder.device)
        _, whole, prior = encoder(standard)
        centers, available = locate(encoder, locator, canvas, layouts)
        local = global_embedding(encoder, crop(canvas, centers))
        middle = global_embedding(encoder, crop(canvas, torch.full_like(centers, 0.5)))
        whole = whole.cpu()
        row = {"whole": whole, "center_only": middle.cpu(), "roi_only": local.cpu(), "available": available.cpu(), "truth": truth,
               "incumbent": previous_scores({"prior": prior.cpu(), "dino": whole}, *components)}
        if original is not None:
            row["original_AC42"] = original(standard).cpu()
        buffers.append(row)
    data = {key: torch.cat([row[key] for row in buffers]) for key in buffers[0]}
    data["center_fusion"] = (data["whole"] + data["center_only"]) * 0.5
    data["roi_fusion"] = (data["whole"] + data["roi_only"]) * 0.5
    print(f"Encoded {name}: {len(data['truth'])} images", flush=True)
    return data


def route(arm, data, evidence, fraction):
    incumbent = data["incumbent"]
    if arm.startswith("roi_"):
        evidence = torch.where(data["available"], evidence, incumbent[:, 1] - incumbent[:, 2])
    return combine(incumbent, evidence, fraction)


def agreed_margin(data, center, previous):
    incumbent = data["incumbent"][:, 1] - data["incumbent"][:, 2]
    mean = (center + previous) / 2
    agree = (center >= 0) == (previous >= 0)
    return torch.where(agree, mean, incumbent)


def fit_crop(arm, seed, fit, rows, dev, start, malik_n):
    seed_everything(seed)
    mask = fit["truth"] > 0
    labels = fit["truth"][mask]
    y = (labels == 1).float()
    x = fit[arm][mask]
    weights = source_weights([row for row, use in zip(rows, mask) if use], labels)
    head = SubtypeProbe(x.mean(0), x.std(0, unbiased=False).clamp_min(0.02))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.004, weight_decay=0.05)
    shuffle = torch.Generator().manual_seed(seed)
    best, history = None, []
    for epoch in range(1, SNAPSHOTS[-1] + 1):
        loss_sum = 0.
        for indices in torch.randperm(len(x), generator=shuffle).split(HEAD_BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(head(x[indices]), y[indices], weight=weights[indices])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        if epoch not in SNAPSHOTS:
            continue
        with torch.no_grad():
            evidence = head(dev[arm])
            for fraction in FRACTIONS:
                item = card(route(arm, dev, evidence, fraction), dev, malik_n, start, epoch, fraction, loss_sum / len(x))
                history.append(item["record"])
                if best is None or ranking(item) > ranking(best):
                    best = item
                    state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    folder = RUN / f"{arm}_s{seed}"
    folder.mkdir(exist_ok=True)
    write_rows(folder / "history.csv", history)
    weight = folder / "probe.pt"
    temporary = weight.with_suffix(".tmp")
    torch.save({"state": state, "arm": arm, "fraction": best["record"]["fraction"]}, temporary)
    temporary.replace(weight)
    best["weight_sha256"] = sha256(weight)
    print(folder.name, best["record"], flush=True)
    return best


def control(arm, seed, roi_decision):
    key = f"{CONTROLS[arm]}_s{seed}"
    return load_probe(ROI_RUN / key / "probe.pt", roi_decision["selections"][key]["weight_sha256"])


def crop_head(arm, seed, chosen):
    key = f"{arm}_s{seed}"
    return load_probe(RUN / key / "probe.pt", chosen[key]["weight_sha256"])


@torch.no_grad()
def agreement_selection(dev, start, malik_n, roi_decision, results, seed):
    pure, archive = crop_head("center_only", seed, results)
    mixed, _ = control("center_fusion", seed, roi_decision)
    evidence = agreed_margin(dev, pure(dev["center_only"]), mixed(dev["center_fusion"]))
    choices = [card(route("center_agreement", dev, evidence, factor), dev, malik_n, start, 0, factor, None)
               for factor in FRACTIONS]
    path = RUN / f"center_agreement_s{seed}"
    path.mkdir(exist_ok=True)
    write_rows(path / "history.csv", [item["record"] for item in choices])
    best = max(choices, key=ranking)
    best["crop_epoch"] = results[f"center_only_s{seed}"]["record"]["epoch"]
    print(path.name, best["record"], flush=True)
    return best


def freeze_plan(paths, roi_path, early_path):
    plan = {"source_sha256": sha256(Path(__file__)), "roi_decision_sha256": sha256(roi_path),
            "incumbent_sha256": sha256(early_path), "cohort": {key: sha256(path) for key, path in paths.items()},
            "new_arms": list(NEW), "archived_control_mapping": CONTROLS, "seeds": list(SEEDS),
            "epochs": list(SNAPSHOTS), "fractions": list(FRACTIONS),
            "selection": "Maintain pooled and Malik disease recalls and Macro-F1. Reduce pooled subtype errors. "
                         "Require seed 42 and at least one other seed. Select only on development.",
            "crop_agreement": "Use the mean of crop-only and archived whole-plus-center margins when their signs agree; otherwise keep the incumbent margin.",
            "evaluation": "Historical replays are retrospective. PlantSeg is supplemental and disease-only if its frozen manifest is available; no model selection uses any evaluation score. "
                          "New field and PlantDoc are audits. Promotion requires independent three-class confirmation."}
    path = RUN / "protocol.json"
    if path.exists():
        if read_json(path) != plan:
            raise ValueError("The frozen protocol differs")
    else:
        write_json(path, plan)
    return sha256(path)


def train(encoder, locator, components, protocol, roi_path, early_path):
    paths, records, _, _ = verified_cohort(protocol)
    RUN.mkdir(exist_ok=True)
    lock = RUN / "decision.json"
    if lock.exists():
        raise FileExistsError("A development selection is already locked")
    plan_digest = freeze_plan(paths, roi_path, early_path)
    fixed = read_json(roi_path)
    fit, field, malik = (encode(encoder, locator, components, paths[key], key)
                         for key in ("augmented", "development_field", "development_malik"))
    dev = {key: torch.cat((field[key], malik[key])) for key in field}
    malik_n = len(malik["truth"])
    baseline = development_metrics(dev["incumbent"], dev, malik_n)
    if baseline["confusion"] != fixed["incumbent"]["confusion"] or baseline["malik"]["confusion"] != fixed["incumbent"]["malik"]["confusion"]:
        raise ValueError("The incumbent development baseline differs")
    controls = {}
    for arm in CONTROLS:
        head, saved = control(arm, 42, fixed)
        with torch.no_grad():
            logits = route(arm, dev, head(dev[arm]), saved["fraction"])
        metrics = development_metrics(logits, dev, malik_n)
        expected = fixed["selections"][CONTROLS[arm] + "_s42"]["metrics"]
        if metrics["confusion"] != expected["confusion"] or metrics["malik"]["confusion"] != expected["malik"]["confusion"]:
            raise ValueError(f"Frozen archived feature control changed: {arm}")
        controls[arm] = metrics
    results = {f"{arm}_s{seed}": fit_crop(arm, seed, fit, records["augmented"], dev, baseline, malik_n)
               for arm, seed in product(NEW[:2], SEEDS)}
    for seed in SEEDS:
        results[f"center_agreement_s{seed}"] = agreement_selection(dev, baseline, malik_n, fixed, results, seed)
    eligible = [f"{arm}_s42" for arm in NEW if results[f"{arm}_s42"]["record"]["eligible"]
                and any(results[f"{arm}_s{seed}"]["record"]["eligible"] for seed in SEEDS[1:])]
    chosen = max(eligible, key=lambda key: ranking(results[key]), default="incumbent")
    write_json(lock, {"candidate": chosen, "qualified": eligible, "protocol_sha256": plan_digest,
                      "development_incumbent": baseline, "development_controls": controls, "selections": results})
    print(f"Locked choice: {chosen}", flush=True)


def model_scores(data, fixed, decision):
    values = {key: data[key] for key in ("original_AC42", "incumbent")}
    for arm in CONTROLS:
        head, saved = control(arm, 42, fixed)
        values[arm] = route(arm, data, head(data[arm]), saved["fraction"])
    keys = [f"{arm}_s42" for arm in NEW]
    if decision["candidate"] != "incumbent":
        prefix = decision["candidate"].rsplit("_s", 1)[0]
        keys.extend(f"{prefix}_s{seed}" for seed in SEEDS[1:])
    for key in keys:
        arm, seed = key.rsplit("_s", 1)
        if arm == "center_agreement":
            pure, _ = crop_head("center_only", int(seed), decision["selections"])
            mixed, _ = control("center_fusion", int(seed), fixed)
            evidence = agreed_margin(data, pure(data["center_only"]), mixed(data["center_fusion"]))
        else:
            head, _ = crop_head(arm, int(seed), decision["selections"])
            evidence = head(data[arm])
        values[key] = route(arm, data, evidence, decision["selections"][key]["record"]["fraction"])
    return values


@torch.no_grad()
def evaluate(encoder, locator, components, protocol, roi_path, early_path):
    _, _, forbidden, _ = verified_cohort(protocol)
    lock = RUN / "decision.json"
    decision = read_json(lock)
    plan = read_json(verified(RUN / "protocol.json", decision["protocol_sha256"]))
    verified(Path(__file__), plan["source_sha256"])
    fixed = read_json(verified(roi_path, plan["roi_decision_sha256"]))
    verified(early_path, plan["incumbent_sha256"])
    previous_plan = read_json(verified(ROI_RUN / "protocol.json", fixed["protocol_sha256"]))
    forbidden |= set(previous_plan["training_localizer_rows"]) | set(previous_plan["localizer_check_rows"])
    audits = read_json(EXPANDED / "evaluation_freeze.json")["external_audits"]
    tests = {**test_receipts(protocol), **read_json(TAIWAN)["tests"], **read_json(AGRI_AUDIT)["tests"], **audits}
    if PLANTSEG.is_file():
        tests["plantseg_subtype_stress"] = read_json(PLANTSEG)
    original = create_model("resnet18")
    original.load_state_dict(original_checkpoint("AC42")["model_state"])
    original.eval().requires_grad_(False).to(encoder.device, memory_format=torch.channels_last)
    digest = sha256(lock)
    for name, spec in tests.items():
        result = RUN / f"{name}_results.json"
        if result.exists():
            existing = read_json(result)
            if existing["decision_sha256"] != digest or existing["manifest_sha256"] != spec["sha256"]:
                raise ValueError(f"Evaluation lock differs: {name}")
            continue
        path = verified(Path(spec["path"]), spec["sha256"])
        rows = read_rows(path)
        hashes = {row["sha256"] for row in rows}
        if len(hashes) != len(rows) or hashes & forbidden:
            raise ValueError(f"Overlapping or repeated identities: {name}")
        data = encode(encoder, locator, components, path, name, original)
        truth = data["truth"]
        metrics, predictions = {}, []
        for key, scores in model_scores(data, fixed, decision).items():
            if truth.unique().numel() == len(CLASSES):
                metrics[key], labels, _ = measurement(scores, truth)
            else:
                metrics[key], labels = measure(scores, truth)
            predictions.extend({"model": key, "sha256": row["sha256"], "true_label": row["label"], "prediction": CLASSES[label]}
                               for row, label in zip(rows, labels.tolist()))
            print(f"{name} {key}: {metrics[key]['accuracy']:.5f}", flush=True)
        write_rows(RUN / f"{name}_predictions.csv", predictions)
        write_json(result, {"decision_sha256": digest, "manifest_sha256": spec["sha256"], "models": metrics,
                            "role": "external_audit" if name in audits else spec.get("role", "historical_replay")})
    decision_file = RUN / "promotion.json"
    if not decision_file.exists():
        totals = {}
        for name in HISTORICAL:
            metrics = read_json(RUN / f"{name}_results.json")["models"]
            for key, values in metrics.items():
                totals[key] = totals.get(key, 0) + sum(values["confusion"][i][i] for i in range(len(CLASSES)))
        write_json(decision_file, {"candidate": decision["candidate"], "qualifies": False,
                                   "independent_three_class_confirmed": False, "historical_correct_totals": totals})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    phase = parser.parse_args().phase
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roi_path, early_path = ROI_RUN / "decision.json", EARLY_RUN / "decision.json"
    fixed = read_json(roi_path)
    early = read_json(verified(early_path, fixed["incumbent_sha256"]))
    health = read_json(verified(HEALTH_RUN / "decision.json", early["health_decision_sha256"]))
    subtype = read_json(verified(SUBTYPE_RUN / "decision.json", health["subtype_decision_sha256"]))
    health_key, early_key = health["preferred_by_development"], early["preferred_by_development"]
    components = (load_heads(subtype), load_gate(health_key, health["selections"][health_key]),
                  restore(early_key, early["selections"][early_key]))
    protocol = read_json(SUBTYPE_RUN / "protocol.json")
    encoder = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", protocol["parent_sha256"]), device)
    locator, _ = load_probe(ROI_RUN / "localizer.pt", fixed["locator_sha256"], device)
    stage = train if phase == "train" else evaluate
    stage(encoder, locator, components, protocol, roi_path, early_path)


if __name__ == "__main__":
    main()
