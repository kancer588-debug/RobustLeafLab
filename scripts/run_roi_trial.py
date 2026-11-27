"""Compare whole-image, center-crop and supervised lesion-crop subtype heads."""

import argparse
from itertools import product
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader

import roi_localizer
from roi_localizer import crop, fit_canvas, locate, split_regions, train_localizer
from run_backbone_trial import move_images
from run_dino_health_trial import RUN as HEALTH_RUN, load_gate
from run_early_review_trial import RUN as EARLY_RUN, restore
from run_late_protection_trial import load_heads
from run_local_leaf_trial import measure
from run_matched_capture_trial import verified_cohort
from run_patch_subtype_trial import AGRI_AUDIT, TAIWAN, card, previous_scores, ranking, source_weights
from run_subtype_trial import (
    CLASSES, EXPANDED, FIRST_RUN, RUN as SUBTYPE_RUN, SEEDS, Specialists, SubtypeProbe, combine,
    development_metrics, measurement, read_rows, test_receipts, verified, write_rows,
)
from stage_roi_evaluation import TEST
from stage_roi_training import STAGE, read
from train_online_candidate import (
    WORKSPACE, ManifestDataset, checked_checkpoint, create_model, original_checkpoint, read_json, seed_everything, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "roi_trial"
ARMS = ("whole", "matched_center_crop", "localized_crop")
SNAPSHOTS = (1, 2, 4, 8, 12)
FRACTIONS = (0.125, 0.25, 0.5, 0.75)
HISTORICAL = ("plantcity_replay", "plantwild_replay", "imtiaz_replay", "rachin_replay")
IMAGE_BATCH = 12
HEAD_BATCH = 64


class InputViews(ManifestDataset):
    def __getitem__(self, ix):
        row = self.rows[ix]
        with Image.open(row["path"]) as source:
            image = source.convert("RGB")
        canvas, layout = fit_canvas(image)
        return self.tf(image), canvas, layout, CLASSES.index(row["label"])


def global_embedding(encoder, images):
    tokens = encoder.dino.forward_features(images.contiguous())
    return torch.cat((tokens["x_norm_clstoken"], tokens["x_norm_patchtokens"].mean(1)), dim=1)


@torch.no_grad()
def encode(encoder, localizer, components, path, name, original=None):
    batches = []
    for standard, canvas, layouts, truth in DataLoader(InputViews(path), batch_size=IMAGE_BATCH):
        standard = move_images(standard, encoder.device)
        canvas = canvas.to(encoder.device)
        _, whole, prior = encoder(standard)
        centers, available = locate(encoder, localizer, canvas, layouts)
        control = crop(canvas, torch.full_like(centers, 0.5))
        local = crop(canvas, centers)
        center_features = global_embedding(encoder, control)
        local_features = global_embedding(encoder, local)
        batches.append({"whole": whole.cpu(), "matched_center_crop": (whole + center_features).mul(0.5).cpu(),
                        "localized_crop": (whole + local_features).mul(0.5).cpu(), "available": available.cpu(), "truth": truth,
                        "incumbent": previous_scores({"prior": prior.cpu(), "dino": whole.cpu()}, *components)})
        if original is not None:
            batches[-1]["original_AC42"] = original(standard).cpu()
    result = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
    print(f"Encoded {name}: {len(result['truth'])} images, {int(result['available'].sum())} localized crops enabled", flush=True)
    return result


def adjust(arm, data, evidence, fraction):
    incumbent = data["incumbent"]
    if arm == "localized_crop":
        evidence = torch.where(data["available"], evidence, incumbent[:, 1] - incumbent[:, 2])
    return combine(incumbent, evidence, fraction)


def fit_subtype(arm, seed, data, records, dev, malik_n, reference):
    seed_everything(seed)
    disease = data["truth"] > 0
    labels = data["truth"][disease]
    target = (labels == 1).float()
    x = data[arm][disease]
    fit_rows = [row for row, include in zip(records, disease) if include]
    weights = source_weights(fit_rows, labels)
    model = SubtypeProbe(x.mean(0), x.std(0, unbiased=False).clamp_min(0.02))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.05)
    shuffle = torch.Generator().manual_seed(seed)
    selected, history = None, []
    for epoch in range(1, SNAPSHOTS[-1] + 1):
        total = 0.
        for indices in torch.randperm(len(x), generator=shuffle).split(HEAD_BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x[indices]), target[indices], weight=weights[indices])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        if epoch not in SNAPSHOTS:
            continue
        with torch.no_grad():
            evidence = model(dev[arm])
            for fraction in FRACTIONS:
                item = card(adjust(arm, dev, evidence, fraction), dev, malik_n, reference, epoch, fraction, total / len(target))
                history.append(item["record"])
                if selected is None or ranking(item) > ranking(selected):
                    selected = item
                    state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    name = f"{arm}_s{seed}"
    folder = RUN / name
    folder.mkdir(exist_ok=True)
    write_rows(folder / "history.csv", history)
    weight = folder / "probe.pt"
    temp = weight.with_suffix(".tmp")
    torch.save({"state": state, "arm": arm, "fraction": selected["record"]["fraction"]}, temp)
    temp.replace(weight)
    selected["weight_sha256"] = sha256(weight)
    print(name, selected["record"], flush=True)
    return selected


def load_probe(path, expected, device="cpu"):
    saved = torch.load(verified(path, expected), map_location="cpu", weights_only=True)
    state = saved["state"]
    head = SubtypeProbe(state["mean"], state["scale"])
    head.load_state_dict(state, strict=True)
    return head.requires_grad_(False).eval().to(device), saved


def freeze_plan(prior_digest, paths, fit, checks):
    source = STAGE / "screened.json"
    review = STAGE / "early_region_review.json"
    test = TEST / "freeze.json"
    assessment = TEST / "assessment.json"
    if not test.exists() and not assessment.exists():
        raise ValueError("The potential external source must be assessed before training")
    fit_paths = dict(paths)
    if test.exists():
        receipt = read(test)
        holdout = verified(Path(receipt["path"]), receipt["sha256"])
        eligible = {row["sha256"] for row in read_rows(holdout)}
        prior_ids = {row["sha256"] for path in paths.values() for row in read_rows(path)} | {row["sha256"] for row in fit + checks}
        if eligible & prior_ids:
            raise ValueError("The new-source evaluation overlaps fitting or development")
        fit_paths["new_source_holdout"] = holdout
    plan = {"incumbent_sha256": prior_digest, "training_annotation_sha256": sha256(source),
            "early_region_review_sha256": sha256(review), "new_test_receipt_sha256": sha256(test) if test.exists() else None,
            "source_exclusion_sha256": sha256(assessment) if not test.exists() else None,
            "manifests": {key: sha256(path) for key, path in fit_paths.items()}, "arms": list(ARMS), "seeds": list(SEEDS),
            "head_snapshots": list(SNAPSHOTS), "strengths": list(FRACTIONS), "localizer_resolution": roi_localizer.RESOLUTION,
            "localizer_epochs": roi_localizer.LOCALIZER_EPOCHS, "crop_fraction": roi_localizer.LOCAL_FRACTION,
            "training_localizer_rows": [row["sha256"] for row in fit], "localizer_check_rows": [row["sha256"] for row in checks],
            "source_sha256": {"trial": sha256(Path(__file__)), "localizer": sha256(Path(roi_localizer.__file__))},
            "selection": "The pooled and Malik development disease recalls and Macro-F1 must be maintained; pooled subtype errors "
                         "must decrease. Require seed 42 and at least one confirmation seed. Only development can select.",
            "promotion": "The selected model must improve new-source accuracy, maintain its Macro-F1 and both disease recalls, "
                         "and maintain the sum of correct predictions on four historical replays. The new source and both external "
                         "audits are read only after the decision lock. Audit scores never determine promotion."}
    file = RUN / "protocol.json"
    if file.exists():
        if read_json(file) != plan:
            raise ValueError("The experiment protocol has changed")
    else:
        write_json(file, plan)
    return sha256(file)


def train(encoder, protocol, components, prior_digest, early):
    RUN.mkdir(exist_ok=True)
    if (RUN / "decision.json").exists():
        raise FileExistsError("The experiment is already locked")
    paths, rows, _, _ = verified_cohort(protocol)
    reviewed = read(STAGE / "early_region_review.json")["boxes_by_review_id"]
    screened = read(STAGE / "screened.json")["records"]
    local_fit, local_check = split_regions(screened, reviewed, rows["augmented"])
    plan_digest = freeze_plan(prior_digest, paths, local_fit, local_check)
    print(f"Locked protocol with {len(local_fit)} locator training frames and {len(local_check)} locator check frames", flush=True)
    weight = RUN / "localizer.pt"
    info = RUN / "localizer.json"
    if info.exists():
        locator_card = read_json(info)
        if locator_card["protocol_sha256"] != plan_digest:
            raise ValueError("Localizer belongs to another protocol")
        localizer, _ = load_probe(weight, locator_card["sha256"], encoder.device)
    else:
        localizer, locator_card = train_localizer(encoder, local_fit, local_check)
        pending = weight.with_suffix(".tmp")
        torch.save({"state": localizer.state_dict()}, pending)
        pending.replace(weight)
        locator_card.update(sha256=sha256(weight), protocol_sha256=plan_digest)
        write_json(info, locator_card)
        localizer.to(encoder.device)
    print("Localizer check", locator_card["checks"], flush=True)
    fits = {key: encode(encoder, localizer, components, path, key) for key, path in paths.items()}
    field, malik = fits["development_field"], fits["development_malik"]
    development = {key: torch.cat((field[key], malik[key])) for key in field}
    malik_n = len(malik["truth"])
    start = development_metrics(development["incumbent"], development, malik_n)
    expected = early["selections"][early["preferred_by_development"]]["metrics"]
    if start["confusion"] != expected["confusion"] or start["malik"]["confusion"] != expected["malik"]["confusion"]:
        raise ValueError("The incumbent does not match its archived development results")
    results = {f"{arm}_s{seed}": fit_subtype(arm, seed, fits["augmented"], rows["augmented"], development,
                                               malik_n, start) for arm, seed in product(ARMS, SEEDS)}
    qualified = [f"{arm}_s42" for arm in ARMS if results[f"{arm}_s42"]["record"]["eligible"]
                 and any(results[f"{arm}_s{seed}"]["record"]["eligible"] for seed in SEEDS[1:])]
    choice = max(qualified, key=lambda key: ranking(results[key]), default="incumbent")
    write_json(RUN / "decision.json", {"preferred_by_development": choice, "qualified": qualified, "incumbent": start,
                                      "protocol_sha256": plan_digest, "locator_sha256": locator_card["sha256"],
                                      "incumbent_sha256": prior_digest, "selections": results})
    print(f"Subtype decision locked: {choice}", flush=True)


@torch.no_grad()
def evaluate(encoder, protocol, components, prior_digest):
    _, _, forbidden, _ = verified_cohort(protocol)
    lock_path = RUN / "decision.json"
    handle = read_json(lock_path)
    plan = read_json(verified(RUN / "protocol.json", handle["protocol_sha256"]))
    if handle["incumbent_sha256"] != prior_digest:
        raise ValueError("Incumbent changed after the lock")
    verified(Path(__file__), plan["source_sha256"]["trial"])
    verified(Path(roi_localizer.__file__), plan["source_sha256"]["localizer"])
    localizer, _ = load_probe(RUN / "localizer.pt", handle["locator_sha256"], encoder.device)
    audits = read_json(EXPANDED / "evaluation_freeze.json")["external_audits"]
    suites = {**test_receipts(protocol), **read_json(TAIWAN)["tests"], **read_json(AGRI_AUDIT)["tests"], **audits}
    if plan["new_test_receipt_sha256"]:
        suites["sheikh_2026_holdout"] = read_json(verified(TEST / "freeze.json", plan["new_test_receipt_sha256"]))
    else:
        verified(TEST / "assessment.json", plan["source_exclusion_sha256"])
    forbidden |= set(plan["training_localizer_rows"]) | set(plan["localizer_check_rows"])
    baseline = create_model("resnet18")
    baseline.load_state_dict(original_checkpoint("AC42")["model_state"])
    baseline.requires_grad_(False).to(encoder.device, memory_format=torch.channels_last).eval()
    names = [f"{arm}_s42" for arm in ARMS]
    preferred = handle["preferred_by_development"]
    if preferred != "incumbent":
        arm = preferred.rsplit("_s", 1)[0]
        names.extend(f"{arm}_s{seed}" for seed in SEEDS[1:])
    models = {key: load_probe(RUN / key / "probe.pt", handle["selections"][key]["weight_sha256"]) for key in names}
    lock_digest = sha256(lock_path)
    for name, spec in suites.items():
        result_path = RUN / f"{name}_results.json"
        if result_path.exists():
            previous = read_json(result_path)
            if previous["decision_sha256"] != lock_digest or previous["manifest_sha256"] != spec["sha256"]:
                raise ValueError(f"Prior evaluation has a different lock: {name}")
            continue
        manifest = verified(Path(spec["path"]), spec["sha256"])
        rows = read_rows(manifest)
        hashes = {row["sha256"] for row in rows}
        if len(hashes) != len(rows) or hashes & forbidden:
            raise ValueError(f"Duplicate or overlapping evaluation photos: {name}")
        data = encode(encoder, localizer, components, manifest, name, baseline)
        scores = {"original_AC42": data["original_AC42"], "incumbent": data["incumbent"]}
        for key, (model, checkpoint) in models.items():
            arm = checkpoint["arm"]
            scores[key] = adjust(arm, data, model(data[arm]), checkpoint["fraction"])
        truth = data["truth"]
        metrics, predictions = {}, []
        for key, logits in scores.items():
            metrics[key], labels = (measurement(logits, truth)[:2] if truth.unique().numel() == len(CLASSES)
                                     else measure(logits, truth))
            predictions.extend({"model": key, "sha256": row["sha256"], "true_label": row["label"], "prediction": CLASSES[label]}
                               for row, label in zip(rows, labels.tolist()))
            print(f"{name} {key}: {metrics[key]['accuracy']:.5f}", flush=True)
        write_rows(RUN / f"{name}_predictions.csv", predictions)
        write_json(result_path, {"decision_sha256": lock_digest, "manifest_sha256": spec["sha256"],
                                 "role": "external_audit" if name in audits else spec.get("role", "historical_replay"),
                                 "localizer_eligible_count": int(data["available"].sum()), "models": metrics})


def assess_release():
    decision = read_json(RUN / "decision.json")
    candidate = decision["preferred_by_development"]
    disposition = {"candidate": candidate, "qualifies": False}
    fresh = RUN / "sheikh_2026_holdout_results.json"
    if candidate != "incumbent" and fresh.exists():
        tests = read_json(fresh)["models"]
        original, updated = tests["incumbent"], tests[candidate]
        supported = updated["accuracy"] > original["accuracy"] and all(
            updated[key] >= original[key] for key in ("macro_f1", "early_recall", "late_recall"))
        totals = {key: 0 for key in ("incumbent", candidate)}
        for name in HISTORICAL:
            entry = read_json(RUN / f"{name}_results.json")["models"]
            for key in totals:
                totals[key] += sum(entry[key]["confusion"][index][index] for index in range(len(CLASSES)))
        disposition.update(qualifies=supported and totals[candidate] >= totals["incumbent"],
                           new_source_criterion=supported, historical_correct_totals=totals)
    elif candidate == "incumbent":
        disposition["reason"] = "No configuration passed locked development selection"
    else:
        disposition["reason"] = "The independent candidate source did not pass identity screening"
    target = RUN / "promotion.json"
    if target.exists():
        if read_json(target) != disposition:
            raise ValueError("The recorded promotion decision differs")
    else:
        write_json(target, disposition)
    print("Promotion audit", disposition, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    phase = parser.parse_args().phase
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_lock = EARLY_RUN / "decision.json"
    early = read_json(early_lock)
    health = read_json(verified(HEALTH_RUN / "decision.json", early["health_decision_sha256"]))
    subtype = read_json(verified(SUBTYPE_RUN / "decision.json", health["subtype_decision_sha256"]))
    health_key, early_key = health["preferred_by_development"], early["preferred_by_development"]
    components = (load_heads(subtype), load_gate(health_key, health["selections"][health_key]),
                  restore(early_key, early["selections"][early_key]))
    protocol = read_json(SUBTYPE_RUN / "protocol.json")
    encoder = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", protocol["parent_sha256"]), device)
    if phase == "train":
        train(encoder, protocol, components, sha256(early_lock), early)
    else:
        evaluate(encoder, protocol, components, sha256(early_lock))
        assess_release()


if __name__ == "__main__":
    main()
