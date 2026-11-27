"""Test an early-blight review that can bypass the incumbent's conditional disease rank."""

import argparse
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import torch
from torch.nn import functional as F

from run_dino_health_trial import RUN as HEALTH_RUN, load_gate
from run_early_review_trial import RUN as EARLY_RUN, fitting_weights, restore
from run_health_rescue_trial import (
    DATA as PREVIOUS_DATA, PLANTSEG, SHARED_DEVELOPMENT_GROUP, VIEW_RUN, RescueHead, checked_data, encoded, features,
    group_counts, pack, record,
)
from run_late_protection_trial import load_heads
from run_local_leaf_trial import measure
from run_patch_subtype_trial import AGRI_AUDIT, TAIWAN
from run_roi_trial import HISTORICAL
from run_subtype_trial import (
    CLASSES, EXPANDED, FIRST_RUN, RUN as SUBTYPE_RUN, SEEDS, Specialists, development_metrics, measurement,
    read_rows, test_receipts, verified, write_rows,
)
from train_online_candidate import (
    WORKSPACE, checked_checkpoint, create_model, original_checkpoint, read_json, seed_everything, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "early_override_trial"
DATA = WORKSPACE / "datasets" / "early_override_trial"
ARMS = {"mean": False, "local": True}
MODES = ("ranked", "direct")
EPOCHS = (1, 2, 4, 8, 12)
THRESHOLDS = (0., 0.5, 1., 1.5, 2., 3.)
GAPS = (0.5, 1., 2., 4.)
BATCH = 64
MIN_RESCUED_GROUPS = 2


def route(scores, evidence, threshold, maximum_gap, mode):
    """Only an incumbent healthy prediction can be replaced by a new early prediction."""
    gap = scores[:, 0] - scores[:, 1:].amax(1)
    condition = (scores.argmax(1) == 0) & (gap <= maximum_gap) & (evidence >= threshold)
    if mode == "ranked":
        condition &= scores[:, 1] >= scores[:, 2]
    updated = scores.clone()
    updated[condition, 1] = scores[condition, 0] + 1
    return updated


def inspect_setting(dev, evidence, context, epoch, loss, threshold, gap, mode):
    start = context[1]
    scores = route(dev["incumbent"], evidence, threshold, gap, mode)
    result = record(scores, dev["truth"], *context, epoch, 1., loss)
    card = result["record"]
    rescued_groups = sum(group["early_missed"] < start["groups"][key]["early_missed"]
                         for key, group in result["groups"].items())
    changed = scores.argmax(1) != dev["incumbent"].argmax(1)
    bypassed = (dev["incumbent"][:, 2] > dev["incumbent"][:, 1]) & changed
    card.pop("strength")
    card.update({"mode": mode, "threshold": threshold, "maximum_healthy_gap": gap,
                 "recovered_groups": rescued_groups, "eligible": card["eligible"] and rescued_groups >= MIN_RESCUED_GROUPS,
                 "bypassed_early_misses": int((bypassed & (dev["truth"] == 1)).sum())})
    return result


def rank(result):
    card = result["record"]
    return (card["eligible"], card["guardrails"], card["recovered_groups"], card["early_recovery_group_mean"],
            card["early_recovered_all"], -card["new_healthy_false_alarms"], -card["additional_late_to_early"],
            -card["maximum_healthy_gap"], card["threshold"], -card["epoch"])


def scan(dev, evidence, context, epoch, loss):
    return [inspect_setting(dev, evidence, context, epoch, loss, threshold, gap, mode)
            for mode, threshold, gap in product(MODES, THRESHOLDS, GAPS)]


def train_head(arm, seed, fit, fit_rows, dev, context):
    seed_everything(seed)
    head = RescueHead(ARMS[arm])
    target = (fit["truth"] == 1).float()
    weights = fitting_weights(fit_rows, fit["incumbent"], fit["truth"])
    # Emphasize early samples the existing health path misses, without changing the total early class weight.
    wrong_early = (fit["incumbent"].argmax(1) == 0) & (target == 1)
    early_weight = weights[target == 1].sum()
    weights[wrong_early] *= 2.5
    weights[target == 1] *= early_weight / weights[target == 1].sum()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.002, weight_decay=0.05)
    generator = torch.Generator().manual_seed(seed)
    selection, states, history = {}, {}, []
    for epoch in range(1, EPOCHS[-1] + 1):
        loss_total = 0.
        for indices in torch.randperm(len(target), generator=generator).split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(features(head, fit, indices), target[indices], weight=weights[indices])
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(indices)
        if epoch not in EPOCHS:
            continue
        with torch.no_grad():
            checked = scan(dev, features(head, dev), context, epoch, loss_total / len(target))
        history.extend(item["record"] for item in checked)
        for mode in MODES:
            best = max((item for item in checked if item["record"]["mode"] == mode), key=rank)
            if mode not in selection or rank(best) > rank(selection[mode]):
                selection[mode] = best
                states[mode] = {key: value.detach().clone() for key, value in head.state_dict().items()}
    folder = RUN / f"{arm}_s{seed}"
    folder.mkdir(exist_ok=True)
    write_rows(folder / "history.csv", history)
    for mode, card in selection.items():
        weight = folder / f"{mode}.pt"
        temporary = weight.with_suffix(".tmp")
        torch.save({"state": states[mode], "local": ARMS[arm]}, temporary)
        temporary.replace(weight)
        card["weight_sha256"] = sha256(weight)
        print(arm, seed, mode, card["record"], flush=True)
    return {f"{arm}_{mode}_s{seed}": card for mode, card in selection.items()}


def checked_splits(protocol):
    paths, rows, forbidden = checked_data(protocol)
    lock = read_json(DATA / "freeze.json")
    path = verified(DATA / "development.csv", lock["manifest_sha256"])
    verified(DATA / "inventory.json", lock["inventory_sha256"])
    supplement = read_rows(path)
    hashes = {row["sha256"] for row in supplement}
    if len(hashes) != len(supplement) or hashes & forbidden or len(hashes) != sum(lock["counts"].values()):
        raise ValueError("Public extension development images overlap or do not match the receipt")
    if any(sha256(Path(row["path"])) != row["sha256"] for row in supplement):
        raise ValueError("A frozen public extension image has changed")
    paths["extension_dev"], rows["extension_dev"] = path, supplement
    return paths, rows, forbidden | hashes


def make_plan(paths, incumbent):
    return {"source_sha256": sha256(Path(__file__)), "incumbent_sha256": sha256(incumbent),
            "ext_development_sha256": sha256(DATA / "freeze.json"), "prior_data_sha256": sha256(PREVIOUS_DATA / "freeze.json"),
            "manifests": {name: sha256(path) for name, path in paths.items()}, "epochs": list(EPOCHS), "seeds": list(SEEDS),
            "thresholds": list(THRESHOLDS), "healthy_gap_limits": list(GAPS), "arms": list(ARMS), "modes": list(MODES),
            "selection": "Zero new healthy false alarms and late-to-early errors, preserve each group's late correctness and "
                         "legacy/Malik Macro-F1. Require early rescues in at least two development groups; require seed 42 "
                         "and a confirmatory seed. Rank by number of groups then mean group recovery. Static comparison uses "
                         "only a fixed incumbent health-gap cap and no learned evidence.",
            "paired_evaluation": "Evaluate every selected direct head again with the original disease-rank constraint at "
                                 "the same threshold and checkpoint, to isolate removing the constraint.",
            "role": "Extension images hold out from fitting. Aggregate archive suites are retrospective. New field and "
                    "PlantDoc are post-lock audits. No independent three-class release test has been confirmed."}


def train(encoder, components, paths, rows, plan):
    RUN.mkdir(exist_ok=True)
    lock = RUN / "decision.json"
    plan_path = RUN / "protocol.json"
    if lock.exists():
        raise FileExistsError("The development decision is already frozen")
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError("The frozen development protocol differs")
    else:
        write_json(plan_path, plan)
    order = ("augmented", "healthy_fit", "development_field", "development_malik", "early_development", "extension_dev")
    fit, extra, field, malik, sheikh, extension = [encoded(encoder, components, paths[key], key, rows[key]) for key in order]
    training = pack((fit, extra))
    dev = pack((field, malik, sheikh, extension))
    old_n, malik_n = len(field["truth"]) + len(malik["truth"]), len(malik["truth"])
    historic_n = old_n + len(sheikh["truth"])
    ordered_rows = [row for name in order[2:] for row in rows[name]]
    groups = defaultdict(list)
    for i, row in enumerate(ordered_rows):
        group = row["capture_group"]
        groups[SHARED_DEVELOPMENT_GROUP.get(group, group)].append(i)
    legacy = development_metrics(dev["incumbent"][:old_n], {"truth": dev["truth"][:old_n]}, malik_n)
    before = {"legacy": legacy, "groups": group_counts(dev["truth"], dev["incumbent"].argmax(1), groups)}
    previous = read_json(WORKSPACE / "outputs" / "health_rescue_trial" / "decision.json")["development_incumbent"]
    old_groups = {group: ids for group, ids in groups.items() if ids[0] < historic_n}
    if any(legacy[key] != previous["legacy"][key] for key in ("confusion", "malik")) or any(
            before["groups"][key] != previous["groups"][key] for key in old_groups):
        raise ValueError("The original development baseline differs")
    context = groups, before, old_n, malik_n
    static = torch.full_like(dev["incumbent"][:, 0], 8.)
    fixed = {mode: [inspect_setting(dev, static, context, 0, None, THRESHOLDS[0], gap, mode) for gap in GAPS]
             for mode in MODES}
    write_rows(RUN / "static_history.csv", [item["record"] for mode in MODES for item in fixed[mode]])
    selections = {f"static_{mode}": max(fixed[mode], key=rank) for mode in MODES}
    train_rows = rows["augmented"] + rows["healthy_fit"]
    print("Development cohort:", before["groups"], flush=True)
    for arm, seed in product(ARMS, SEEDS):
        selections.update(train_head(arm, seed, training, train_rows, dev, context))
    eligible = [f"{arm}_{mode}_s42" for arm, mode in product(ARMS, MODES)
                if selections[f"{arm}_{mode}_s42"]["record"]["eligible"]
                and any(selections[f"{arm}_{mode}_s{seed}"]["record"]["eligible"] for seed in SEEDS[1:])]
    eligible.extend(key for key in ("static_ranked", "static_direct") if selections[key]["record"]["eligible"])
    chosen = max(eligible, key=lambda key: rank(selections[key]), default="incumbent")
    write_json(lock, {"candidate": chosen, "qualified": eligible, "protocol_sha256": sha256(plan_path),
                      "development_incumbent": before, "selections": selections})
    print("Locked choice:", chosen, flush=True)


def evaluation_receipts(protocol):
    audits = read_json(EXPANDED / "evaluation_freeze.json")["external_audits"]
    tests = {**test_receipts(protocol), **read_json(TAIWAN)["tests"], **read_json(AGRI_AUDIT)["tests"],
             "plantseg_subtype_stress": read_json(PLANTSEG), **audits}
    return tests, audits


def restore_head(key, card):
    arm, mode, seed = key.split("_")
    file = verified(RUN / f"{arm}_{seed}" / f"{mode}.pt", card["weight_sha256"])
    checkpoint = torch.load(file, map_location="cpu", weights_only=True)
    layer = RescueHead(checkpoint["local"])
    layer.load_state_dict(checkpoint["state"])
    return layer.eval()


@torch.no_grad()
def evaluate(encoder, components, plan, protocol, forbidden):
    lock = RUN / "decision.json"
    decision = read_json(lock)
    frozen = read_json(verified(RUN / "protocol.json", decision["protocol_sha256"]))
    if plan != frozen:
        raise ValueError("The frozen training context changed")
    original = create_model("resnet18")
    original.load_state_dict(original_checkpoint("AC42")["model_state"])
    original.to(encoder.device, memory_format=torch.channels_last).eval().requires_grad_(False)
    selections = decision["selections"]
    heads = {key: restore_head(key, item) for key, item in selections.items() if not key.startswith("static_")}
    decision_digest = sha256(lock)
    tests, audits = evaluation_receipts(protocol)
    for suite, spec in tests.items():
        result = RUN / f"{suite}_results.json"
        if result.exists():
            archived = read_json(result)
            if archived["decision_sha256"] != decision_digest or archived["manifest_sha256"] != spec["sha256"]:
                raise ValueError(f"Existing evaluation hash changed: {suite}")
            continue
        path = verified(Path(spec["path"]), spec["sha256"])
        rows = read_rows(path)
        hashes = {row["sha256"] for row in rows}
        if len(hashes) != len(rows) or hashes & forbidden:
            raise ValueError(f"Evaluation overlaps fitting or development: {suite}")
        data = encoded(encoder, components, path, suite, rows, original)
        baseline, truth = data["incumbent"], data["truth"]
        scores = {"original_AC42": data["original"], "incumbent": baseline}
        for key, item in selections.items():
            card = item["record"]
            evidence = features(heads[key], data) if key in heads else torch.full_like(baseline[:, 0], 8.)
            scores[key] = route(baseline, evidence, card["threshold"], card["maximum_healthy_gap"], card["mode"])
            if card["mode"] == "direct":
                scores[f"{key}_matched_ranked"] = route(baseline, evidence, card["threshold"], card["maximum_healthy_gap"], "ranked")
        prior = baseline.argmax(1)
        leading_late = baseline[:, 2] > baseline[:, 1]
        metrics, predictions = {}, []
        for key, logits in scores.items():
            values, predicted = (measurement(logits, truth)[:2] if truth.unique().numel() == len(CLASSES)
                                 else measure(logits, truth))
            switches = predicted != prior
            if key not in ("original_AC42", "incumbent") and (switches & ((prior != 0) | (predicted != 1))).any():
                raise ValueError(f"Invalid rescue route: {suite}, {key}")
            evaluation_rows = {"early_rescues": int((switches & (truth == 1) & (predicted == 1)).sum()),
                               "new_healthy_false_alarms": int((switches & (truth == 0) & (prior == 0)).sum()),
                               "new_late_as_early": int((switches & (truth == 2) & (prior == 0) & (predicted == 1)).sum()),
                               "rescues_with_late_leading": int((switches & leading_late & (truth == 1) & (predicted == 1)).sum())}
            if key != "original_AC42":
                values.update(evaluation_rows)
            metrics[key] = values
            predictions.extend({"model": key, "sha256": row["sha256"], "true_label": row["label"],
                                "prediction": CLASSES[guess], "incumbent_leading_late": leading}
                               for row, guess, leading in zip(rows, predicted.tolist(), leading_late.tolist(), strict=True))
        write_rows(RUN / f"{suite}_predictions.csv", predictions)
        write_json(result, {"decision_sha256": decision_digest, "manifest_sha256": spec["sha256"], "models": metrics,
                            "role": "external_audit" if suite in audits else spec.get("role", "historical_replay")})
        print(suite, {key: round(metrics[key]["accuracy"], 5) for key in ("incumbent", "static_direct", "mean_direct_s42", "local_direct_s42")}, flush=True)
    promotion = RUN / "promotion.json"
    if not promotion.exists():
        totals = Counter()
        for suite in HISTORICAL:
            for key, values in read_json(RUN / f"{suite}_results.json")["models"].items():
                totals[key] += sum(values["confusion"][i][i] for i in range(len(CLASSES)))
        write_json(promotion, {"candidate": decision["candidate"], "qualifies": False,
                               "independent_three_class_confirmed": False, "historical_correct_totals": dict(totals)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    phase = parser.parse_args().phase
    protocol = read_json(SUBTYPE_RUN / "protocol.json")
    paths, rows, forbidden = checked_splits(protocol)
    early_path = verified(EARLY_RUN / "decision.json", read_json(VIEW_RUN / "protocol.json")["incumbent_sha256"])
    early = read_json(early_path)
    health = read_json(verified(HEALTH_RUN / "decision.json", early["health_decision_sha256"]))
    subtype = read_json(verified(SUBTYPE_RUN / "decision.json", health["subtype_decision_sha256"]))
    gate, review = health["preferred_by_development"], early["preferred_by_development"]
    context = (load_heads(subtype), load_gate(gate, health["selections"][gate]), restore(review, early["selections"][review]))
    plan = make_plan(paths, early_path)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", protocol["parent_sha256"]), device)
    if phase == "train":
        train(encoder, context, paths, rows, plan)
    else:
        evaluate(encoder, context, plan, protocol, forbidden)


if __name__ == "__main__":
    main()
