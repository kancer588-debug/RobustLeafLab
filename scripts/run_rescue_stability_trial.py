"""Replicate early rescue training and evaluate pre-registered seed agreement."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch

import run_early_override_trial as override
from run_dino_health_trial import RUN as HEALTH_RUN, load_gate
from run_early_review_trial import RUN as EARLY_RUN, restore
from run_health_rescue_trial import SHARED_DEVELOPMENT_GROUP, VIEW_RUN, encoded, features, group_counts, pack, record
from run_late_protection_trial import load_heads
from run_local_leaf_trial import measure
from run_roi_trial import HISTORICAL
from run_subtype_trial import (
    CLASSES, FIRST_RUN, RUN as SUBTYPE_RUN, Specialists, development_metrics, measurement, read_rows, verified, write_rows,
)
from train_online_candidate import (
    WORKSPACE, checked_checkpoint, create_model, original_checkpoint, read_json, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "rescue_stability_trial"
PREVIOUS = override.RUN
REPLICATION = RUN / "replication"
ORIGINAL_SEEDS = (42, 43, 44)
FRESH_SEEDS = (45, 46, 47)
SEEDS = ORIGINAL_SEEDS + FRESH_SEEDS
THRESHOLD_DELTA = 0.25
PRIMARY = "both_majorities"


def vote_masks(per_seed):
    """Require two votes in each triplet even when testing the loss of one member."""
    first = torch.stack([per_seed[seed] for seed in ORIGINAL_SEEDS]).sum(0)
    second = torch.stack([per_seed[seed] for seed in FRESH_SEEDS]).sum(0)
    prior, fresh = first >= 2, second >= 2
    return {"prior_majority": prior, "fresh_majority": fresh,
            "prior_unanimous": first == 3, "fresh_unanimous": second == 3,
            "either_majority": prior | fresh, PRIMARY: prior & fresh,
            **{f"without_s{seed}": ((first - per_seed[seed].long()) >= 2) & fresh for seed in ORIGINAL_SEEDS},
            **{f"without_s{seed}": prior & ((second - per_seed[seed].long()) >= 2) for seed in FRESH_SEEDS}}


def deploy_vote(scores, vote):
    return override.route(scores, vote.float(), 0.5, float("inf"), "direct")


@torch.no_grad()
def score_models(data, heads, cards):
    base = data["incumbent"]
    evidence = {seed: features(heads[seed], data) for seed in SEEDS}
    masks = {seed: (override.route(base, evidence[seed], cards[seed]["threshold"],
                                  cards[seed]["maximum_healthy_gap"], "direct").argmax(1) == 1) & (base.argmax(1) == 0)
             for seed in SEEDS}
    comparison = {f"local_direct_s{seed}": deploy_vote(base, masks[seed]) for seed in SEEDS}
    comparison.update({name: deploy_vote(base, mask) for name, mask in vote_masks(masks).items()})
    for direction, delta in (("looser", -THRESHOLD_DELTA), ("tighter", THRESHOLD_DELTA)):
        shifted = {seed: (override.route(base, evidence[seed], cards[seed]["threshold"] + delta,
                                        cards[seed]["maximum_healthy_gap"], "direct").argmax(1) == 1) & (base.argmax(1) == 0)
                   for seed in SEEDS}
        comparison[f"both_majorities_{direction}"] = deploy_vote(base, vote_masks(shifted)[PRIMARY])
    return comparison


def restore_models(previous, fresh):
    heads, cards = {}, {}
    for seeds, selection, root in ((ORIGINAL_SEEDS, previous["selections"], PREVIOUS), (FRESH_SEEDS, fresh, REPLICATION)):
        for seed in seeds:
            key = f"local_direct_s{seed}"
            selected = selection[key]
            weight = verified(root / f"local_s{seed}" / "direct.pt", selected["weight_sha256"])
            checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
            if checkpoint["local"] is not True:
                raise ValueError(f"A rescue vote needs a local evidence head: {key}")
            head = override.RescueHead(True)
            head.load_state_dict(checkpoint["state"])
            heads[seed], cards[seed] = head.eval(), selected["record"]
    return heads, cards


def fit_confirmatory(training, training_rows, dev, context):
    REPLICATION.mkdir(exist_ok=True)
    selections = {}
    override.RUN = REPLICATION
    try:
        for seed in FRESH_SEEDS:
            selections.update(override.train_head("local", seed, training, training_rows, dev, context))
    finally:
        override.RUN = PREVIOUS
    return selections


def make_plan(paths, previous_lock, previous_plan):
    return {"source_sha256": sha256(Path(__file__)), "prior_decision_sha256": sha256(previous_lock),
            "prior_protocol_sha256": previous_plan["protocol_sha256"],
            "manifests": {name: sha256(path) for name, path in paths.items()},
            "original_seeds": list(ORIGINAL_SEEDS), "confirmatory_seeds": list(FRESH_SEEDS),
            "primary": PRIMARY, "threshold_sensitivity_delta": THRESHOLD_DELTA,
            "training": "Repeat the frozen local-rescue training and direct threshold selection for all three new seeds. "
                        "Keep all six seeds, their individual checkpoints, thresholds, and maximum healthy gaps.",
            "decision": "Preselect the two-of-three vote in each triplet and require agreement between triplets. "
                        "Require at least two development groups to gain, with zero new healthy false alarms, "
                        "zero additional late-to-early errors, unchanged per-group late correct counts, and unchanged or "
                        "better legacy and Malik Macro-F1, also at both predeclared threshold perturbations. "
                        "Report unanimity, agreement in either group, and single-member removal as diagnostics.",
            "evaluation": "All archived test inference occurs after locking. New field and PlantDoc are external audits only. "
                          "Archived replays cannot qualify a release as independently confirmed."}


def train(encoder, components, paths, rows, plan, previous):
    RUN.mkdir(exist_ok=True)
    lock, plan_path = RUN / "decision.json", RUN / "protocol.json"
    if lock.exists():
        raise FileExistsError("The stability decision is already frozen")
    if not plan_path.exists():
        write_json(plan_path, plan)
    elif read_json(plan_path) != plan:
        raise ValueError("The frozen trial protocol differs")
    order = ("augmented", "healthy_fit", "development_field", "development_malik", "early_development", "extension_dev")
    fit, extra, field, malik, sheikh, extension = [encoded(encoder, components, paths[name], name, rows[name]) for name in order]
    training = pack((fit, extra))
    dev = pack((field, malik, sheikh, extension))
    malik_n = len(malik["truth"])
    old_n = len(field["truth"]) + malik_n
    groups = defaultdict(list)
    for i, row in enumerate(row for name in order[2:] for row in rows[name]):
        source = row["capture_group"]
        groups[SHARED_DEVELOPMENT_GROUP.get(source, source)].append(i)
    start = {"legacy": development_metrics(dev["incumbent"][:old_n], {"truth": dev["truth"][:old_n]}, malik_n),
             "groups": group_counts(dev["truth"], dev["incumbent"].argmax(1), groups)}
    if start != previous["development_incumbent"]:
        raise ValueError("The frozen development baseline differs")
    context = groups, start, old_n, malik_n
    fresh = fit_confirmatory(training, rows["augmented"] + rows["healthy_fit"], dev, context)
    variants = score_models(dev, *restore_models(previous, fresh))
    stats = {}
    for name, scores in variants.items():
        item = record(scores, dev["truth"], *context, 0, 0., None)
        gains = [group for group, values in item["groups"].items()
                 if values["early_missed"] < start["groups"][group]["early_missed"]]
        item["record"].update({"groups_with_early_gain": gains, "passes_stability_limits":
                               item["record"]["guardrails"] and len(gains) >= override.MIN_RESCUED_GROUPS})
        stats[name] = item
    for seed in SEEDS:
        key = f"local_direct_s{seed}"
        selected = previous["selections"][key] if seed in ORIGINAL_SEEDS else fresh[key]
        if stats[key]["groups"] != selected["groups"] or stats[key]["legacy"] != selected["legacy"]:
            raise ValueError(f"The retained {key} checkpoint does not reproduce its development selection")
    keys = (PRIMARY, "both_majorities_looser", "both_majorities_tighter")
    passes = all(stats[key]["record"]["passes_stability_limits"] for key in keys)
    write_json(lock, {"candidate": PRIMARY if passes else "incumbent", "development_passed": passes,
                      "protocol_sha256": sha256(plan_path), "fresh_selections": fresh, "development": stats})
    print("Development lock:", {key: stats[key]["record"] for key in keys}, flush=True)


@torch.no_grad()
def evaluate(encoder, components, plan, protocol, forbidden, previous):
    lock = RUN / "decision.json"
    decision = read_json(lock)
    if plan != read_json(verified(RUN / "protocol.json", decision["protocol_sha256"])):
        raise ValueError("The frozen stability protocol differs")
    heads, cards = restore_models(previous, decision["fresh_selections"])
    tests, audits = override.evaluation_receipts(protocol)
    original = create_model("resnet18")
    original.load_state_dict(original_checkpoint("AC42")["model_state"])
    original.to(encoder.device, memory_format=torch.channels_last).eval().requires_grad_(False)
    decision_digest = sha256(lock)
    for suite, spec in tests.items():
        result = RUN / f"{suite}_results.json"
        if result.exists():
            completed = read_json(result)
            if completed["manifest_sha256"] != spec["sha256"] or completed["decision_sha256"] != decision_digest:
                raise ValueError(f"Mismatched result history: {suite}")
            continue
        path = verified(Path(spec["path"]), spec["sha256"])
        rows = read_rows(path)
        hashes = {row["sha256"] for row in rows}
        if len(hashes) != len(rows) or hashes & forbidden:
            raise ValueError(f"Evaluation overlaps fitting or development: {suite}")
        data = encoded(encoder, components, path, suite, rows, original)
        base, truth = data["incumbent"], data["truth"]
        inputs = {"original_AC42": data["original"], "incumbent": base, **score_models(data, heads, cards)}
        baseline = base.argmax(1)
        late_leading = base[:, 2] > base[:, 1]
        three_class = truth.unique().numel() == len(CLASSES)
        predictions, metrics = [], {}
        for name, scores in inputs.items():
            values, labels = measurement(scores, truth)[:2] if three_class else measure(scores, truth)
            if name != "original_AC42":
                changed = labels != baseline
                if (changed & ((labels != 1) | (baseline != 0))).any():
                    raise ValueError(f"A model changed a decision outside allowed early rescue: {suite} {name}")
                values.update({"early_rescues": int((changed & (truth == 1)).sum()),
                               "new_healthy_false_alarms": int((changed & (truth == 0)).sum()),
                               "new_late_as_early": int((changed & (truth == 2)).sum()),
                               "rescues_with_late_leading": int((changed & late_leading & (truth == 1)).sum())})
            metrics[name] = values
            predictions.extend({"model": name, "sha256": row["sha256"], "true_label": row["label"],
                                "prediction": CLASSES[label]} for row, label in zip(rows, labels.tolist(), strict=True))
        write_rows(RUN / f"{suite}_predictions.csv", predictions)
        write_json(result, {"manifest_sha256": spec["sha256"], "decision_sha256": decision_digest,
                            "role": "external_audit" if suite in audits else spec.get("role", "historical_replay"), "models": metrics})
        print(suite, {key: round(metrics[key]["accuracy"], 5) for key in ("incumbent", "prior_majority", "fresh_majority", PRIMARY)}, flush=True)
    promotion = RUN / "promotion.json"
    if not promotion.exists():
        pooled = defaultdict(Counter)
        for suite in HISTORICAL:
            for key, values in read_json(RUN / f"{suite}_results.json")["models"].items():
                pooled[key]["correct"] += sum(values["confusion"][i][i] for i in range(len(CLASSES)))
                for metric in ("early_rescues", "new_healthy_false_alarms", "new_late_as_early", "rescues_with_late_leading"):
                    if metric in values:
                        pooled[key][metric] += values[metric]
        write_json(promotion, {"candidate": decision["candidate"], "development_passed": decision["development_passed"],
                               "qualifies": False, "independent_three_class_confirmed": False, "historical": pooled})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    phase = parser.parse_args().phase
    protocol = read_json(SUBTYPE_RUN / "protocol.json")
    paths, rows, forbidden = override.checked_splits(protocol)
    previous_lock = PREVIOUS / "decision.json"
    previous = read_json(previous_lock)
    previous_plan = read_json(verified(PREVIOUS / "protocol.json", previous["protocol_sha256"]))
    verified(Path(override.__file__), previous_plan["source_sha256"])
    if {name: sha256(path) for name, path in paths.items()} != previous_plan["manifests"]:
        raise ValueError("A prior data split changed")
    early_path = verified(EARLY_RUN / "decision.json", read_json(VIEW_RUN / "protocol.json")["incumbent_sha256"])
    verified(early_path, previous_plan["incumbent_sha256"])
    early = read_json(early_path)
    health = read_json(verified(HEALTH_RUN / "decision.json", early["health_decision_sha256"]))
    subtype = read_json(verified(SUBTYPE_RUN / "decision.json", health["subtype_decision_sha256"]))
    gate, review = health["preferred_by_development"], early["preferred_by_development"]
    context = (load_heads(subtype), load_gate(gate, health["selections"][gate]), restore(review, early["selections"][review]))
    plan = make_plan(paths, previous_lock, previous)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", protocol["parent_sha256"]), device)
    if phase == "train":
        train(encoder, context, paths, rows, plan, previous)
    else:
        evaluate(encoder, context, plan, protocol, forbidden, previous)


if __name__ == "__main__":
    main()
