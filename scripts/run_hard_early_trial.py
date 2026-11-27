"""Compare supplemental field pictures and targeted hard-positive training weights."""

import argparse
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import torch
from torch.nn import functional as F

import run_early_override_trial as override
import run_last_block_lora_trial as matched
import run_rescue_stability_trial as stability
from run_dino_health_trial import RUN as HEALTH_RUN, load_gate
from run_early_review_trial import RUN as EARLY_RUN, fitting_weights, restore
from run_health_rescue_trial import SHARED_DEVELOPMENT_GROUP, VIEW_RUN, encoded, group_counts, pack
from run_late_protection_trial import load_heads
from run_local_leaf_trial import measure
from run_roi_trial import HISTORICAL
from run_subtype_trial import (
    CLASSES, FIRST_RUN, RUN as SUBTYPE_RUN, Specialists, development_metrics, measurement, read_rows, verified, write_rows,
)
from stage_hard_early_trial import ROOT
from train_online_candidate import (
    WORKSPACE, checked_checkpoint, create_model, original_checkpoint, read_json, seed_everything, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "hard_early_trial"
ARMS = ("base", "expanded", "hard_weighted")
SEEDS = matched.SEEDS
MAX_ADDITIONS = 48
GROUP_CAP = 4
NEW_HARD_FACTOR = 3.


def pool_info(forbidden):
    receipt = read_json(ROOT / "review_freeze.json")
    verified(ROOT / "inventory.json", receipt["inventory_sha256"])
    pool = verified(ROOT / "reviewed_training_pool.csv", receipt["sha256"])
    records = read_rows(pool)
    hashes = {row["sha256"] for row in records}
    if len(hashes) != len(records) or hashes & forbidden or len(records) != sum(receipt["counts"].values()):
        raise ValueError("The supplemental training pool is not disjoint or changed")
    if any(row["label"] != "early_blight" or sha256(Path(row["path"])) != row["sha256"] for row in records):
        raise ValueError("A supplemental early image changed")
    return pool, records, forbidden | hashes


def choose_pool(data, records):
    scores = data["incumbent"]
    hard = scores.argmax(1) == 0
    health_gap = scores[:, 0] - scores[:, 1:].amax(1)
    priority = sorted(range(len(records)), key=lambda i: (not hard[i].item(),
                      -health_gap[i].item() if hard[i] else 0.,
                      records[i]["view_type"] != "publisher_original_field_frame", records[i]["sha256"]))
    groups, chosen = Counter(), []
    for i in priority:
        group = records[i]["capture_group"]
        if groups[group] < GROUP_CAP:
            chosen.append(i)
            groups[group] += 1
        if len(chosen) == MAX_ADDITIONS:
            break
    mined = [{**records[i], "incumbent_healthy_miss": bool(hard[i]), "incumbent_health_gap": health_gap[i].item()}
             for i in chosen]
    write_rows(RUN / "supplement_training.csv", mined)
    stats = {"n_pool": len(records), "pool_true_hard_misses": int(hard.sum()), "n_added": len(chosen),
             "added_true_hard_misses": int(hard[chosen].sum()), "groups": dict(groups),
             "source_counts": dict(Counter(row["source"] for row in mined)),
             "hard_source_counts": dict(Counter(row["source"] for row in mined if row["incumbent_healthy_miss"]))}
    return {key: val[chosen] for key, val in data.items()}, mined, stats


def weight_vector(rows, data, original_n, targeted):
    targets = data["truth"]
    weights = fitting_weights(rows, data["incumbent"], targets)
    early = targets == 1
    miss = early & (data["incumbent"].argmax(1) == 0)
    added = torch.arange(len(targets)) >= original_n
    total_early = weights[early].sum()
    weights[miss & ~added] *= 2.5
    if targeted:
        weights[miss & added] *= NEW_HARD_FACTOR
    weights[early] *= total_early / weights[early].sum()
    return weights


def fit_head(encoder, identity, seed, data, dev, weights, context, epoch_size):
    seed_everything(seed)
    model = matched.RescueModel(encoder, adapted=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=matched.HEAD_LR, weight_decay=0.05)
    generator = torch.Generator().manual_seed(seed)
    labels = (data["truth"] == 1).float()
    best, history = None, []
    for epoch in range(1, matched.EPOCHS[-1] + 1):
        loss_sum = 0.
        for indices in torch.randperm(len(labels), generator=generator)[:epoch_size].split(matched.BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(data, indices), labels[indices].to(encoder.device),
                                                       weight=weights[indices].to(encoder.device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        if epoch not in matched.EPOCHS:
            continue
        evidence = matched.evidence(model, dev)
        thresholds = {threshold + delta for threshold, delta in product(override.THRESHOLDS, matched.SHIFTS)}
        checked = {(value, gap): matched.diagnostics(override.route(dev["incumbent"], evidence, value, gap, "direct"), dev, context)
                   for value, gap in product(thresholds, override.GAPS)}
        for threshold, gap in product(override.THRESHOLDS, override.GAPS):
            row = {"epoch": epoch, "threshold": threshold, "gap": gap, "loss": loss_sum / epoch_size,
                   **matched.summarize([checked[threshold + delta, gap] for delta in matched.SHIFTS])}
            history.append(row)
            if best is None or matched.rank(row) > matched.rank(best):
                best, state = row, model.saved_state()
    target = RUN / identity
    target.mkdir(exist_ok=True)
    write_rows(target / "history.csv", history)
    weight = target / "model.pt"
    temporary = weight.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(weight)
    best["weight_sha256"] = sha256(weight)
    print(identity, best, flush=True)
    return best


def restore_set(encoder, selection):
    models = {}
    for name, card in selection.items():
        if name.startswith("base_"):
            target = matched.RUN / name.replace("base_", "head_only_", 1)
        else:
            target = RUN / name
        state = torch.load(verified(target / "model.pt", card["weight_sha256"]), map_location="cpu", weights_only=True)
        head = matched.RescueModel(encoder, adapted=False)
        head.load_saved_state(state)
        models[name] = head.eval()
    return models


def score_models(data, models, cards):
    base, answers = data["incumbent"], {}
    healthy = base.argmax(1) == 0
    evidence = {key: matched.evidence(model, data) for key, model in models.items()}
    for arm in ARMS:
        for delta, suffix in zip(matched.SHIFTS, ("_looser", "", "_tighter"), strict=True):
            votes = []
            for seed in SEEDS:
                key = f"{arm}_s{seed}"
                logits = override.route(base, evidence[key], cards[key]["threshold"] + delta, cards[key]["gap"], "direct")
                votes.append(healthy & (logits.argmax(1) == 1))
                if delta == 0.:
                    answers[key] = logits
            answers[arm + "_majority" + suffix] = stability.deploy_vote(base, torch.stack(votes).sum(0) >= 2)
    return answers


def train(encoder, components, paths, split_rows, pool, supplemental, plan, previous):
    RUN.mkdir(exist_ok=True)
    lock, protocol = RUN / "decision.json", RUN / "protocol.json"
    if lock.exists():
        raise FileExistsError("The hard-early decision is already frozen")
    if protocol.exists():
        if read_json(protocol) != plan:
            raise ValueError("The existing protocol differs")
    else:
        write_json(protocol, plan)
    names = ("augmented", "healthy_fit", "development_field", "development_malik", "early_development", "extension_dev")
    train_a, train_b, *dev_parts = [encoded(encoder, components, paths[k], k, split_rows[k]) for k in names]
    fit, dev = pack((train_a, train_b)), pack(dev_parts)
    supplemental_data = encoded(encoder, components, pool, "training_candidates_only", supplemental)
    additions, added_rows, mining = choose_pool(supplemental_data, supplemental)
    if not added_rows:
        raise RuntimeError("The training supplement is empty")
    print("Training-only mining:", mining, flush=True)
    expanded = pack((fit, additions))
    training_rows = split_rows["augmented"] + split_rows["healthy_fit"] + added_rows
    groups = defaultdict(list)
    for i, row in enumerate(row for key in names[2:] for row in split_rows[key]):
        group = row["capture_group"]
        groups[SHARED_DEVELOPMENT_GROUP.get(group, group)].append(i)
    malik_n = len(split_rows["development_malik"])
    n_legacy = len(split_rows["development_field"]) + malik_n
    start = {"legacy": development_metrics(dev["incumbent"][:n_legacy], {"truth": dev["truth"][:n_legacy]}, malik_n),
             "groups": group_counts(dev["truth"], dev["incumbent"].argmax(1), groups)}
    if start != previous["development_incumbent"]:
        raise ValueError("The frozen development baseline changed")
    context = groups, start, n_legacy, malik_n
    prior = read_json(verified(matched.RUN / "decision.json", plan["head_only_decision_sha256"]))
    selection = {f"base_s{seed}": prior["selection"][f"head_only_s{seed}"] for seed in SEEDS}
    for arm in ARMS[1:]:
        weights = weight_vector(training_rows, expanded, len(fit["truth"]), arm == "hard_weighted")
        for seed in SEEDS:
            key = f"{arm}_s{seed}"
            selection[key] = fit_head(encoder, key, seed, expanded, dev, weights, context, len(fit["truth"]))
    answers = score_models(dev, restore_set(encoder, selection), selection)
    dev_metrics = {key: matched.diagnostics(value, dev, context) for key, value in answers.items()}
    comparison = {arm: matched.summarize([dev_metrics[f"{arm}_majority{suffix}"] for suffix in ("_looser", "", "_tighter")])
                  for arm in ARMS}
    for arm in ARMS:
        comparison[arm]["individually_passing_seeds"] = sum(selection[f"{arm}_s{seed}"]["passes"] for seed in SEEDS)
    if comparison["base"] != {**{k: v for k, v in prior["comparison"]["head_only"].items() if k != "passing_single_seeds"},
                               "individually_passing_seeds": prior["comparison"]["head_only"]["passing_single_seeds"]}:
        raise ValueError("The archived equal-budget control did not reproduce")
    quality = lambda arm: (comparison[arm]["minimum_rescues"], comparison[arm]["individually_passing_seeds"])
    eligible = [arm for arm in ARMS[1:] if comparison[arm]["passes"]
                and comparison[arm]["individually_passing_seeds"] >= 2 and quality(arm) > quality("base")]
    candidate = max(eligible, key=quality, default="incumbent")
    write_json(lock, {"candidate": candidate, "selection": selection, "development": dev_metrics, "comparison": comparison,
                      "mining": mining, "supplement_sha256": sha256(RUN / "supplement_training.csv"), "protocol_sha256": sha256(protocol)})
    print("Development choice", candidate, comparison, flush=True)


@torch.no_grad()
def evaluate(encoder, components, protocol, plan, forbidden, previous):
    decision_path = RUN / "decision.json"
    decision = read_json(decision_path)
    if read_json(verified(RUN / "protocol.json", decision["protocol_sha256"])) != plan:
        raise ValueError("The frozen evaluation protocol differs")
    verified(RUN / "supplement_training.csv", decision["supplement_sha256"])
    calibrated = read_json(verified(matched.VOTE_PLAN, plan["calibrated_vote_sha256"]))
    voting, cards = stability.restore_models(previous, calibrated["fresh_selections"])
    cards = {seed: {**card, "threshold": card["threshold"] + calibrated["offset"]} for seed, card in cards.items()}
    models = restore_set(encoder, decision["selection"])
    original = create_model("resnet18")
    original.load_state_dict(original_checkpoint("AC42")["model_state"])
    original.to(encoder.device, memory_format=torch.channels_last).eval().requires_grad_(False)
    tests, audits = override.evaluation_receipts(protocol)
    digest = sha256(decision_path)
    totals = defaultdict(Counter)
    for name, spec in tests.items():
        path = verified(Path(spec["path"]), spec["sha256"])
        rows = read_rows(path)
        hashes = {row["sha256"] for row in rows}
        if len(hashes) != len(rows) or hashes & forbidden:
            raise ValueError(f"A test overlaps fitting or development: {name}")
        result = RUN / f"{name}_results.json"
        if result.exists():
            payload = read_json(result)
            if payload["decision_sha256"] != digest or payload["manifest_sha256"] != spec["sha256"]:
                raise ValueError(f"Existing evaluation differs: {name}")
            metrics = payload["models"]
        else:
            data = encoded(encoder, components, path, name, rows, original)
            truth, base = data["truth"], data["incumbent"]
            six_vote = stability.score_models(data, voting, cards)[stability.PRIMARY]
            scores = {"original_AC42": data["original"], "incumbent": base, "calibrated_six_vote": six_vote,
                      **score_models(data, models, decision["selection"])}
            metrics, predictions = {}, []
            initial = base.argmax(1)
            three_classes = truth.unique().numel() == len(CLASSES)
            for key, logits in scores.items():
                values, predicted = measurement(logits, truth)[:2] if three_classes else measure(logits, truth)
                if key != "original_AC42":
                    changed = initial != predicted
                    if (changed & ((initial != 0) | (predicted != 1))).any():
                        raise ValueError(f"Unexpected direct rescue change: {name} {key}")
                    values.update(early_rescues=int((changed & (truth == 1)).sum()),
                                  healthy_fp_added=int((changed & (truth == 0)).sum()),
                                  late_as_early_added=int((changed & (truth == 2)).sum()))
                metrics[key] = values
                predictions.extend({"model": key, "sha256": row["sha256"], "true_label": row["label"], "prediction": CLASSES[label]}
                                   for row, label in zip(rows, predicted.tolist(), strict=True))
            write_rows(RUN / f"{name}_predictions.csv", predictions)
            write_json(result, {"manifest_sha256": spec["sha256"], "decision_sha256": digest, "models": metrics,
                                "role": "external_audit" if name in audits else spec.get("role", "historical_replay")})
            print(name, {k: round(metrics[k]["accuracy"], 5) for k in ("incumbent", "base_majority", "expanded_majority", "hard_weighted_majority")}, flush=True)
        if name in HISTORICAL:
            for key, values in metrics.items():
                totals[key]["correct"] += sum(values["confusion"][i][i] for i in range(len(CLASSES)))
                for metric in ("early_rescues", "healthy_fp_added", "late_as_early_added"):
                    if metric in values:
                        totals[key][metric] += values[metric]
    if not (RUN / "promotion.json").exists():
        write_json(RUN / "promotion.json", {"candidate": decision["candidate"], "historical": totals, "qualifies": False,
                                            "independent_three_class_confirmed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    phase = parser.parse_args().phase
    foundation = read_json(SUBTYPE_RUN / "protocol.json")
    paths, rows, forbidden = override.checked_splits(foundation)
    pool, supplemental, forbidden = pool_info(forbidden)
    prior_path = override.RUN / "decision.json"
    previous = read_json(prior_path)
    prior_protocol = read_json(verified(override.RUN / "protocol.json", previous["protocol_sha256"]))
    digests = {key: sha256(path) for key, path in paths.items()}
    if digests != prior_protocol["manifests"]:
        raise ValueError("The base training and development manifests changed")
    early_file = verified(EARLY_RUN / "decision.json", read_json(VIEW_RUN / "protocol.json")["incumbent_sha256"])
    early = read_json(verified(early_file, prior_protocol["incumbent_sha256"]))
    health = read_json(verified(HEALTH_RUN / "decision.json", early["health_decision_sha256"]))
    subtype = read_json(verified(SUBTYPE_RUN / "decision.json", health["subtype_decision_sha256"]))
    gate, review = health["preferred_by_development"], early["preferred_by_development"]
    components = (load_heads(subtype), load_gate(gate, health["selections"][gate]), restore(review, early["selections"][review]))
    plan = {"source_sha256": sha256(Path(__file__)), "incumbent_sha256": sha256(early_file), "previous_sha256": sha256(prior_path),
            "head_only_decision_sha256": sha256(matched.RUN / "decision.json"), "calibrated_vote_sha256": sha256(matched.VOTE_PLAN),
            "manifests": digests, "reviewed_pool_sha256": sha256(pool), "pool_review_receipt_sha256": sha256(ROOT / "review_freeze.json"),
            "seeds": list(SEEDS), "training_epochs": list(matched.EPOCHS), "batch_size": matched.BATCH,
            "max_additions": MAX_ADDITIONS, "per_group_limit": GROUP_CAP, "hard_weight_multiplier": NEW_HARD_FACTOR,
            "mining": "Rank the visually reviewed training-only pool by frozen incumbent healthy mistakes, then margin. "
                      "Use the same sampled source images for both expanded arms and preserve 909 draws per epoch.",
            "selection": "Require two development groups with fewer early misses and zero new healthy or late-to-early "
                         "errors at the central and both perturbed majority thresholds, plus two individually passing seeds. "
                         "Require improvement over the frozen equal-budget head-only control.",
            "roles": "Selection uses development. Historical replay and external audits run only after locking."}
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", foundation["parent_sha256"]), device)
    if phase == "train":
        train(encoder, components, paths, rows, pool, supplemental, plan, previous)
    else:
        evaluate(encoder, components, foundation, plan, forbidden, previous)


if __name__ == "__main__":
    main()
