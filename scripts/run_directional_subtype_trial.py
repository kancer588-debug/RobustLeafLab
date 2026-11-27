"""Train direction-conditioned early/late corrections around a locked health route."""

import argparse
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import run_early_override_trial as override
import run_hard_early_trial as rescue
from run_dino_health_trial import RUN as HEALTH_RUN, load_gate, source_name
from run_early_review_trial import RUN as EARLY_RUN, restore
from run_health_rescue_trial import SHARED_DEVELOPMENT_GROUP, VIEW_RUN, encoded, pack
from run_last_block_lora_trial import VOTE_PLAN, evidence
from run_late_protection_trial import load_heads
from run_local_leaf_trial import measure
from run_rescue_stability_trial import PRIMARY, deploy_vote, restore_models, score_models as previous_votes
from run_roi_trial import HISTORICAL
from run_subtype_trial import (
    CLASSES, FIRST_RUN, RUN as SUBTYPE_RUN, Specialists, measurement, read_rows, verified, write_rows,
)
from train_online_candidate import (
    WORKSPACE, checked_checkpoint, create_model, original_checkpoint, read_json, seed_everything, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "directional_subtype_trial"
ARMS = ("shared", "directional")
SEEDS = (42, 45, 46)
EPOCHS = (1, 2, 4, 8)
THRESHOLDS = (0.5, 1., 2., 3.)
MAX_GAPS = (0.5, 1., 2., 4.)
SHIFTS = (-0.25, 0., 0.25)
BATCH = 64


class Probe(nn.Module):
    """Use a shared image projection and one shared or two conditional logit outputs."""

    def __init__(self, directional):
        super().__init__()
        self.directional = directional
        self.image = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 48), nn.GELU())
        shared_output = nn.Linear(48, 1)
        if directional:
            self.output = nn.Linear(48, 2)
            with torch.no_grad():
                self.output.weight.copy_(shared_output.weight.expand_as(self.output.weight))
                self.output.bias.copy_(shared_output.bias.expand_as(self.output.bias))
        else:
            self.output = shared_output

    def forward(self, image, late_leads):
        logits = self.output(self.image(image))
        if self.directional:
            return logits.gather(1, late_leads.long().unsqueeze(1)).flatten()
        return logits.flatten()


def lead(data):
    raw = data["incumbent"]
    return raw[:, 2] > raw[:, 1]


def rescue_baseline(data, models, locked):
    base = data["incumbent"]
    votes = []
    for seed in SEEDS:
        key = f"expanded_s{seed}"
        card = locked[key]
        proposed = override.route(base, evidence(models[key], data), card["threshold"], card["gap"], "direct")
        votes.append((base.argmax(1) == 0) & (proposed.argmax(1) == 1))
    return deploy_vote(base, torch.stack(votes).sum(0) >= 2)


def replace_subtype(base, switch):
    scores = base.clone()
    incumbent = base.argmax(1)
    other = 3 - incumbent[switch]
    scores[switch, other] = base[switch].amax(1) + 1.
    return scores


def proposed(data, logit, early_threshold, late_threshold, gap_limit):
    raw = data["incumbent"]
    gap = (raw[:, 1] - raw[:, 2]).abs()
    late_first = lead(data)
    active = (raw.argmax(1) != 0) & (gap <= gap_limit)
    return active & torch.where(late_first, logit >= early_threshold, logit <= -late_threshold)


def weights_for(rows, truth, late_leads):
    strata = [(side.item(), row["label"], source_name(row)) for row, side in zip(rows, late_leads, strict=True)]
    counts = Counter(strata)
    weights = torch.tensor([max(8, counts[key]) ** -0.5 for key in strata])
    for side, label in product((False, True), (1, 2)):
        indices = (late_leads == side) & (truth == label)
        if not indices.any():
            raise ValueError("Both diseases must be represented in each training direction")
        weights[indices] *= (len(rows) / 4) / weights[indices].sum()
    return weights


def grouped_stats(scores, truth, group_keys, groups, old_correct, counts):
    correct = (scores.argmax(1) == truth).long()
    totals = torch.zeros_like(old_correct).scatter_add_(0, group_keys, correct)
    gains = (totals - old_correct).reshape(len(groups), 3)
    covered = counts.reshape(len(groups), 3)[:, 1:] > 0
    changes = gains[:, 1:]
    change_groups = [group for group, increase in zip(groups, changes.sum(1).tolist(), strict=True) if increase > 0]
    relative = (changes / counts.reshape(len(groups), 3)[:, 1:].clamp_min(1))[covered]
    return {"passes": bool((changes >= 0).all()) and len(change_groups) >= 2,
            "guardrails": bool((changes >= 0).all()), "groups": change_groups,
            "net_correct": int(changes.sum()), "early_delta": int(changes[:, 0].sum()), "late_delta": int(changes[:, 1].sum()),
            "mean_group_class_recall_delta": float(relative.mean())}


def summarize(variants):
    return {"pass": all(row["passes"] for row in variants), "guardrails": all(row["guardrails"] for row in variants),
            "min_groups": min(len(row["groups"]) for row in variants), "min_correct_gain": min(row["net_correct"] for row in variants),
            "min_balanced_gain": min(row["mean_group_class_recall_delta"] for row in variants),
            "center_early_delta": variants[1]["early_delta"], "center_late_delta": variants[1]["late_delta"]}


def rank(row):
    return (row["pass"], row["guardrails"], row["min_groups"], row["min_balanced_gain"], row["min_correct_gain"],
            -row["gap"], row["early_threshold"] + row["late_threshold"], -row["epoch"])


def train_head(key, arm, seed, fit, dev, base, weights, metrics):
    seed_everything(seed)
    head = Probe(arm == "directional")
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.002, weight_decay=0.05)
    generator = torch.Generator().manual_seed(seed)
    targets = (fit["truth"] == 1).float()
    direction = lead(fit)
    best, history = None, []
    for epoch in range(1, EPOCHS[-1] + 1):
        for indices in torch.randperm(len(targets), generator=generator).split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            predictions = head(fit["dino"][indices], direction[indices])
            loss = F.binary_cross_entropy_with_logits(predictions, targets[indices], weight=weights[indices])
            loss.backward()
            optimizer.step()
        if epoch not in EPOCHS:
            continue
        with torch.no_grad():
            evidence_score = head(dev["dino"], lead(dev))
        for early, late, gap in product(THRESHOLDS, THRESHOLDS, MAX_GAPS):
            shifts = [metrics(replace_subtype(base, proposed(dev, evidence_score, early + d, late + d, gap))) for d in SHIFTS]
            card = {"epoch": epoch, "early_threshold": early, "late_threshold": late, "gap": gap, **summarize(shifts)}
            history.append(card)
            if best is None or rank(card) > rank(best):
                best, state = card, {name: value.detach().clone() for name, value in head.state_dict().items()}
    directory = RUN / key
    directory.mkdir(exist_ok=True)
    write_rows(directory / "history.csv", history)
    weight = directory / "probe.pt"
    temp = weight.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(weight)
    best["sha256"] = sha256(weight)
    print(key, best, flush=True)
    return best


def restore_probes(cards):
    heads = {}
    for key, item in cards.items():
        model = Probe(key.startswith("directional_"))
        path = verified(RUN / key / "probe.pt", item["sha256"])
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        heads[key] = model.eval()
    return heads


@torch.no_grad()
def results(data, base, heads, cards):
    scores = {key: model(data["dino"], lead(data)) for key, model in heads.items()}
    variants = {}
    for arm in ARMS:
        for delta, suffix in zip(SHIFTS, ("_looser", "", "_tighter"), strict=True):
            votes = []
            for seed in SEEDS:
                key = f"{arm}_s{seed}"
                setting = cards[key]
                vote = proposed(data, scores[key], setting["early_threshold"] + delta,
                                setting["late_threshold"] + delta, setting["gap"])
                votes.append(vote)
                if delta == 0.:
                    variants[key] = replace_subtype(base, vote)
            variants[f"{arm}_majority{suffix}"] = replace_subtype(base, torch.stack(votes).sum(0) >= 2)
    return variants


def train(encoder, components, paths, rows, added_file, added, plan, rescued, rescue_cards):
    RUN.mkdir(exist_ok=True)
    protocol, lock = RUN / "protocol.json", RUN / "decision.json"
    if lock.exists():
        raise FileExistsError("The subtype selection is already locked")
    if not protocol.exists():
        write_json(protocol, plan)
    elif read_json(protocol) != plan:
        raise ValueError("The frozen protocol differs")
    dev_order = ("development_field", "development_malik", "early_development", "extension_dev")
    full_fit = pack([encoded(encoder, components, paths["augmented"], "augmented", rows["augmented"]),
                     encoded(encoder, components, added_file, "supplement_training", added)])
    dev = pack([encoded(encoder, components, paths[name], name, rows[name]) for name in dev_order])
    base = rescue_baseline(dev, rescued, rescue_cards)
    diseases = full_fit["truth"] != 0
    fit = {key: value[diseases] for key, value in full_fit.items()}
    fit_rows = [row for row, keep in zip(rows["augmented"] + added, diseases.tolist(), strict=True) if keep]
    direction = lead(fit)
    weights = weights_for(fit_rows, fit["truth"], direction)
    groups = {}
    indices = []
    for row in (row for key in dev_order for row in rows[key]):
        group = row["capture_group"]
        canonical = SHARED_DEVELOPMENT_GROUP.get(group, group)
        indices.append(groups.setdefault(canonical, len(groups)))
    keys = torch.tensor(indices) * 3 + dev["truth"]
    counts = torch.bincount(keys, minlength=len(groups) * 3)
    before = torch.zeros_like(counts).scatter_add_(0, keys, (base.argmax(1) == dev["truth"]).long())

    def metrics(scores):
        return grouped_stats(scores, dev["truth"], keys, groups, before, counts)

    selection = {f"{arm}_s{seed}": train_head(f"{arm}_s{seed}", arm, seed, fit, dev, base, weights, metrics)
                 for arm, seed in product(ARMS, SEEDS)}
    evaluated = {key: metrics(value) for key, value in results(dev, base, restore_probes(selection), selection).items()}
    comparison = {arm: summarize([evaluated[arm + "_majority" + suffix] for suffix in ("_looser", "", "_tighter")])
                  for arm in ARMS}
    for arm in ARMS:
        comparison[arm]["passing_seeds"] = sum(selection[f"{arm}_s{seed}"]["pass"] for seed in SEEDS)
    eligible = [arm for arm in ARMS if comparison[arm]["pass"] and comparison[arm]["passing_seeds"] >= 2]
    candidate = max(eligible, key=lambda arm: (comparison[arm]["min_balanced_gain"], comparison[arm]["min_correct_gain"]),
                    default="early_rescue_reference")
    composition = {f"incumbent_{'late' if side else 'early'}_truth_{CLASSES[label]}": int(((direction == side) & (fit['truth'] == label)).sum())
                   for side, label in product((False, True), (1, 2))}
    write_json(lock, {"candidate": candidate, "selections": selection, "development": evaluated, "comparison": comparison,
                      "training_direction_counts": composition, "protocol_sha256": sha256(protocol)})
    print("Development choice:", candidate, comparison, flush=True)


@torch.no_grad()
def evaluate(encoder, components, foundation, forbidden, plan, rescued, rescue_cards, prior_override):
    lock = RUN / "decision.json"
    decision = read_json(lock)
    if read_json(verified(RUN / "protocol.json", decision["protocol_sha256"])) != plan:
        raise ValueError("The frozen subtype protocol differs")
    heads = restore_probes(decision["selections"])
    prior_vote = read_json(verified(VOTE_PLAN, plan["calibrated_vote_sha256"]))
    six, six_cards = restore_models(prior_override, prior_vote["fresh_selections"])
    six_cards = {seed: {**card, "threshold": card["threshold"] + prior_vote["offset"]} for seed, card in six_cards.items()}
    original = create_model("resnet18")
    original.load_state_dict(original_checkpoint("AC42")["model_state"])
    original.to(encoder.device, memory_format=torch.channels_last).eval().requires_grad_(False)
    tests, audits = override.evaluation_receipts(foundation)
    digest = sha256(lock)
    totals = defaultdict(Counter)
    for name, spec in tests.items():
        path = verified(Path(spec["path"]), spec["sha256"])
        rows = read_rows(path)
        hashes = {row["sha256"] for row in rows}
        if len(hashes) != len(rows) or hashes & forbidden:
            raise ValueError(f"Fitting or development identity found in evaluation: {name}")
        file = RUN / f"{name}_results.json"
        if file.exists():
            stored = read_json(file)
            if stored["manifest_sha256"] != spec["sha256"] or stored["decision_sha256"] != digest:
                raise ValueError(f"Evaluation lock differs: {name}")
            table = stored["models"]
        else:
            data = encoded(encoder, components, path, name, rows, original)
            truth = data["truth"]
            rescue_output = rescue_baseline(data, rescued, rescue_cards)
            scores = {"original_AC42": data["original"], "incumbent": data["incumbent"],
                      "calibrated_six_vote": previous_votes(data, six, six_cards)[PRIMARY], "early_rescue_reference": rescue_output,
                      **results(data, rescue_output, heads, decision["selections"])}
            table, predictions = {}, []
            before = rescue_output.argmax(1)
            for key, value in scores.items():
                metrics, guess = measurement(value, truth)[:2] if truth.unique().numel() == 3 else measure(value, truth)
                if key.startswith(ARMS):
                    changed = guess != before
                    if (changed & ((before == 0) | (guess == 0) | (data["incumbent"].argmax(1) == 0))).any():
                        raise ValueError(f"A subtype model moved the rescue boundary: {name} {key}")
                table[key] = metrics
                predictions.extend({"model": key, "sha256": row["sha256"], "true_label": row["label"],
                                    "prediction": CLASSES[label]} for row, label in zip(rows, guess.tolist(), strict=True))
            write_rows(RUN / f"{name}_predictions.csv", predictions)
            write_json(file, {"models": table, "manifest_sha256": spec["sha256"], "decision_sha256": digest,
                              "role": "external_audit" if name in audits else spec.get("role", "historical_replay")})
            print(name, {key: round(table[key]["accuracy"], 5) for key in ("early_rescue_reference", "shared_majority", "directional_majority")}, flush=True)
        if name in HISTORICAL:
            for key, value in table.items():
                matrix = value["confusion"]
                totals[key].update({"correct": sum(matrix[i][i] for i in range(3)), "early_correct": matrix[1][1],
                                    "late_correct": matrix[2][2], "early_to_late": matrix[1][2],
                                    "late_to_early": matrix[2][1], "healthy_fp": matrix[0][1] + matrix[0][2]})
    if not (RUN / "promotion.json").exists():
        write_json(RUN / "promotion.json", {"candidate": decision["candidate"], "historical": totals,
                                            "qualifies": False, "independent_three_class_confirmed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    phase = parser.parse_args().phase
    foundation = read_json(SUBTYPE_RUN / "protocol.json")
    paths, rows, forbidden = override.checked_splits(foundation)
    _, _, forbidden = rescue.pool_info(forbidden)
    last_path = rescue.RUN / "decision.json"
    last = read_json(last_path)
    last_protocol = read_json(verified(rescue.RUN / "protocol.json", last["protocol_sha256"]))
    verified(Path(rescue.__file__), last_protocol["source_sha256"])
    hashes = {key: sha256(path) for key, path in paths.items()}
    if hashes != last_protocol["manifests"] or last["candidate"] != "expanded":
        raise ValueError("The locked early rescue reference differs")
    added_file = verified(rescue.RUN / "supplement_training.csv", last["supplement_sha256"])
    added = read_rows(added_file)
    if any(sha256(Path(row["path"])) != row["sha256"] for row in added):
        raise ValueError("A frozen supplemental leaf image changed")
    early_path = verified(EARLY_RUN / "decision.json", read_json(VIEW_RUN / "protocol.json")["incumbent_sha256"])
    early = read_json(verified(early_path, last_protocol["incumbent_sha256"]))
    health = read_json(verified(HEALTH_RUN / "decision.json", early["health_decision_sha256"]))
    subtype = read_json(verified(SUBTYPE_RUN / "decision.json", health["subtype_decision_sha256"]))
    prior_override = read_json(verified(override.RUN / "decision.json", last_protocol["previous_sha256"]))
    gate, review = health["preferred_by_development"], early["preferred_by_development"]
    components = (load_heads(subtype), load_gate(gate, health["selections"][gate]), restore(review, early["selections"][review]))
    plan = {"source_sha256": sha256(Path(__file__)), "early_rescue_sha256": sha256(last_path),
            "calibrated_vote_sha256": sha256(VOTE_PLAN), "manifests": hashes, "supplement_sha256": sha256(added_file),
            "arms": list(ARMS), "seeds": list(SEEDS), "epochs": list(EPOCHS), "thresholds": list(THRESHOLDS),
            "max_gaps": list(MAX_GAPS), "threshold_shifts": list(SHIFTS),
            "learning": "Train on labeled early and late images. Give equal total training weight to each truth-by-incumbent "
                        "direction stratum, balance publishers within strata, and compare one shared versus two direction-specific outputs.",
            "selection": "No per-group early or late recall decline, at least two groups with improved recall, and two individually "
                         "passing seeds. Enforce the guardrails at the central and both perturbed majority thresholds. "
                         "Early rescores from the locked health rescue are protected.",
            "roles": "No archival metrics enter training, checkpointing, or threshold selection. New field and PlantDoc are audits."}
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", foundation["parent_sha256"]), device)
    cards = {key: item for key, item in last["selection"].items() if key.startswith("expanded_")}
    rescued = rescue.restore_set(encoder, cards)
    if phase == "train":
        train(encoder, components, paths, rows, added_file, added, plan, rescued, cards)
    else:
        evaluate(encoder, components, foundation, forbidden, plan, rescued, cards, prior_override)


if __name__ == "__main__":
    main()
