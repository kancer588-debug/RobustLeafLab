"""Train one-way early-blight rescue heads under frozen healthy false-alarm limits."""

import argparse
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from run_dino_health_trial import RUN as HEALTH_RUN, load_gate
from run_early_review_trial import RUN as EARLY_RUN, apply_review, fitting_weights, restore
from run_late_protection_trial import load_heads
from run_local_leaf_trial import measure
from run_matched_capture_trial import verified_cohort
from run_patch_subtype_trial import AGRI_AUDIT, EMBED, TAIWAN, encode_patches, previous_scores
from run_roi_trial import HISTORICAL
from run_subtype_trial import (
    CLASSES, EXPANDED, FIRST_RUN, RUN as SUBTYPE_RUN, SEEDS, Specialists, development_metrics, measurement,
    read_rows, test_receipts, verified, write_rows,
)
from train_online_candidate import (
    WORKSPACE, checked_checkpoint, create_model, original_checkpoint, read_json, seed_everything, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "health_rescue_trial"
DATA = WORKSPACE / "datasets" / "health_rescue_trial"
VIEW_RUN = WORKSPACE / "outputs" / "view_ablation"
PLANTSEG = WORKSPACE / "datasets" / "view_ablation" / "plantseg" / "freeze.json"
ARMS = {"mean_original": (False, False), "local_original": (True, False),
        "mean_extra_healthy": (False, True), "local_extra_healthy": (True, True)}
EPOCHS = (1, 2, 4, 8, 12)
STRENGTHS = (0.25, 0.5, 1., 2., 4., 8.)
BATCH = 64
SHARED_DEVELOPMENT_GROUP = {"sheikh_2026|samsung Galaxy M35 5G|07/12/25": "field_extra|Galaxy M35 5G|2025-12-07"}


class RescueHead(nn.Module):
    """Combine global context and shared regional evidence with mean or soft-maximum pooling."""

    def __init__(self, local):
        super().__init__()
        self.local = local
        self.region = nn.Sequential(nn.Linear(EMBED, 48), nn.GELU(), nn.Linear(48, 1))
        self.context = nn.Linear(EMBED, 1)

    def forward(self, whole, patches):
        local = self.region(patches).squeeze(-1)
        if self.local:
            temperature = 0.35
            visual = temperature * (torch.logsumexp(local / temperature, dim=1) - math.log(local.shape[1]))
        else:
            visual = local.mean(1)
        return visual + self.context(whole).flatten()


def pack(parts):
    return {field: torch.cat([part[field] for part in parts]) for field in parts[0]}


def features(head, data, indices=slice(None)):
    return head(data["cls"][indices], data["patches"][indices])


def checked_data(protocol):
    paths, rows, forbidden, _ = verified_cohort(protocol)
    freeze = read_json(DATA / "freeze.json")
    for name in ("healthy_fit", "early_development"):
        item = freeze["manifests"][name]
        path = verified(DATA / item["file"], item["sha256"])
        if name == "healthy_fit":
            visual = read_json(DATA / "healthy_fit_visual_lock.json")
            if visual["source_manifest_sha256"] != item["sha256"]:
                raise ValueError("The visual screen is for a different training cohort")
            published = {row["sha256"] for row in read_rows(path)}
            path = verified(DATA / visual["manifest_file"], visual["manifest_sha256"])
            item = {"n": visual["n"]}
        records = read_rows(path)
        hashes = {row["sha256"] for row in records}
        category = "healthy" if name == "healthy_fit" else "early_blight"
        if not records or len(records) != item["n"] or len(hashes) != len(records) or hashes & forbidden:
            raise ValueError(f"Invalid source separation: {name}")
        if name == "healthy_fit" and not hashes <= published:
            raise ValueError("The visually reviewed negatives must be a subset of the publisher sample")
        if any(row["label"] != category or sha256(Path(row["path"])) != row["sha256"] for row in records):
            raise ValueError(f"Unexpected class or changed source picture: {name}")
        forbidden.update(hashes)
        paths[name], rows[name] = path, records
    return paths, rows, forbidden


def group_counts(labels, predictions, groups):
    result = {}
    for key, indices in groups.items():
        truth, guess = labels[indices], predictions[indices]
        result[key] = {"n": len(indices), "early_count": int((truth == 1).sum()),
                       "early_missed": int(((truth == 1) & (guess == 0)).sum()),
                       "healthy_false_alarm": int(((truth == 0) & (guess != 0)).sum()),
                       "late_to_early": int(((truth == 2) & (guess == 1)).sum()),
                       "late_correct": int(((truth == 2) & (guess == 2)).sum())}
    return result


def record(scores, truth, groups, start, old_n, malik_n, epoch, strength, loss):
    original = development_metrics(scores[:old_n], {"truth": truth[:old_n]}, malik_n)
    group_metrics = group_counts(truth, scores.argmax(1), groups)
    before = start["groups"]
    changes = {key: before[key]["early_missed"] - row["early_missed"] for key, row in group_metrics.items()}
    false_alarms = sum(row["healthy_false_alarm"] - before[key]["healthy_false_alarm"] for key, row in group_metrics.items())
    late_to_early = sum(row["late_to_early"] - before[key]["late_to_early"] for key, row in group_metrics.items())
    late_protected = all(row["late_correct"] == before[key]["late_correct"] for key, row in group_metrics.items())
    guardrails = (false_alarms == 0 and late_to_early == 0 and late_protected
                  and original["macro_f1"] >= start["legacy"]["macro_f1"]
                  and original["malik"]["macro_f1"] >= start["legacy"]["malik"]["macro_f1"])
    recoverable_groups = [count / before[key]["early_missed"] for key, count in changes.items()
                          if before[key]["early_missed"] > 0]
    return {"record": {"epoch": epoch, "strength": strength, "loss": loss,
                       "eligible": guardrails and sum(changes.values()) > 0, "guardrails": guardrails,
                       "early_recovery_group_mean": sum(recoverable_groups) / max(1, len(recoverable_groups)),
                       "early_recovered_legacy": start["legacy"]["early_to_healthy"] - original["early_to_healthy"],
                       "early_recovered_all": sum(changes.values()), "new_healthy_false_alarms": false_alarms,
                       "additional_late_to_early": late_to_early, "legacy_macro_f1": original["macro_f1"],
                       "malik_macro_f1": original["malik"]["macro_f1"]},
            "legacy": original, "groups": group_metrics}


def rank(card):
    row = card["record"]
    return (row["eligible"], row["guardrails"], row["early_recovery_group_mean"], row["early_recovered_all"],
            row["legacy_macro_f1"], -row["new_healthy_false_alarms"], -row["strength"], -row["epoch"])


def check_strengths(evidence, dev, context, epoch, loss):
    return [record(apply_review(dev["incumbent"], evidence, strength, "rescue_only"), dev["truth"],
                   *context, epoch, strength, loss) for strength in STRENGTHS]


def fit_head(name, seed, fit, fit_rows, dev, context, rows_per_epoch):
    seed_everything(seed)
    local, _ = ARMS[name]
    head = RescueHead(local)
    weights = fitting_weights(fit_rows, fit["incumbent"], fit["truth"])
    target = (fit["truth"] == 1).float()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.002, weight_decay=0.05)
    generator = torch.Generator().manual_seed(seed)
    selected, history = None, []
    for epoch in range(1, EPOCHS[-1] + 1):
        loss_total = 0.
        shuffled = torch.randperm(len(target), generator=generator)[:rows_per_epoch]
        for indices in shuffled.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(features(head, fit, indices), target[indices], weight=weights[indices])
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(indices)
        if epoch not in EPOCHS:
            continue
        with torch.no_grad():
            checked = check_strengths(features(head, dev), dev, context, epoch, loss_total / rows_per_epoch)
        history.extend(item["record"] for item in checked)
        best = max(checked, key=rank)
        if selected is None or rank(best) > rank(selected):
            selected = best
            state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    key = f"{name}_s{seed}"
    folder = RUN / key
    folder.mkdir(exist_ok=True)
    write_rows(folder / "history.csv", history)
    weight = folder / "rescue.pt"
    temporary = weight.with_suffix(".tmp")
    torch.save({"state": state, "local": local}, temporary)
    temporary.replace(weight)
    selected["weight_sha256"] = sha256(weight)
    print(key, selected["record"], flush=True)
    return selected


def make_plan(paths, incumbent_digest):
    return {"source_sha256": sha256(Path(__file__)), "incumbent_sha256": incumbent_digest,
            "data_freeze_sha256": sha256(DATA / "freeze.json"),
            "visual_fit_sha256": sha256(DATA / "healthy_fit_visual_lock.json"),
            "manifests": {key: sha256(path) for key, path in paths.items()},
            "shared_development_group": SHARED_DEVELOPMENT_GROUP,
            "arms": list(ARMS), "epochs": list(EPOCHS), "seeds": list(SEEDS), "strengths": list(STRENGTHS),
            "selection": "Require an early miss recovered, no new healthy false positives or late-to-early errors, "
                         "late correct counts preserved in every capture group, and legacy pooled and Malik Macro-F1 preserved. "
                         "Rank on mean fraction of early misses recovered per development group, then total early misses recovered. "
                         "Require seed 42 and at least one confirmatory seed. Compare a static healthy margin shift, "
                         "whose deterministic threshold is selected with the same constraints.",
            "roles": "Oni healthy only augments training. Sheikh early only augments development. Prior public tests "
                     "are retrospective audits. New field and PlantDoc are post-lock external audits only."}


def encoded(encoder, components, path, name, rows, original=None):
    data = encode_patches(encoder, path, name, original)
    if [CLASSES[label] for label in data["truth"].tolist()] != [row["label"] for row in rows]:
        raise ValueError(f"Manifest labels changed: {name}")
    data["incumbent"] = previous_scores(data, *components)
    return data


def train(encoder, components, paths, rows, plan, early):
    RUN.mkdir(exist_ok=True)
    lock = RUN / "decision.json"
    plan_path = RUN / "protocol.json"
    if lock.exists():
        raise FileExistsError("The development decision is already frozen")
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError("The existing protocol differs")
    else:
        write_json(plan_path, plan)
    order = ("augmented", "healthy_fit", "development_field", "development_malik", "early_development")
    fit, extra, field, malik, sheikh = [encoded(encoder, components, paths[key], key, rows[key]) for key in order]
    fit_all = pack((fit, extra))
    dev = pack((field, malik, sheikh))
    old_n, malik_n = len(field["truth"]) + len(malik["truth"]), len(malik["truth"])
    group_rows = rows["development_field"] + rows["development_malik"] + rows["early_development"]
    groups = defaultdict(list)
    for i, row in enumerate(group_rows):
        capture_group = row["capture_group"]
        groups[SHARED_DEVELOPMENT_GROUP.get(capture_group, capture_group)].append(i)
    start = {"legacy": development_metrics(dev["incumbent"][:old_n], {"truth": dev["truth"][:old_n]}, malik_n),
             "groups": group_counts(dev["truth"], dev["incumbent"].argmax(1), groups)}
    previous = early["selections"][early["preferred_by_development"]]["metrics"]
    if any(start["legacy"][key] != previous[key] for key in ("confusion", "malik")):
        raise ValueError("The incumbent development predictions changed")
    context = groups, start, old_n, malik_n
    static_history = check_strengths(torch.full_like(dev["truth"], 40.).float(), dev, context, 0, None)
    write_rows(RUN / "static_shift_history.csv", [item["record"] for item in static_history])
    selections = {"static_shift": max(static_history, key=rank)}
    for name, seed in product(ARMS, SEEDS):
        _, expanded = ARMS[name]
        fit_data = fit_all if expanded else fit
        fit_rows = rows["augmented"] + rows["healthy_fit"] if expanded else rows["augmented"]
        selections[f"{name}_s{seed}"] = fit_head(name, seed, fit_data, fit_rows, dev, context, len(fit["truth"]))
    eligible = [f"{name}_s42" for name in ARMS if selections[f"{name}_s42"]["record"]["eligible"]
                and any(selections[f"{name}_s{seed}"]["record"]["eligible"] for seed in SEEDS[1:])]
    if selections["static_shift"]["record"]["eligible"]:
        eligible.append("static_shift")
    chosen = max(eligible, key=lambda key: rank(selections[key]), default="incumbent")
    write_json(lock, {"candidate": chosen, "qualified": eligible, "protocol_sha256": sha256(plan_path),
                      "development_incumbent": start, "selections": selections})
    print("Locked choice:", chosen, flush=True)


@torch.no_grad()
def evaluate(encoder, components, plan, protocol, forbidden):
    lock = RUN / "decision.json"
    decision = read_json(lock)
    if read_json(verified(RUN / "protocol.json", decision["protocol_sha256"])) != plan:
        raise ValueError("The frozen training context changed")
    audits = read_json(EXPANDED / "evaluation_freeze.json")["external_audits"]
    tests = {**test_receipts(protocol), **read_json(TAIWAN)["tests"], **read_json(AGRI_AUDIT)["tests"],
             "plantseg_subtype_stress": read_json(PLANTSEG), **audits}
    original = create_model("resnet18")
    original.load_state_dict(original_checkpoint("AC42")["model_state"])
    original.to(encoder.device, memory_format=torch.channels_last).eval().requires_grad_(False)
    heads = {}
    for name, item in decision["selections"].items():
        if name == "static_shift":
            continue
        weight = verified(RUN / name / "rescue.pt", item["weight_sha256"])
        model = torch.load(weight, map_location="cpu", weights_only=True)
        head = RescueHead(model["local"])
        head.load_state_dict(model["state"])
        heads[name] = head.eval()
    decision_digest = sha256(lock)
    for name, spec in tests.items():
        result_path = RUN / f"{name}_results.json"
        if result_path.exists():
            old = read_json(result_path)
            if old["decision_sha256"] != decision_digest or old["manifest_sha256"] != spec["sha256"]:
                raise ValueError(f"An existing evaluation was produced using a different lock: {name}")
            continue
        path = verified(Path(spec["path"]), spec["sha256"])
        records = read_rows(path)
        hashes = {row["sha256"] for row in records}
        if len(hashes) != len(records) or hashes & forbidden:
            raise ValueError(f"The evaluation overlaps training or development: {name}")
        data = encoded(encoder, components, path, name, records, original)
        baseline = data["incumbent"]
        scores = {"original_AC42": data["original"], "incumbent": baseline}
        for key, item in decision["selections"].items():
            evidence = features(heads[key], data) if key in heads else torch.full_like(baseline[:, 0], 40.)
            scores[key] = apply_review(baseline, evidence, item["record"]["strength"], "rescue_only")
        truth = data["truth"]
        metrics, predictions = {}, []
        for key, logits in scores.items():
            metrics[key], predicted = (measurement(logits, truth)[:2] if truth.unique().numel() == 3
                                       else measure(logits, truth))
            if key in decision["selections"]:
                prior = baseline.argmax(1)
                if ((predicted != prior) & ((prior != 0) | (predicted != 1))).any():
                    raise ValueError(f"Unexpected non-early-rescue switch: {name} {key}")
            predictions.extend({"model": key, "sha256": row["sha256"], "true_label": row["label"],
                                "prediction": CLASSES[guess]} for row, guess in zip(records, predicted.tolist(), strict=True))
        write_rows(RUN / f"{name}_predictions.csv", predictions)
        write_json(result_path, {"decision_sha256": decision_digest, "manifest_sha256": spec["sha256"],
                                 "role": "external_audit" if name in audits else spec.get("role", "historical_replay"),
                                 "models": metrics})
        print(name, {key: round(metrics[key]["accuracy"], 5) for key in ("incumbent", "static_shift", *heads)}, flush=True)
    promotion = RUN / "promotion.json"
    if not promotion.exists():
        totals = Counter()
        for name in HISTORICAL:
            for key, values in read_json(RUN / f"{name}_results.json")["models"].items():
                totals[key] += sum(values["confusion"][i][i] for i in range(len(CLASSES)))
        write_json(promotion, {"candidate": decision["candidate"], "qualifies": False,
                               "independent_three_class_confirmed": False, "historical_correct_totals": dict(totals)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    phase = parser.parse_args().phase
    protocol = read_json(SUBTYPE_RUN / "protocol.json")
    paths, rows, forbidden = checked_data(protocol)
    view = read_json(VIEW_RUN / "protocol.json")
    early_path = verified(EARLY_RUN / "decision.json", view["incumbent_sha256"])
    early = read_json(early_path)
    health = read_json(verified(HEALTH_RUN / "decision.json", early["health_decision_sha256"]))
    subtype = read_json(verified(SUBTYPE_RUN / "decision.json", health["subtype_decision_sha256"]))
    gate, review = health["preferred_by_development"], early["preferred_by_development"]
    components = (load_heads(subtype), load_gate(gate, health["selections"][gate]), restore(review, early["selections"][review]))
    plan = make_plan(paths, sha256(early_path))
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", protocol["parent_sha256"]), device)
    if phase == "train":
        train(encoder, components, paths, rows, plan, early)
    else:
        evaluate(encoder, components, plan, protocol, forbidden)


if __name__ == "__main__":
    main()
