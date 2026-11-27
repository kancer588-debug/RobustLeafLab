"""Calibrate late-blight overrides and average three locked DINOv2 subtype heads."""

import argparse
import math
from itertools import product
from pathlib import Path

import torch

from run_subtype_trial import (
    CLASSES, EXPANDED, FIRST_RUN, RUN as SUBTYPE_RUN, SEEDS, Specialists, SubtypeProbe,
    combine, development_metrics, encode, measurement, read_rows, test_receipts, verified, write_rows,
)
from train_online_candidate import (
    WORKSPACE, checked_checkpoint, create_model, original_checkpoint, read_json, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "late_protection_trial"
STRATEGIES = ("single_head", "mean_heads")
CUTOFFS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
PRIOR_WEIGHTS = (0.0, 0.25)


def load_heads(selection):
    arm = selection["preferred_by_development"].rsplit("_s", 1)[0]
    heads = {}
    for seed in SEEDS:
        key = f"{arm}_s{seed}"
        expected = selection["selections"][key]["weight_sha256"]
        path = verified(SUBTYPE_RUN / key / "probe.pt", expected)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        state = saved["state"]
        head = SubtypeProbe(state["mean"], state["scale"])
        head.load_state_dict(state)
        heads[seed] = (head.eval(), saved["feature"], saved["fraction"])
    return heads


@torch.no_grad()
def margins(data, heads):
    parent = data["prior"][:, 1] - data["prior"][:, 2]
    return {seed: torch.lerp(parent, head(data[feature]), weight)
            for seed, (head, feature, weight) in heads.items()}


def make_predictions(prior, margin, cutoff, prior_weight):
    parent_margin = prior[:, 1] - prior[:, 2]
    required = cutoff + prior_weight * (-parent_margin).clamp_min(0)
    reject_early = (prior.argmax(1) == 2) & (margin > 0) & (margin < required)
    return combine(prior, torch.where(reject_early, parent_margin, margin), 1.0)


def correct(metrics):
    return metrics["confusion"][1][1], metrics["confusion"][2][2]


def constraints(value, old, starting, *, primary):
    early, late = correct(value)
    old_early, old_late = correct(old)
    start_early, _ = correct(starting)
    early_floor = start_early + math.ceil(max(old_early - start_early, 0) / 2)
    malik = value["malik"]
    old_malik = old["malik"]
    return {
        "early_gain_retained": early >= early_floor,
        "late_recall_preserved": late >= old_late,
        "pooled_macro_f1_preserved": value["macro_f1"] >= old["macro_f1"],
        "malik_macro_f1_preserved": malik["macro_f1"] >= old_malik["macro_f1"],
        "malik_late_recall_preserved": correct(malik)[1] >= correct(old_malik)[1],
        "strict_gain": not primary or late > old_late or value["macro_f1"] > old["macro_f1"],
    }


def run_configuration(strategy, cutoff, prior_weight, raw, development, malik_n, baseline, starting):
    prior = development["prior"]
    single = strategy == "single_head"
    primary_margin = raw[42] if single else torch.stack(list(raw.values())).mean(0)
    current = development_metrics(make_predictions(prior, primary_margin, cutoff, prior_weight), development, malik_n)
    checks = constraints(current, baseline[42], starting, primary=True)
    confirmations = {}
    for seed in SEEDS[1:]:
        confirm_margin = raw[seed] if single else torch.stack([score for key, score in raw.items() if key != seed]).mean(0)
        metrics = development_metrics(make_predictions(prior, confirm_margin, cutoff, prior_weight), development, malik_n)
        confirmations[seed] = {"metrics": metrics,
                               "checks": constraints(metrics, baseline[seed], starting, primary=False)}
    early, late = correct(current)
    passed = sum(all(item["checks"].values()) for item in confirmations.values())
    record = {"strategy": strategy, "cutoff": cutoff, "prior_weight": prior_weight,
              "eligible": all(checks.values()) and passed >= 1, "passed_confirmations": passed,
              "failed_primary_checks": sum(not check for check in checks.values()),
              "early_correct": early, "late_correct": late, "macro_f1": current["macro_f1"],
              "malik_macro_f1": current["malik"]["macro_f1"]}
    return {"record": record, "metrics": current, "checks": checks, "confirmations": confirmations}


def rank(choice):
    row = choice["record"]
    return (row["eligible"], -row["failed_primary_checks"], row["passed_confirmations"],
            (row["macro_f1"] + row["malik_macro_f1"]) / 2, row["late_correct"],
            -row["prior_weight"], -row["cutoff"])


def select(model, protocol, heads, previous_lock):
    splits = {name: protocol["manifests"][name] for name in ("development_field", "development_malik")}
    plan = {"prior_decision_sha256": previous_lock, "development": splits, "strategies": list(STRATEGIES),
            "cutoffs": list(CUTOFFS), "prior_weights": list(PRIOR_WEIGHTS),
            "policy": "Only veto a late-to-early override on disease-routed images. Gate thresholds are "
                      "global and chosen on development only. Require preservation of incumbent late recall, "
                      "pooled and Malik Macro-F1, Malik late recall, and half the early gain over the starting "
                      "candidate. Require a primary improvement and one confirmation. For the mean arm, "
                      "confirm on the two held-head-out pairs that omit seed 43 or 44.",
            "protection": "New field and PlantDoc are opened only for post-lock external inference."}
    RUN.mkdir(exist_ok=True)
    if (RUN / "decision.json").exists():
        raise FileExistsError("A development decision has already been locked")
    plan_path = RUN / "protocol.json"
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError("The existing experiment has a different protocol")
    else:
        write_json(plan_path, plan)
    parts = [encode(model, verified(Path(item["path"]), item["sha256"]), name) for name, item in splits.items()]
    development = {kind: torch.cat([part[kind] for part in parts]) for kind in parts[0]}
    malik_n = len(parts[1]["truth"])
    prior = development["prior"]
    raw = margins(development, heads)
    baseline = {seed: development_metrics(combine(prior, value, 1.0), development, malik_n)
                for seed, value in raw.items()}
    starting = development_metrics(prior, development, malik_n)
    candidates = [run_configuration(strategy, cutoff, weight, raw, development, malik_n, baseline, starting)
                  for strategy, cutoff, weight in product(STRATEGIES, CUTOFFS, PRIOR_WEIGHTS)]
    write_rows(RUN / "development_search.csv", [row["record"] for row in candidates])
    best = {strategy: max((row for row in candidates if row["record"]["strategy"] == strategy
                          and (strategy == "mean_heads" or row["record"]["cutoff"] or row["record"]["prior_weight"])),
                          key=rank) for strategy in STRATEGIES}
    eligible = {key: choice for key, choice in best.items() if choice["record"]["eligible"]}
    preferred = max(eligible, key=lambda name: rank(eligible[name]), default="dino_incumbent")
    write_json(RUN / "decision.json", {"prior_decision_sha256": previous_lock, "preferred_by_development": preferred,
                                     "locked_rules": best, "starting": starting, "baseline": baseline})
    for choice in best.values():
        print(choice["record"], flush=True)
    print(f"Decision locked: {preferred}", flush=True)


def evaluate(model, protocol, heads, previous_lock):
    path = RUN / "decision.json"
    decision = read_json(path)
    decision_hash = sha256(path)
    if decision["prior_decision_sha256"] != previous_lock:
        raise ValueError("The incumbent changed after selection")
    receipts = test_receipts(protocol)
    protected = read_json(EXPANDED / "evaluation_freeze.json")["external_audits"]
    forbidden = {row["sha256"] for split in protocol["manifests"].values()
                 for row in read_rows(verified(Path(split["path"]), split["sha256"]))}
    original = create_model("resnet18")
    original.load_state_dict(original_checkpoint("AC42")["model_state"])
    original.requires_grad_(False).to(model.device, memory_format=torch.channels_last).eval()
    for name, receipt in {**receipts, **protected}.items():
        test_path = verified(Path(receipt["path"]), receipt["sha256"])
        rows = read_rows(test_path)
        keys = {row["sha256"] for row in rows}
        if len(keys) != len(rows) or keys & forbidden:
            raise ValueError(f"Evaluation images overlap training or development: {name}")
        result = RUN / f"{name}_results.json"
        if result.exists():
            recorded = read_json(result)
            if recorded["decision_sha256"] != decision_hash or recorded["manifest_sha256"] != receipt["sha256"]:
                raise ValueError(f"Test already uses a different decision or manifest: {name}")
            continue
        data = encode(model, test_path, name, original)
        raw = margins(data, heads)
        prior = data["prior"]
        models = {"original_AC42": data["original"], "starting_candidate": prior,
                  "dino_incumbent": combine(prior, raw[42], 1.0)}
        for strategy, item in decision["locked_rules"].items():
            record = item["record"]
            margin = raw[42] if strategy == "single_head" else torch.stack(list(raw.values())).mean(0)
            models[strategy] = make_predictions(prior, margin, record["cutoff"], record["prior_weight"])
        metrics, predictions = {}, []
        for label, logits in models.items():
            metrics[label], predicted, subtype = measurement(logits, data["truth"])
            for row, guess, conditional in zip(rows, predicted.tolist(), subtype.tolist()):
                predictions.append({"model": label, "sha256": row["sha256"], "true_label": row["label"],
                                    "prediction": CLASSES[guess], "subtype_if_diseased": CLASSES[conditional]})
            confusion = metrics[label]["confusion"]
            print(f"{name} {label} accuracy={metrics[label]['accuracy']:.5f} early={confusion[1][1]} "
                  f"late={confusion[2][2]}", flush=True)
        write_rows(RUN / f"{name}_predictions.csv", predictions)
        write_json(result, {"decision_sha256": decision_hash, "manifest_sha256": receipt["sha256"],
                            "role": "external_audit" if name in protected else "historical_replay", "models": metrics})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "evaluate"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    starting_hash = read_json(FIRST_RUN / "selection.json")["candidate_sha256"]
    previous = SUBTYPE_RUN / "decision.json"
    previous_lock = sha256(previous)
    protocol = read_json(SUBTYPE_RUN / "protocol.json")
    if protocol["parent_sha256"] != starting_hash:
        raise ValueError("The starting checkpoint differs from the previous subtype experiment")
    model = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", starting_hash), device)
    heads = load_heads(read_json(previous))
    phase = select if args.phase == "select" else evaluate
    phase(model, protocol, heads, previous_lock)


if __name__ == "__main__":
    main()
