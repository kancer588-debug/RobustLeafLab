"""Gate subtype changes on agreement between two full-frame views."""

from itertools import product
from pathlib import Path
from shutil import copyfile

import torch

from run_full_frame_trial import encode_views
from run_matched_capture_trial import verified_cohort
from run_patch_subtype_trial import TAIWAN, card, ranking
from run_source_residual_trial import disease_margin
from run_subtype_trial import CLASSES, FIRST_RUN, RUN as SUBTYPE_RUN, Specialists, combine, read_rows, test_receipts, verified, write_rows
from run_dino_health_trial import RUN as HEALTH_RUN, load_gate
from run_early_review_trial import RUN as EARLY_RUN, restore
from run_late_protection_trial import load_heads
from run_local_leaf_trial import measure
from train_online_candidate import WORKSPACE, checked_checkpoint, read_json, sha256, write_json


RUN = WORKSPACE / "outputs" / "consensus_frame_trial"
STRENGTHS = (0.35, 0.7, 1.0, 1.5)
GAPS = (0.0, 0.5, 1.0)
RULES = ("late_only", "symmetric")


def consensus(data, strength, gap, rule):
    center = disease_margin(data["center"])
    full, mirror = data["full"], data["mirror"]
    late_support = -torch.maximum(full, mirror)
    early_support = torch.minimum(full, mirror)
    to_late = (center >= 0) & (strength * late_support > center + gap)
    to_early = (center < 0) & (strength * early_support > -center + gap) & (rule == "symmetric")
    margins = torch.where(to_late, center - strength * late_support, center)
    margins = torch.where(to_early, center + strength * early_support, margins)
    return combine(data["center"], margins, 1.0)


def select(dev, malik_n, prior):
    items = []
    for rule, strength, gap in product(RULES, STRENGTHS, GAPS):
        item = card(consensus(dev, strength, gap, rule), dev, malik_n, prior, 0, strength, None)
        conservative = card(consensus(dev, strength * 0.8, gap, rule), dev, malik_n, prior, 0, strength * 0.8, None)
        item["record"].update(rule=rule, strength=strength, gap=gap, conservative_pass=conservative["record"]["eligible"])
        items.append(item)
    passed = [item for item in items if item["record"]["eligible"] and item["record"]["conservative_pass"]]
    chosen = max(passed or items, key=ranking)
    write_rows(RUN / "development_search.csv", [item["record"] for item in items])
    return chosen, bool(passed)


def evaluate(encoder, protocol, components, forbidden, decision, lock_path):
    tests = {**test_receipts(protocol), **read_json(TAIWAN)["tests"]}
    choice = decision["selected"]["record"]
    lock_sha256 = sha256(lock_path)
    for name, receipt in tests.items():
        result_path = RUN / f"{name}_results.json"
        predictions_path = RUN / f"{name}_predictions.csv"
        if result_path.exists():
            existing = read_json(result_path)
            if (existing["decision_sha256"] != lock_sha256 or existing["manifest_sha256"] != receipt["sha256"]
                    or not predictions_path.is_file()):
                raise ValueError(f"Incomplete evaluation or different freeze: {name}")
            continue
        path = verified(Path(receipt["path"]), receipt["sha256"])
        rows = read_rows(path)
        ids = {row["sha256"] for row in rows}
        if len(ids) != len(rows) or ids & forbidden:
            raise ValueError(f"Overlapping evaluation photos: {name}")
        data = encode_views(encoder, components, path, name)
        logits = {"incumbent": data["center"], "consensus": consensus(data, choice["strength"], choice["gap"], choice["rule"])}
        metrics, predictions = {}, []
        for variant, scores in logits.items():
            metrics[variant], labels = measure(scores, data["truth"])
            predictions.extend({"model": variant, "sha256": row["sha256"], "true_label": row["label"],
                                "prediction": CLASSES[label]} for row, label in zip(rows, labels.tolist()))
        write_rows(predictions_path, predictions)
        write_json(result_path, {"decision_sha256": lock_sha256, "manifest_sha256": receipt["sha256"], "models": metrics})
        print(name, {key: round(item["accuracy"], 5) for key, item in metrics.items()}, flush=True)


def main():
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
    splits, _, forbidden, _ = verified_cohort(protocol)
    prior = early["selections"][early_key]["metrics"]
    RUN.mkdir(exist_ok=True)
    plan_path, lock_path = RUN / "protocol.json", RUN / "decision.json"
    selection_source = RUN / "selection_source.py"
    if not plan_path.exists() and not selection_source.exists():
        copyfile(Path(__file__), selection_source)
    plan = {"incumbent_sha256": sha256(early_lock), "script_sha256": sha256(selection_source),
            "development_sha256": {name: sha256(path) for name, path in splits.items() if name != "augmented"},
            "rules": list(RULES), "strengths": list(STRENGTHS), "gaps": list(GAPS),
            "selection": "Both full-frame views must favor changing the subtype, and the weaker view must beat the "
                         "center margin by the gap. Apply the same development gates as the original full-frame trial. "
                         "Require the configuration at 80% of the same strength to meet the same gates."}
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError("Consensus protocol has changed")
    else:
        write_json(plan_path, plan)
    encoder = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", protocol["parent_sha256"]), device)
    if lock_path.exists():
        decision = read_json(lock_path)
        if decision["protocol_sha256"] != sha256(plan_path):
            raise ValueError("Decision belongs to another protocol")
    else:
        field = encode_views(encoder, components, splits["development_field"], "development_field")
        malik = encode_views(encoder, components, splits["development_malik"], "development_malik")
        development = {name: torch.cat((field[name], malik[name])) for name in field}
        if measure(development["center"], development["truth"])[0]["confusion"] != prior["confusion"]:
            raise ValueError("Incumbent does not match the frozen development results")
        choice, approved = select(development, len(malik["truth"]), prior)
        decision = {"preferred_by_development": "consensus" if approved else "incumbent",
                    "selected": choice, "incumbent": prior, "protocol_sha256": sha256(plan_path)}
        write_json(lock_path, decision)
    print("Consensus decision:", decision["preferred_by_development"], decision["selected"]["record"], flush=True)
    if decision["preferred_by_development"] == "consensus":
        evaluate(encoder, protocol, components, forbidden, decision, lock_path)


if __name__ == "__main__":
    main()
