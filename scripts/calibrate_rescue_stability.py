"""Choose one shared early-evidence threshold offset from development only."""

from collections import defaultdict
from pathlib import Path

import torch

import run_rescue_stability_trial as stability
from run_health_rescue_trial import SHARED_DEVELOPMENT_GROUP, encoded, group_counts, pack, record
from run_subtype_trial import development_metrics
from train_online_candidate import read_json, sha256, write_json


BASE = stability.RUN
RUN = BASE / "calibrated"
OFFSETS = (-0.75, -0.5, -0.25, 0.0, 0.25)
SHIFT_TESTS = ("both_majorities_looser", "both_majorities", "both_majorities_tighter")


def plan(paths, previous_lock, previous):
    base = read_json(BASE / "decision.json")
    protocol = stability.verified(BASE / "protocol.json", base["protocol_sha256"])
    stability.verified(Path(stability.__file__), read_json(protocol)["source_sha256"])
    return {"source_sha256": sha256(Path(__file__)), "base_decision_sha256": sha256(BASE / "decision.json"),
            "base_protocol_sha256": sha256(protocol), "prior_decision_sha256": sha256(previous_lock),
            "prior_protocol_sha256": previous["protocol_sha256"],
            "manifests": {name: sha256(path) for name, path in paths.items()}, "offsets": list(OFFSETS),
            "rule": "Fit no weights. Apply one common offset to all six individually locked early-evidence thresholds. "
                    "Choose the smallest absolute offset whose central, looser, and tighter variants all satisfy "
                    "the original two-group development and error guardrails. Ties prefer the higher threshold. "
                    "Archived replay and audit results are not calibration inputs."}


def offset_cards(cards, offset):
    return {seed: {**card, "threshold": card["threshold"] + offset} for seed, card in cards.items()}


@torch.no_grad()
def calibrate(encoder, components, paths, rows, frozen, previous):
    RUN.mkdir(exist_ok=True)
    lock, protocol = RUN / "decision.json", RUN / "protocol.json"
    if lock.exists():
        raise FileExistsError("The shared offset is already frozen")
    if protocol.exists():
        if read_json(protocol) != frozen:
            raise ValueError("The calibration protocol differs")
    else:
        write_json(protocol, frozen)
    order = ("development_field", "development_malik", "early_development", "extension_dev")
    dev = pack([encoded(encoder, components, paths[name], name, rows[name]) for name in order])
    groups = defaultdict(list)
    for i, row in enumerate(row for name in order for row in rows[name]):
        group = row["capture_group"]
        groups[SHARED_DEVELOPMENT_GROUP.get(group, group)].append(i)
    malik_n = len(rows["development_malik"])
    old_n = len(rows["development_field"]) + malik_n
    start = {"legacy": development_metrics(dev["incumbent"][:old_n], {"truth": dev["truth"][:old_n]}, malik_n),
             "groups": group_counts(dev["truth"], dev["incumbent"].argmax(1), groups)}
    if start != previous["development_incumbent"]:
        raise ValueError("The calibration development baseline differs")
    source = read_json(BASE / "decision.json")
    heads, cards = stability.restore_models(previous, source["fresh_selections"])
    evaluations = {}
    for delta in {offset + perturbation for offset in OFFSETS
                  for perturbation in (-stability.THRESHOLD_DELTA, 0., stability.THRESHOLD_DELTA)}:
        scores = stability.score_models(dev, heads, offset_cards(cards, delta))[stability.PRIMARY]
        item = record(scores, dev["truth"], groups, start, old_n, malik_n, 0, 0., None)
        gains = [key for key, values in item["groups"].items() if values["early_missed"] < start["groups"][key]["early_missed"]]
        evaluations[delta] = {**item["record"], "gain_groups": gains,
                              "passes": item["record"]["guardrails"] and len(gains) >= stability.override.MIN_RESCUED_GROUPS}
    comparisons = {offset: {name: evaluations[offset + delta] for name, delta in zip(
        SHIFT_TESTS, (-stability.THRESHOLD_DELTA, 0., stability.THRESHOLD_DELTA), strict=True)} for offset in OFFSETS}
    eligible = [offset for offset, variants in comparisons.items() if all(item["passes"] for item in variants.values())]
    selected = max(eligible, key=lambda offset: (-abs(offset), offset), default=None)
    write_json(lock, {"candidate": stability.PRIMARY if selected is not None else "incumbent",
                     "development_passed": selected is not None, "offset": selected, "protocol_sha256": sha256(protocol),
                     "fresh_selections": source["fresh_selections"], "offset_development": comparisons})
    print("Calibrated offset:", selected, flush=True)


def evaluate(encoder, components, frozen, protocol, forbidden, previous):
    offset = read_json(RUN / "decision.json")["offset"]
    if offset is None:
        print("No new calibration passed; the uncalibrated replays are retained.", flush=True)
        return
    scorer = stability.score_models

    def adjusted(data, heads, cards):
        return scorer(data, heads, offset_cards(cards, offset))

    stability.score_models = adjusted
    try:
        original_evaluate(encoder, components, frozen, protocol, forbidden, previous)
    finally:
        stability.score_models = scorer


if __name__ == "__main__":
    original_evaluate = stability.evaluate
    stability.RUN = RUN
    stability.make_plan = plan
    stability.train = calibrate
    stability.evaluate = evaluate
    stability.main()
