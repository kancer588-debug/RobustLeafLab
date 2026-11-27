"""Measure late-blight supplementation with matched one-way correction heads."""

import argparse
from collections import Counter
from itertools import product
from pathlib import Path

import torch
from torch.nn import functional as F

import run_directional_subtype_trial as prior
from stage_late_supplement_trial import ROOT as DATA


RUN = prior.WORKSPACE / "outputs" / "late_supplement_trial"
ARMS = ("continued", "supplemented")
EPOCHS = (1, 2, 4)
THRESHOLDS = (0.5, 1., 2., 3.)
GAPS = (0.5, 1., 2.)
SHIFTS = prior.SHIFTS


def votes(data, base, model, threshold, gap):
    incumbent = data["incumbent"]
    uncertainty = (incumbent[:, 1] - incumbent[:, 2]).abs()
    evidence = model(data["dino"], prior.lead(data))
    return ((base.argmax(1) == 1) & (incumbent.argmax(1) != 0) & (uncertainty <= gap)
            & (evidence <= -threshold))


def evaluate_groups(dev, base, records):
    truth = dev["truth"]
    groups = {}
    indices = []
    for row in records:
        raw = row["capture_group"]
        group = prior.SHARED_DEVELOPMENT_GROUP.get(raw, raw)
        indices.append(groups.setdefault(group, len(groups)))
    index = torch.tensor(indices) * 3 + truth
    total = torch.bincount(index, minlength=3 * len(groups))
    correct = (base.argmax(1) == truth).long()
    baseline = torch.zeros_like(total).scatter_add_(0, index, correct)

    def compare(change):
        scores = torch.zeros_like(baseline).scatter_add_(0, index, (prior.replace_subtype(base, change).argmax(1) == truth).long())
        gains = (scores - baseline).reshape(-1, 3)[:, 1:]
        counts = total.reshape(-1, 3)[:, 1:]
        late_groups = [group for group, improved in zip(groups, gains[:, 1] > 0, strict=True) if improved]
        return {"passes": bool((gains >= 0).all()) and len(late_groups) >= 2,
                "guardrails": bool((gains >= 0).all()), "groups": late_groups, "net_correct": int(gains.sum()),
                "early_delta": int(gains[:, 0].sum()), "late_delta": int(gains[:, 1].sum()),
                "mean_group_class_recall_delta": float((gains / counts.clamp_min(1))[counts > 0].mean())}

    return compare


def fit(key, model, seed, data, rows, dev, base, compare):
    prior.seed_everything(seed)
    model = model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    generator = torch.Generator().manual_seed(seed)
    truth = data["truth"]
    direction = prior.lead(data)
    weights = prior.weights_for(rows, truth, direction)
    history, best = [], None
    for epoch in range(1, EPOCHS[-1] + 1):
        for i in torch.randperm(len(rows), generator=generator).split(prior.BATCH):
            optimizer.zero_grad(set_to_none=True)
            logits = model(data["dino"][i], direction[i])
            loss = F.binary_cross_entropy_with_logits(logits, (truth[i] == 1).float(), weight=weights[i])
            loss.backward()
            optimizer.step()
        if epoch not in EPOCHS:
            continue
        with torch.no_grad():
            logits = model(dev["dino"], prior.lead(dev))
            incumbent = dev["incumbent"]
            uncertainty = (incumbent[:, 1] - incumbent[:, 2]).abs()
            allowed = (base.argmax(1) == 1) & (incumbent.argmax(1) != 0)
            for threshold, gap in product(THRESHOLDS, GAPS):
                checks = [compare(allowed & (uncertainty <= gap) & (logits <= -(threshold + shift))) for shift in SHIFTS]
                card = {"epoch": epoch, "threshold": threshold, "gap": gap, **prior.summarize(checks)}
                history.append(card)
                if best is None or rank(card) > rank(best):
                    best = card
                    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    folder = RUN / key
    folder.mkdir(exist_ok=True)
    prior.write_rows(folder / "history.csv", history)
    weights_file = folder / "probe.pt"
    temporary = weights_file.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(weights_file)
    return {**best, "sha256": prior.sha256(weights_file)}


def rank(card):
    return (card["pass"], card["guardrails"], card["min_groups"], card["min_correct_gain"], card["min_balanced_gain"],
            -card["gap"], card["threshold"], -card["epoch"])


def restore(key, card):
    model = prior.Probe(True)
    file = prior.verified(RUN / key / "probe.pt", card["sha256"])
    model.load_state_dict(torch.load(file, map_location="cpu", weights_only=True))
    return model.eval()


@torch.no_grad()
def arm_votes(arm, cards, data, base, shift=0.):
    selected = [(key, card) for key, card in cards.items() if key.startswith(arm + "_")]
    return torch.stack([votes(data, base, restore(key, card), card["threshold"] + shift, card["gap"])
                        for key, card in selected]).sum(0) >= 2


def prior_models(foundation, old_plan):
    last = prior.read_json(prior.verified(prior.rescue.RUN / "decision.json", old_plan["early_rescue_sha256"]))
    rescue_plan = prior.read_json(prior.verified(prior.rescue.RUN / "protocol.json", last["protocol_sha256"]))
    early_path = prior.verified(prior.EARLY_RUN / "decision.json", prior.read_json(prior.VIEW_RUN / "protocol.json")["incumbent_sha256"])
    early = prior.read_json(prior.verified(early_path, rescue_plan["incumbent_sha256"]))
    health = prior.read_json(prior.verified(prior.HEALTH_RUN / "decision.json", early["health_decision_sha256"]))
    subtype = prior.read_json(prior.verified(prior.SUBTYPE_RUN / "decision.json", health["subtype_decision_sha256"]))
    gate, review = health["preferred_by_development"], early["preferred_by_development"]
    components = (prior.load_heads(subtype), prior.load_gate(gate, health["selections"][gate]), prior.restore(review, early["selections"][review]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = prior.Specialists(prior.checked_checkpoint(prior.FIRST_RUN / "candidate.pt", foundation["parent_sha256"]), device)
    cards = {key: card for key, card in last["selection"].items() if key.startswith("expanded_")}
    rescued = prior.rescue.restore_set(encoder, cards)
    return encoder, components, rescued, cards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    phase = parser.parse_args().phase
    torch.set_num_threads(4)
    load, verify, sha = prior.read_json, prior.verified, prior.sha256
    foundation = load(prior.SUBTYPE_RUN / "protocol.json")
    previous_path = prior.RUN / "decision.json"
    previous = load(previous_path)
    old_plan = load(verify(prior.RUN / "protocol.json", previous["protocol_sha256"]))
    verify(Path(prior.__file__), old_plan["source_sha256"])
    if previous["candidate"] != "directional":
        raise ValueError("Expected the development-locked directional baseline")
    paths, records, protected = prior.override.checked_splits(foundation)
    if {name: sha(path) for name, path in paths.items()} != old_plan["manifests"]:
        raise ValueError("The prior development manifests changed")
    _, _, protected = prior.rescue.pool_info(protected)
    fit_root = prior.rescue.RUN / "supplement_training.csv"
    paths["early_supplement"] = verify(fit_root, old_plan["supplement_sha256"])
    paths["late_supplement"] = verify(DATA / "supplement_training.csv", load(DATA / "training_freeze.json")["manifest_sha256"])
    records["early_supplement"] = prior.read_rows(paths["early_supplement"])
    added = records["late_supplement"] = prior.read_rows(paths["late_supplement"])
    identities = {row["sha256"] for row in added}
    if identities & protected or len(identities) != len(added) or any(row["label"] != "late_blight" or sha(Path(row["path"])) != row["sha256"] for row in added):
        raise ValueError("An added image has the wrong identity or label")
    test_screen = DATA / "test_screen.json"
    fresh = load(test_screen).get("independent_test")
    if fresh:
        verify(Path(fresh["path"]), fresh["sha256"])
    protocol = {"source_sha256": sha(Path(__file__)), "previous_decision_sha256": sha(previous_path),
                "manifest_sha256": {name: sha(path) for name, path in paths.items()}, "test_screen_sha256": sha(test_screen),
                "seed": list(prior.SEEDS), "epochs": list(EPOCHS), "thresholds": list(THRESHOLDS), "gaps": list(GAPS),
                "shifts": list(SHIFTS), "criterion": "Preserve each development group's early and late recall. Require late recall gains in at least two groups in all three threshold settings and at least two passing seeds. Prefer the supplemented arm only when it surpasses the continuation control.",
                "roles": "Only approved source images and original fitting data enter training. Audits are read after development lock."}
    RUN.mkdir(exist_ok=True)
    protocol_file, decision_file = RUN / "protocol.json", RUN / "decision.json"
    if protocol_file.exists():
        if load(protocol_file) != protocol:
            raise ValueError("The frozen protocol changed")
    elif phase == "train":
        prior.write_json(protocol_file, protocol)
    else:
        raise FileNotFoundError(protocol_file)
    encoder, components, rescue, rescue_cards = prior_models(foundation, old_plan)
    baseline_cards = {key: card for key, card in previous["selections"].items() if key.startswith("directional_")}
    baseline_heads = prior.restore_probes(baseline_cards)

    def baseline(data):
        early = prior.rescue_baseline(data, rescue, rescue_cards)
        switches = [prior.proposed(data, baseline_heads[key](data["dino"], prior.lead(data)), card["early_threshold"], card["late_threshold"], card["gap"])
                    for key, card in baseline_cards.items()]
        return early, prior.replace_subtype(early, torch.stack(switches).sum(0) >= 2)

    if phase == "train":
        if decision_file.exists():
            raise FileExistsError(decision_file)
        train(paths, records, encoder, components, baseline, baseline_cards, protocol_file, decision_file)
    else:
        evaluate(encoder, components, foundation, protected | identities, baseline, protocol_file, decision_file, fresh)


def train(paths, records, encoder, components, baseline, baseline_cards, protocol_file, decision_file):
    names = ("augmented", "early_supplement", "late_supplement", "development_field", "development_malik", "early_development", "extension_dev")
    features = {name: prior.encoded(encoder, components, paths[name], f"late_trial_{name}", records[name]) for name in names}
    dev = prior.pack([features[name] for name in names[3:]])
    _, base = baseline(dev)
    compare = evaluate_groups(dev, base, [row for name in names[3:] for row in records[name]])
    cards, composition = {}, {}
    for arm in ARMS:
        selected = names[:2] if arm == "continued" else names[:3]
        raw = prior.pack([features[name] for name in selected])
        disease = raw["truth"] != 0
        data = {key: value[disease] for key, value in raw.items()}
        rows = [row for row, keep in zip([row for name in selected for row in records[name]], disease.tolist(), strict=True) if keep]
        composition[arm] = {"n": len(rows), "late_early_leading": int(((data["truth"] == 2) & ~prior.lead(data)).sum()),
                            "steps_per_epoch": (len(rows) + prior.BATCH - 1) // prior.BATCH}
        for seed in prior.SEEDS:
            old_key = f"directional_s{seed}"
            key = f"{arm}_s{seed}"
            model = prior.restore_probes({old_key: baseline_cards[old_key]})[old_key]
            cards[key] = fit(key, model, seed, data, rows, dev, base, compare)
            print(key, cards[key], flush=True)
    comparisons, variants = {}, {}
    for arm in ARMS:
        variants[arm] = [compare(arm_votes(arm, cards, dev, base, shift)) for shift in SHIFTS]
        comparisons[arm] = prior.summarize(variants[arm])
        comparisons[arm]["passing_seeds"] = sum(card["pass"] for key, card in cards.items() if key.startswith(arm))
    continued, supplemented = (comparisons[arm] for arm in ARMS)
    quality = lambda card: (card["min_groups"], card["min_correct_gain"], card["min_balanced_gain"])
    eligible = lambda card: card["pass"] and card["passing_seeds"] >= 2
    candidate = "locked_directional"
    if eligible(supplemented) and (not eligible(continued) or quality(supplemented) > quality(continued)):
        candidate = "supplemented"
    elif eligible(continued):
        candidate = "continued"
    late = features["late_supplement"]
    mined = [{**row, "incumbent_conditional_early": not lead, "incumbent_prediction": prior.CLASSES[prediction]}
             for row, lead, prediction in zip(records["late_supplement"], prior.lead(late).tolist(), late["incumbent"].argmax(1).tolist(), strict=True)]
    prior.write_rows(RUN / "supplement_with_baseline.csv", mined)
    prior.write_json(decision_file, {"candidate": candidate, "cards": cards, "comparisons": comparisons, "variants": variants,
                                    "composition": composition, "protocol_sha256": prior.sha256(protocol_file)})
    print("Development choice", candidate, comparisons, flush=True)


@torch.no_grad()
def evaluate(encoder, components, foundation, protected, baseline, protocol, decision_path, fresh):
    decision = prior.read_json(decision_path)
    prior.verified(protocol, decision["protocol_sha256"])
    original = prior.create_model("resnet18")
    original.load_state_dict(prior.original_checkpoint("AC42")["model_state"])
    original.to(encoder.device, memory_format=torch.channels_last).eval().requires_grad_(False)
    suites, audits = prior.override.evaluation_receipts(foundation)
    if fresh:
        suites = {**suites, "new_source_test": fresh}
    for name, spec in suites.items():
        path = prior.verified(Path(spec["path"]), spec["sha256"])
        rows = prior.read_rows(path)
        hashes = {row["sha256"] for row in rows}
        if len(hashes) != len(rows) or hashes & protected:
            raise ValueError(f"Evaluation identity conflicts: {name}")
        role = "external_audit" if name in audits else spec.get("role", "historical_replay")
        data = prior.encoded(encoder, components, path, f"late_trial_{name}", rows, original)
        early, base = baseline(data)
        outputs = {"original_AC42": data["original"], "incumbent": data["incumbent"], "early_rescue_reference": early, "locked_directional": base}
        for arm in ARMS:
            outputs[arm] = prior.replace_subtype(base, arm_votes(arm, decision["cards"], data, base))
        table, predictions = {}, []
        truth = data["truth"]
        for key, scores in outputs.items():
            table[key], labels = (prior.measurement(scores, truth)[:2] if truth.unique().numel() == 3 else prior.measure(scores, truth))
            predictions.extend({"model": key, "sha256": row["sha256"], "true_label": row["label"], "prediction": prior.CLASSES[label]}
                               for row, label in zip(rows, labels.tolist(), strict=True))
        prior.write_rows(RUN / f"{name}_predictions.csv", predictions)
        prior.write_json(RUN / f"{name}_results.json", {"models": table, "role": role, "manifest_sha256": spec["sha256"],
                         "decision_sha256": prior.sha256(decision_path)})
        print(name, {key: (value["n"], value["accuracy"]) for key, value in table.items()}, flush=True)


if __name__ == "__main__":
    main()
