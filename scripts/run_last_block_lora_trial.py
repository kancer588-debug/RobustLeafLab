"""Compare a frozen rescue head with rank-four adaptation of the last DINOv2 block."""

import argparse
import copy
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import run_early_override_trial as override
import run_rescue_stability_trial as stability
from run_dino_health_trial import RUN as HEALTH_RUN, load_gate
from run_early_review_trial import RUN as EARLY_RUN, fitting_weights, restore
from run_health_rescue_trial import SHARED_DEVELOPMENT_GROUP, VIEW_RUN, encoded, group_counts, pack, record
from run_late_protection_trial import load_heads
from run_local_leaf_trial import measure
from run_roi_trial import HISTORICAL
from run_subtype_trial import (
    CLASSES, FIRST_RUN, RUN as SUBTYPE_RUN, Specialists, development_metrics, measurement, read_rows, verified, write_rows,
)
from train_online_candidate import (
    WORKSPACE, checked_checkpoint, create_model, original_checkpoint, read_json, seed_everything, sha256, write_json,
)


RUN = WORKSPACE / "outputs" / "last_block_lora_trial"
VOTE_PLAN = stability.RUN / "calibrated" / "decision.json"
ARMS = ("head_only", "lora")
SEEDS = (42, 45, 46)
EPOCHS = (1, 2, 4, 8)
BATCH = 32
ADAPTER_RANK = 4
HEAD_LR = 0.002
ADAPTER_LR = 0.0002
SHIFTS = (-0.25, 0., 0.25)


class QKVAdapter(nn.Module):
    """Learn a low-rank update around the unchanged QKV projection."""

    def __init__(self, projection):
        super().__init__()
        self.original = projection
        self.down = nn.Linear(projection.in_features, ADAPTER_RANK, bias=False)
        self.up = nn.Linear(ADAPTER_RANK, projection.out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, features):
        return self.original(features) + self.up(self.down(features))


class RescueModel(nn.Module):
    def __init__(self, encoder, adapted):
        super().__init__()
        self.head = override.RescueHead(True)
        self.block = None
        self.device = encoder.device
        if adapted:
            self.block = copy.deepcopy(encoder.dino.blocks[-1]).eval().requires_grad_(False)
            self.block.attn.qkv = QKVAdapter(self.block.attn.qkv)
            self.norm = copy.deepcopy(encoder.dino.norm).eval().requires_grad_(False)
        self.to(self.device)

    def forward(self, data, indices):
        if self.block is None:
            return self.head(data["cls"][indices].to(self.device), data["patches"][indices].to(self.device))
        tokens = self.norm(self.block(data["last_input"][indices].to(self.device)))
        cls, patches = tokens[:, 0], tokens[:, 1:]
        side = int(patches.shape[1] ** 0.5)
        pooled = F.avg_pool2d(patches.transpose(1, 2).reshape(len(patches), patches.shape[2], side, side), 2)
        return self.head(cls, pooled.flatten(2).transpose(1, 2))

    def saved_state(self):
        state = {"head": {name: value.detach().cpu().clone() for name, value in self.head.state_dict().items()}}
        if self.block is not None:
            adapter = self.block.attn.qkv
            state["down"] = adapter.down.weight.detach().cpu().clone()
            state["up"] = adapter.up.weight.detach().cpu().clone()
        return state

    def load_saved_state(self, state):
        self.head.load_state_dict(state["head"])
        if self.block is not None:
            adapter = self.block.attn.qkv
            with torch.no_grad():
                adapter.down.weight.copy_(state["down"])
                adapter.up.weight.copy_(state["up"])


def extract(encoder, components, path, name, rows, original=None):
    if encoder.dino.chunked_blocks or encoder.dino.num_register_tokens:
        raise ValueError("The adaptation requires the frozen unchunked backbone with no register tokens")
    batches = []
    hook = encoder.dino.blocks[-1].register_forward_pre_hook(lambda _module, inputs: batches.append(inputs[0].detach().cpu()))
    try:
        data = encoded(encoder, components, path, name, rows, original)
    finally:
        hook.remove()
    data["last_input"] = torch.cat(batches)
    if len(data["last_input"]) != len(rows):
        raise ValueError(f"Unexpected number of cached last-block inputs: {name}")
    return data


@torch.no_grad()
def evidence(model, data):
    return torch.cat([model(data, slice(i, i + BATCH)).cpu() for i in range(0, len(data["truth"]), BATCH)])


def diagnostics(scores, dev, context):
    before = context[1]["groups"]
    card = record(scores, dev["truth"], *context, 0, 0., None)
    changes = sum(group["early_missed"] < before[name]["early_missed"] for name, group in card["groups"].items())
    row = card["record"]
    return {"groups": changes, "rescues": row["early_recovered_all"], "guardrails": row["guardrails"],
            "healthy_fp_added": row["new_healthy_false_alarms"], "late_as_early_added": row["additional_late_to_early"],
            "legacy_macro_f1": row["legacy_macro_f1"], "malik_macro_f1": row["malik_macro_f1"]}


def summarize(cards):
    guarded = all(value["guardrails"] for value in cards)
    groups = min(value["groups"] for value in cards)
    recovered = min(value["rescues"] for value in cards)
    return {"passes": guarded and groups >= 2, "guardrails": guarded, "minimum_groups": groups,
            "minimum_rescues": recovered, "central_rescues": cards[1]["rescues"],
            "central_healthy_fp_added": cards[1]["healthy_fp_added"], "central_late_as_early_added": cards[1]["late_as_early_added"]}


def rank(card):
    return (card["passes"], card["guardrails"], card["minimum_groups"], card["minimum_rescues"], card["central_rescues"],
            -card["gap"], card["threshold"], -card["epoch"])


def train_one(encoder, arm, seed, fit, dev, weights, context):
    seed_everything(seed)
    model = RescueModel(encoder, arm == "lora")
    initial = model.saved_state()
    params = [{"params": model.head.parameters(), "lr": HEAD_LR}]
    if arm == "lora":
        live = model.block.attn.qkv
        params.extend([{"params": [live.down.weight, live.up.weight], "lr": ADAPTER_LR}])
        if not torch.allclose(model(fit, slice(0, 4)), model.head(fit["cls"][:4].to(encoder.device),
                                                                fit["patches"][:4].to(encoder.device)), atol=2e-5, rtol=1e-4):
            raise ValueError("A zero adapter changed its starting local-evidence score")
    optimizer = torch.optim.AdamW(params, weight_decay=0.05)
    generator = torch.Generator().manual_seed(seed)
    labels = (fit["truth"] == 1).float()
    best, history = None, []
    for epoch in range(1, EPOCHS[-1] + 1):
        total = 0.
        for indices in torch.randperm(len(labels), generator=generator).split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            logits = model(fit, indices)
            loss = F.binary_cross_entropy_with_logits(logits, labels[indices].to(encoder.device),
                                                       weight=weights[indices].to(encoder.device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        if epoch not in EPOCHS:
            continue
        scores = evidence(model, dev)
        deltas = {offset + threshold for offset, threshold in product(SHIFTS, override.THRESHOLDS)}
        checked = {(threshold, gap): diagnostics(override.route(dev["incumbent"], scores, threshold, gap, "direct"), dev, context)
                   for threshold, gap in product(deltas, override.GAPS)}
        for threshold, gap in product(override.THRESHOLDS, override.GAPS):
            checks = [checked[threshold + delta, gap] for delta in SHIFTS]
            row = {"epoch": epoch, "threshold": threshold, "gap": gap, "loss": total / len(labels), **summarize(checks)}
            history.append(row)
            if best is None or rank(row) > rank(best):
                best, state = row, model.saved_state()
    folder = RUN / f"{arm}_s{seed}"
    folder.mkdir(exist_ok=True)
    write_rows(folder / "history.csv", history)
    weight = folder / "model.pt"
    temp = weight.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(weight)
    changed = (state["up"].double() @ state["down"].double()).norm().item() if arm == "lora" else 0.
    if arm == "lora" and (changed <= 0. or torch.equal(state["down"], initial["down"])):
        raise ValueError("The selected adapter contains no learned update")
    result = {**best, "weight_sha256": sha256(weight), "adapter_update_norm": changed}
    print(folder.name, result, flush=True)
    return result


def ensemble(data, scores, selection):
    original = data["incumbent"]
    top = original.argmax(1)
    variants = {}
    for arm in ARMS:
        for offset, suffix in zip(SHIFTS, ("_looser", "", "_tighter"), strict=True):
            votes = []
            for seed in SEEDS:
                key = f"{arm}_s{seed}"
                card = selection[key]
                logits = override.route(original, scores[key], card["threshold"] + offset, card["gap"], "direct")
                vote = (top == 0) & (logits.argmax(1) == 1)
                votes.append(vote)
                if offset == 0.:
                    variants[key] = logits
            variants[arm + "_majority" + suffix] = stability.deploy_vote(original, torch.stack(votes).sum(0) >= 2)
    return variants


def load_models(encoder, selections):
    models = {}
    for key, card in selections.items():
        model = RescueModel(encoder, key.startswith("lora_"))
        weight = verified(RUN / key / "model.pt", card["weight_sha256"])
        model.load_saved_state(torch.load(weight, weights_only=True, map_location="cpu"))
        models[key] = model.eval()
    return models


def train(encoder, components, paths, rows, plan, previous):
    RUN.mkdir(exist_ok=True)
    lock, protocol = RUN / "decision.json", RUN / "protocol.json"
    if lock.exists():
        raise FileExistsError("A development decision is already frozen")
    if not protocol.exists():
        write_json(protocol, plan)
    elif read_json(protocol) != plan:
        raise ValueError("The frozen fine-tuning protocol differs")
    names = ("augmented", "healthy_fit", "development_field", "development_malik", "early_development", "extension_dev")
    cached = [extract(encoder, components, paths[name], name, rows[name]) for name in names]
    fit, dev = pack(cached[:2]), pack(cached[2:])
    del cached
    fit_rows = rows["augmented"] + rows["healthy_fit"]
    group_ids = defaultdict(list)
    for index, row in enumerate(row for name in names[2:] for row in rows[name]):
        group = row["capture_group"]
        group_ids[SHARED_DEVELOPMENT_GROUP.get(group, group)].append(index)
    malik_n = len(rows["development_malik"])
    old_n = len(rows["development_field"]) + malik_n
    initial = {"legacy": development_metrics(dev["incumbent"][:old_n], {"truth": dev["truth"][:old_n]}, malik_n),
               "groups": group_counts(dev["truth"], dev["incumbent"].argmax(1), group_ids)}
    if initial != previous["development_incumbent"]:
        raise ValueError("The frozen incumbent development scores changed")
    context = group_ids, initial, old_n, malik_n
    weights = fitting_weights(fit_rows, fit["incumbent"], fit["truth"])
    early = fit["truth"] == 1
    early_weight = weights[early].sum()
    weights[early & (fit["incumbent"].argmax(1) == 0)] *= 2.5
    weights[early] *= early_weight / weights[early].sum()
    selection = {f"{arm}_s{seed}": train_one(encoder, arm, seed, fit, dev, weights, context)
                 for arm, seed in product(ARMS, SEEDS)}
    loaded = load_models(encoder, selection)
    variants = ensemble(dev, {key: evidence(model, dev) for key, model in loaded.items()}, selection)
    development = {key: diagnostics(scores, dev, context) for key, scores in variants.items()}
    comparison = {arm: summarize([development[f"{arm}_majority{suffix}"] for suffix in ("_looser", "", "_tighter")])
                  for arm in ARMS}
    for arm in ARMS:
        comparison[arm]["passing_single_seeds"] = sum(selection[f"{arm}_s{seed}"]["passes"] for seed in SEEDS)
    tune, control = comparison["lora"], comparison["head_only"]
    better = (tune["minimum_rescues"], tune["passing_single_seeds"]) > (control["minimum_rescues"], control["passing_single_seeds"])
    chosen = "lora_majority" if tune["passes"] and tune["passing_single_seeds"] >= 2 and better else "incumbent"
    write_json(lock, {"candidate": chosen, "protocol_sha256": sha256(protocol), "selection": selection,
                      "development": development, "comparison": comparison, "fit_early_misses": int((early & (fit["incumbent"].argmax(1) == 0)).sum())})
    print("Locked development comparison:", comparison, "chosen:", chosen, flush=True)


@torch.no_grad()
def evaluate(encoder, components, protocol, plan, forbidden, previous):
    decision_path = RUN / "decision.json"
    decision = read_json(decision_path)
    if read_json(verified(RUN / "protocol.json", decision["protocol_sha256"])) != plan:
        raise ValueError("The evaluation protocol changed")
    calibrated = read_json(verified(VOTE_PLAN, plan["calibrated_vote_sha256"]))
    vote_heads, vote_cards = stability.restore_models(previous, calibrated["fresh_selections"])
    vote_cards = {seed: {**card, "threshold": card["threshold"] + calibrated["offset"]} for seed, card in vote_cards.items()}
    models = load_models(encoder, decision["selection"])
    original = create_model("resnet18")
    original.load_state_dict(original_checkpoint("AC42")["model_state"])
    original.to(encoder.device, memory_format=torch.channels_last).eval().requires_grad_(False)
    tests, audits = override.evaluation_receipts(protocol)
    lock_hash = sha256(decision_path)
    totals = defaultdict(Counter)
    for name, spec in tests.items():
        path = verified(Path(spec["path"]), spec["sha256"])
        rows = read_rows(path)
        digests = {row["sha256"] for row in rows}
        if len(digests) != len(rows) or digests & forbidden:
            raise ValueError(f"Overlapping test identities: {name}")
        result = RUN / f"{name}_results.json"
        if result.exists():
            saved = read_json(result)
            if saved["decision_sha256"] != lock_hash or saved["manifest_sha256"] != spec["sha256"]:
                raise ValueError(f"An evaluation lock changed: {name}")
            metrics = saved["models"]
        else:
            data = extract(encoder, components, path, name, rows, original)
            base, truth = data["incumbent"], data["truth"]
            six_vote = stability.score_models(data, vote_heads, vote_cards)[stability.PRIMARY]
            scores = {"original_AC42": data["original"], "incumbent": base, "calibrated_six_vote": six_vote,
                      **ensemble(data, {key: evidence(model, data) for key, model in models.items()}, decision["selection"])}
            metrics, predictions = {}, []
            prior = base.argmax(1)
            for key, logits in scores.items():
                values, guessed = measurement(logits, truth)[:2] if truth.unique().numel() == len(CLASSES) else measure(logits, truth)
                if key != "original_AC42":
                    switched = prior != guessed
                    if (switched & ((prior != 0) | (guessed != 1))).any():
                        raise ValueError(f"Invalid rescue route: {name} {key}")
                    values.update(early_rescues=int((switched & (truth == 1)).sum()),
                                  healthy_fp_added=int((switched & (truth == 0)).sum()),
                                  late_as_early_added=int((switched & (truth == 2)).sum()))
                metrics[key] = values
                predictions.extend({"model": key, "sha256": row["sha256"], "true_label": row["label"], "prediction": CLASSES[guess]}
                                   for row, guess in zip(rows, guessed.tolist(), strict=True))
            write_rows(RUN / f"{name}_predictions.csv", predictions)
            write_json(result, {"decision_sha256": lock_hash, "manifest_sha256": spec["sha256"], "models": metrics,
                                "role": "external_audit" if name in audits else spec.get("role", "historical_replay")})
            print(name, {key: round(metrics[key]["accuracy"], 5) for key in ("incumbent", "calibrated_six_vote", "head_only_majority", "lora_majority")}, flush=True)
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
    protocol = read_json(SUBTYPE_RUN / "protocol.json")
    paths, rows, forbidden = override.checked_splits(protocol)
    prior_path = override.RUN / "decision.json"
    previous = read_json(prior_path)
    old_plan = read_json(verified(override.RUN / "protocol.json", previous["protocol_sha256"]))
    digests = {key: sha256(path) for key, path in paths.items()}
    if digests != old_plan["manifests"]:
        raise ValueError("A frozen data manifest changed")
    early_path = verified(EARLY_RUN / "decision.json", read_json(VIEW_RUN / "protocol.json")["incumbent_sha256"])
    early = read_json(verified(early_path, old_plan["incumbent_sha256"]))
    health = read_json(verified(HEALTH_RUN / "decision.json", early["health_decision_sha256"]))
    subtype = read_json(verified(SUBTYPE_RUN / "decision.json", health["subtype_decision_sha256"]))
    gate, review = health["preferred_by_development"], early["preferred_by_development"]
    components = (load_heads(subtype), load_gate(gate, health["selections"][gate]), restore(review, early["selections"][review]))
    plan = {"source_sha256": sha256(Path(__file__)), "incumbent_sha256": sha256(early_path), "prior_decision_sha256": sha256(prior_path),
            "calibrated_vote_sha256": sha256(VOTE_PLAN), "manifests": digests, "arms": list(ARMS), "seeds": list(SEEDS),
            "epochs": list(EPOCHS), "thresholds": list(override.THRESHOLDS), "maximum_gaps": list(override.GAPS),
            "threshold_sensitivity": list(SHIFTS), "rank": ADAPTER_RANK, "batch": BATCH, "head_lr": HEAD_LR, "adapter_lr": ADAPTER_LR,
            "selection": "Use the same batches and initial local head under both arms. Fine-tune only a rank-four QKV "
                         "update in the last DINOv2 block alongside the head. The majority vote must rescue at least two "
                         "development groups with no new healthy errors or late-to-early errors at all three thresholds. "
                         "Require two individually passing seeds and compare worst-case recoveries then passing seeds to head-only.",
            "roles": "Previously frozen development only for selection. Archives and external audits run after locking. "
                     "No independent three-class release cohort is present."}
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", protocol["parent_sha256"]), device)
    if phase == "train":
        train(encoder, components, paths, rows, plan, previous)
    else:
        evaluate(encoder, components, protocol, plan, forbidden, previous)


if __name__ == "__main__":
    main()
