"""Compare shared-backbone whole-image and local-view residual subtype classifiers."""

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TV

from run_backbone_trial import FIRST_RUN, TRIAL_SPLITS, manifest
from run_expanded_field_trial import eligible, score, set_limits, verified
from train_online_candidate import (
    CLASSES, ORIGINAL_CHECKPOINTS, WORKSPACE, checked_checkpoint, create_model, original_checkpoint,
    read_json, seed_everything, sha256, write_json,
)

from common import transform


SPLITS = WORKSPACE / "datasets" / "local_leaf_trial"
RUN = WORKSPACE / "outputs" / "local_leaf_trial_base873"
EXPANDED = WORKSPACE / "datasets" / "expanded_field_trial"
ARMS = ("whole", "whole_and_local")
SEEDS = (42, 43, 44)
EPOCHS = 10
LR = 8e-4
MAX_MARGIN_SHIFT = 1.0
CROP_FRACTION = 0.55
IMAGE_BATCH_SIZE = 24
HEAD_BATCH_SIZE = 64
REPLAY_NAMES = ("plantcity_replay", "plantwild_replay", "imtiaz_replay")


def manifest_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class FrozenViewEncoder(nn.Module):
    """Encode a standard crop and a higher-magnification crop of a responsive area."""

    def __init__(self, parent, device):
        super().__init__()
        self.encoder = create_model("resnet18")
        self.encoder.load_state_dict(parent["model_state"])
        self.encoder.requires_grad_(False)
        self.encoder.to(device, memory_format=torch.channels_last).eval()
        self.device = device
        self.normalize = transform()

    def maps(self, x):
        model = self.encoder
        x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        return model.layer4(model.layer3(model.layer2(model.layer1(x))))

    def image_tensors(self, views):
        return torch.stack(views).to(self.device, memory_format=torch.channels_last)

    def local_crop(self, image, resized_size, peak, grid_size):
        width, height = resized_size
        grid_w, grid_h = grid_size
        cx = (round((width - 224) / 2) + (peak % grid_w + 0.5) * 224 / grid_w) * image.width / width
        cy = (round((height - 224) / 2) + (peak // grid_w + 0.5) * 224 / grid_h) * image.height / height
        side = max(1, round(224 * CROP_FRACTION * min(image.width / width, image.height / height)))
        x = min(max(0, round(cx - side / 2)), image.width - side)
        y = min(max(0, round(cy - side / 2)), image.height - side)
        return image.crop((x, y, x + side, y + side))

    @torch.no_grad()
    def encode(self, images):
        resized = [TV.resize(image, 256) for image in images]
        whole = self.image_tensors([self.normalize(image) for image in images])
        spatial = self.maps(whole)
        whole_features = self.encoder.avgpool(spatial).flatten(1)
        prior = self.encoder.fc(whole_features)
        activation = self.encoder.fc(spatial.permute(0, 2, 3, 1))
        response = activation[..., 1:].logsumexp(-1) - activation[..., 0]
        response = F.avg_pool2d(response.unsqueeze(1), 3, stride=1, padding=1, count_include_pad=False)
        peaks = response.flatten(1).argmax(1).tolist()
        grid_size = spatial.shape[-1], spatial.shape[-2]
        regions = [self.local_crop(image, scaled.size, index, grid_size)
                   for image, scaled, index in zip(images, resized, peaks)]
        local = self.image_tensors([self.normalize(region) for region in regions])
        local_features = self.encoder.avgpool(self.maps(local)).flatten(1)
        whole_features = F.normalize(whole_features) * whole_features.shape[1] ** 0.5
        local_features = F.normalize(local_features) * local_features.shape[1] ** 0.5
        return whole_features.cpu(), ((whole_features + local_features) * 0.5).cpu(), prior.cpu()


@torch.no_grad()
def cached(encoder, name, path, baseline=None):
    support = manifest_rows(path)
    buffers = []
    baseline_parts = []
    print(f"Encoding {name}: {len(support)} images", flush=True)
    for at in range(0, len(support), IMAGE_BATCH_SIZE):
        batch = support[at:at + IMAGE_BATCH_SIZE]
        images = []
        for row in batch:
            with Image.open(row["path"]) as image:
                images.append(image.convert("RGB"))
        buffers.append(encoder.encode(images))
        if baseline is not None:
            views = encoder.image_tensors([encoder.normalize(image) for image in images])
            baseline_parts.append(baseline(views).cpu())
    whole, mixed, prior = (torch.cat(parts) for parts in zip(*buffers))
    truth = torch.tensor([CLASSES.index(row["label"]) for row in support])
    return {"whole": whole, "whole_and_local": mixed, "prior": prior, "truth": truth, "rows": support,
            "original_AC42": torch.cat(baseline_parts) if baseline_parts else None}


class SubtypeResidual(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.linear = nn.Linear(dimensions, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def margin(self, features, prior):
        return prior[:, 1] - prior[:, 2] + MAX_MARGIN_SHIFT * self.linear(features).flatten().tanh()

    def forward(self, features, prior):
        margin = self.margin(features, prior)
        maximum = prior[:, 1:].max(dim=1).values
        early = maximum + margin.clamp(max=0)
        late = maximum + (-margin).clamp(max=0)
        return torch.stack((prior[:, 0], early, late), dim=1)


def measure(logits, truth):
    guesses = logits.argmax(dim=1)
    matrix = torch.bincount(3 * truth + guesses, minlength=9).reshape(3, 3).tolist()
    counts = [sum(row) for row in matrix]
    correct = [matrix[i][i] for i in range(3)]
    f1 = [
        2 * correct[i] / max(counts[i] + sum(row[i] for row in matrix), 1)
        for i in range(3)
    ]
    metrics = {
        "n": len(truth), "accuracy": sum(correct) / len(truth),
        "macro_f1": sum(f1) / 3 if all(counts) else None,
        "early_recall": correct[1] / counts[1] if counts[1] else None,
        "late_recall": correct[2] / counts[2] if counts[2] else None,
        "healthy_false_alarm_count": counts[0] - correct[0],
        "late_to_healthy": matrix[2][0], "late_to_early": matrix[2][1],
        "class_counts": dict(zip(CLASSES, counts)), "class_f1": dict(zip(CLASSES, f1)),
        "confusion": matrix,
    }
    return metrics, guesses


def classify(head, data, arm):
    if head is None:
        return data["prior"]
    with torch.no_grad():
        prediction = head(data[arm], data["prior"])
    if not torch.equal(prediction.argmax(1) == 0, data["prior"].argmax(1) == 0):
        raise RuntimeError("The subtype residual changed a healthy decision")
    return prediction


def compare_development(head, arm, field, malik):
    pooled = torch.cat((classify(head, field, arm), classify(head, malik, arm)), dim=0)
    target = torch.cat((field["truth"], malik["truth"]))
    result = measure(pooled, target)[0]
    result["malik"] = measure(classify(head, malik, arm), malik["truth"])[0]
    return result


def train_head(arm, seed, fit, field, malik, limits):
    seed_everything(seed)
    disease = fit["truth"] > 0
    features = fit[arm][disease]
    anchor = fit["prior"][disease]
    labels = (fit["truth"][disease] == 1).float()
    weights = torch.where(labels > 0, labels.numel() / (2 * labels.sum()),
                          labels.numel() / (2 * (labels == 0).sum()))
    head = SubtypeResidual(features.shape[1])
    initial = classify(head, field, arm)
    torch.testing.assert_close(initial, field["prior"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.02)
    best = None
    best_rank = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        for index in torch.randperm(len(labels)).split(HEAD_BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            margin = head.margin(features[index], anchor[index])
            loss = F.binary_cross_entropy_with_logits(margin, labels[index], weight=weights[index])
            loss.backward()
            optimizer.step()
        metrics = compare_development(head, arm, field, malik)
        passes = eligible(metrics, limits)
        record = {
            "epoch": epoch, "score": score(metrics), "macro_f1": metrics["macro_f1"],
            "early_f1": metrics["class_f1"]["early_blight"], "late_correct": metrics["confusion"][2][2],
            "late_to_early": metrics["late_to_early"], "eligible": passes,
        }
        history.append(record)
        rank = passes, record["score"], record["early_f1"]
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = {"record": record, "metrics": metrics}
            state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    directory = RUN / f"{arm}_s{seed}"
    directory.mkdir()
    with (directory / "history.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    artifact = directory / "head.pt"
    torch.save({"arm": arm, "seed": seed, "head_state": state, "input_features": features.shape[1]}, artifact)
    best["checkpoint_sha256"] = sha256(artifact)
    write_json(directory / "selection.json", best)
    print(f"{arm} seed {seed} {json.dumps(best['record'])}", flush=True)
    return best


def train(encoder, parent_hash, *, cohort):
    expanded = read_json(EXPANDED / "training_freeze.json")
    fit_path = verified(EXPANDED / "expanded_train.csv", expanded["training_sha256"]["expanded_train"])
    validation = read_json(TRIAL_SPLITS / "freeze.json")
    paths = {
        "train": fit_path, "field": manifest("development_field", validation),
        "malik": manifest("development_malik", validation),
    }
    if cohort == "chen_pilot":
        extra = read_json(SPLITS / "training_additions.json")
        paths["train"] = verified(SPLITS / "train.csv", extra["manifest_sha256"])
    replay = read_json(EXPANDED / "evaluation_freeze.json")["historical_replays"]
    replay = {name: replay[name] for name in REPLAY_NAMES}
    RUN.mkdir(exist_ok=False)
    write_json(RUN / "protocol.json", {
        "parent_sha256": parent_hash, "cohort": cohort, "seed": SEEDS, "epochs": EPOCHS, "head_lr": LR,
        "original_ac42_sha256": ORIGINAL_CHECKPOINTS["AC42"][1],
        "view": {"crop_fraction": CROP_FRACTION, "max_margin_shift": MAX_MARGIN_SHIFT, "local_feature_weight": 0.5},
        "manifests": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "replays": replay, "selection": "Retain existing development guardrails. Rank by mean Macro-F1 across "
        "pooled development and Malik, then pooled early F1. Replacement must increase score, keep pooled "
        "Macro-F1, and improve early F1 versus starting, for seed 42 and at least one confirmatory seed. "
        "The local input must also outperform the matched whole-image head on development score.",
        "protection": "New field and PlantDoc supply only duplicate-screening identities. Neither is used "
        "for ranking, early stopping, threshold fitting, or model inference in this experiment.",
    })
    fit, field, malik = (cached(encoder, name, paths[name]) for name in ("train", "field", "malik"))
    start = compare_development(None, "whole", field, malik)
    limits = set_limits(start)
    selections = {
        f"{arm}_s{seed}": train_head(arm, seed, fit, field, malik, limits)
        for seed in SEEDS for arm in ARMS
    }
    passes = {}
    for seed in SEEDS:
        for arm in ARMS:
            key = f"{arm}_s{seed}"
            chosen = selections[key]
            metrics = chosen["metrics"]
            control_score = score(start) if arm == "whole" else selections[f"whole_s{seed}"]["record"]["score"]
            passes[key] = bool(chosen["record"]["eligible"] and score(metrics) > max(score(start), control_score)
                               and metrics["macro_f1"] >= start["macro_f1"]
                               and metrics["class_f1"]["early_blight"] > start["class_f1"]["early_blight"])
    candidates = [f"{arm}_s42" for arm in ARMS if passes[f"{arm}_s42"]
                  and any(passes[f"{arm}_s{seed}"] for seed in SEEDS[1:])]
    preferred = max(candidates, key=lambda key: selections[key]["record"]["score"], default="starting_candidate")
    write_json(RUN / "decision.json", {
        "preferred_by_development": preferred, "promotion_passes": passes,
        "starting_development": start, "limits": limits, "selections": selections,
    })


def evaluate(encoder, *, extra_test):
    protocol = read_json(RUN / "protocol.json")
    decision = read_json(RUN / "decision.json")
    baseline = create_model("resnet18")
    baseline.load_state_dict(original_checkpoint("AC42")["model_state"])
    baseline.requires_grad_(False).to(encoder.device, memory_format=torch.channels_last).eval()
    entries = {"original_AC42": (None, "whole"), "starting_candidate": (None, "whole")}
    for arm in ARMS:
        name = f"{arm}_s42"
        path = verified(RUN / name / "head.pt", decision["selections"][name]["checkpoint_sha256"])
        saved = torch.load(path, map_location="cpu", weights_only=True)
        head = SubtypeResidual(saved["input_features"])
        head.load_state_dict(saved["head_state"])
        entries[name] = (head, arm)
    tests = dict(protocol["replays"])
    if extra_test:
        receipt = read_json(extra_test)
        if receipt["compatible_training_sha256"] != protocol["manifests"]["train"]["sha256"]:
            raise ValueError("The fresh source was not approved for this training cohort")
        tests.update(receipt["tests"])
    for name, item in tests.items():
        path = verified(Path(item["path"]), item["sha256"])
        result_path = RUN / f"{name}_results.json"
        if result_path.exists():
            if read_json(result_path)["manifest_sha256"] != item["sha256"]:
                raise ValueError(f"Existing result uses a different manifest: {name}")
            continue
        data = cached(encoder, name, path, baseline)
        results = {}
        predictions_path = RUN / f"{name}_predictions.csv"
        pending_predictions = predictions_path.with_suffix(".tmp")
        with pending_predictions.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("model", "sha256", "true_label", "prediction"))
            writer.writeheader()
            for key, (head, arm) in entries.items():
                logits = data[key] if key == "original_AC42" else classify(head, data, arm)
                results[key], guessed = measure(logits, data["truth"])
                for row, prediction in zip(data["rows"], guessed.tolist()):
                    writer.writerow({"model": key, "sha256": row["sha256"], "true_label": row["label"],
                                     "prediction": CLASSES[prediction]})
                print(name, key, json.dumps(results[key]), flush=True)
        pending_predictions.replace(predictions_path)
        write_json(result_path, {"manifest_sha256": item["sha256"], "models": results})


def main():
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate", "all"))
    parser.add_argument("--cohort", choices=("base873", "chen_pilot"), default="base873")
    parser.add_argument("--extra-test", type=Path)
    args = parser.parse_args()
    if args.cohort == "chen_pilot":
        RUN = WORKSPACE / "outputs" / "local_leaf_trial"
    torch.set_num_threads(4)
    torch.set_grad_enabled(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent_hash = read_json(FIRST_RUN / "selection.json")["candidate_sha256"]
    parent = checked_checkpoint(FIRST_RUN / "candidate.pt", parent_hash)
    encoder = FrozenViewEncoder(parent, device)
    if args.phase in ("train", "all"):
        train(encoder, parent_hash, cohort=args.cohort)
    if args.phase in ("evaluate", "all"):
        evaluate(encoder, extra_test=args.extra_test)


if __name__ == "__main__":
    main()
