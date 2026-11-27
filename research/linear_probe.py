"""Fit or evaluate one three-class classifier on frozen DINOv2 features."""

import argparse
import csv
import hashlib
import json
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from robustleaflab.backbone import load_dino, verify
from robustleaflab.models import CLASSES, read_image


FIT_SETTINGS = {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1_000, "random_state": 42}


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_manifest(spec, root):
    manifest = verify(root / spec["path"], spec["sha256"])
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    for row in rows:
        image = root / row["path"]
        key = row["sha256"]
        if key in seen or row["label"] not in CLASSES or sha256(image) != key:
            raise ValueError(f"Duplicate, invalid label, or changed image in {manifest}: {image}")
        row["path"] = image
        seen.add(key)
    return rows


@torch.inference_mode()
def features(model, rows, device):
    vectors = []
    cursor = iter(rows)
    while batch := list(islice(cursor, 16)):
        images = torch.stack([read_image(row["path"]) for row in batch]).to(device)
        tokens = model.forward_features(images)
        vectors.append(torch.cat((tokens["x_norm_clstoken"], tokens["x_norm_patchtokens"].mean(1)), 1).cpu().numpy())
    return np.concatenate(vectors).astype(np.float64)


def labels(rows):
    return np.array([CLASSES.index(row["label"]) for row in rows])


def score(archive, array):
    standardized = (array - archive["mean"]) / archive["scale"]
    return (standardized @ archive["coefficients"].T + archive["intercept"]).argmax(1)


def metrics(truth, prediction):
    present = np.unique(truth)
    return {"n": len(truth), "correct": int((truth == prediction).sum()),
            "accuracy": float((truth == prediction).mean()),
            "macro_f1": float(f1_score(truth, prediction, labels=present, average="macro", zero_division=0)),
            "confusion": confusion_matrix(truth, prediction, labels=range(len(CLASSES))).tolist()}


def write_json(path, content):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2)
        handle.write("\n")


def fit(config, args, encoder, device):
    fit_rows = [row for spec in config["training"].values() for row in load_manifest(spec, args.root)]
    dev_rows = [row for spec in config["development"].values() for row in load_manifest(spec, args.root)]
    fit_ids, dev_ids = ({row["sha256"] for row in rows} for rows in (fit_rows, dev_rows))
    if len(fit_ids) != len(fit_rows) or len(dev_ids) != len(dev_rows) or fit_ids & dev_ids:
        raise ValueError("Training and development must be disjoint, with unique identities")
    target = labels(fit_rows)
    if len(np.unique(target)) != len(CLASSES):
        raise ValueError("All three classes are required for fitting")
    print(f"Encoding {len(fit_rows)} fit images", flush=True)
    array = features(encoder, fit_rows, device)
    scaler = StandardScaler().fit(array)
    head = LogisticRegression(**FIT_SETTINGS).fit(scaler.transform(array), target)
    if (head.n_iter_ >= FIT_SETTINGS["max_iter"]).any():
        raise ValueError("Probe did not converge within the fixed iteration limit")
    weights = args.output / "probe.npz"
    with weights.open("xb") as handle:
        np.savez_compressed(handle, mean=scaler.mean_, scale=scaler.scale_, coefficients=head.coef_, intercept=head.intercept_)
    print(f"Scoring {len(dev_rows)} development images", flush=True)
    with np.load(weights, allow_pickle=False) as archive:
        development = metrics(labels(dev_rows), score(archive, features(encoder, dev_rows, device)))
    write_json(args.output / "decision.json", {
        "settings": FIT_SETTINGS, "weights_sha256": sha256(weights), "fitting": config["training"],
        "development_manifests": config["development"], "fit_count_by_class": np.bincount(target).tolist(),
        "development": development, "class_order": CLASSES, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "preprocess": "224 RGB center crop; concatenated DINOv2 class token and mean patch tokens",
        "selection": "Fixed balanced logistic classifier. No hyperparameter, checkpoint, or audit selection.",
    })


def evaluation(config, args, encoder, device):
    decision = json.loads((args.output / "decision.json").read_text(encoding="utf-8"))
    if decision["fitting"] != config["training"] or decision["development_manifests"] != config["development"]:
        raise ValueError("Fit or development configuration changed after lock")
    forbidden = set()
    for spec in (*config["training"].values(), *config["development"].values()):
        with verify(args.root / spec["path"], spec["sha256"]).open(encoding="utf-8-sig", newline="") as handle:
            forbidden.update(row["sha256"] for row in csv.DictReader(handle))
    results, pooled_truth, pooled_predictions = {}, [], []
    with np.load(verify(args.output / "probe.npz", decision["weights_sha256"]), allow_pickle=False) as archive:
        for name, spec in config["evaluation"].items():
            rows = load_manifest(spec, args.root)
            if any(row["sha256"] in forbidden for row in rows):
                raise ValueError(f"Training or development image found in {name}")
            truth = labels(rows)
            predicted = score(archive, features(encoder, rows, device))
            results[name] = {**metrics(truth, predicted), "role": spec["role"], "manifest_sha256": spec["sha256"]}
            print(name, results[name]["correct"], '/', len(rows), flush=True)
            with (args.output / f"{name}_probe.csv").open("x", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("sha256", "true_label", "prediction"))
                writer.writerows((row["sha256"], row["label"], CLASSES[label]) for row, label in zip(rows, predicted, strict=True))
            if spec["role"] == "historical_replay":
                pooled_truth.append(truth)
                pooled_predictions.append(predicted)
    write_json(args.output / "results.json", {"cohorts": results, "historical_pool": metrics(np.concatenate(pooled_truth),
               np.concatenate(pooled_predictions)), "decision_sha256": sha256(args.output / "decision.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.phase == "fit" and (args.output / "decision.json").exists():
        raise FileExistsError("The probe decision is already locked")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    encoder = load_dino(args.cache, device).eval().requires_grad_(False)
    (fit if args.phase == "fit" else evaluation)(config, args, encoder, device)


if __name__ == "__main__":
    main()
