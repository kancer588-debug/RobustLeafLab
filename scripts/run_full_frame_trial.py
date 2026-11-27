"""Evaluate full-frame views as additional early-versus-late blight evidence."""

import argparse
from pathlib import Path
from shutil import copyfile

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from run_backbone_trial import loader, move_images
from run_dino_health_trial import RUN as HEALTH_RUN, load_gate
from run_early_review_trial import RUN as EARLY_RUN, restore
from run_late_protection_trial import load_heads
from run_local_leaf_trial import measure
from run_matched_capture_trial import verified_cohort
from run_patch_subtype_trial import AGRI_AUDIT, TAIWAN, card, previous_scores, ranking
from run_source_residual_trial import disease_margin
from run_subtype_trial import (
    CLASSES, EXPANDED, FIRST_RUN, RUN as SUBTYPE_RUN, Specialists, combine, development_metrics,
    measurement, read_rows, test_receipts, verified, write_rows,
)
from train_online_candidate import WORKSPACE, ManifestDataset, checked_checkpoint, read_json, sha256, write_json


RUN = WORKSPACE / "outputs" / "full_frame_trial"
FRACTIONS = (0.125, 0.25, 0.5, 0.75, 1.0)
SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FULL_TRANSFORM = transforms.Compose((transforms.ToTensor(), transforms.Normalize(MEAN, STD)))


class FrameDataset(ManifestDataset):
    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as raw:
            image = raw.convert("RGB")
        center = self.tf(image)
        factor = SIZE / max(image.size)
        dimensions = tuple(max(1, round(dimension * factor)) for dimension in image.size)
        resized = image.resize(dimensions, Image.Resampling.BILINEAR)
        full = ImageOps.pad(resized, (SIZE, SIZE), color=tuple(round(channel * 255) for channel in MEAN))
        return center, FULL_TRANSFORM(full), CLASSES.index(row["label"])


@torch.no_grad()
def encode_views(encoder, components, path, name):
    buffers = []
    for center, full, truth in loader(FrameDataset(path)):
        views = {}
        for label, batch in (("center", center), ("full", full), ("mirror", full.flip(-1))):
            _, dino, prior = encoder(move_images(batch, encoder.device))
            scores = previous_scores({"dino": dino.cpu(), "prior": prior.cpu()}, *components)
            views[label] = scores if label == "center" else disease_margin(scores)
        buffers.append({**views, "truth": truth})
    data = {key: torch.cat([batch[key] for batch in buffers]) for key in buffers[0]}
    print(f"Encoded {name}: {len(data['truth'])} images, three views per image", flush=True)
    return data


def fused(data, fraction, rule, view="both"):
    center = data["center"]
    average = (data["full"] + data["mirror"]) / 2 if view == "both" else data[view]
    if rule == "agree" and view == "both":
        same_sign = (data["full"] >= 0) == (data["mirror"] >= 0)
        average = torch.where(same_sign, average, disease_margin(center))
    return combine(center, average, fraction)


def select(encoder, protocol, components, early_lock, early):
    paths, _, _, _ = verified_cohort(protocol)
    RUN.mkdir(exist_ok=True)
    lock = RUN / "decision.json"
    if lock.exists():
        raise FileExistsError("Full-frame selection is already locked")
    plan = {"incumbent_sha256": early_lock, "script_sha256": sha256(Path(__file__)),
            "development_manifests": {name: sha256(path) for name, path in paths.items() if name != "augmented"},
            "fractions": list(FRACTIONS), "rules": ["mean", "agree"],
            "method": "Run the locked classifier on centered crop, square-padded full image, and horizontally flipped "
                      "full image. Change only the early-minus-late score. No extra parameter fitting.",
            "selection": "Keep both disease recalls and Macro-F1 on pooled development and Malik, and reduce subtype "
                         "errors. Require at least one single full-frame view to meet the same gates as the combined views."}
    plan_file = RUN / "protocol.json"
    if plan_file.exists():
        if read_json(plan_file) != plan:
            raise ValueError("The trial protocol has changed")
    else:
        write_json(plan_file, plan)
    selection_source = RUN / "selection_source.py"
    if not selection_source.exists():
        copyfile(Path(__file__), selection_source)
    verified(selection_source, plan["script_sha256"])
    field = encode_views(encoder, components, paths["development_field"], "development_field")
    malik = encode_views(encoder, components, paths["development_malik"], "development_malik")
    dev = {key: torch.cat((field[key], malik[key])) for key in field}
    malik_n = len(malik["truth"])
    start = development_metrics(dev["center"], dev, malik_n)
    expected = early["selections"][early["preferred_by_development"]]["metrics"]
    if start["confusion"] != expected["confusion"] or start["malik"]["confusion"] != expected["malik"]["confusion"]:
        raise ValueError("Incumbent differs from archived development predictions")
    candidates = []
    for rule in plan["rules"]:
        for fraction in FRACTIONS:
            candidate = card(fused(dev, fraction, rule), dev, malik_n, start, 0, fraction, None)
            mirrors = [card(fused(dev, fraction, rule, view), dev, malik_n, start, 0, fraction, None)["record"]
                       for view in ("full", "mirror")]
            candidate["record"].update(rule=rule, single_view_passes=sum(item["eligible"] for item in mirrors))
            candidate["single_view_checks"] = mirrors
            candidates.append(candidate)
    passing = [item for item in candidates if item["record"]["eligible"] and item["record"]["single_view_passes"]]
    chosen = max(passing or candidates, key=ranking)
    decision = {"preferred_by_development": "full_frame" if passing else "incumbent", "full_frame": chosen,
                "incumbent": start, "protocol_sha256": sha256(plan_file), "configurations": len(candidates)}
    write_rows(RUN / "development_search.csv", [candidate["record"] for candidate in candidates])
    write_json(lock, decision)
    print(f"Full-frame decision: {decision['preferred_by_development']}, {chosen['record']}", flush=True)


def evaluate(encoder, protocol, components, early_lock):
    _, _, forbidden, _ = verified_cohort(protocol)
    lock = RUN / "decision.json"
    decision = read_json(lock)
    plan = read_json(verified(RUN / "protocol.json", decision["protocol_sha256"]))
    verified(RUN / "selection_source.py", plan["script_sha256"])
    if early_lock != plan["incumbent_sha256"]:
        raise ValueError("Incumbent changed after selection")
    audits = read_json(EXPANDED / "evaluation_freeze.json")["external_audits"]
    tests = {**test_receipts(protocol), **read_json(TAIWAN)["tests"], **read_json(AGRI_AUDIT)["tests"], **audits}
    for name, entry in tests.items():
        path = verified(Path(entry["path"]), entry["sha256"])
        rows = read_rows(path)
        ids = {row["sha256"] for row in rows}
        if len(ids) != len(rows) or ids & forbidden:
            raise ValueError(f"Duplicate or overlapping evaluation images: {name}")
        result = RUN / f"{name}_results.json"
        if result.exists():
            saved = read_json(result)
            if saved["decision_sha256"] != sha256(lock) or saved["manifest_sha256"] != entry["sha256"]:
                raise ValueError(f"Evaluation has a different freeze: {name}")
            continue
        data = encode_views(encoder, components, path, name)
        gate = decision["full_frame"]["record"]
        models = {"incumbent": data["center"], "full_frame": fused(data, gate["fraction"], gate["rule"])}
        metrics, predictions = {}, []
        truth = data["truth"]
        for key, logits in models.items():
            if truth.unique().numel() == len(CLASSES):
                metrics[key], labels, subtype = measurement(logits, truth)
            else:
                metrics[key], labels = measure(logits, truth)
                subtype = logits[:, 1:].argmax(1) + 1
            predictions.extend({"model": key, "sha256": row["sha256"], "true_label": row["label"],
                                "prediction": CLASSES[label], "subtype_if_diseased": CLASSES[disease]}
                               for row, label, disease in zip(rows, labels.tolist(), subtype.tolist()))
            print(f"{name} {key} accuracy={metrics[key]['accuracy']:.5f}", flush=True)
        write_rows(RUN / f"{name}_predictions.csv", predictions)
        write_json(result, {"decision_sha256": sha256(lock), "manifest_sha256": entry["sha256"],
                            "evaluation_script_sha256": sha256(Path(__file__)),
                            "role": "external_audit" if name in audits else entry.get("role", "historical_replay"),
                            "models": metrics})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "evaluate"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lock = EARLY_RUN / "decision.json"
    early = read_json(lock)
    health = read_json(verified(HEALTH_RUN / "decision.json", early["health_decision_sha256"]))
    subtype = read_json(verified(SUBTYPE_RUN / "decision.json", health["subtype_decision_sha256"]))
    health_key, early_key = health["preferred_by_development"], early["preferred_by_development"]
    components = (load_heads(subtype), load_gate(health_key, health["selections"][health_key]),
                  restore(early_key, early["selections"][early_key]))
    protocol = read_json(SUBTYPE_RUN / "protocol.json")
    encoder = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", protocol["parent_sha256"]), device)
    if args.phase == "select":
        select(encoder, protocol, components, sha256(lock), early)
    else:
        evaluate(encoder, protocol, components, sha256(lock))


if __name__ == "__main__":
    main()
