"""Probe the frozen ROI candidate using development images and box-check groups."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import roi_localizer
from roi_localizer import GRID, LOCAL_FRACTION, area_mask, crop, fit_canvas, split_regions
from run_backbone_trial import move_images
from run_dino_health_trial import RUN as HEALTH_RUN, load_gate
from run_early_review_trial import RUN as EARLY_RUN, restore
from run_late_protection_trial import load_heads
from run_matched_capture_trial import verified_cohort
from run_patch_subtype_trial import previous_scores
from run_roi_trial import RUN as ROI_RUN, adjust, global_embedding, load_probe
from run_subtype_trial import CLASSES, FIRST_RUN, RUN as SUBTYPE_RUN, Specialists, verified, write_rows
from stage_roi_training import STAGE, read
from train_online_candidate import WORKSPACE, ManifestDataset, checked_checkpoint, read_json, sha256, write_json


RUN = WORKSPACE / "outputs" / "roi_debug"
BATCH = 12


class Images(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]["path"]) as source:
            return fit_canvas(source.convert("RGB"))


def peak_scores(encoder, head, images, layouts):
    tokens = encoder.dino.forward_features(images)["x_norm_patchtokens"]
    logits = head(tokens).reshape(len(images), -1)
    valid = torch.stack([area_mask(layout) for layout in layouts]).flatten(1).to(images.device)
    peak, position = logits.masked_fill(~valid, -torch.inf).max(1)
    points = torch.stack((position.remainder(GRID), position // GRID), dim=1).float().add(0.5).div(GRID)
    return peak.cpu(), points


def summary(values):
    scores = torch.tensor(values)
    quantiles = scores[torch.isfinite(scores)].quantile(torch.tensor([0.1, 0.5, 0.9]))
    return {"n": len(scores), "nonnegative": int((scores >= 0).sum()), "nonfinite": int((~scores.isfinite()).sum()),
            "p10": float(quantiles[0]), "median": float(quantiles[1]), "p90": float(quantiles[2])}


def development_scores(encoder, locator, rows):
    metrics = defaultdict(list)
    result = []
    for start, (images, layouts) in enumerate(DataLoader(Images(rows), batch_size=BATCH)):
        scores, _ = peak_scores(encoder, locator, images.to(encoder.device), layouts)
        for row, value in zip(rows[start * BATCH:], scores.tolist()):
            metrics[row["label"]].append(value)
            result.append({"sha256": row["sha256"], "label": row["label"], "capture_group": row["capture_group"],
                           "maximum_patch_logit": value, "roi_applied": value >= 0})
    print(f"Checked full inference locator on {len(result)} development images", flush=True)
    return result, {key: summary(values) for key, values in metrics.items()}


def boxes_on_canvas(boxes, layout):
    left, top, width, height = layout.tolist()
    return [(left + x0 * width, top + y0 * height, left + x1 * width, top + y1 * height)
            for x0, y0, x1, y1 in boxes]


def region_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def coverage(point, boxes):
    x, y = point
    half = LOCAL_FRACTION / 2
    x, y = (min(max(position, half), 1 - half) for position in (x, y))
    scores = []
    for x0, y0, x1, y1 in boxes:
        width = max(0, min(x + half, x1) - max(x - half, x0))
        height = max(0, min(y + half, y1) - max(y - half, y0))
        scores.append(width * height / region_area((x0, y0, x1, y1)))
    return max(scores)


def box_checks(encoder, locator, subtype, fraction, components, rows, transform):
    observations = []
    for row in rows:
        with Image.open(row["path"]) as src:
            image = src.convert("RGB")
        standard = move_images(transform(image).unsqueeze(0), encoder.device)
        canvas, layout = fit_canvas(image)
        canvas = canvas.unsqueeze(0).to(encoder.device)
        _, whole, prior = encoder(standard)
        base = previous_scores({"prior": prior.cpu(), "dino": whole.cpu()}, *components)
        peak, prediction = peak_scores(encoder, locator, canvas, layout.unsqueeze(0))
        boxes = boxes_on_canvas(row["target"], layout)
        largest = max(boxes, key=region_area)
        oracle = ((largest[0] + largest[2]) / 2, (largest[1] + largest[3]) / 2)
        centers = {"located": prediction[0].tolist(), "center": [0.5, 0.5], "annotation": list(oracle)}
        record = {"sha256": row["sha256"], "truth": row["label"], "capture_group": row["capture_group"],
                  "baseline_subtype": CLASSES[int(base[:, 1:].argmax(1)) + 1], "maximum_patch_logit": float(peak[0]),
                  "baseline_margin": float(base[0, 1] - base[0, 2]),
                  "point_strictly_in_a_box": any(x0 <= centers["located"][0] <= x1 and y0 <= centers["located"][1] <= y1
                                                  for x0, y0, x1, y1 in boxes)}
        for key, center in centers.items():
            image_crop = crop(canvas, canvas.new_tensor([center]))
            features = (whole + global_embedding(encoder, image_crop)).mul(0.5).cpu()
            take_crop = peak >= 0 if key == "located" else torch.ones(1, dtype=torch.bool)
            data = {"incumbent": base, "available": take_crop}
            evidence = subtype(features)
            scores = adjust("localized_crop", data, evidence, fraction)
            record[key + "_box_coverage"] = round(coverage(center, boxes), 5)
            record[key + "_probe_margin"] = float(evidence[0])
            record[key + "_fused_margin"] = float(scores[0, 1] - scores[0, 2])
            record[key + "_subtype"] = CLASSES[int(scores[:, 1:].argmax(1)) + 1]
        observations.append(record)
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-margins-only", action="store_true", help="Save crop comparison scores without re-encoding development")
    options = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decision_path = ROI_RUN / "decision.json"
    decision = read_json(decision_path)
    plan = read_json(verified(ROI_RUN / "protocol.json", decision["protocol_sha256"]))
    verified(Path(roi_localizer.__file__), plan["source_sha256"]["localizer"])
    early = read_json(verified(EARLY_RUN / "decision.json", decision["incumbent_sha256"]))
    health = read_json(verified(HEALTH_RUN / "decision.json", early["health_decision_sha256"]))
    subtype_record = read_json(verified(SUBTYPE_RUN / "decision.json", health["subtype_decision_sha256"]))
    health_key, early_key = health["preferred_by_development"], early["preferred_by_development"]
    components = (load_heads(subtype_record), load_gate(health_key, health["selections"][health_key]),
                  restore(early_key, early["selections"][early_key]))
    protocol = read_json(SUBTYPE_RUN / "protocol.json")
    paths, records, _, _ = verified_cohort(protocol)
    locator, _ = load_probe(ROI_RUN / "localizer.pt", decision["locator_sha256"], device)
    key = "localized_crop_s42"
    head, saved = load_probe(ROI_RUN / key / "probe.pt", decision["selections"][key]["weight_sha256"])
    annotations = read(verified(STAGE / "screened.json", plan["training_annotation_sha256"]))["records"]
    reviewed = read(verified(STAGE / "early_region_review.json", plan["early_region_review_sha256"]))["boxes_by_review_id"]
    fitting, checked = split_regions(annotations, reviewed, records["augmented"])
    if [row["sha256"] for row in checked] != plan["localizer_check_rows"]:
        raise ValueError("Box diagnostic frames changed")
    encoder = Specialists(checked_checkpoint(FIRST_RUN / "candidate.pt", protocol["parent_sha256"]), device)
    transform = ManifestDataset(paths["development_malik"]).tf
    metrics_path = RUN / ("box_margins.csv" if options.box_margins_only else "diagnostics.json")
    if metrics_path.exists():
        raise FileExistsError("This diagnostic is already archived")
    RUN.mkdir(exist_ok=True)
    with torch.no_grad():
        checks = box_checks(encoder, locator, head, saved["fraction"], components, checked, transform)
        if options.box_margins_only:
            write_rows(metrics_path, checks)
            return
        score_rows, by_label = development_scores(encoder, locator, records["development_field"] + records["development_malik"])
    areas = {label: [region_area(box) for row in fitting if row["label"] == label for box in row["target"]]
             for label in CLASSES[1:]}
    write_rows(RUN / "box_checks.csv", checks)
    write_rows(RUN / "development_peaks.csv", score_rows)
    result = {"roi_decision_sha256": sha256(decision_path), "development_peaks_by_class": by_label,
              "median_training_box_area_fraction": {label: float(torch.tensor(values).quantile(0.5)) for label, values in areas.items()},
              "strict_point_hits": sum(row["point_strictly_in_a_box"] for row in checks), "boxes_checked": len(checks),
              "correct_conditional_subtype": {key: sum(row[key] == row["truth"] for row in checks)
                                              for key in ("baseline_subtype", "located_subtype", "center_subtype", "annotation_subtype")},
              "covers_half_or_more_of_one_box": {key: sum(row[key + "_box_coverage"] >= 0.5 for row in checks)
                                                 for key in ("located", "center", "annotation")}}
    write_json(metrics_path, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
