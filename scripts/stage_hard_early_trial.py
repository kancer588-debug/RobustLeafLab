"""Review source-labeled early-blight training candidates without opening test pixels."""

import argparse
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageOps

from prepare_online_run import FIELDS, rows, sha256, write_rows
from stage_local_leaf_sources import image_hashes
from stage_roi_training import STAGE, read, references, write_once
from train_online_candidate import WORKSPACE


DATA = WORKSPACE / "datasets"
ROOT = DATA / "hard_early_trial"
FRAME = re.compile(r"(?:_|/)(\d{13})(?:_|\.)")
TOM = DATA / "early_source_trial" / "tom2024_staged.csv"
CHEN = DATA / "local_leaf_trial" / "chen_review.csv"


def candidates():
    data = []
    raw = ROOT / "publisher_originals.csv"
    if raw.exists():
        data.extend(rows(raw))
    for row in rows(TOM):
        if row["label"] != "early_blight" or row["publisher_label"] != "alternaria-d":
            raise ValueError("The TOM2024 early candidate catalog differs")
        data.append({**row, "capture_group": f"TOM2024|{row['capture_date']}", "view_type": "publisher_labeled_crop"})
    for row in rows(CHEN):
        source = PurePosixPath(row["origin"])
        if row["label"] == "early_blight" and source.parts[:3] == ("Tomato-Disease", "train", "images"):
            camera = source.name.split("_IMG_")[0].split("_DSC")[0]
            if len(camera) > 30:
                camera = "publisher_train_unspecified"
            data.append({**row, "capture_group": f"Chen|{camera}", "view_type": "publisher_detection_training_image"})
    return data


def extra_references():
    extra = list(read(STAGE / "screened.json")["records"])
    for name in ("view_ablation/plantseg/test.csv", "health_rescue_trial/healthy_fit_reviewed.csv",
                 "health_rescue_trial/early_development.csv", "early_override_trial/development.csv",
                 "backbone_trial/development_field.csv", "backbone_trial/development_malik.csv"):
        extra.extend(rows(DATA / name))
    return extra


def contact_sheets(records):
    side, columns, page_size = 134, 7, 35
    for start in range(0, len(records), page_size):
        group = records[start:start + page_size]
        canvas = Image.new("RGB", (side * columns, (side + 18) * ((len(group) + columns - 1) // columns)), "white")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(group):
            x, y = i % columns * side, i // columns * (side + 18)
            with Image.open(row["path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((side, side))
            canvas.paste(image, (x + (side - image.width) // 2, y + (side - image.height) // 2))
            draw.text((x + 3, y + side + 2), f"{row['review_id']:03d} {row['source'].split()[0]}", fill="black")
        canvas.save(ROOT / f"review_{start // page_size + 1}.jpg", quality=88)


def stage():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "inventory.json").exists():
        raise FileExistsError("The candidate inventory already exists")
    exact, dhash, phash, unreadable = references()
    frames = set()
    for row in rows(DATA / "matched_capture_trial" / "training.csv"):
        if found := FRAME.search(row.get("origin") or row["path"]):
            frames.add(found[1])
    for row in extra_references():
        exact.add(row["sha256"])
        a, b = image_hashes(row["path"])[0]
        dhash.add(a)
        phash.add(b)
    keep, rejected = [], Counter()
    for row in candidates():
        source = row["source"]
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"A source image changed: {row['path']}")
        frame = FRAME.search(row["origin"])
        signatures = image_hashes(row["path"])
        if row["sha256"] in exact or (frame and frame[1] in frames) or any(
                dhash.matches(a) or phash.matches(b) for a, b in signatures):
            rejected[source] += 1
            continue
        exact.add(row["sha256"])
        if frame:
            frames.add(frame[1])
        for a, b in signatures:
            dhash.add(a)
            phash.add(b)
        keep.append({**row, "review_id": len(keep) + 1,
                     "dataset": "tom2024" if source.startswith("TOM") else "chen_early"})
    if not keep:
        raise RuntimeError("No supplemental candidates passed identity screening")
    write_once(ROOT / "inventory.json", {"records": keep, "rejected_duplicates": dict(rejected), "unreadable": unreadable})
    contact_sheets(keep)
    print("Screened supplemental early candidates", dict(Counter(r["source"] for r in keep)), "rejected", dict(rejected), flush=True)


def freeze(exclusions):
    inventory = ROOT / "inventory.json"
    data = read(inventory)["records"]
    if not set(exclusions) <= {row["review_id"] for row in data}:
        raise ValueError("An excluded photo ID is absent from visual review")
    keep = [row for row in data if row["review_id"] not in exclusions]
    if not keep:
        raise RuntimeError("Visual review excluded the entire pool")
    corrected = []
    for row in keep:
        with Image.open(row["path"]) as source:
            row["width"], row["height"] = source.size
            if source.getexif().get(274, 1) == 1:
                continue
            output = ROOT / "orientation"
            output.mkdir(exist_ok=True)
            name = output / f"photo_{row['review_id']:03d}.png"
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.save(name)
            row.update(path=str(name), sha256=sha256(name), width=image.width, height=image.height)
            corrected.append(row["review_id"])
    path = ROOT / "reviewed_training_pool.csv"
    write_rows(path, keep, (*FIELDS, "review_id", "view_type"))
    write_once(ROOT / "review_freeze.json", {"sha256": sha256(path), "inventory_sha256": sha256(inventory),
                      "counts": dict(Counter(r["source"] for r in keep)), "visual_exclusions": sorted(exclusions),
                      "orientation_normalized": corrected, "role": "Training candidates only; selected before any model training."})
    print("Frozen visually screened early pool", len(keep), "by source", dict(Counter(r["source"] for r in keep)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "freeze"))
    parser.add_argument("--exclude", type=int, nargs="*", default=[])
    args = parser.parse_args()
    stage() if args.phase == "stage" else freeze(args.exclude)
