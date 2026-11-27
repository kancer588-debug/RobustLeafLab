"""Stage TOM2024 subtype additions and a publisher-held-out field test."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256 as digest
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image

from prepare_online_run import FIELDS
from stage_local_leaf_sources import (
    EXPANDED, MANIFEST_DIR, TRIAL as PREVIOUS, image_hashes, reference_images,
    rows, sha256, write_csv, write_json,
)


ROOT = MANIFEST_DIR.parent / "subtype_trial"
TOM_URL = "https://data.mendeley.com/datasets/3d4yg89rtr/1"
FIELD_URL = "https://data.mendeley.com/datasets/93h9p62kg4/1"
TOM_LABELS = {"alternaria-d": "early_blight", "tomato-late-blight-d": "late_blight"}
FIELD_LABELS = {"Healthy": "healthy", "Early Blight": "early_blight", "Late Blight": "late_blight"}
FRAME = re.compile(r"_(\d{13})(?:_|\.)")
SERIAL = re.compile(r"^IMG_(\d+)(?:[ _.\-]|$)", re.IGNORECASE)
TARGET_COUNTS = {"healthy": 103, "early_blight": 204, "late_blight": 166}
MIN_PER_CLASS = 15
SERIAL_BLOCK = 5


def with_image(data, label, origin, group, source):
    checksum = digest(data).hexdigest()
    destination = ROOT / "images" / source.replace(" ", "_").lower()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{checksum}{PurePosixPath(origin).suffix.lower()}"
    if path.exists():
        if sha256(path) != checksum:
            raise ValueError(f"Staged image has changed: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    return {"path": str(path), "label": label, "sha256": checksum, "source": source,
            "origin": origin, "capture_group": group}


def tom_candidates(path):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    features = json.loads(parquet.schema_arrow.metadata[b"huggingface"])["info"]["features"]
    labels = features["label"]["names"]
    classes = defaultdict(set)
    variants = {}
    for batch in parquet.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            if row["crop_type"] != "Tomato":
                continue
            image = row["image"]
            match = FRAME.search(image["path"])
            if not match:
                continue
            frame = match[1]
            label = labels[row["label"]]
            classes[frame].add(label)
            if label in TOM_LABELS:
                previous = variants.get(frame)
                if previous is None or (len(image["bytes"]), image["path"]) > (len(previous["bytes"]), previous["path"]):
                    variants[frame] = image
    return [with_image(image["bytes"], TOM_LABELS[next(iter(classes[frame]))], image["path"],
                       f"tom2024_frame_{frame}", "TOM2024 subtype")
            for frame, image in sorted(variants.items()) if len(classes[frame]) == 1]


def field_candidates(path):
    result = []
    with ZipFile(path) as archive:
        for member in archive.infolist():
            origin = PurePosixPath(member.filename)
            label = FIELD_LABELS.get(origin.parent.name)
            if label is None or origin.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            serial = SERIAL.match(origin.name)
            if serial is None:
                raise ValueError(f"No camera serial in {origin}")
            content = archive.read(member)
            with Image.open(BytesIO(content)) as shot:
                metadata = shot.getexif()
                camera = metadata.get(272, "unknown_camera")
                date = metadata.get_ifd(34665).get(36867, "unknown_date")
            group = f"{camera}|{str(date)[:10]}|{int(serial[1]) // SERIAL_BLOCK}"
            result.append(with_image(content, label, member.filename, group, "Khagan Charabag"))
    counts = dict(Counter(row["label"] for row in result))
    if counts != TARGET_COUNTS:
        raise ValueError(f"The local archive has different class counts from the publisher: {counts}")
    return result


def count(records):
    return dict(Counter(row["label"] for row in records))


def stage(tom, field):
    split = ROOT / "staged.json"
    if split.exists():
        raise FileExistsError(split)
    fit = tom_candidates(tom)
    test = field_candidates(field)
    print(f"Loaded {len(fit)} TOM2024 frames and {len(test)} Khagan/Charabag images", flush=True)
    exact, differences, perceptual = reference_images(True, include_original=True)
    test_existing = rows(PREVIOUS / "taiwan_test.csv") + rows(PREVIOUS / "chen_review.csv")
    for row in test_existing:
        exact.add(row["sha256"])
        d, p = image_hashes(row["path"])[0]
        differences.add(d)
        perceptual.add(p)
    references = len(exact)
    ids = Counter(row["sha256"] for row in fit + test)
    test_origins = Counter((row["capture_group"], SERIAL.match(PurePosixPath(row["origin"]).name)[1]) for row in test)
    seen_groups = set()
    outputs, exclusions = {}, {}
    for name, cohort in (("field", test), ("tom2024", fit)):
        accepted, reasons = [], Counter()
        ordered = sorted(cohort, key=lambda row: (row["label"], row["sha256"]))
        with ThreadPoolExecutor(max_workers=6) as pool:
            for row, hashes in zip(ordered, pool.map(image_hashes, [item["path"] for item in ordered])):
                key = (row["capture_group"], row["label"])
                if row["sha256"] in exact or ids[row["sha256"]] > 1:
                    reasons["exact_or_reserved"] += 1
                    continue
                if any(differences.matches(d) or perceptual.matches(p) for d, p in hashes):
                    reasons["perceptual_overlap"] += 1
                    continue
                if name == "field":
                    frame_id = SERIAL.match(PurePosixPath(row["origin"]).name)[1]
                    if test_origins[(row["capture_group"], frame_id)] > 1 or key in seen_groups:
                        reasons["camera_sequence_thinning"] += 1
                        continue
                    seen_groups.add(key)
                accepted.append({**row, "review_id": len(accepted) + 1})
                exact.add(row["sha256"])
                differences.add(hashes[0][0])
                perceptual.add(hashes[0][1])
        write_csv(ROOT / f"{name}_review.csv", accepted)
        make_sheets_for_trial(accepted, name)
        outputs[name] = {"before": count(cohort), "review": count(accepted)}
        exclusions[name] = dict(reasons)
        print(name, json.dumps({**outputs[name], "excluded": exclusions[name]}), flush=True)
    write_json(split, {"tom2024_parquet_sha256": sha256(tom), "field_archive_sha256": sha256(field),
                       "historical_reference_identities": references, "counts": outputs, "excluded": exclusions,
                       "field_publisher": FIELD_URL, "tom2024_publisher": TOM_URL, "serial_block": SERIAL_BLOCK})


def make_sheets_for_trial(records, name):
    from PIL import ImageDraw, ImageOps

    location = ROOT / "review"
    location.mkdir(exist_ok=True)
    for start in range(0, len(records), 30):
        block = records[start:start + 30]
        canvas = Image.new("RGB", (1080, 146 * ((len(block) + 5) // 6)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(block):
            x, y = index % 6 * 180 + 4, index // 6 * 146 + 3
            with Image.open(row["path"]) as image:
                image.draft("RGB", (420, 420))
                thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (168, 112))
            canvas.paste(thumb, (x + (168 - thumb.width) // 2, y))
            draw.text((x, y + 114), f"{row['review_id']} {row['label']}", fill="black")
            draw.text((x, y + 128), PurePosixPath(row["origin"]).name[:24], fill="black")
        canvas.save(location / f"{name}-{start // 30 + 1:02d}.jpg", quality=87)


def freeze(review):
    decisions = json.loads(review.read_text(encoding="utf-8"))
    field, tom = (rows(ROOT / f"{kind}_review.csv") for kind in ("field", "tom2024"))
    test_rejections = set(decisions["test_rejected"])
    fit_accepts = set(decisions["tom_accepted"])
    if test_rejections - {int(row["review_id"]) for row in field} or fit_accepts - {int(row["review_id"]) for row in tom}:
        raise ValueError("Unknown review ID in the decision")
    test = [row for row in field if int(row["review_id"]) not in test_rejections]
    additions = [row for row in tom if int(row["review_id"]) in fit_accepts]
    if not additions:
        raise ValueError("The augmented trial needs one or more visually reviewed additions")
    base = rows(EXPANDED / "expanded_train.csv")
    if {row["sha256"] for row in test} & {row["sha256"] for row in base + additions}:
        raise ValueError("A test image appears in training")
    cohorts = {"training": base + additions, "additions": additions}
    if min(count(test).get(label, 0) for label in FIELD_LABELS.values()) >= MIN_PER_CLASS:
        cohorts["test"] = test
    paths = {name: ROOT / f"{name}.csv" for name in cohorts}
    for name, records in cohorts.items():
        with paths[name].open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(*FIELDS, "capture_date", "review_id"), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    write_json(ROOT / "freeze.json", {
        "files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "counts": {name: count(records) for name, records in cohorts.items()},
        "review_sha256": sha256(review), "stage_sha256": sha256(ROOT / "staged.json"),
        "scope": "The field publisher may enter the test only if all three categories survive screening. "
                 "TOM2024 additions are used only for subtype training.",
    })
    print("Frozen test:", count(cohorts.get("test", [])), "fitting additions:", count(additions), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    staging = commands.add_parser("stage")
    staging.add_argument("tom_parquet", type=Path)
    staging.add_argument("field_archive", type=Path)
    final = commands.add_parser("freeze")
    final.add_argument("review", type=Path)
    args = parser.parse_args()
    if args.phase == "stage":
        stage(args.tom_parquet, args.field_archive)
    else:
        freeze(args.review)


if __name__ == "__main__":
    main()
