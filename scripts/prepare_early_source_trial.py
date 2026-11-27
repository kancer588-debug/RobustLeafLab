"""Build a matched early-blight training comparison and a separate field test."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageOps

from prepare_online_run import (
    EXTERNAL_MANIFESTS, FIELDS, MANIFEST_DIR, SOURCE_DIR, NearDuplicateIndex,
    fingerprint, rows, sha256,
)


TRIAL = MANIFEST_DIR.parent / "early_source_trial"
BACKBONE = MANIFEST_DIR.parent / "backbone_trial"
PUBLISHER_CATALOG = SOURCE_DIR / "imtiaz-e205d0ab-a9e6-4e7b-a734-d50b0351249c.json"
TOMATO_CAPTURE = re.compile(r"_(\d{13})(?:_|\.)")
IMTIAZ_CAPTURE = re.compile(r"^IMG_\d+", re.IGNORECASE)
IMTIAZ_LABELS = {1: "early_blight", 2: "healthy", 3: "late_blight"}
MIN_TEST_PER_CLASS = 10
EARLY = "early_blight"
PUBLISHERS = {
    "TOM2024": "https://data.mendeley.com/datasets/3d4yg89rtr/1",
    "Imtiaz v2": "https://data.mendeley.com/datasets/bpfd9cns5g/2",
}
STAGED_FIELDS = (*FIELDS, "capture_date", "publisher_label", "review_id")


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, records, fields):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def parquet_rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
        yield from batch.to_pylist()


def select_tom2024(path):
    metadata = json.loads(pq.ParquetFile(path).schema_arrow.metadata[b"huggingface"])
    labels = metadata["info"]["features"]["label"]["names"]
    variants = defaultdict(list)
    by_frame = defaultdict(set)
    counts = Counter()
    for row in parquet_rows(path):
        if row["crop_type"] != "Tomato":
            continue
        label = labels[row["label"]]
        image = row["image"]
        match = TOMATO_CAPTURE.search(image["path"])
        if not match:
            counts["unparsed_target"] += label == "alternaria-d"
            continue
        frame = match[1]
        by_frame[frame].add(label)
        if label == "alternaria-d":
            counts["publisher_target_rows"] += 1
            variants[frame].append(image)
    selected = []
    for frame, images in sorted(variants.items()):
        if by_frame[frame] != {"alternaria-d"}:
            counts["conflicting_capture_frames"] += 1
            continue
        image = max(images, key=lambda value: (len(value["bytes"]), value["path"]))
        selected.append((frame, EARLY, "alternaria-d", image))
    counts["same_frame_variants"] = sum(len(images) - 1 for images in variants.values())
    return selected, counts


def select_imtiaz(path):
    catalog = json_read(PUBLISHER_CATALOG)
    released = {item["content_details"]["sha256_hash"]: item["filename"] for item in catalog}
    catalog_groups = Counter(match[0].lower() for item in catalog
                             if (match := IMTIAZ_CAPTURE.match(item["filename"])))
    by_frame = defaultdict(set)
    variants = defaultdict(list)
    counts = Counter()
    for row in parquet_rows(path):
        image = row["image"]
        counts["mirror_rows"] += 1
        match = IMTIAZ_CAPTURE.match(image["path"])
        if not match:
            counts["unparsed_frame"] += 1
            continue
        frame = match[0].lower()
        by_frame[frame].update(row["objects"]["categories"])
        digest = hashlib.sha256(image["bytes"]).hexdigest()
        publisher_name = released.get(digest, "")
        publisher_frame = IMTIAZ_CAPTURE.match(publisher_name)
        if publisher_frame and publisher_frame[0].lower() == frame:
            counts["publisher_byte_matches"] += 1
            variants[frame].append({**image, "path": publisher_name})
    selected = []
    for frame, images in sorted(variants.items()):
        labels = by_frame[frame]
        if len(labels) != 1 or not labels <= IMTIAZ_LABELS.keys():
            counts["unusable_single_export_class"] += 1
            continue
        label = next(iter(labels))
        image = max(images, key=lambda value: (len(value["bytes"]), value["path"]))
        selected.append((frame, IMTIAZ_LABELS[label], str(label), image))
    counts["catalog_images"] = len(catalog)
    counts["raw_catalog_frames"] = len(catalog_groups)
    counts["single_raw_export_frames"] = sum(n == 1 for n in catalog_groups.values())
    return selected, counts


def reference_identities():
    history = json_read(MANIFEST_DIR / "historical_index.json")
    for path, expected in history["manifests"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Historical manifest has changed: {path}")
    exact = set(history["exact"])
    near = NearDuplicateIndex()
    for signature in history["dhash"]:
        near.add(signature)
    additional = [
        *(MANIFEST_DIR / f"{name}.csv" for name in ("train", "validation", "test")),
        *(BACKBONE / f"{name}.csv" for name in
          ("train", "development_field", "development_malik", "fresh_test", "plantwild_staged")),
        *(SOURCE_DIR / f"{name}-raw.csv" for name in ("plantcity", "malik")),
    ]
    scanned = set()
    for path in additional:
        for row in rows(path):
            digest = row["sha256"]
            exact.add(digest)
            if digest not in scanned:
                signature, _ = fingerprint(row["path"], resize_for_hash=True)
                near.add(signature[0])
                scanned.add(digest)
    for path in EXTERNAL_MANIFESTS:
        exact.update(row["sha256"] for row in rows(path))
    return exact, near


def stage_records(source, candidates, exact, near):
    destination = TRIAL / "images" / source.split()[0].lower()
    destination.mkdir(parents=True)
    staged, rejected = [], []
    for frame, label, publisher_label, image in candidates:
        digest = hashlib.sha256(image["bytes"]).hexdigest()
        reason = ""
        if digest in exact:
            reason = "Exact duplicate"
        else:
            try:
                with Image.open(BytesIO(image["bytes"])) as frame_image:
                    frame_image.load()
                path = destination / f"{digest}.jpg"
                path.write_bytes(image["bytes"])
                signatures, (width, height) = fingerprint(path, mirrored=True)
                if any(near.matches(signature) for signature in signatures):
                    reason = "Perceptual duplicate or close match"
            except OSError:
                reason = "Unreadable image"
        if reason:
            rejected.append({"source": source, "origin": image["path"], "sha256": digest,
                             "capture_group": frame, "reason": reason})
            continue
        exact.add(digest)
        near.add(signatures[0])
        date = ""
        if source == "TOM2024":
            date = datetime.fromtimestamp(int(frame) / 1000, UTC).date().isoformat()
        staged.append(dict(
            path=str(path), mask_path="", label=label, sha256=digest, source=source, dataset=source,
            origin=image["path"], capture_group=frame, width=width, height=height,
            capture_date=date, publisher_label=publisher_label, review_id=len(staged) + 1,
        ))
    return staged, rejected


def review_sheets(source, records):
    folder = TRIAL / "review"
    folder.mkdir(exist_ok=True)
    per_sheet = 30
    for start in range(0, len(records), per_sheet):
        batch = records[start:start + per_sheet]
        canvas = Image.new("RGB", (1100, 164 * ((len(batch) + 5) // 6)), "#f5f5f5")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(batch):
            x, y = 183 * (index % 6) + 4, 164 * (index // 6) + 4
            with Image.open(row["path"]) as picture:
                preview = ImageOps.contain(ImageOps.exif_transpose(picture).convert("RGB"), (172, 132))
            canvas.paste(preview, (x + (172 - preview.width) // 2, y))
            draw.text((x + 4, y + 134), f"{row['review_id']:03d} {row['label']}", fill="#171717")
        canvas.save(folder / f"{source.lower()}-{start // per_sheet + 1:02d}.jpg", quality=88)


def stage(tom2024_path, imtiaz_path):
    exact, near = reference_identities()
    sources = [
        ("Imtiaz v2", imtiaz_path, select_imtiaz, "imtiaz"),
        ("TOM2024", tom2024_path, select_tom2024, "tom2024"),
    ]
    metadata, exclusions = {}, []
    for source, path, select, stem in sources:
        candidates, counts = select(path)
        accepted, rejected = stage_records(source, candidates, exact, near)
        write_csv(TRIAL / f"{stem}_staged.csv", accepted, STAGED_FIELDS)
        review_sheets(stem, accepted)
        metadata[stem] = {
            "publisher": PUBLISHERS[source], "parquet_sha256": sha256(path), "screening": dict(counts),
            "before_duplicate_filter": len(candidates), "after_duplicate_filter": len(accepted),
            "counts_after_filter": dict(Counter(row["label"] for row in accepted)),
        }
        exclusions.extend(rejected)
        print(stem, metadata[stem], flush=True)
    write_csv(TRIAL / "duplicates.csv", exclusions, (*FIELDS, "reason"))
    (TRIAL / "source_inventory.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def freeze(review_path):
    base = MANIFEST_DIR / "train.csv"
    if sha256(base) != json_read(MANIFEST_DIR / "freeze.json")["manifest_sha256"]["train"]:
        raise ValueError("Original fit manifest has changed")
    approved = json_read(review_path)
    control = rows(base)
    replaceable = [i for i, row in enumerate(control)
                   if row["source"] == "plantvillage" and row["label"] == EARLY]
    test = [row for row in rows(TRIAL / "imtiaz_staged.csv")
            if int(row["review_id"]) in approved["imtiaz"]]
    pool = [row for row in rows(TRIAL / "tom2024_staged.csv")
            if int(row["review_id"]) in approved["tom2024"]]
    if len(pool) != len(set(approved["tom2024"])) or len(test) != len(set(approved["imtiaz"])):
        raise ValueError("Reviewed IDs are missing or repeated")
    # Spread selected training frames across capture dates before taking a second frame from each date.
    by_date = defaultdict(list)
    for row in sorted(pool, key=lambda row: row["sha256"]):
        by_date[row["capture_date"]].append(row)
    replacements = []
    for rank in range(max(map(len, by_date.values()), default=0)):
        replacements.extend(items[rank] for _, items in sorted(by_date.items()) if len(items) > rank)
    replacements = replacements[:len(replaceable)]
    if not replacements:
        raise ValueError("No reviewed training replacements")
    test_counts = Counter(row["label"] for row in test)
    if any(test_counts[label] < MIN_TEST_PER_CLASS for label in ("healthy", EARLY, "late_blight")):
        raise ValueError(f"New test has fewer than {MIN_TEST_PER_CLASS} images in a class: {test_counts}")
    paired = control.copy()
    removed = []
    for index, incoming in zip(replaceable, replacements):
        removed.append(control[index])
        paired[index] = incoming
    partitions = {"control": control, "replacement": paired, "fresh_test": test, "replaced": removed}
    test_hashes = {row["sha256"] for row in test}
    if (len(test_hashes) != len(test) or any(test_hashes & {row["sha256"] for row in arm}
                                           for arm in (control, paired))):
        raise ValueError("Test overlaps a training arm")
    if Counter(row["label"] for row in control) != Counter(row["label"] for row in paired):
        raise ValueError("Training class counts must match")
    for name, records in partitions.items():
        write_csv(TRIAL / f"{name}.csv", records, STAGED_FIELDS)
    frozen = {
        "sha256": {name: sha256(TRIAL / f"{name}.csv") for name in partitions},
        "review_sha256": sha256(review_path), "replacements": len(replacements),
        "training_counts": dict(Counter(row["label"] for row in control)),
        "fresh_test_counts": dict(test_counts), "training_capture_dates": len(
            {row["capture_date"] for row in replacements}
        ),
        "publishers": PUBLISHERS,
    }
    with (TRIAL / "freeze.json").open("x", encoding="utf-8") as stream:
        json.dump(frozen, stream, indent=2)
        stream.write("\n")
    print(json.dumps(frozen, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("stage")
    staging.add_argument("tom2024", type=Path)
    staging.add_argument("imtiaz", type=Path)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("review", type=Path)
    args = parser.parse_args()
    if args.command == "stage":
        stage(args.tom2024, args.imtiaz)
    else:
        freeze(args.review)


if __name__ == "__main__":
    main()
