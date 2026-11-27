"""Freeze source-disjoint manifests before running the AC42 fine-tune."""

import argparse
import os
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE_DIR = WORKSPACE / "datasets" / "online_sources"
MANIFEST_DIR = WORKSPACE / "datasets" / "independent_test"
REFERENCE = Path(os.environ.get("ROBUSTLEAFLAB_REFERENCE", WORKSPACE / "reference"))
PREVIOUS = Path(os.environ.get("ROBUSTLEAFLAB_PRIOR_WORKSPACE", WORKSPACE / "prior_workspace"))
FIT = REFERENCE / "outputs/train_improve_late/ac42_field_ft_s42/train.csv"
REFERENCE_MANIFESTS = [
    REFERENCE / name for name in (
        "data/val.csv", "data/test.csv", "data_v5/train.csv", "data_v5/field_val.csv",
        "data_v5/field_test.csv", "data_v5/greenhouse_healthy.csv",
        "outputs/train_improve_late/ac42_field_ft_s42/external_audit.csv",
    )
] + [FIT, PREVIOUS / "work/ac42-real-scene-spot-check/manifest.csv"]
EXTERNAL_MANIFESTS = [REFERENCE / "data_v5/new_field_test.csv", REFERENCE / "data/external_plantdoc.csv"]
LABELS = {"healthy", "early_blight", "late_blight"}
MALIK_CAMERA_NAME = re.compile(r"IMG_20251102_\d{6}\.jpg", re.IGNORECASE)
FIELDS = ("path", "mask_path", "label", "sha256", "source", "dataset", "origin",
          "capture_group", "width", "height")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/reference_source/scripts"))
from prepare_multisource import dhash


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def fingerprint(path, *, mirrored=False, resize_for_hash=False):
    with Image.open(path) as image:
        if resize_for_hash:
            image.draft("RGB", (320, 320))
        oriented = ImageOps.exif_transpose(image)
        oriented.load()
        signatures = [dhash(oriented)]
        if mirrored:
            signatures.append(dhash(ImageOps.mirror(oriented)))
        return signatures, oriented.size


def create_history_index():
    manifests = {str(path): sha256(path) for path in REFERENCE_MANIFESTS}
    records = {row["sha256"]: row["path"] for path in REFERENCE_MANIFESTS for row in rows(path)}
    failures = []
    values = []

    def scan(record):
        digest, filename = record
        try:
            return digest, fingerprint(filename, resize_for_hash=True)[0][0], None
        except (OSError, ValueError) as exc:
            return digest, None, str(exc)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for index, result in enumerate(pool.map(scan, records.items()), start=1):
            digest, signature, error = result
            if signature is None:
                failures.append({"sha256": digest, "error": error})
            else:
                values.append(signature)
            if index % 1500 == 0:
                print("Historical identities screened:", index, flush=True)
    data = {"manifests": manifests, "exact": list(records), "dhash": sorted(set(values)),
            "unreadable": failures}
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "historical_index.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


class NearDuplicateIndex:
    """At Hamming distance two, one of three disjoint bit ranges must match."""

    def __init__(self):
        self.bins = defaultdict(set)

    @staticmethod
    def bins_for(signature):
        return ((0, signature & 0x1FFFFF), (1, (signature >> 21) & 0x1FFFFF),
                (2, signature >> 42))

    def add(self, signature):
        for key in self.bins_for(signature):
            self.bins[key].add(signature)

    def matches(self, signature):
        return any((signature ^ value).bit_count() <= 2
                   for key in self.bins_for(signature) for value in self.bins.get(key, ()))


def malik_conflicting_origins():
    labels_by_name = defaultdict(set)
    entries = json.loads((SOURCE_DIR / "malik-zip-members.json").read_text(encoding="utf-8"))
    for entry in entries:
        origin = PurePosixPath(entry["name"])
        if (origin.parts[:-2] == ("Tomato Dataset", "Dataset (raw)")
                and MALIK_CAMERA_NAME.fullmatch(origin.name)):
            labels_by_name[origin.name.casefold()].add(origin.parent.name)
    return {name for name, labels in labels_by_name.items() if len(labels) > 1}


def filter_source(source, duplicates, reserved, reviewed_exclusions, rejected):
    candidates = rows(SOURCE_DIR / f"{source}-raw.csv")
    conflicting_names = malik_conflicting_origins() if source == "malik" else set()
    labels_by_hash = defaultdict(set)
    for row in candidates:
        labels_by_hash[row["sha256"]].add(row["label"])
    accepted = []
    for row in sorted(candidates, key=lambda item: item["origin"]):
        digest = row["sha256"]
        origin = row["origin"]
        filename = PurePosixPath(origin).name
        reason = reviewed_exclusions.get(origin, "")
        if not reason and source == "malik" and not MALIK_CAMERA_NAME.fullmatch(filename):
            reason = "outside the documented camera filename cohort"
        if not reason and filename.casefold() in conflicting_names:
            reason = "same camera filename assigned to different publisher labels"
        if not reason and len(labels_by_hash[digest]) > 1:
            reason = "conflicting publisher labels for the same file"
        if not reason and digest in reserved:
            reason = "exact duplicate"
        if not reason:
            try:
                signatures, (width, height) = fingerprint(row["path"], mirrored=True)
                if any(duplicates.matches(signature) for signature in signatures):
                    reason = "perceptual near-duplicate quarantine"
            except (OSError, ValueError):
                reason = "unreadable image"
        if reason:
            rejected.append({"source": source, "origin": origin, "label": row["label"],
                             "sha256": digest, "reason": reason})
            continue
        reserved.add(digest)
        duplicates.add(signatures[0])
        accepted.append({**row, "dataset": source, "capture_group": source,
                         "width": width, "height": height})
    return accepted


def write_rows(path, records, fields):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def freeze(exclusions_path):
    history_path = MANIFEST_DIR / "historical_index.json"
    if not history_path.exists():
        history = create_history_index()
    else:
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if any(sha256(Path(path)) != digest for path, digest in history["manifests"].items()):
            raise ValueError("Historical manifest changed. Rebuild its identity index.")
    reviewed_exclusions = {}
    if exclusions_path:
        reviewed_exclusions = {row["origin"]: row["reason"] for row in rows(exclusions_path)}
    exact = set(history["exact"])
    # These external audits only contribute already-recorded file identities.
    exact.update(row["sha256"] for path in EXTERNAL_MANIFESTS for row in rows(path))
    near = NearDuplicateIndex()
    for signature in history["dhash"]:
        near.add(signature)
    rejected = []
    test = filter_source("plantcity", near, exact, reviewed_exclusions, rejected)
    validation = filter_source("malik", near, exact, reviewed_exclusions, rejected)
    fit = rows(FIT)
    partitions = {"train": fit, "validation": validation, "test": test}
    for name, partition in partitions.items():
        if {row["label"] for row in partition} != LABELS or any(not Path(row["path"]).is_file() for row in partition):
            raise ValueError(f"Incomplete or missing data in {name}")
    train_ids, val_ids, test_ids = ({row["sha256"] for row in part} for part in partitions.values())
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("The partitions contain the same image")
    if (MANIFEST_DIR / "freeze.json").exists():
        raise FileExistsError("The split is already frozen")
    result = {"class_counts": {}, "manifest_sha256": {}, "source_split": {
        "train": "Historical 513-image public-data fine-tuning manifest",
        "validation": "Malik v2 raw camera-name cohort from 2025-11-02; at most 16 files per class ranked by archive-path SHA-256 before review",
        "test": "PlantCity v4, original publisher test.zip",
    }, "historical_unreadable": history["unreadable"], "rejected": {}}
    for name, partition in partitions.items():
        path = MANIFEST_DIR / f"{name}.csv"
        write_rows(path, partition, FIELDS)
        result["class_counts"][name] = dict(Counter(row["label"] for row in partition))
        result["manifest_sha256"][name] = sha256(path)
    for source in ("malik", "plantcity"):
        result["rejected"][source] = dict(Counter(row["reason"] for row in rejected if row["source"] == source))
    write_rows(MANIFEST_DIR / "excluded.csv", rejected, ("source", "origin", "label", "sha256", "reason"))
    (MANIFEST_DIR / "freeze.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-history", action="store_true")
    parser.add_argument("--review-exclusions", type=Path)
    args = parser.parse_args()
    if args.index_history:
        result = create_history_index()
        print("Historical index:", len(result["exact"]), "images;",
              len(result["unreadable"]), "unreadable", flush=True)
    else:
        freeze(args.review_exclusions)


if __name__ == "__main__":
    main()
