"""Stage and freeze the three tomato classes from PlantWild v1's official test split."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile
from zlib import crc32

from fetch_online_images import RemoteZip, batches, extract_member
from prepare_online_run import (
    EXTERNAL_MANIFESTS, MANIFEST_DIR, SOURCE_DIR, NearDuplicateIndex, fingerprint,
    rows, sha256,
)


BASE = MANIFEST_DIR.parent
META_DIR = BASE / "backbone_weights"
TRIAL = BASE / "backbone_trial"
IMAGE_DIR = TRIAL / "plantwild_images"
LABELS = {
    "tomato leaf": ("83", "healthy"),
    "tomato early blight": ("81", "early_blight"),
    "tomato late blight": ("82", "late_blight"),
}
FIELDS = ("path", "mask_path", "label", "sha256", "source", "origin", "width", "height")


def stage():
    metadata = json.loads((META_DIR / "plantwild-v1-publisher.json").read_text(encoding="utf-8"))
    remote = RemoteZip(metadata)
    split_path = META_DIR / "plantwild-trainval.txt"
    definitions = {}
    for row in split_path.read_text(encoding="utf-8-sig").splitlines():
        origin, class_id, domain = row.rsplit("=", 2)
        group = origin.split("/", 1)[0]
        if group in LABELS and domain == "0":
            expected, label = LABELS[group]
            if class_id != expected:
                raise ValueError(f"Unexpected publisher label: {origin}")
            definitions[origin] = label
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    with ZipFile(remote) as archive:
        entries = [(archive.getinfo(f"plantwild/images/{origin}"), label)
                   for origin, label in definitions.items()]
        entries.sort(key=lambda entry: entry[0].header_offset)
        print("Publisher test labels", dict(Counter(definitions.values())), flush=True)

        def save_batch(block):
            start, end, members = block
            existing = [(IMAGE_DIR / f"{member.header_offset}.jpg").exists()
                        for member, _ in members]
            contents = None if all(existing) else remote.fetch(start, end)
            output = []
            for (member, label), was_downloaded in zip(members, existing):
                path = IMAGE_DIR / f"{member.header_offset}.jpg"
                payload = path.read_bytes() if was_downloaded else extract_member(contents, start, member)
                if was_downloaded:
                    if len(payload) != member.file_size or crc32(payload) != member.CRC:
                        raise ValueError(f"Cached image failed publisher CRC: {path}")
                else:
                    temporary = path.with_suffix(".part")
                    temporary.write_bytes(payload)
                    temporary.replace(path)
                output.append({
                    "path": str(path.resolve()), "mask_path": "", "label": label,
                    "sha256": hashlib.sha256(payload).hexdigest(), "source": "PlantWild v1",
                    "origin": member.filename.removeprefix("plantwild/images/"),
                })
            return output

        with ThreadPoolExecutor(max_workers=8) as workers:
            records = []
            for result in workers.map(save_batch, batches(entries)):
                records.extend(result)
                if len(records) % 40 < len(result):
                    print("Staged PlantWild images:", len(records), flush=True)
    with (TRIAL / "plantwild_staged.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print("Staged PlantWild official test membership:", len(records), flush=True)


def freeze(review_exclusions):
    if (TRIAL / "fresh_freeze.json").exists():
        raise FileExistsError("PlantWild test was already frozen")
    history = json.loads((MANIFEST_DIR / "historical_index.json").read_text(encoding="utf-8"))
    for path, digest in history["manifests"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Historical index no longer matches {path}")
    exact = set(history["exact"])
    sources = [*EXTERNAL_MANIFESTS, SOURCE_DIR / "malik-raw.csv", SOURCE_DIR / "plantcity-raw.csv"]
    near = NearDuplicateIndex()
    for dhash in history["dhash"]:
        near.add(dhash)
    for path in sources:
        for row in rows(path):
            exact.add(row["sha256"])
            if path not in EXTERNAL_MANIFESTS:
                signatures, _ = fingerprint(row["path"], resize_for_hash=True)
                near.add(signatures[0])
    excluded = {}
    if review_exclusions:
        excluded = {row["origin"]: row["reason"] for row in rows(review_exclusions)}
    accepted = []
    quarantined = []
    for row in sorted(rows(TRIAL / "plantwild_staged.csv"), key=lambda item: item["origin"]):
        file_hash = row["sha256"]
        reason = excluded.get(row["origin"], "")
        if not reason and file_hash in exact:
            reason = "exact duplicate"
        if not reason:
            try:
                fingerprints, (width, height) = fingerprint(row["path"], mirrored=True)
                if any(near.matches(value) for value in fingerprints):
                    reason = "potential perceptual duplicate"
            except (OSError, ValueError):
                reason = "unreadable image"
        if reason:
            quarantined.append({**row, "reason": reason})
        else:
            accepted.append({**row, "width": width, "height": height})
            exact.add(file_hash)
            near.add(fingerprints[0])
    counts = Counter(row["label"] for row in accepted)
    if set(counts) != {label for _, label in LABELS.values()}:
        raise ValueError("Three-class coverage did not survive screening")
    for filename, data, fields in (
        ("fresh_test.csv", accepted, FIELDS),
        ("fresh_excluded.csv", quarantined, (*FIELDS, "reason")),
    ):
        with (TRIAL / filename).open("x", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    output = {
        "dataset": "PlantWild v1",
        "publisher": "https://huggingface.co/datasets/uqtwei2/PlantWild",
        "membership": "All available images under the three target tomato classes with domain 0 in the author trainval.txt",
        "sha256": {"fresh_test": sha256(TRIAL / "fresh_test.csv")},
        "counts": counts,
        "quarantined": Counter(row["reason"] for row in quarantined),
    }
    with (TRIAL / "fresh_freeze.json").open("x", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    print(json.dumps(output, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage", "freeze"))
    parser.add_argument("--review-exclusions", type=Path)
    args = parser.parse_args()
    if args.command == "stage":
        stage()
    else:
        freeze(args.review_exclusions)


if __name__ == "__main__":
    main()
