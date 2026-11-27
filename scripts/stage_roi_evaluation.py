"""Freeze an original-archive, source-held-out tomato evaluation sample."""

import hashlib
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image, ImageDraw

from fetch_online_images import RemoteZip, batches, extract_member
from prepare_online_run import FIELDS, rows, sha256, write_rows
from stage_local_leaf_sources import image_hashes
from stage_roi_candidates import RUN
from stage_roi_training import STAGE, read, references, write_once


TEST = RUN / "bangladesh_public"
ORIGINAL = TEST / "original_archive.json"
PUBLISHER = "https://data.mendeley.com/datasets/9jxvtgh325/1"
ARCHIVE_HASH = "098a3ceaa776d853d19119eb59d3717104c32995afb68e18046a8de3ab9c7531"
CLASSES = {"Healthy Leaf": "healthy", "Early Blight Leaf": "early_blight", "Late Blight Leaf": "late_blight"}
OPTIONS_PER_CLASS = 85
TEST_PER_CLASS = 60


def select_members(archive):
    groups = defaultdict(list)
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.parts[0] == "Resize Dataset" and path.parent.name in CLASSES and path.suffix.lower() == ".png":
            groups[CLASSES[path.parent.name]].append(member)
    if any(len(groups[label]) != 500 for label in CLASSES.values()):
        raise ValueError("Original archive class counts differ from the reviewed catalog")
    pools = {label: sorted(pool, key=lambda row: hashlib.sha256(row.filename.encode()).hexdigest())[:OPTIONS_PER_CLASS]
             for label, pool in groups.items()}
    return sorted([(item, label) for label, pool in pools.items() for item in pool], key=lambda item: item[0].header_offset)


def fetch_candidates():
    original = read(ORIGINAL)
    if original["content_details"]["sha256_hash"] != ARCHIVE_HASH:
        raise ValueError("The publisher archive identity changed")
    remote = RemoteZip(original)
    with ZipFile(remote) as archive:
        entries = select_members(archive)
    destination = TEST / "images"
    destination.mkdir(exist_ok=True)

    def path_for(member):
        return destination / (hashlib.sha256(member.filename.encode()).hexdigest()[:20] + ".png")

    def fetch(task):
        start, end, members = task
        payload = remote.fetch(start, end)
        result = []
        for member, label in members:
            image = extract_member(payload, start, member)
            origin = member.filename
            path = path_for(member)
            digest = hashlib.sha256(image).hexdigest()
            if path.exists():
                if sha256(path) != digest:
                    raise ValueError(f"Cached image differs: {origin}")
            else:
                pending = path.with_suffix(".tmp")
                pending.write_bytes(image)
                pending.replace(path)
            result.append((label, origin, path, digest))
        return result

    data, missing = [], []
    for member, label in entries:
        path = path_for(member)
        if not path.is_file():
            missing.append((member, label))
            continue
        payload = path.read_bytes()
        if len(payload) != member.file_size or zlib.crc32(payload) != member.CRC:
            raise ValueError(f"Cached image fails the publisher ZIP checksum: {member.filename}")
        data.append((label, member.filename, path, hashlib.sha256(payload).hexdigest()))
    with ThreadPoolExecutor(max_workers=12) as executor:
        for chunk in executor.map(fetch, batches(missing)):
            data.extend(chunk)
    return data


def screen(candidates):
    exact, difference, perceptual, unreadable = references()
    for item in read(STAGE / "screened.json")["records"]:
        exact.add(item["sha256"])
        for d_hash, p_hash in image_hashes(item["path"]):
            difference.add(d_hash)
            perceptual.add(p_hash)
    eligible, rejected = [], defaultdict(Counter)
    counts = Counter()
    for label, origin, path, digest in sorted(candidates, key=lambda row: hashlib.sha256(row[1].encode()).hexdigest()):
        if counts[label] == TEST_PER_CLASS:
            continue
        if digest in exact:
            rejected[label]["exact_repeat"] += 1
            continue
        signatures = image_hashes(path)
        if any(difference.matches(d_hash) or perceptual.matches(p_hash) for d_hash, p_hash in signatures):
            rejected[label]["perceptual_repeat"] += 1
            continue
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        eligible.append({"path": str(path), "mask_path": "", "label": label, "sha256": digest, "source": "sheikh_2026",
                         "dataset": "sheikh_2026", "origin": origin, "capture_group": "sheikh_2026_publisher",
                         "width": width, "height": height})
        counts[label] += 1
        exact.add(digest)
        for d_hash, p_hash in signatures:
            difference.add(d_hash)
            perceptual.add(p_hash)
    return eligible, counts, rejected, unreadable


def show_sample(eligible):
    side = 160
    canvas = Image.new("RGB", (side * 6, (side + 18) * 3), "white")
    draw = ImageDraw.Draw(canvas)
    for line, label in enumerate(CLASSES.values()):
        subset = [row for row in eligible if row["label"] == label][:6]
        for column, row in enumerate(subset):
            x, y = column * side, line * (side + 18)
            with Image.open(row["path"]) as image:
                canvas.paste(image.convert("RGB").resize((side, side), Image.Resampling.BILINEAR), (x, y))
            draw.text((x + 3, y + side + 1), label, fill="black")
    canvas.save(TEST / "review.jpg", quality=90)


def main():
    destination = TEST / "test.csv"
    receipt = TEST / "freeze.json"
    assessment = TEST / "assessment.json"
    if receipt.exists():
        expected = read(receipt)["sha256"]
        if sha256(destination) != expected or any(sha256(Path(row["path"])) != row["sha256"] for row in rows(destination)):
            raise ValueError("A frozen evaluation artifact changed")
        return
    if assessment.exists():
        print(read(assessment), flush=True)
        return
    candidates = fetch_candidates()
    print(f"Verified {len(candidates)} original-archive candidates", flush=True)
    eligible, counts, rejected, unreadable = screen(candidates)
    if any(counts[label] < TEST_PER_CLASS for label in CLASSES.values()):
        write_once(assessment, {"publisher": PUBLISHER, "archive_sha256": ARCHIVE_HASH,
                                "status": "excluded_from_evaluation", "target_per_class": TEST_PER_CLASS,
                                "candidates": dict(Counter(label for label, *_ in candidates)),
                                "screened_counts": {label: counts[label] for label in CLASSES.values()},
                                "duplicate_flags": {label: dict(values) for label, values in rejected.items()},
                                "unreadable_reference_paths": unreadable,
                                "reason": "Too few distinct labeled originals after exact and perceptual identity screening."})
        print(read(assessment), flush=True)
        return
    write_rows(destination, eligible, FIELDS)
    write_once(receipt, {"path": str(destination), "sha256": sha256(destination), "publisher": PUBLISHER,
                         "archive_sha256": ARCHIVE_HASH, "counts": dict(counts), "duplicates_excluded": dict(rejected),
                         "unreadable_reference_paths": unreadable, "role": "new_source_holdout",
                         "scope": "Publisher-original archive (resized images), one source held out in full; no plot, date or plant IDs."})
    show_sample(eligible)
    print("Frozen test sample", dict(counts), "duplicate exclusions", dict(rejected), flush=True)


if __name__ == "__main__":
    main()
