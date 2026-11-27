"""Stage labelled public images for a local-view experiment and cross-source audit."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256 as digest
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn
import numpy as np

from fetch_online_images import RemoteZip, extract_member
from prepare_expanded_field_trial import EARLY_TRIAL, excluded_evaluation
from prepare_online_run import EXTERNAL_MANIFESTS, MANIFEST_DIR, REFERENCE, REFERENCE_MANIFESTS
from prepare_online_run import NearDuplicateIndex, dhash, rows, sha256


TRIAL = MANIFEST_DIR.parent / "local_leaf_trial"
EXPANDED = MANIFEST_DIR.parent / "expanded_field_trial"
CHEN_ARCHIVE = {
    "content_details": {"size": 544524808,
                        "download_url": "https://zenodo.org/records/15868289/files/Tomato-Disease.zip?download=1"},
}
CHEN_CLASSES = {0: "early_blight", 1: "healthy", 2: "late_blight"}
TAIWAN_CLASSES = {"health": "healthy", "Late blight": "late_blight"}
TAIWAN_SOURCE = "https://doi.org/10.17632/ngdgg79rzb.1"
TAIWAN_ARCHIVE_SHA256 = "f623d13a7f0388dada08ca5b5f22716557defce3832f50ff6af7153e0eab676d"
FIELD_LIMIT = 40
FIELDS = ("path", "mask_path", "label", "sha256", "source", "origin", "capture_group", "review_id")
SUFFIXES = {".jpg", ".jpeg", ".png"}


def write_csv(path, records):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def image_hashes(path):
    with Image.open(path) as picture:
        picture = ImageOps.exif_transpose(picture).convert("RGB")
    signatures = []
    for variant in (picture, ImageOps.mirror(picture)):
        sample = np.asarray(variant.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        low = dctn(sample, norm="ortho")[:8, :8].flatten()
        packed = int.from_bytes(np.packbits(low > np.median(low[1:])).tobytes(), "big")
        signatures.append((dhash(variant), packed))
    return signatures


def reference_images(include_new_train, *, include_original=False):
    refs = [EXPANDED / "expanded_train.csv", EXPANDED / "rachin_evaluation.csv",
            *excluded_evaluation().values(), *EXTERNAL_MANIFESTS,
            EARLY_TRIAL / "replacement.csv"]
    if include_new_train:
        refs.append(TRIAL / "train.csv")
    if include_original:
        refs.extend([*REFERENCE_MANIFESTS, REFERENCE / "data/train.csv",
                     REFERENCE / "outputs/train_improve_late/folds/field_extra_merged/train.csv"])
    by_hash = {item["sha256"]: item["path"] for path in refs for item in rows(path)}
    differences, perceptual = NearDuplicateIndex(), NearDuplicateIndex()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for signatures in pool.map(image_hashes, by_hash.values()):
            first, second = signatures[0]
            differences.add(first)
            perceptual.add(second)
    return set(by_hash), differences, perceptual


def same_photo(filename):
    stem = PurePosixPath(filename).stem.split(".rf.")[0].casefold()
    stem = re.sub(r"(?:[-_](?:jpg|jpeg|png))+$", "", stem)
    return re.sub(r"[^a-z0-9]", "", stem)


def candidate_members(archive, inventory):
    labels = {}
    for row in inventory:
        label_path = PurePosixPath(row["origin"])
        if label_path.parts[1] == "train" and len(row["classes"]) == 1:
            class_id = row["classes"][0]
            if class_id in CHEN_CLASSES:
                labels[label_path.stem] = CHEN_CLASSES[class_id]
    pools = defaultdict(list)
    used_origins = set()
    for entry in archive.infolist():
        path = PurePosixPath(entry.filename)
        if path.parts[:-1] != ("Tomato-Disease", "train", "images") or path.stem not in labels:
            continue
        lower = path.name.casefold()
        group = same_photo(entry.filename)
        if (re.search(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-", lower)
                or any(fragment in lower for fragment in ("transformed", "rotated", "augmented"))
                or path.suffix.lower() not in SUFFIXES or group in used_origins):
            continue
        used_origins.add(group)
        pools[labels[path.stem]].append(entry)
    return [(entry, label) for label, entries in pools.items()
            for entry in sorted(entries, key=lambda item: digest(item.filename.encode()).hexdigest())[:FIELD_LIMIT]]


def range_groups(members):
    groups = []
    for entry, label in sorted(members, key=lambda item: item[0].header_offset):
        if (not groups or entry.header_offset - groups[-1][-1][0]._end_offset > 64 * 1024
                or entry._end_offset - groups[-1][0][0].header_offset > 512 * 1024):
            groups.append([])
        groups[-1].append((entry, label))
    return groups


def get_chen_candidates():
    source = RemoteZip(CHEN_ARCHIVE)
    inventory = json.loads((TRIAL / "chen_label_inventory.json").read_text(encoding="utf-8"))
    destination = TRIAL / "chen_images"
    destination.mkdir(exist_ok=True)
    with ZipFile(source) as archive:
        entries = candidate_members(archive, inventory)
    groups = range_groups(entries)
    print(f"Chen selection: {len(entries)} images, {len(groups)} byte ranges", flush=True)

    def fetch(group):
        start, end = group[0][0].header_offset, group[-1][0]._end_offset
        payload = source.fetch(start, end)
        batch = []
        for entry, label in group:
            path = destination / f"{entry.header_offset}{PurePosixPath(entry.filename).suffix.lower()}"
            content = extract_member(payload, start, entry)
            if not path.exists() or path.read_bytes() != content:
                path.write_bytes(content)
            batch.append(dict(path=str(path), label=label, sha256=sha256(path), source="Chen Tomato-Disease",
                              origin=entry.filename, capture_group=same_photo(entry.filename)))
        return batch

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=14) as pool:
        futures = {pool.submit(fetch, group): group for group in groups}
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except (OSError, ValueError) as error:
                errors.append({"members": [entry.filename for entry, _ in futures[future]], "error": str(error)})
            if len(results) and len(results) % 20 < 3:
                print("Verified archive images:", len(results), flush=True)
    write_json(TRIAL / "chen_download_failures.json", errors)
    return results


def make_sheets(records, stem):
    review = TRIAL / "review"
    review.mkdir(exist_ok=True)
    for start in range(0, len(records), 30):
        block = records[start:start + 30]
        canvas = Image.new("RGB", (1080, 142 * ((len(block) + 5) // 6)), "#f5f5f5")
        drawing = ImageDraw.Draw(canvas)
        for at, row in enumerate(block):
            x, y = at % 6 * 180 + 4, at // 6 * 142 + 4
            with Image.open(row["path"]) as picture:
                thumbnail = ImageOps.contain(picture.convert("RGB"), (167, 111))
            canvas.paste(thumbnail, (x + (167-thumbnail.width)//2, y))
            drawing.text((x + 4, y + 116), f"{row['review_id']} {row['label']}", fill="#161616")
        canvas.save(review / f"{stem}-{start//30+1:02d}.jpg", quality=88)


def screen(candidates, protect_new_train=False, *, include_original=False):
    exact, dindex, pindex = reference_images(protect_new_train, include_original=include_original)
    approval, exclusions = [], Counter()
    for row in sorted(candidates, key=lambda value: (value["label"], value["sha256"])):
        if row["sha256"] in exact:
            exclusions["exact_or_reserved"] += 1
            continue
        try:
            pairs = image_hashes(row["path"])
        except OSError:
            exclusions["unreadable"] += 1
            continue
        if any(dindex.matches(d) or pindex.matches(p) for d, p in pairs):
            exclusions["perceptual_match"] += 1
            continue
        exact.add(row["sha256"])
        dindex.add(pairs[0][0])
        pindex.add(pairs[0][1])
        approval.append({**row, "review_id": len(approval) + 1})
    return approval, dict(exclusions)


def stage_chen():
    staged = get_chen_candidates()
    checked, excluded = screen(staged)
    write_csv(TRIAL / "chen_review.csv", checked)
    make_sheets(checked, "chen")
    stats = {"staged": dict(Counter(row["label"] for row in staged)),
             "after_duplicates": dict(Counter(row["label"] for row in checked)), "excluded": excluded}
    write_json(TRIAL / "chen_inventory.json", stats)
    print(json.dumps(stats), flush=True)


def freeze_training(review):
    ids = set(json.loads(review.read_text(encoding="utf-8"))["accepted_review_ids"])
    candidates = [row for row in rows(TRIAL / "chen_review.csv") if int(row["review_id"]) in ids]
    if len(candidates) != len(ids):
        raise ValueError("Review contains unknown image IDs")
    counts = Counter(row["label"] for row in candidates)
    n = min(counts[label] for label in CHEN_CLASSES.values())
    if not n:
        raise ValueError("The published supplement lacks one of the matched classes")
    extras = [row for label in CHEN_CLASSES.values()
              for row in [item for item in candidates if item["label"] == label][:n]]
    base = rows(EXPANDED / "expanded_train.csv")
    path = TRIAL / "train.csv"
    write_csv(path, base + extras)
    write_csv(TRIAL / "chen_additions.csv", extras)
    write_json(TRIAL / "training_additions.json", {
        "manifest_sha256": sha256(path), "review_sha256": sha256(review), "additions_per_class": n,
        "counts": dict(Counter(row["label"] for row in base + extras)),
        "source": "https://doi.org/10.5281/zenodo.15868289",
    })
    print("Approved additions per class:", n, "final training size:", len(base) + len(extras), flush=True)


def freeze_taiwan():
    if sha256(TRIAL / "taiwan.7z") != TAIWAN_ARCHIVE_SHA256:
        raise ValueError("Unexpected Taiwan archive checksum")
    raw = TRIAL / "taiwan_raw" / "taiwan" / "Preprocessed data"
    originals = [path for path in raw.glob("*/*/*") if path.suffix.lower() in SUFFIXES]
    if len(originals) != 622:
        raise ValueError("The archived nonaugmented subset should contain 622 images")
    candidates = [
        {"path": str(path), "label": TAIWAN_CLASSES[path.parent.name], "sha256": sha256(path),
         "source": "Huang and Chang Taiwan", "origin": path.relative_to(raw).as_posix(),
         "capture_group": "taiwan_release_unspecified_capture"}
        for path in originals if path.parent.name in TAIWAN_CLASSES
    ]
    counts = dict(Counter(row["label"] for row in candidates))
    if counts != {"healthy": 106, "late_blight": 98}:
        raise ValueError(f"Unexpected Taiwan class counts: {counts}")
    taiwan_names = {same_photo(row["origin"]): row["origin"] for row in candidates}
    chen_overlap = [
        {"chen": row["origin"], "taiwan": taiwan_names[same_photo(row["origin"])]}
        for row in rows(TRIAL / "chen_additions.csv") if same_photo(row["origin"]) in taiwan_names
    ]
    print("Shared image stems with Chen pilot:", len(chen_overlap), flush=True)
    checked, excluded = screen(candidates, include_original=True)
    class_counts = dict(Counter(row["label"] for row in checked))
    if set(class_counts) != set(TAIWAN_CLASSES.values()):
        raise ValueError("Both target classes must survive duplicate screening")
    path = TRIAL / "taiwan_test.csv"
    write_csv(path, checked)
    training = EXPANDED / "expanded_train.csv"
    write_json(TRIAL / "fresh_test_freeze.json", {
        "source": TAIWAN_SOURCE, "archive_sha256": TAIWAN_ARCHIVE_SHA256,
        "compatible_training_sha256": sha256(training), "chen_pilot_name_overlaps": chen_overlap,
        "before_screening": counts, "excluded": excluded,
        "tests": {"taiwan_healthy_late": {"path": str(path), "sha256": sha256(path),
                 "class_counts": class_counts,
                 "role": "New publisher source for the base873 cohort, only nonaugmented healthy and late"}},
    })
    print("Taiwan public test:", json.dumps({"retained": class_counts, "excluded": excluded}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("stage-chen")
    commands.add_parser("freeze-taiwan")
    freeze = commands.add_parser("freeze-training")
    freeze.add_argument("review", type=Path)
    args = parser.parse_args()
    if args.phase == "stage-chen":
        stage_chen()
    elif args.phase == "freeze-taiwan":
        freeze_taiwan()
    else:
        freeze_training(args.review)


if __name__ == "__main__":
    main()
