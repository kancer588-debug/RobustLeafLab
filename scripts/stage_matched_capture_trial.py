"""Freeze reuse of unused Tomato-Village late-blight frames from the old training fold."""

import argparse
import csv
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256 as digest
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

from PIL import Image, ImageDraw, ImageOps

from prepare_expanded_field_trial import excluded_evaluation
from prepare_online_run import EXTERNAL_MANIFESTS, FIELDS, MANIFEST_DIR, REFERENCE, NearDuplicateIndex, rows, sha256
from stage_local_leaf_sources import image_hashes, write_json


ROOT = MANIFEST_DIR.parent / "matched_capture_trial"
SOURCE_DIR = REFERENCE / "outputs" / "train_improve_late" / "folds" / "field_extra_merged"
BASE = MANIFEST_DIR.parent / "subtype_trial" / "training.csv"
EARLY = "early_blight"
LATE = "late_blight"
PUBLISHER = "https://github.com/mamta-joshi-gehlot/Tomato-Village"
COLUMNS = (*FIELDS, "capture_date", "review_id")
PUBLIC_LABELS = "https://raw.githubusercontent.com/mamta-joshi-gehlot/Tomato-Village/main/Variant-b(MultiLabel%20Classification)/Multi-Label%20dataset%20-%20with%20augmented.csv"
FRAME = re.compile(r"IMG_?(20\d{12})\.(?:jpg|jpeg|png)", re.IGNORECASE)


def write_rows(path, records):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def protected_paths():
    data = MANIFEST_DIR.parent
    return list(dict.fromkeys([BASE, *excluded_evaluation().values(), *EXTERNAL_MANIFESTS,
                               data / "expanded_field_trial" / "rachin_evaluation.csv",
                               data / "local_leaf_trial" / "taiwan_test.csv",
                               data / "early_review_trial" / "agrivision_health" / "test.csv"]))


def reference_index(paths):
    known = {item["sha256"]: item["path"] for manifest in paths for item in rows(manifest)}
    differences, perceptual = NearDuplicateIndex(), NearDuplicateIndex()
    with ThreadPoolExecutor(max_workers=6) as workers:
        for signatures in workers.map(image_hashes, known.values()):
            d, p = signatures[0]
            differences.add(d)
            perceptual.add(p)
    return set(known), differences, perceptual


def montage(records):
    for offset in range(0, len(records), 24):
        subset = records[offset:offset+24]
        panel = Image.new("RGB", (1080, 146 * ((len(subset)+5)//6)), "white")
        draw = ImageDraw.Draw(panel)
        for index, row in enumerate(subset):
            x, y = index % 6 * 180 + 4, index // 6 * 146 + 4
            with Image.open(row["path"]) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (170, 115))
            panel.paste(thumb, (x + (170 - thumb.width)//2, y))
            draw.text((x, y + 120), f"{row['review_id']} {row['capture_group'].rsplit('|', 1)[-1]}", fill="black")
        panel.save(ROOT / f"review-{offset//24+1}.jpg", quality=89)


def stage():
    ROOT.mkdir(exist_ok=True)
    train = SOURCE_DIR / "train.csv"
    expected = json.loads((SOURCE_DIR / "fold.json").read_text(encoding="utf-8"))["fold_manifests_sha256"]["train.csv"]
    if sha256(train) != expected:
        raise ValueError("The original trained-fold manifest has changed")
    base = rows(BASE)
    base_ids = {item["sha256"] for item in base}
    eligible_batches = {item["capture_group"] for item in base if item["label"] == EARLY
                        and item.get("dataset") == "tomato_village" and item["capture_group"]}
    candidates = [item for item in rows(train) if item["label"] == LATE and item["dataset"] == "tomato_village"
                  and item["capture_group"] in eligible_batches and item["sha256"] not in base_ids]
    print(f"Found {len(candidates)} unused late frames in {len(eligible_batches)} existing mixed batches", flush=True)
    paths = protected_paths()
    original_excluded, accepted = Counter(), []
    if candidates:
        exact, dhash, phash = reference_index(paths)
    for item in sorted(candidates, key=lambda row: row["origin"]):
        if item["sha256"] in exact:
            original_excluded["exact_duplicate"] += 1
            continue
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"An archive image changed: {path}")
        signatures = image_hashes(path)
        if any(dhash.matches(d) or phash.matches(p) for d, p in signatures):
            original_excluded["perceptual_duplicate"] += 1
            continue
        exact.add(item["sha256"])
        d, p = signatures[0]
        dhash.add(d)
        phash.add(p)
        accepted.append({**item, "review_id": len(accepted) + 1})
    write_rows(ROOT / "review.csv", accepted)
    montage(accepted)
    write_json(ROOT / "screening.json", {"source": PUBLISHER, "source_manifest_sha256": expected,
                                         "source_candidates": len(candidates), "screened": len(accepted),
                                         "exclusions": dict(original_excluded),
                                         "reference_manifests": {str(path): sha256(path) for path in paths}})
    print(f"Screened for review: {len(accepted)}; exclusions: {dict(original_excluded)}", flush=True)


def freeze(rejected):
    review_path = ROOT / "review.csv"
    candidates = rows(review_path)
    if rejected - {int(item["review_id"]) for item in candidates}:
        raise ValueError("Unknown visual review ID")
    additions = [item for item in candidates if int(item["review_id"]) not in rejected]
    base = rows(BASE)
    original = SOURCE_DIR / "train.csv"
    screened = json.loads((ROOT / "screening.json").read_text(encoding="utf-8"))
    if sha256(original) != screened["source_manifest_sha256"]:
        raise ValueError("The archived capture roster has changed")
    archived = {item["sha256"]: item for item in rows(original) if item["dataset"] == "tomato_village"}
    archived_frames = {item[1] for row in archived.values() if (item := FRAME.fullmatch(PurePosixPath(row["origin"]).name))}
    with urlopen(PUBLIC_LABELS, timeout=20) as response:
        catalog_data = response.read()
    catalog = list(csv.DictReader(catalog_data.decode("utf-8-sig").splitlines()))
    published_frames = {item[1] for row in catalog if row["SUM"] == "1"
                        and any(row[label] == "1" for label in ("Healthy", "Early blight", "Late blight"))
                        and (item := FRAME.fullmatch(row["filename"].strip()))}
    restored = Counter()
    for item in base:
        if item.get("dataset") != "tomato_village":
            continue
        info = archived[item["sha256"]]
        if info["label"] != item["label"] or not info["capture_group"]:
            raise ValueError("Missing or conflicting archived capture metadata")
        if item["capture_group"] and info["capture_group"] != item["capture_group"]:
            raise ValueError("Capture groups disagree across archived sources")
        if not item["capture_group"]:
            item["capture_group"] = info["capture_group"]
            restored[item["label"]] += 1
    paired = Counter((item["capture_group"], item["label"]) for item in base + additions
                     if item.get("dataset") == "tomato_village")
    batches = sorted({batch for batch, _ in paired if paired[batch, EARLY] and paired[batch, LATE]})
    groups = {batch: {EARLY: paired[batch, EARLY], LATE: paired[batch, LATE]} for batch in batches}
    files = {"training": ROOT / "training.csv", "additions": ROOT / "additions.csv"}
    write_rows(files["training"], base + additions)
    write_rows(files["additions"], additions)
    write_json(ROOT / "freeze.json", {"base_sha256": sha256(BASE), "source": PUBLISHER,
                                      "screening_sha256": sha256(ROOT / "screening.json"),
                                      "review_sha256": sha256(review_path), "visually_rejected": sorted(rejected),
                                      "counts": {"base": len(base), "added_late": len(additions),
                                                 "training": len(base)+len(additions)}, "mixed_group_counts": groups,
                                      "capture_groups_restored": dict(restored),
                                      "publisher_variant_b": {"catalog_sha256": digest(catalog_data).hexdigest(),
                                                              "rows": len(catalog), "single_target_original_frames": len(published_frames),
                                                              "unseen_original_frames": len(published_frames - archived_frames)},
                                      "files": {key: {"path": str(value), "sha256": sha256(value)}
                                                for key, value in files.items()}})
    print(f"Frozen {len(base)} baseline images, {len(additions)} additions, batch metadata restored: {dict(restored)}",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "freeze"))
    parser.add_argument("--reject", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if args.phase == "stage":
        stage()
    else:
        freeze(set(args.reject))


if __name__ == "__main__":
    main()
