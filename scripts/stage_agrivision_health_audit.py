"""Screen the Agri-Vision4 publisher test split for a healthy-only audit."""

import argparse
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256 as digest
from pathlib import PurePosixPath
from zipfile import ZipFile

from fetch_online_images import RemoteZip, batches, extract_member
from prepare_online_run import MANIFEST_DIR, rows, sha256
from stage_local_leaf_sources import TRIAL as LOCAL, image_hashes, reference_images, write_csv, write_json


DESTINATION = MANIFEST_DIR.parent / "early_review_trial" / "agrivision_health"
PUBLISHER = "https://data.mendeley.com/datasets/8t6k37ztxc/3"
ARCHIVE = {"content_details": {
    "download_url": "https://data.mendeley.com/public-files/datasets/8t6k37ztxc/files/1f98f7d1-4cc8-46e2-88fa-c8a73b5182e9/file_downloaded",
    "size": 2227350048,
}}
FRAME = re.compile(r"^IMG_(\d{8})_(\d{4})\d{2}", re.IGNORECASE)
SUBTYPE_TRAIN = MANIFEST_DIR.parent / "subtype_trial" / "training.csv"


def fetch_group(archive, group):
    start, end, members = group
    chunk = archive.fetch(start, end)
    return [(member.filename, extract_member(chunk, start, member)) for member, _ in members]


def stage():
    remote = RemoteZip(ARCHIVE)
    with ZipFile(remote) as archive:
        entries = [(item, None) for item in archive.infolist()
                   if PurePosixPath(item.filename).parts[-3:-1] == ("Test", "Tomato_Healthy")
                   and PurePosixPath(item.filename).suffix.lower() in {".jpg", ".png", ".jpeg"}]
        if len(entries) != 43:
            raise ValueError(f"Expected the 43 nonaugmented healthy test images, got {len(entries)}")
        groups = list(batches(sorted(entries, key=lambda item: item[0].header_offset)))
    photos = DESTINATION / "images"
    photos.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        assets = [item for block in pool.map(lambda group: fetch_group(remote, group), groups) for item in block]
    print(f"Downloaded and CRC-verified {len(assets)} publisher-test images", flush=True)
    exact, dhash, phash = reference_images(True, include_original=True)
    for manifest in (SUBTYPE_TRAIN, LOCAL / "taiwan_test.csv"):
        for item in rows(manifest):
            if item["sha256"] in exact:
                continue
            exact.add(item["sha256"])
            d, p = image_hashes(item["path"])[0]
            dhash.add(d)
            phash.add(p)
    excluded, trial, groups_seen = Counter(), [], set()
    for origin, content in sorted(assets):
        filename = PurePosixPath(origin).name
        match = FRAME.match(filename)
        if not match:
            excluded["unparsed_capture"] += 1
            continue
        group = f"agrivision|{match[1]}|{match[2]}"
        checksum = digest(content).hexdigest()
        if checksum in exact:
            excluded["exact_duplicate"] += 1
            continue
        path = photos / f"{checksum}{PurePosixPath(filename).suffix.lower()}"
        path.write_bytes(content)
        signatures = image_hashes(path)
        if any(dhash.matches(d) or phash.matches(p) for d, p in signatures):
            excluded["perceptual_duplicate"] += 1
            continue
        if group in groups_seen:
            excluded["same_capture_minute"] += 1
            continue
        d, p = signatures[0]
        exact.add(checksum)
        dhash.add(d)
        phash.add(p)
        groups_seen.add(group)
        trial.append({"path": str(path), "label": "healthy", "sha256": checksum, "source": "Agri-Vision4 v3",
                      "origin": origin, "capture_group": group, "review_id": len(trial) + 1})
    write_csv(DESTINATION / "review.csv", trial)
    from PIL import Image, ImageDraw, ImageOps

    for offset in range(0, len(trial), 24):
        subset = trial[offset:offset + 24]
        panel = Image.new("RGB", (1080, 145 * ((len(subset) + 5) // 6)), "white")
        draw = ImageDraw.Draw(panel)
        for at, item in enumerate(subset):
            x, y = at % 6 * 180 + 4, at // 6 * 145 + 4
            with Image.open(item["path"]) as image:
                preview = ImageOps.contain(image.convert("RGB"), (170, 112))
            panel.paste(preview, (x + (170 - preview.width) // 2, y))
            draw.text((x, y + 117), f"{item['review_id']} {item['capture_group'].split('|', 1)[1]}", fill="black")
        panel.save(DESTINATION / f"review-{offset // 24 + 1}.jpg", quality=88)
    write_json(DESTINATION / "review_stats.json", {"publisher": PUBLISHER, "released_test_class_count": 43,
                                                   "excluded": dict(excluded), "after_screening": len(trial)})
    print("After screening:", len(trial), dict(excluded), flush=True)


def freeze(reject_ids):
    source = rows(DESTINATION / "review.csv")
    if reject_ids - {int(item["review_id"]) for item in source}:
        raise ValueError("Unknown rejected review ID")
    kept = [item for item in source if int(item["review_id"]) not in reject_ids]
    if not kept:
        raise ValueError("No visually approved healthy images remain")
    path = DESTINATION / "test.csv"
    write_csv(path, kept)
    write_json(DESTINATION / "freeze.json", {"publisher": PUBLISHER, "compatible_train_sha256": sha256(SUBTYPE_TRAIN),
                                            "visually_rejected_ids": sorted(reject_ids),
                                            "capture_dates": sorted({item["capture_group"].split("|")[1] for item in kept}),
                                            "image_scope": "Publisher-processed isolated leaves with a transparent background.",
                                            "tests": {"agrivision_healthy": {"path": str(path), "sha256": sha256(path),
                                                       "count": len(kept), "role": "fresh_processed_healthy_only_audit"}}})
    print(f"Frozen {len(kept)} healthy photos from 1 independent publisher", flush=True)


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
