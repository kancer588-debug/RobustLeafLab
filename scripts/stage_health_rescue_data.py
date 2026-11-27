"""Freeze new public healthy training images and a separate early-only development probe."""

import hashlib
import re
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageOps

from fetch_online_images import RemoteZip, extract_member
from prepare_online_run import FIELDS, rows, sha256, write_rows
from stage_local_leaf_sources import image_hashes
from stage_roi_evaluation import TEST as SHEIKH, fetch_candidates
from stage_roi_training import STAGE, read, references, write_once
from train_online_candidate import WORKSPACE


ROOT = WORKSPACE / "datasets" / "health_rescue_trial"
ONI = "https://data.mendeley.com/datasets/rnbsw72zb5/1"
ONI_ARCHIVE = {"content_details": {
    "size": 1084433928,
    "download_url": "https://data.mendeley.com/public-files/datasets/rnbsw72zb5/files/3eea427c-1e61-4e94-85cd-f5e0ce87d34a/file_downloaded",
}}
ONI_NAME = re.compile(r"Tomato_Leaves/Images/train/H \((\d+)\)\.jpg")
ONI_COUNT = 36


def oni_training_candidates():
    remote = RemoteZip(ONI_ARCHIVE)
    with ZipFile(remote) as archive:
        members = [(int(match[1]), item) for item in archive.infolist() if (match := ONI_NAME.fullmatch(item.filename))]
    members.sort(key=lambda pair: pair[0])
    # Sample across the numbered catalog while limiting transfers to complete, lower-size originals.
    selected = [min(members[i * len(members) // ONI_COUNT:(i + 1) * len(members) // ONI_COUNT],
                    key=lambda pair: pair[1].compress_size) for i in range(ONI_COUNT)]
    directory = ROOT / "oni_training"
    directory.mkdir(exist_ok=True)

    def retrieve(item):
        number, member = item
        path = directory / f"oni_healthy_{number:04}.jpg"
        if path.exists():
            payload = path.read_bytes()
            if len(payload) != member.file_size or zlib.crc32(payload) != member.CRC:
                raise ValueError(f"Cached original differs: {member.filename}")
        else:
            start = member.header_offset
            content = remote.fetch(start, member._end_offset)
            path.write_bytes(extract_member(content, start, member))
        with Image.open(path) as image:
            exif = image.getexif()
            date = str(exif.get(306, ""))[:10].replace(":", "-")
            group = "|".join(part for part in ("oni_bd", str(exif.get(272, "")), date) if part)
        return ("healthy", member.filename, path, sha256(path), "oni_bd", group)

    with ThreadPoolExecutor(max_workers=8) as workers:
        return list(workers.map(retrieve, selected))


def sheikh_development_candidates():
    if read(SHEIKH / "assessment.json")["status"] != "excluded_from_evaluation":
        raise ValueError("Only the excluded Sheikh source can be used as developmental evidence")
    metadata = {row["File_Name"]: row for row in rows(SHEIKH / "related_raw_metadata.csv")
                if "early blight" in row["Class_Label"].casefold()}
    candidates = []
    for label, origin, path, digest in fetch_candidates():
        details = metadata.get(PurePosixPath(origin).name)
        if label != "early_blight" or details is None or details["Augmented"].casefold() != "no":
            continue
        group = f"sheikh_2026|{details['Camera_Device']}|{details['Capture_Date']}"
        candidates.append((label, origin, path, digest, "sheikh_2026", group))
    return sorted(candidates, key=lambda row: hashlib.sha256(row[1].encode()).hexdigest())


def screen(candidates):
    exact, difference, perceptual, unreadable = references()
    extra = [(row["sha256"], row["path"]) for row in read(STAGE / "screened.json")["records"]]
    extra.extend((row["sha256"], row["path"]) for row in rows(
        WORKSPACE / "datasets" / "view_ablation" / "plantseg" / "test.csv"))
    for digest, path in extra:
        exact.add(digest)
        first, second = image_hashes(path)[0]
        difference.add(first)
        perceptual.add(second)
    records, rejected = [], Counter()
    for label, origin, path, digest, source, group in candidates:
        signatures = image_hashes(path)
        if digest in exact or any(difference.matches(a) or perceptual.matches(b) for a, b in signatures):
            rejected[source] += 1
            continue
        with Image.open(path) as image:
            width, height = image.size
        records.append({"path": str(path), "mask_path": "", "label": label, "sha256": digest, "source": source,
                        "dataset": source, "origin": origin, "capture_group": group, "width": width, "height": height})
        exact.add(digest)
        for a, b in signatures:
            difference.add(a)
            perceptual.add(b)
    return records, rejected, unreadable


def contact_sheet(path, records):
    side = 128
    canvas = Image.new("RGB", (8 * side, ((len(records) + 7) // 8) * (side + 18)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(records):
        x, y = (i % 8) * side, (i // 8) * (side + 18)
        with Image.open(row["path"]) as photo:
            image = ImageOps.exif_transpose(photo).convert("RGB")
            image.thumbnail((side, side))
        canvas.paste(image, (x, y))
        draw.text((x + 2, y + side + 2), PurePosixPath(row["origin"]).name, fill="black")
    canvas.save(path, quality=90)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "freeze.json"
    if lock.exists():
        for item in read(lock)["manifests"].values():
            if sha256(ROOT / item["file"]) != item["sha256"]:
                raise ValueError("A frozen data manifest changed")
        return
    with ThreadPoolExecutor(max_workers=2) as workers:
        train_job = workers.submit(oni_training_candidates)
        dev_options = sheikh_development_candidates()
        train_options = train_job.result()
    print(f"Verified public downloads: {len(train_options)} healthy fit, {len(dev_options)} early development", flush=True)
    screened, rejected, unreadable = screen(train_options + dev_options)
    manifests = {}
    for role, label in (("healthy_fit", "healthy"), ("early_development", "early_blight")):
        selected = [row for row in screened if row["label"] == label]
        if not selected:
            raise ValueError(f"No images survived screening: {role}")
        path = ROOT / f"{role}.csv"
        write_rows(path, selected, FIELDS)
        contact_sheet(ROOT / f"{role}_review.jpg", selected)
        manifests[role] = {"file": path.name, "sha256": sha256(path), "n": len(selected),
                           "groups": dict(Counter(row["capture_group"] for row in selected))}
    write_once(lock, {"manifests": manifests, "unreadable_reference_paths": unreadable, "rejected": dict(rejected),
                      "n_candidates": {"healthy_fit": len(train_options), "early_development": len(dev_options)},
                      "sources": {"healthy_fit": ONI, "early_development": read(SHEIKH / "assessment.json")["publisher"]},
                      "roles": "Oni publisher H train filenames are training only. Sheikh early is development only; "
                               "its related series was excluded from independent testing. Neither collection is a release test."})
    print("Data frozen", manifests, flush=True)


if __name__ == "__main__":
    main()
