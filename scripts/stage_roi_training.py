"""Stage and screen box supervision from the pinned public image sources."""

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image, ImageDraw

from fetch_online_images import RemoteZip, batches, extract_member
from prepare_expanded_field_trial import excluded_evaluation
from prepare_online_run import EXTERNAL_MANIFESTS, NearDuplicateIndex, REFERENCE, rows, sha256
from stage_local_leaf_sources import CHEN_ARCHIVE, image_hashes
from stage_roi_candidates import RUN, WORKSPACE, load_objects


STAGE = RUN / "localization_data"
ANNOTATIONS = RUN / "annotation_inventory_all.json"
CHEN_DIR = WORKSPACE / "datasets" / "local_leaf_trial"
CHEN_REVIEW = CHEN_DIR / "chen_review.csv"
CHEN_APPROVAL = CHEN_DIR / "chen_visual_review.json"
CONTROL = WORKSPACE / "datasets" / "matched_capture_trial" / "training.csv"
HISTORY = WORKSPACE / "datasets" / "independent_test" / "historical_index.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path, data):
    encoded = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"Existing staging artifact differs: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def normalized_size(path):
    with Image.open(path) as image:
        image.load()
        return image.size


def village_frames():
    inventory = read(ANNOTATIONS)
    folder = STAGE / "village_c"
    folder.mkdir(exist_ok=True)
    frames = [row for row in inventory["frames"] if any(box["class"] == "late_blight" for box in row["boxes"])]
    missing = sorted({row["image_git_oid"] for row in frames if not (folder / (row["image_git_oid"] + ".jpg")).exists()})
    payloads = load_objects(missing) if missing else {}
    candidates = []
    for row in frames:
        oid = row["image_git_oid"]
        path = folder / (oid + ".jpg")
        if oid in payloads:
            temp = path.with_suffix(".tmp")
            temp.write_bytes(payloads[oid])
            temp.replace(path)
        payload = path.read_bytes()
        object_header = f"blob {len(payload)}\0".encode("ascii")
        if hashlib.sha1(object_header + payload).hexdigest() != oid:
            raise ValueError(f"Image Git object mismatch: {row['frame']}")
        width, height = normalized_size(path)
        if [width, height] != row["size"]:
            raise ValueError(f"Annotation dimensions changed: {row['frame']}")
        target = [[x / width, y / height, a / width, b / height]
                  for x, y, a, b in (item["xyxy"] for item in row["boxes"] if item["class"] == "late_blight")]
        ignored = [[x / width, y / height, a / width, b / height]
                   for x, y, a, b in (item["xyxy"] for item in row["boxes"] if item["class"] != "late_blight")]
        candidates.append({"path": str(path), "sha256": sha256(path), "origin": row["image"], "label": "late_blight",
                           "source": "tomato_village_c", "capture_group": row["capture_group"],
                           "target": target, "ignore": ignored})
    return candidates


def chen_labels(entries):
    cache = STAGE / "chen_yolo_labels.json"
    if cache.exists():
        result = read(cache)
        if set(result) != set(entries):
            raise ValueError("The cached public label list changed")
        return result
    remote = RemoteZip(CHEN_ARCHIVE)
    with ZipFile(remote) as archive:
        members = sorted([(archive.getinfo(name), name) for name in entries], key=lambda item: item[0].header_offset)

    def fetch(batch):
        start, end, items = batch
        block = remote.fetch(start, end)
        return {name: extract_member(block, start, item).decode("utf-8-sig") for item, name in items}

    result = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for block in executor.map(fetch, batches(members)):
            result.update(block)
    write_once(cache, result)
    return result


def chen_frames():
    approved = set(read(CHEN_APPROVAL)["accepted_review_ids"])
    staged = {str(PurePosixPath(row["origin"]).parent.parent / "labels" / (PurePosixPath(row["origin"]).stem + ".txt")): row
              for row in rows(CHEN_REVIEW) if row["label"] == "early_blight" and int(row["review_id"]) in approved}
    candidates = []
    for name, annotation in chen_labels(staged).items():
        row = staged[name]
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Existing public photo changed: review {row['review_id']}")
        target = []
        for line in annotation.splitlines():
            if not line.strip():
                continue
            label, cx, cy, width, height = (float(value) for value in line.split())
            if label != 0 or width <= 0 or height <= 0:
                continue
            left, right = max(0., cx - width / 2), min(1., cx + width / 2)
            top, bottom = max(0., cy - height / 2), min(1., cy + height / 2)
            if left < right and top < bottom:
                target.append([left, top, right, bottom])
        if target:
            candidates.append({"path": str(path), "sha256": row["sha256"], "origin": row["origin"], "label": "early_blight",
                               "source": "chen_early", "capture_group": row["capture_group"], "target": target, "ignore": [],
                               "previous_review_id": int(row["review_id"])})
    return candidates


def references():
    data = WORKSPACE / "datasets"
    historical = read(HISTORY)
    exact = set(historical["exact"])
    difference, perceptual = NearDuplicateIndex(), NearDuplicateIndex()
    for signature in historical["dhash"]:
        difference.add(signature)
    for path in EXTERNAL_MANIFESTS:
        exact.update(row["sha256"] for row in rows(path))
    ordinary = [CONTROL, *excluded_evaluation().values(), data / "expanded_field_trial" / "rachin_evaluation.csv",
                data / "local_leaf_trial" / "taiwan_test.csv", data / "early_review_trial" / "agrivision_health" / "test.csv",
                REFERENCE / "outputs/train_improve_late/folds/field_extra_merged/train.csv"]
    by_hash = {row["sha256"]: row["path"] for path in ordinary for row in rows(path)}
    exact.update(by_hash)
    unreadable = 0

    def signatures(path):
        try:
            return image_hashes(path)[0]
        except OSError:
            return None

    with ThreadPoolExecutor(max_workers=6) as executor:
        for pair in executor.map(signatures, by_hash.values()):
            if pair is None:
                unreadable += 1
                continue
            difference.add(pair[0])
            perceptual.add(pair[1])
    print(f"Indexed {len(by_hash)} ordinary reference identities ({unreadable} unreadable)", flush=True)
    return exact, difference, perceptual, unreadable


def screen(candidates):
    exact, differences, perceptual, unreadable = references()
    kept, rejected = [], []
    for row in sorted(candidates, key=lambda item: (item["label"], item["capture_group"], item["origin"])):
        signatures = image_hashes(row["path"])
        reason = None
        if row["sha256"] in exact:
            reason = "exact_identity"
        elif any(differences.matches(digest) or perceptual.matches(p_hash) for digest, p_hash in signatures):
            reason = "near_duplicate"
        if reason:
            rejected.append({"origin": row["origin"], "source": row["source"], "reason": reason})
            continue
        kept.append({**row, "review_id": len(kept) + 1})
        exact.add(row["sha256"])
        for digest, p_hash in signatures:
            differences.add(digest)
            perceptual.add(p_hash)
    return kept, rejected, unreadable


def sheets(records):
    side, columns, page_size = 195, 6, 24
    for start in range(0, len(records), page_size):
        subset = records[start:start + page_size]
        sheet = Image.new("RGB", (side * columns, (side + 28) * ((len(subset) + columns - 1) // columns)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(subset):
            x, y = index % columns * side, index // columns * (side + 28)
            with Image.open(row["path"]) as image:
                tile = image.convert("RGB").resize((side, side), Image.Resampling.BILINEAR)
            sheet.paste(tile, (x, y))
            for left, top, right, bottom in row["target"]:
                draw.rectangle((x + left * side, y + top * side, x + right * side, y + bottom * side), outline="#ffbe33", width=2)
            draw.text((x + 4, y + side + 2), f"{row['review_id']} {row['label']} {row['capture_group'][:16]}", fill="black")
        sheet.save(STAGE / f"review_{start // page_size + 1:02d}.jpg", quality=90)


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    manifest = STAGE / "screened.json"
    if manifest.exists():
        frozen = read(manifest)
        if any(sha256(Path(row["path"])) != row["sha256"] for row in frozen["records"]):
            raise ValueError("A staged training photo changed")
        sheets(frozen["records"])
        print(frozen["counts"], flush=True)
        return
    final = village_frames() + chen_frames()
    print("Staged public candidates", dict(Counter(row["label"] for row in final)), flush=True)
    accepted, rejected, unreadable = screen(final)
    counts = dict(Counter(row["label"] for row in accepted))
    write_once(manifest, {"candidate_counts": dict(Counter(row["label"] for row in final)), "counts": counts,
                          "ignored_unreadable_reference_paths": unreadable,
                          "public_annotation_inventory_sha256": sha256(ANNOTATIONS), "chen_review_sha256": sha256(CHEN_REVIEW),
                          "excluded": rejected, "records": accepted})
    sheets(accepted)
    print("After deduplication", counts, "excluded", len(rejected), flush=True)


if __name__ == "__main__":
    main()
