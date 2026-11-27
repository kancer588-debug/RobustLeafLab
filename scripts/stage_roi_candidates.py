"""Inventory public Tomato-Village detection annotations before imaging experiments."""

import csv
import os
import io
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree


WORKSPACE = Path(__file__).resolve().parents[1]
RUN = WORKSPACE / "datasets" / "roi_trial"
REPO = RUN / "tomato-village-index"
FOLDER = "Variant-c(Object Detection)"
PUBLISHER = "https://github.com/mamta-joshi-gehlot/Tomato-Village"
REVISION = "dfc246014558293987b381c1edb49398d9f570ac"
GIT = ["git", "-c", "http.sslBackend=openssl", "-c", "fetch.negotiationAlgorithm=noop", "-C", str(REPO)]
CAMERA_FRAME = re.compile(r"IMG_?(20\d{12})(?:_\d+)?", re.IGNORECASE)
AUGMENT_SUFFIX = re.compile(r"_aug\d+$", re.IGNORECASE)
KNOWN_FOLD = Path(os.environ.get("ROBUSTLEAFLAB_REFERENCE", WORKSPACE / "reference")) / "outputs/train_improve_late/folds/field_extra_merged"
CLASSES = ("early_blight", "healthy", "late_blight", "leaf_miner", "magnesium_deficiency",
           "nitrogen_deficiency", "potassium_deficiency", "spotted_wilt_virus")


def git(*args):
    return subprocess.check_output([*GIT, *args])


def publisher_tree():
    tree = {}
    entries = git("ls-tree", "-rz", REVISION, "--", FOLDER).rstrip(b"\0").split(b"\0")
    for entry in entries:
        metadata, filename = entry.split(b"\t", 1)
        tree[PurePosixPath(filename.decode())] = metadata.decode().split()[-1]
    return tree


def known_training_stems():
    seen = set()
    paths = [KNOWN_FOLD / "train.csv", WORKSPACE / "datasets/matched_capture_trial/training.csv"]
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("dataset") == "tomato_village":
                    frame = CAMERA_FRAME.fullmatch(PurePosixPath(row["origin"]).stem)
                    if frame:
                        seen.add(frame[1])
    return seen


def annotations(tree):
    split_families = defaultdict(set)
    originals = []
    for path, oid in tree.items():
        if len(path.parts) != 4 or path.parts[2] != "pascal_voc" or path.suffix != ".xml":
            continue
        frame = AUGMENT_SUFFIX.sub("", path.stem)
        split_families[frame].add(path.parts[1])
        if frame != path.stem:
            continue
        for ext in (".jpg", ".JPG", ".jpeg", ".png"):
            image = path.parent.parent / "images" / path.with_suffix(ext).name
            if image in tree:
                originals.append((frame, path, oid, image, tree[image]))
                break
    return originals, split_families


def load_objects(oids):
    selection = b"\n".join(oid.encode("ascii") for oid in oids) + b"\n"
    subprocess.run([*GIT, "fetch", "--no-tags", "--no-write-fetch-head", "--no-auto-gc", "--filter=blob:none",
                    "origin", "--stdin"], input=selection, check=True, timeout=90)
    result = subprocess.run([*GIT, "cat-file", "--batch"], input=selection, capture_output=True, check=True, timeout=60)
    stream = io.BytesIO(result.stdout)
    data = {}
    for expected in oids:
        oid, kind, length = stream.readline().decode("ascii").split()
        if oid != expected or kind != "blob":
            raise ValueError(f"Unexpected object in annotation response: {oid}")
        data[oid] = stream.read(int(length))
        if stream.read(1) != b"\n":
            raise ValueError(f"Truncated annotation object: {oid}")
    return data


def parse_box_labels(payload, image_name):
    document = ElementTree.fromstring(payload)
    width, height = (int(document.findtext(f"size/{side}")) for side in ("width", "height"))
    if document.findtext("filename") != image_name or min(width, height) <= 0:
        raise ValueError(f"Annotation image identity mismatch: {image_name}")
    boxes = []
    for annotation in document.findall("object"):
        raw_label = float(annotation.findtext("name"))
        label = int(raw_label)
        if label != raw_label or not 0 <= label < len(CLASSES):
            raise ValueError(f"Unexpected publisher label: {raw_label}")
        box = [int(annotation.findtext(f"bndbox/{side}")) for side in ("xmin", "ymin", "xmax", "ymax")]
        if 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height:
            boxes.append({"class": CLASSES[label], "xyxy": box})
    return [width, height], boxes


def main():
    if git("remote", "get-url", "origin").decode().strip() != PUBLISHER + ".git":
        raise ValueError("Detection annotation index must point to the original publisher")
    tree = publisher_tree()
    originals, families = annotations(tree)
    known = known_training_stems()
    oids = sorted({item[2] for item in originals})
    if not oids:
        raise ValueError("No raw-frame annotations found in the pinned release")
    data = load_objects(oids)
    counts = defaultdict(Counter)
    records = []
    for frame, path, oid, image, image_oid in originals:
        dimensions, boxes = parse_box_labels(data[oid], image.name)
        stamp = CAMERA_FRAME.fullmatch(frame)
        capture_group = stamp[1][:8] if stamp else frame.split(" (")[0]
        seen = bool(stamp and stamp[1] in known)
        classes = {item["class"] for item in boxes}
        for label in CLASSES[:3]:
            if label in classes:
                counts[capture_group][label + "_frames"] += 1
                counts[capture_group][label + "_boxes"] += sum(item["class"] == label for item in boxes)
        counts[capture_group]["frames"] += 1
        counts[capture_group]["known_training_stem"] += seen
        records.append({"frame": frame, "capture_group": capture_group, "known_training_stem": seen,
                        "publisher_splits": sorted(families[frame]), "image": str(image), "image_git_oid": image_oid,
                        "annotation": str(path), "annotation_git_oid": oid, "size": dimensions, "boxes": boxes})
    result = {"publisher": PUBLISHER, "revision": REVISION, "label_order": CLASSES,
              "annotation_stem_families": len(families), "families_spanning_publisher_splits": sum(len(splits) > 1 for splits in families.values()),
              "unaugmented_image_annotation_pairs": len(records), "distinct_original_frames": len({row["frame"] for row in records}),
              "known_training_stems": len(known), "candidate_counts_by_filename_group": counts,
              "status": "Publisher metadata only. Image similarity, region validity and capture independence require review.",
              "frames": records}
    target = RUN / "annotation_inventory_all.json"
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != encoded:
        raise FileExistsError("The existing inventory differs from the pinned annotations")
    if not target.exists():
        with target.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(json.dumps({key: value for key, value in result.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
