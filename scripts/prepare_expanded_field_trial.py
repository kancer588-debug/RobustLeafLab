"""Freeze equal-class additions and an evaluation cohort for the AC42 expansion trial."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from prepare_online_run import (
    EXTERNAL_MANIFESTS, FIELDS, MANIFEST_DIR, REFERENCE, SOURCE_DIR,
    NearDuplicateIndex, fingerprint, rows, sha256,
)
from stage_rachin_evaluation import TRIAL


CLASSES = ("healthy", "early_blight", "late_blight")
ADDITIONS_PER_CLASS = 120
BACKBONE = MANIFEST_DIR.parent / "backbone_trial"
EARLY_TRIAL = MANIFEST_DIR.parent / "early_source_trial"
V5 = REFERENCE / "data_v5"
OUTPUT_FIELDS = (*FIELDS, "capture_date")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, records):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def source_of(row):
    return row.get("dataset") or row["source"]


def interleave(records):
    groups = defaultdict(list)
    for row in sorted(records, key=lambda item: item["sha256"]):
        groups[source_of(row)].append(row)
    for at in range(max(map(len, groups.values()), default=0)):
        for _, group in sorted(groups.items()):
            if at < len(group):
                yield group[at]


def excluded_evaluation():
    return {
        "development_field": BACKBONE / "development_field.csv",
        "development_malik": BACKBONE / "development_malik.csv",
        "plantcity_replay": MANIFEST_DIR / "test.csv",
        "plantwild_replay": BACKBONE / "fresh_test.csv",
        "imtiaz_replay": EARLY_TRIAL / "fresh_test.csv",
        "reference_validation": V5 / "val.csv",
        "reference_field_validation": V5 / "field_val.csv",
        "reference_field_test": V5 / "field_test.csv",
        "reference_pv_test": REFERENCE / "data" / "test.csv",
        "plantcity_raw": SOURCE_DIR / "plantcity-raw.csv",
        "malik_raw": SOURCE_DIR / "malik-raw.csv",
        "plantwild_raw": BACKBONE / "plantwild_staged.csv",
    }


def build_training():
    TRIAL.mkdir(exist_ok=True)
    base_path = MANIFEST_DIR / "train.csv"
    if sha256(base_path) != load(MANIFEST_DIR / "freeze.json")["manifest_sha256"]["train"]:
        raise ValueError("The starting training list changed")
    base = rows(base_path)
    references = excluded_evaluation()
    exact, near = {row["sha256"] for row in base}, NearDuplicateIndex()
    to_hash = {}
    for path in [base_path, *references.values()]:
        for row in rows(path):
            exact.add(row["sha256"])
            to_hash.setdefault(row["sha256"], row["path"])
    for path in EXTERNAL_MANIFESTS:
        exact.update(row["sha256"] for row in rows(path))
    for filename in to_hash.values():
        signatures, _ = fingerprint(filename, resize_for_hash=True)
        near.add(signatures[0])

    archival = rows(V5 / "train.csv")
    review_path = EARLY_TRIAL / "visual_review.json"
    reviewed = set(load(review_path)["tom2024"])
    reviewed_early = [row for row in rows(EARLY_TRIAL / "tom2024_staged.csv")
                      if int(row["review_id"]) in reviewed]
    if len(reviewed_early) != len(reviewed):
        raise ValueError("Reviewed TOM2024 images no longer match their archived list")
    field = [
        row for row in archival if row.get("dataset") in {"tom2024", "tomato_village"}
    ] + reviewed_early
    controlled = [row for row in archival if row["source"] == "plantvillage"]
    selected = []
    quarantined = Counter()
    for label in CLASSES:
        additions = []
        pools = (field, controlled)
        for pool in pools:
            for row in interleave(row for row in pool if row["label"] == label):
                if len(additions) == ADDITIONS_PER_CLASS:
                    break
                digest = row["sha256"]
                if digest in exact:
                    quarantined["exact_or_reserved"] += 1
                    continue
                signatures, _ = fingerprint(row["path"], mirrored=True)
                if any(near.matches(signature) for signature in signatures):
                    quarantined["perceptual_match"] += 1
                    continue
                if sha256(Path(row["path"])) != digest:
                    raise ValueError(f"Image content differs from its provenance: {row['path']}")
                exact.add(digest)
                near.add(signatures[0])
                additions.append(row)
        if len(additions) != ADDITIONS_PER_CLASS:
            raise ValueError(f"Insufficient screened additions for {label}: {len(additions)}")
        selected.extend(additions)

    partitions = {"control_train": base, "expanded_train": base + selected, "additions": selected}
    for name, records in partitions.items():
        write_csv(TRIAL / f"{name}.csv", records)
    frozen = {
        "training_sha256": {name: sha256(TRIAL / f"{name}.csv") for name in partitions},
        "training_counts": {
            key: dict(Counter(row["label"] for row in value)) for key, value in partitions.items()
        },
        "additions_by_source": {
            label: dict(Counter(source_of(row) for row in selected if row["label"] == label))
            for label in CLASSES
        },
        "quarantined": dict(quarantined),
        "source_sha256": {
            "original": sha256(base_path), "archive": sha256(V5 / "train.csv"),
            "tom2024_review": sha256(review_path),
        },
        "excluded_manifest_sha256": {str(path): sha256(path) for path in references.values()},
    }
    write_json(TRIAL / "training_freeze.json", frozen)
    print(json.dumps(frozen["training_counts"], indent=2), flush=True)
    print(json.dumps(frozen["additions_by_source"], indent=2), flush=True)


def freeze_evaluation(review_path):
    approved = set(load(review_path)["accepted_sha256"])
    candidates = rows(TRIAL / "rachin_candidates.csv")
    selected = [row for row in candidates if row["sha256"] in approved]
    if len(selected) != len(approved):
        raise ValueError("The review contains an unknown image")
    known = {row["sha256"] for row in rows(TRIAL / "expanded_train.csv")}
    if any(row["sha256"] in known for row in selected):
        raise ValueError("Evaluation images overlap training")
    index = NearDuplicateIndex()
    for row in rows(TRIAL / "expanded_train.csv"):
        signatures, _ = fingerprint(row["path"], resize_for_hash=True)
        index.add(signatures[0])
    if any(index.matches(value) for row in selected for value in fingerprint(
            row["path"], mirrored=True)[0]):
        raise ValueError("Potential near duplicate between evaluation and expanded training")
    counts = Counter(row["label"] for row in selected)
    if any(counts[label] == 0 for label in CLASSES):
        raise ValueError(f"The reviewed evaluation lacks target coverage: {counts}")
    path = TRIAL / "rachin_evaluation.csv"
    write_csv(path, selected)
    write_json(TRIAL / "evaluation_freeze.json", {
        "sha256": sha256(path), "review_sha256": sha256(review_path), "counts": dict(counts),
        "publisher": "https://data.mendeley.com/datasets/jttrv2w27r/1",
        "scope": "New images from the previously sampled Rachin publisher; prior images excluded",
        "historical_replays": {
            key: {"path": str(value), "sha256": sha256(value)}
            for key, value in excluded_evaluation().items() if key.endswith("_replay")
        },
        "external_audits": {
            name: {"path": str(value), "sha256": sha256(value)}
            for name, value in zip(("new_field_audit", "plantdoc_audit"), EXTERNAL_MANIFESTS)
        },
    })
    print(json.dumps(counts), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("training")
    evaluate = commands.add_parser("evaluation")
    evaluate.add_argument("review", type=Path)
    args = parser.parse_args()
    if args.phase == "training":
        build_training()
    else:
        freeze_evaluation(args.review)


if __name__ == "__main__":
    main()
