"""Freeze label-reviewed late-blight training frames against existing holdouts."""

from collections import Counter
from pathlib import Path, PurePosixPath

from prepare_online_run import FIELDS, rows, sha256, write_rows
from stage_hard_early_trial import DATA, extra_references
from stage_local_leaf_sources import image_hashes, same_photo
from stage_roi_training import read, references, write_once


ROOT = DATA / "late_blind_trial"
CHEN = DATA / "local_leaf_trial"
NATURALIST = DATA / "late_protection_trial" / "public_late_review"


def candidates():
    review = read(CHEN / "chen_visual_review.json")
    labels = {PurePosixPath(item["origin"]).stem: item["classes"] for item in read(CHEN / "chen_label_inventory.json")}
    taiwan_stems = {same_photo(row["origin"]) for row in rows(CHEN / "taiwan_test.csv")}
    for row in rows(CHEN / "chen_review.csv"):
        origin = PurePosixPath(row["origin"])
        if (row["label"] == "late_blight" and int(row["review_id"]) in review["accepted_review_ids"]
                and origin.parts[:3] == ("Tomato-Disease", "train", "images") and labels.get(origin.stem) == [2]
                and same_photo(row["origin"]) not in taiwan_stems):
            yield {**row, "dataset": "chen_late", "capture_group": "Chen|unknown", "license": "CC BY 4.0",
                   "provenance": "Publisher YOLO annotations and frozen tomato-leaf review"}
    approval = read(NATURALIST / "visual_review.json")
    for photo in read(NATURALIST / "observations.json")["photos"]:
        if photo["review_id"] not in approval["retained_review_ids"]:
            continue
        path = Path(photo["path"])
        yield {"path": str(path), "label": "late_blight", "sha256": sha256(path), "source": "iNaturalist",
               "origin": photo["source"], "dataset": "inaturalist_late",
               "capture_group": f"iNaturalist|{photo['observer_key']}|{photo['date']}", "license": photo["license"],
               "provenance": f"Photo {photo['photo']}; {photo['attribution']}", "review_id": photo["review_id"]}


def main():
    ROOT.mkdir(exist_ok=True)
    manifest = ROOT / "supplement_training.csv"
    if manifest.exists():
        raise FileExistsError(manifest)
    exact, dhash, phash, unreadable = references()
    for row in (*extra_references(), *rows(DATA / "hard_early_trial" / "reviewed_training_pool.csv")):
        exact.add(row["sha256"])
        difference, perceptual = image_hashes(row["path"])[0]
        dhash.add(difference)
        phash.add(perceptual)
    kept, excluded = [], []
    for row in candidates():
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"Image changed {row['path']}")
        signatures = image_hashes(row["path"])
        if row["sha256"] in exact or any(dhash.matches(a) or phash.matches(b) for a, b in signatures):
            excluded.append({"source": row["source"], "origin": row["origin"], "reason": "exact or perceptual duplicate"})
            continue
        exact.add(row["sha256"])
        for difference, perceptual in signatures:
            dhash.add(difference)
            phash.add(perceptual)
        kept.append(row)
    if not kept:
        raise ValueError("No reviewed late-blight images survived the reference screen")
    write_rows(manifest, kept, (*FIELDS, "license", "provenance", "review_id"))
    counts = dict(Counter(row["source"] for row in kept))
    write_once(ROOT / "training_freeze.json", {"manifest_sha256": sha256(manifest), "counts": counts,
               "unreadable_references": unreadable, "excluded": excluded,
               "group_counts": dict(Counter(row["capture_group"] for row in kept)),
               "role": "Training only. Unknown Chen capture batches are conservatively grouped together."})
    print("Accepted late-blight supplement", counts, "duplicates", len(excluded), flush=True)


if __name__ == "__main__":
    main()
