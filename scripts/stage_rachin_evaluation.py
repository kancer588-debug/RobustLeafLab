"""Verify a publisher-sampled Rachin cohort for a separate three-class evaluation."""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from prepare_online_run import (
    EXTERNAL_MANIFESTS, MANIFEST_DIR, NearDuplicateIndex, fingerprint, rows,
)


TRIAL = MANIFEST_DIR.parent / "expanded_field_trial"
CATALOGS = {
    "healthy": "rachin_4d9f0de7-e68c-4851-8d95-96e54e79e4e4_files.json",
    "early_blight": "rachin_ed945ce8-e8e6-4b6a-8730-76af8c32cedb_files.json",
    "late_blight": "rachin_f1fb28eb-4ae9-4efa-b4dc-f40006f5c42f_files.json",
}
PUBLISHER = "https://data.mendeley.com/datasets/jttrv2w27r/1"
FIELDS = ("path", "mask_path", "label", "sha256", "source", "origin", "width", "height")
PER_CLASS_LIMIT = 80


def fetch(payload):
    label, entry = payload
    published = entry["content_details"]
    digest = published["sha256_hash"]
    url = published["download_url"]
    if (urlparse(url).scheme, urlparse(url).hostname) != ("https", "data.mendeley.com"):
        raise ValueError("Unexpected publisher download address")
    path = TRIAL / "rachin_images" / f"{digest}.jpg"
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return path
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "RobustLeafLab course dataset audit"})
            with urlopen(req, timeout=12) as response:
                content = response.read(published["size"] + 1)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("Publisher SHA-256 mismatch")
            temp = path.with_suffix(".part")
            temp.write_bytes(content)
            temp.replace(path)
            return path
        except (OSError, ValueError) as error:
            if attempt == 2:
                return str(error)
            time.sleep(0.5 * (attempt + 1))


def stage(catalog_dir):
    history = json.loads((MANIFEST_DIR / "historical_index.json").read_text(encoding="utf-8"))
    exact = set(history["exact"])
    exact.update(row["sha256"] for manifest in EXTERNAL_MANIFESTS for row in rows(manifest))
    index = NearDuplicateIndex()
    for signature in history["dhash"]:
        index.add(signature)
    references = [
        MANIFEST_DIR / "validation.csv", MANIFEST_DIR / "test.csv",
        MANIFEST_DIR.parent / "backbone_trial" / "fresh_test.csv",
        MANIFEST_DIR.parent / "early_source_trial" / "fresh_test.csv",
        MANIFEST_DIR.parent / "early_source_trial" / "replacement.csv",
    ]
    for manifest in references:
        for row in rows(manifest):
            exact.add(row["sha256"])
            signatures, _ = fingerprint(row["path"], resize_for_hash=True)
            index.add(signatures[0])
    (TRIAL / "rachin_images").mkdir(parents=True, exist_ok=True)
    selected = []
    failures = []
    for label, name in CATALOGS.items():
        listing = json.loads((catalog_dir / name).read_text(encoding="utf-8"))
        unique = {r["content_details"]["sha256_hash"]: r for r in listing}
        candidates = [(label, r) for _, r in sorted(unique.items()) if r["content_details"]["sha256_hash"] not in exact]
        chosen = []
        with ThreadPoolExecutor(max_workers=14) as pool:
            for offset in range(0, len(candidates), 32):
                batch = candidates[offset:offset + 32]
                for (_, entry), path in zip(batch, pool.map(fetch, batch)):
                    digest = entry["content_details"]["sha256_hash"]
                    if isinstance(path, str):
                        failures.append({"label": label, "origin": entry["filename"], "reason": path})
                        continue
                    try:
                        signatures, (width, height) = fingerprint(path, mirrored=True)
                    except OSError as error:
                        failures.append({"label": label, "origin": entry["filename"], "reason": str(error)})
                        continue
                    if digest in exact or any(index.matches(signature) for signature in signatures):
                        failures.append({"label": label, "origin": entry["filename"], "reason": "overlapping identity"})
                        continue
                    exact.add(digest)
                    index.add(signatures[0])
                    if len(chosen) < PER_CLASS_LIMIT:
                        chosen.append(dict(
                            path=str(path), mask_path="", label=label, sha256=digest, source="Rachin v1",
                            origin=entry["filename"], width=width, height=height,
                        ))
                if len(chosen) >= PER_CLASS_LIMIT:
                    break
        print(label, len(chosen), "candidate rows", len(candidates), flush=True)
        selected.extend(chosen)
    if len({r["label"] for r in selected}) != len(CATALOGS):
        raise RuntimeError("The publisher cohort is missing a target class")
    with (TRIAL / "rachin_candidates.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    with (TRIAL / "rachin_inventory.json").open("x", encoding="utf-8") as stream:
        json.dump({"publisher": PUBLISHER, "candidates": dict(Counter(r["label"] for r in selected)),
                   "failures_and_quarantines": failures}, stream, indent=2)
        stream.write("\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("catalog_dir", type=Path)
    stage(cli.parse_args().catalog_dir)


if __name__ == "__main__":
    main()
