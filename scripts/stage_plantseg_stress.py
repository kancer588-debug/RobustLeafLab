"""Stage an early/late-only PlantSeg stress set without reading model predictions."""

import csv
import hashlib
import io
import re
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import ZipFile

from PIL import Image, ImageDraw

from fetch_online_images import USER_AGENT, RemoteZip, batches, extract_member
from prepare_online_run import FIELDS, rows, sha256, write_rows
from stage_local_leaf_sources import image_hashes
from stage_roi_training import STAGE, read, references, write_once
from train_online_candidate import WORKSPACE


ROOT = WORKSPACE / "datasets" / "view_ablation" / "plantseg"
DOI = "https://doi.org/10.5281/zenodo.17719108"
ARCHIVE_PATH = ROOT / "publisher.zip"
ARCHIVE_SIZE = 1057281724
ARCHIVE_MD5 = "9358a66dff88cdd15c4fe009763c40a3"
ARCHIVE_URL = "https://zenodo.org/records/17719108/files/plantseg.zip?download=1"
LABELS = {"tomato early blight": "early_blight", "tomato late blight": "late_blight"}
LIMIT = 64
DOMAIN_LIMIT = 12


def release_members(archive):
    payload = archive.read("plantseg/Metadata.csv")
    editions = {PurePosixPath(member.filename).name: member for member in archive.infolist()
                if member.filename.startswith("plantseg/images/") and member.filename.endswith(".jpg")}
    metadata = [row for row in csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
                if row["Plant"].casefold() == "tomato" and row["Disease"].casefold() in LABELS]
    if len(metadata) != 270 or len({row["Name"] for row in metadata}) != len(metadata):
        raise ValueError("Unexpected PlantSeg class catalog")
    candidates = []
    for row in metadata:
        domain = urlparse(row["URL"]).netloc.casefold().removeprefix("www.")
        if domain and domain != "source.roboflow.com":
            candidates.append((editions[row["Name"]], row, domain))
    return sorted(candidates, key=lambda item: item[0].header_offset)


def range_groups(intervals):
    group, total_bytes = [], 0
    for start, end, entries in intervals:
        if group and (len(group) == 16 or total_bytes + end - start > 2 * 1024**2):
            yield group
            group, total_bytes = [], 0
        group.append((start, end, entries))
        total_bytes += end - start
    if group:
        yield group


def multipart_ranges(remote, group):
    if len(group) == 1:
        start, end, _ = group[0]
        return [remote.fetch(start, end)]
    ranges = ",".join(f"{start}-{end - 1}" for start, end, _ in group)
    request = Request(remote.url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={ranges}"})
    byte_limit = sum(end - start for start, end, _ in group) + len(group) * 512 + 512
    for attempt in range(3):
        try:
            with urlopen(request, timeout=40) as response:
                if response.status != 206:
                    raise ValueError("Publisher did not honor multipart byte ranges")
                body = response.read(byte_limit + 1)
            break
        except OSError as error:
            if attempt == 2:
                raise
            if isinstance(error, HTTPError) and error.code == 429:
                time.sleep(int(error.headers.get("Retry-After", 60)))
            else:
                time.sleep(2 ** attempt)
    boundary, end_line, rest = body.lstrip(b"\r\n").partition(b"\n")
    boundary = boundary.rstrip(b"\r")
    if len(body) > byte_limit or not end_line or not boundary.startswith(b"--"):
        raise ValueError("Invalid PlantSeg multipart range response")
    retrieved = []
    for index, (start, end, _) in enumerate(group):
        section = re.search(rb"\r?\n\r?\n", rest[:512])
        expected = f"Content-Range: bytes {start}-{end - 1}/{remote.size}".encode()
        if section is None or expected not in rest[:section.start()].splitlines():
            raise ValueError("PlantSeg archive response did not match the requested range")
        offset = section.end()
        retrieved.append(rest[offset:offset + end - start])
        line, end_line, rest = rest[offset + end - start:].lstrip(b"\r\n").partition(b"\n")
        marker = boundary + (b"--" if index == len(group) - 1 else b"")
        if line.rstrip(b"\r") != marker:
            raise ValueError("PlantSeg archive response had an invalid multipart boundary")
    return retrieved


def retrieve(archive, candidates, remote=None):
    directory = ROOT / "images"
    directory.mkdir(exist_ok=True)
    staged, pending = [], []

    def store(path, payload):
        partial = path.with_suffix(".tmp")
        partial.write_bytes(payload)
        partial.replace(path)

    def extract(group):
        try:
            payloads = multipart_ranges(remote, group)
        except ValueError:
            if len(group) == 1:
                raise
            payloads = [remote.fetch(start, end) for start, end, _ in group]
        for (start, _, entries), payload in zip(group, payloads, strict=True):
            for member, _, _ in entries:
                store(directory / PurePosixPath(member.filename).name, extract_member(payload, start, member))

    for member, row, domain in candidates:
        path = directory / PurePosixPath(member.filename).name
        if path.is_file():
            payload = path.read_bytes()
            if len(payload) != member.file_size or zlib.crc32(payload) != member.CRC:
                raise ValueError(f"Local image differs from archive: {path.name}")
        elif remote is None:
            store(path, archive.read(member))
        else:
            pending.append((member, row, domain))
        staged.append((path, row, domain))
    if pending:
        groups = list(range_groups(batches(pending)))
        with ThreadPoolExecutor(max_workers=4) as executor:
            for index, _ in enumerate(executor.map(extract, groups), start=1):
                print(f"Verified PlantSeg archive ranges {index}/{len(groups)}", flush=True)
    return staged


def select(candidates):
    exact, difference, perceptual, unreadable = references()
    for row in read(STAGE / "screened.json")["records"]:
        exact.add(row["sha256"])
        for first, second in image_hashes(row["path"]):
            difference.add(first)
            perceptual.add(second)
    counts, groups, rejected = Counter(), Counter(), Counter()
    chosen = []
    for path, row, domain in sorted(candidates, key=lambda item: hashlib.sha256(item[0].name.encode()).hexdigest()):
        label = LABELS[row["Disease"].casefold()]
        if counts[label] >= LIMIT or groups[label, domain] >= DOMAIN_LIMIT:
            continue
        digest = sha256(path)
        signatures = image_hashes(path)
        if digest in exact or any(difference.matches(first) or perceptual.matches(second) for first, second in signatures):
            rejected[label] += 1
            continue
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        chosen.append({"path": str(path), "mask_path": "", "label": label, "sha256": digest, "source": "plantseg",
                       "dataset": "plantseg", "origin": path.name, "capture_group": domain, "width": width, "height": height})
        counts[label] += 1
        groups[label, domain] += 1
        exact.add(digest)
        for first, second in signatures:
            difference.add(first)
            perceptual.add(second)
    return chosen, counts, groups, rejected, unreadable


def review_sheet(examples):
    side = 130
    cols = 8
    canvas = Image.new("RGB", (side * cols, (side + 32) * 4), "white")
    draw = ImageDraw.Draw(canvas)
    for row_ix, label in enumerate(LABELS.values()):
        chosen = [row for row in examples if row["label"] == label][:16]
        for index, row in enumerate(chosen):
            x, y = (index % cols) * side, (row_ix * 2 + index // cols) * (side + 32)
            with Image.open(row["path"]) as source:
                image = source.convert("RGB")
                image.thumbnail((side, side))
            canvas.paste(image, (x + (side - image.width) // 2, y + (side - image.height) // 2))
            draw.text((x + 3, y + side + 1), label, fill="black")
            draw.text((x + 3, y + side + 15), row["origin"].removeprefix("tomato_").removesuffix(".jpg")[-20:], fill="black")
    canvas.save(ROOT / "review.jpg", quality=90)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    test = ROOT / "test.csv"
    receipt = ROOT / "freeze.json"
    if receipt.exists():
        if read(receipt)["sha256"] != sha256(test) or any(sha256(Path(row["path"])) != row["sha256"] for row in rows(test)):
            raise ValueError("The frozen PlantSeg data changed")
        return
    if ARCHIVE_PATH.is_file():
        if ARCHIVE_PATH.stat().st_size != ARCHIVE_SIZE:
            raise ValueError("Unexpected PlantSeg archive size")
        with ARCHIVE_PATH.open("rb") as stream:
            if hashlib.file_digest(stream, "md5").hexdigest() != ARCHIVE_MD5:
                raise ValueError("PlantSeg archive differs from the publisher checksum")
        with ZipFile(ARCHIVE_PATH) as archive:
            candidates = retrieve(archive, release_members(archive))
    else:
        remote = RemoteZip({"content_details": {"download_url": ARCHIVE_URL, "size": ARCHIVE_SIZE}})
        with ZipFile(remote) as archive:
            candidates = retrieve(archive, release_members(archive), remote)
    print(f"Verified {len(candidates)} public PlantSeg candidates", flush=True)
    chosen, counts, groups, rejected, unreadable = select(candidates)
    if any(counts[label] == 0 for label in LABELS.values()):
        raise ValueError("One target disease is absent after screening")
    write_rows(test, chosen, FIELDS)
    write_once(receipt, {"path": str(test), "sha256": sha256(test), "source": DOI, "counts": dict(counts),
                         "published_archive_md5": ARCHIVE_MD5, "full_archive_md5_verified": ARCHIVE_PATH.is_file(),
                         "screened_candidate_count": len(candidates),
                         "domains": {label: len({domain for category, domain in groups if category == label}) for label in counts},
                         "rejected_near_duplicates": dict(rejected), "unreadable_reference_paths": unreadable,
                         "role": "source_related_disease_only_stress", "scope": "Publisher-labeled, no healthy class. PlantSeg relates to PlantWild. "
                         "Domain groups are URL domains, not confirmed capture or plant identifiers."})
    review_sheet(chosen)
    print("PlantSeg frozen", dict(counts), dict(rejected), flush=True)


if __name__ == "__main__":
    main()
