"""Extract only the target classes from the publishers' public ZIP archives."""

import csv
import hashlib
import io
import json
import re
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


ROOT = Path(__file__).resolve().parents[1] / "datasets" / "online_sources"
USER_AGENT = "RobustLeafLab/1.0"
LOCAL_HEADER = struct.Struct("<4s5H3I2H")
MALIK_CAMERA_NAME = re.compile(r"IMG_20251102_\d{6}\.jpg", re.IGNORECASE)
MALIK_PER_CLASS = 16
LABELS = {
    "malik": {
        "Tomato_Healthy_leaf": "healthy",
        "Tomato_Early_blight_leaf": "early_blight",
        "Tomato_leaf_late_blight": "late_blight",
    },
    "plantcity": {
        "tomato_healthy_leaf": "healthy",
        "tomato_early_blight": "early_blight",
        "tomato_late_blight": "late_blight",
    },
}


class RemoteZip(io.RawIOBase):
    def __init__(self, metadata):
        self.url = metadata["content_details"]["download_url"]
        self.size = int(metadata["content_details"]["size"])
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0, self.position, self.size)[whence] + offset
        return self.position

    def fetch(self, start, end):
        expected = f"bytes {start}-{end - 1}/{self.size}"
        request = Request(self.url, headers={
            "User-Agent": USER_AGENT, "Range": f"bytes={start}-{end - 1}",
        })
        for attempt in range(4):
            try:
                with urlopen(request, timeout=35) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise OSError(f"Publisher returned an unexpected byte range for {expected}")
                    content = response.read(end - start + 1)
                if len(content) != end - start:
                    raise OSError(f"Incomplete response for {expected}")
                return content
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    def read(self, size=-1):
        start = self.position
        end = self.size if size < 0 else min(start + size, self.size)
        if end <= start:
            return b""
        content = self.fetch(start, end)
        self.position = end
        return content


def select(source, member):
    parts = PurePosixPath(member.filename).parts
    prefix = ("Tomato Dataset", "Dataset (raw)") if source == "malik" else ("test",)
    if parts[:-2] != prefix or member.is_dir():
        return None
    if source == "malik" and not MALIK_CAMERA_NAME.fullmatch(parts[-1]):
        return None
    return LABELS[source].get(parts[-2])


def batches(entries):
    block = []
    start = end = None
    for entry in entries:
        member = entry[0]
        if block and (member.header_offset - end > 131072 or member._end_offset - start > 4 * 1024**2):
            yield start, end, block
            block = []
        if not block:
            start = member.header_offset
        end = member._end_offset
        block.append(entry)
    if block:
        yield start, end, block


def extract_member(data, from_byte, member):
    header_offset = member.header_offset - from_byte
    header = LOCAL_HEADER.unpack_from(data, header_offset)
    if header[0] != b"PK\x03\x04" or header[3] != member.compress_type:
        raise ValueError(f"Invalid local ZIP header for {member.filename}")
    start = header_offset + LOCAL_HEADER.size + header[9] + header[10]
    payload = data[start:start + member.compress_size]
    if member.compress_type == ZIP_DEFLATED:
        payload = zlib.decompress(payload, -zlib.MAX_WBITS)
    elif member.compress_type != ZIP_STORED:
        raise ValueError(f"Unsupported ZIP compression for {member.filename}")
    if len(payload) != member.file_size or zlib.crc32(payload) != member.CRC:
        raise ValueError(f"ZIP member checksum mismatch for {member.filename}")
    return payload


def fetch_source(source, metadata):
    remote = RemoteZip(metadata)
    entries = []
    with ZipFile(remote) as archive:
        members = sorted(archive.infolist(), key=lambda item: item.header_offset)
        for member in members:
            label = select(source, member)
            if label is not None:
                entries.append((member, label))
    members_path = ROOT / f"{source}-zip-members.json"
    members_path.write_text(json.dumps([
        {"name": member.filename, "offset": member.header_offset, "size": member.file_size,
         "compressed": member.compress_size, "crc32": member.CRC}
        for member in members if not member.is_dir()
    ], indent=2), encoding="utf-8")
    if source == "malik":
        sample = []
        for label in LABELS[source].values():
            group = (entry for entry in entries if entry[1] == label)
            sample.extend(sorted(group, key=lambda entry: hashlib.sha256(entry[0].filename.encode()).digest())
                          [:MALIK_PER_CLASS])
        entries = sorted(sample, key=lambda entry: entry[0].header_offset)
    if not entries:
        raise ValueError(f"No target images found for {source}")
    print(source, len(entries), "target members", flush=True)
    destination = ROOT / source
    destination.mkdir(exist_ok=True)

    def fetch_batch(block):
        start, end, members = block
        results = []
        missing = [member for member, _ in members if not (destination / f"{member.header_offset}.jpg").exists()]
        contents = remote.fetch(start, end) if missing else None
        for member, label in members:
            path = destination / f"{member.header_offset}.jpg"
            if path.exists():
                payload = path.read_bytes()
                if len(payload) != member.file_size or zlib.crc32(payload) != member.CRC:
                    raise ValueError(f"Existing member failed checksum. Inspect {path}")
            else:
                payload = extract_member(contents, start, member)
                part = path.with_suffix(".part")
                part.write_bytes(payload)
                part.replace(path)
            results.append({
                "path": str(path.resolve()), "label": label, "source": source,
                "origin": member.filename, "sha256": hashlib.sha256(payload).hexdigest(),
            })
        return results

    rows = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        for result in pool.map(fetch_batch, batches(entries)):
            rows.extend(result)
            if len(rows) % 100 < len(result):
                print(source, len(rows), "verified", flush=True)
    with (ROOT / f"{source}-raw.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(source, len(rows), "complete", flush=True)


def main():
    malik = json.loads((ROOT / "malik-v2.publisher.json").read_text(encoding="utf-8"))
    plantcity = next(
        row for row in json.loads((ROOT / "plantcity-v4.publisher.json").read_text(encoding="utf-8"))
        if row["filename"] == "test.zip"
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(fetch_source, source, meta)
                 for source, meta in (("malik", malik), ("plantcity", plantcity))]
        for task in tasks:
            task.result()


if __name__ == "__main__":
    main()
