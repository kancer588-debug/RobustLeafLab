"""Fetch a date-balanced sample through the public PPeDMaS tomato leaf search."""

import hashlib
import io
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

from PIL import Image

from prepare_online_run import FIELDS, rows, sha256, write_rows
from stage_hard_early_trial import DATA, FRAME, ROOT
from stage_roi_training import write_once


HOST = "https://ppedmas.org/"
SEARCH = HOST + "search_images.php"
IMAGE = re.compile(r'datasets/images/original_images/(\d{13})\.jpg')
LIMIT = 40
AGENT = {"User-Agent": "Mozilla/5.0"}


def public_catalog():
    browser = build_opener(HTTPCookieProcessor(CookieJar()))
    found = set()
    for page in range(5):
        params = {"action": "search", "crop": "Tomato", "labelled_WASCAL": "Alternaria Blight",
                  "image_type": "Diseased leaf"} if page == 0 else {"action": "next"}
        request = Request(SEARCH, data=urlencode(params).encode(), headers=AGENT)
        with browser.open(request, timeout=16) as result:
            html = result.read(2_000_000).decode("utf-8", "replace")
        current = set(IMAGE.findall(html))
        if not current - found:
            break
        found.update(current)
        if 'value="next"' not in html:
            break
    if not found:
        raise RuntimeError("No source-labeled original tomato frames were available")
    return found


def day(frame):
    return datetime.fromtimestamp(int(frame) / 1000, timezone.utc).date().isoformat()


def choose(frames):
    previous = {item[1] for row in rows(DATA / "matched_capture_trial" / "training.csv")
                if (item := FRAME.search(row.get("origin") or row["path"]))}
    by_day = defaultdict(list)
    for frame in sorted(frames - previous, key=lambda key: hashlib.sha256(key.encode()).hexdigest()):
        by_day[day(frame)].append(frame)
    groups = sorted(by_day)
    return [by_day[group][i] for i in range(max(map(len, by_day.values()), default=0))
            for group in groups if i < len(by_day[group])][:LIMIT]


def fetch(frame):
    path = ROOT / "publisher_originals" / f"{frame}.jpg"
    url = HOST + f"datasets/images/original_images/{frame}.jpg"
    try:
        if not path.exists():
            with urlopen(Request(url, headers=AGENT), timeout=18) as result:
                payload = result.read(10_000_001)
            if len(payload) > 10_000_000:
                raise ValueError("Source exceeded the bounded image size")
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            temp = path.with_suffix(".part")
            temp.write_bytes(payload)
            temp.replace(path)
        with Image.open(path) as image:
            width, height = image.size
        return {"path": str(path), "mask_path": "", "label": "early_blight", "sha256": sha256(path),
                "source": "TOM2024 original", "dataset": "tom2024", "origin": url, "capture_group": f"TOM2024|{day(frame)}",
                "width": width, "height": height, "view_type": "publisher_original_field_frame"}
    except (OSError, ValueError) as error:
        return {"origin": url, "error": str(error)}


def main():
    (ROOT / "publisher_originals").mkdir(parents=True, exist_ok=True)
    frames = public_catalog()
    sample = choose(frames)
    with ThreadPoolExecutor(max_workers=8) as pool:
        result = list(pool.map(fetch, sample))
    successful = [row for row in result if "error" not in row]
    output = ROOT / "publisher_originals.csv"
    write_rows(output, successful, (*FIELDS, "view_type"))
    write_once(ROOT / "publisher_originals_receipt.json", {"source": SEARCH, "manifest_sha256": sha256(output),
                      "catalog_frames": len(frames), "sampled": len(sample), "downloaded": len(successful),
                      "groups": dict(Counter(row["capture_group"] for row in successful)),
                      "failures": [row for row in result if "error" in row]})
    print("Source-labeled TOM2024 originals", len(successful), "of", len(sample), "catalog", len(frames), flush=True)


if __name__ == "__main__":
    main()
