"""Hold attributed extension images out of training for early-override development."""

import argparse
import io
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageOps

from prepare_online_run import FIELDS, rows, sha256, write_rows
from stage_local_leaf_sources import image_hashes
from stage_roi_training import STAGE, read, references, write_once
from train_online_candidate import WORKSPACE


ROOT = WORKSPACE / "datasets" / "early_override_trial"
CORNELL = "https://blogs.cornell.edu/livegpath/"
NC = "https://content.ces.ncsu.edu/"
MD = "https://extension.umd.edu/"
PURDUE = "https://ag.purdue.edu/department/arge/swpap/"
INRAE = "https://ephytia.inrae.fr/"
SOURCES = (
    ("cornell_extension", "early_blight", CORNELL + "gallery/tomato/early-blight/", CORNELL, (
        "files/2021/02/early_blight_leavesx1200.jpg", "files/2021/02/early_blight_leafcux1200.jpg",
        "files/2019/09/tomato-early-blight1x2400.jpg", "files/2019/09/tomato-early-blight2x2400.jpg",
        "files/2021/02/IMG_0194.jpg", "files/2021/02/IMG_0195.jpg")),
    ("nc_state_extension", "early_blight", NC + "early-blight-of-tomato", NC + "media/images/", (
        "IMG_1301.JPG", "IMG_1302.jpeg", "Shoemaker_7068.JPG", "IMG_1780.jpeg")),
    ("umd_extension", "early_blight", MD + "resource/early-blight-tomatoes", MD, (
        "sites/extension.umd.edu/files/styles/optimized/public/2021-05/hgic_veg_early%20blight%20lesions%20on%20tomato_1573858_1200.jpg?itok=-1EwR8BC",
        "sites/extension.umd.edu/files/styles/optimized/public/2021-03/hgic_veg_disease_early%20blight%20lesions%20with%20yellow%20haloes_500.jpg?itok=Zi6A64YP",
        "sites/extension.umd.edu/files/styles/optimized/public/2021-05/hgic_veg_early%20blight_severe_600.jpg?itok=AGT0J0lr")),
    ("utah_state_extension", "early_blight", "https://extension.usu.edu/vegetableguide/tomato-pepper-eggplant/early-blight",
     "https://extension.usu.edu/vegetableguide/images/solanaceae-images/", ("early-blight-tomato1.jpg",)),
    ("colorado_state_extension", "early_blight", "https://extension.colostate.edu/resource/tomato-early-blight/",
     "https://extension.colostate.edu/wp-content/uploads/2025/10/", ("CMG-GN-Tomato-Early-Blight-fig1.png",)),
    ("cornell_extension", "late_blight", CORNELL + "gallery/tomato/tomato-late-blight/", CORNELL + "files/2021/02/", (
        "late_blight_tomato_leaf3x1200.jpg", "late_blight_tomato_leaf4x1200.jpg", "late_blight_tomato_leaf5x1200.jpg",
        "lb_leaf_lesions1x1200.jpg", "late_blight_2013_longisland2x1200.jpg", "late_blight_tomato_leaf_johnsonx1200.jpg",
        "late_blight_drought_leaves1x1200.jpg", "late_blight_tomato_unusual-symptomsx1200.jpg")),
    ("nc_state_extension", "late_blight", NC + "tomato-late-blight", NC + "media/images/", (
        "IMG_5808.jpg", "IMG_5813.jpg", "IMG_0609.jpeg", "IMG_0019.jpg")),
    ("purdue_extension", "early_blight", PURDUE + "tomato-diseases.html", PURDUE + "_media/",
     tuple(f"early-blight-of-tomato-{i}.jpg" for i in (1, 3, 4, 5, 6, 8, 9, 10))),
    ("purdue_extension", "late_blight", PURDUE + "tomato-diseases.html", PURDUE + "_media/",
     tuple(f"late-blight-of-tomato-{i}.jpg" for i in (2, 4, 5))),
    ("inrae", "early_blight", INRAE + "fr/C/5202/Tomate-Principaux-symptomes", INRAE, (
        "fr/IC/24681/Alternaria-solani1", "fr/IC/24682/Alternaria-solani2", "fr/IC/24683/Alternaria-solani3")),
    ("mofga_via_uconn", "early_blight", "https://ipm.cahnr.uconn.edu/tomato-early-blight/",
     "https://ipm-cahnr.media.uconn.edu/wp-content/uploads/sites/3216/2026/06/", (
        "Screenshot-2026-06-16-at-1.57.20-PM-e1781633593300-288x400.png",)),
)


def retrieve(item):
    index, group, label, page, url = item
    path = ROOT / "images" / f"photo_{index:02d}{Path(urlsplit(url).path).suffix.lower()}"
    try:
        if not path.is_file():
            with urlopen(Request(url, headers={"User-Agent": "RobustLeafLab course image audit"}), timeout=24) as stream:
                payload = stream.read(10 * 1024**2 + 1)
            if len(payload) > 10 * 1024**2:
                raise ValueError("Publisher image exceeded the download cap")
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            part = path.with_suffix(".part")
            part.write_bytes(payload)
            part.replace(path)
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, ValueError) as exc:
        return {"review_id": index, "url": url, "page": page, "error": str(exc)}
    return {"review_id": index, "path": str(path), "mask_path": "", "label": label, "sha256": sha256(path),
            "source": group, "dataset": "extension_public_development", "origin": url,
            "capture_group": group, "width": width, "height": height, "page": page}


def contact_sheet(records):
    side, columns = 156, 6
    canvas = Image.new("RGB", (columns * side, ((len(records) + columns - 1) // columns) * (side + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(records):
        x, y = i % columns * side, i // columns * (side + 24)
        with Image.open(row["path"]) as image:
            photo = ImageOps.exif_transpose(image).convert("RGB")
            photo.thumbnail((side, side))
        canvas.paste(photo, (x + (side - photo.width) // 2, y + (side - photo.height) // 2))
        draw.text((x + 2, y + side + 3), f"{row['review_id']:02d} {row['label']} {row['source'].split('_')[0]}", fill="black")
    canvas.save(ROOT / "review.jpg", quality=88)


def stage():
    (ROOT / "images").mkdir(parents=True, exist_ok=True)
    photos = [(group, label, page, base + filename) for group, label, page, base, names in SOURCES for filename in names]
    with ThreadPoolExecutor(max_workers=8) as workers:
        result = list(workers.map(retrieve, [(i, *item) for i, item in enumerate(photos, 1)]))
    downloaded = [row for row in result if "error" not in row]
    if not downloaded:
        raise RuntimeError("No public extension images could be decoded")
    exact, difference, perceptual, unreadable = references()
    extra = read(STAGE / "screened.json")["records"]
    datasets = WORKSPACE / "datasets"
    for manifest in (datasets / "view_ablation" / "plantseg" / "test.csv",
                     datasets / "health_rescue_trial" / "healthy_fit_reviewed.csv",
                     datasets / "health_rescue_trial" / "early_development.csv"):
        extra.extend(rows(manifest))
    for row in extra:
        exact.add(row["sha256"])
        first, second = image_hashes(row["path"])[0]
        difference.add(first)
        perceptual.add(second)
    accepted, duplicates = [], []
    for row in downloaded:
        signatures = image_hashes(row["path"])
        if row["sha256"] in exact or any(difference.matches(a) or perceptual.matches(b) for a, b in signatures):
            duplicates.append({"review_id": row["review_id"], "url": row["origin"]})
            continue
        accepted.append(row)
        exact.add(row["sha256"])
        for first, second in signatures:
            difference.add(first)
            perceptual.add(second)
    if not accepted:
        raise RuntimeError("All extension images matched an existing image")
    inventory = {"records": accepted, "network_failures": [r for r in result if "error" in r], "duplicates": duplicates,
                 "unreadable_reference_paths": unreadable}
    write_once(ROOT / "inventory.json", inventory)
    contact_sheet(accepted)
    print("Public extension staging:", len(photos), len(downloaded), Counter(r["label"] for r in accepted), duplicates, flush=True)


def freeze(rejected):
    inventory = ROOT / "inventory.json"
    candidates = read(inventory)["records"]
    if not set(rejected) <= {row["review_id"] for row in candidates}:
        raise ValueError("A visual review ID is absent from the screened photo inventory")
    keep = [row for row in candidates if row["review_id"] not in rejected]
    counts = Counter(row["label"] for row in keep)
    if not counts["early_blight"] or not counts["late_blight"]:
        raise ValueError("The supplemental challenge must include both labeled diseases")
    manifest = ROOT / "development.csv"
    lock = ROOT / "freeze.json"
    if manifest.exists() or lock.exists():
        raise FileExistsError("The supplemental development set is already frozen")
    normalized = []
    for row in keep:
        with Image.open(row["path"]) as image:
            if image.getexif().get(274, 1) == 1:
                continue
            directory = ROOT / "orientation_normalized"
            directory.mkdir(exist_ok=True)
            canonical = directory / f"photo_{row['review_id']:02d}.png"
            photo = ImageOps.exif_transpose(image).convert("RGB")
            photo.save(canonical)
            normalized.append(row["review_id"])
            row.update(path=str(canonical), sha256=sha256(canonical), width=photo.width, height=photo.height)
    write_rows(manifest, keep, FIELDS)
    groups = Counter(row["capture_group"] for row in keep)
    write_once(lock, {"manifest_sha256": sha256(manifest), "inventory_sha256": sha256(inventory), "counts": dict(counts),
                      "groups": dict(groups), "visual_exclusions": sorted(rejected), "exif_orientation_normalized": normalized,
                      "role": "Supplemental development only. Published image captions supply disease labels. "
                              "Publisher groups conservatively combine photographs with unknown dependence. "
                              "Neither capture group independence nor complete three-class coverage is claimed."})
    print("Frozen public development:", dict(counts), dict(groups), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "freeze"))
    parser.add_argument("--exclude", nargs="*", type=int, default=[])
    args = parser.parse_args()
    if args.phase == "stage":
        stage()
    else:
        freeze(args.exclude)


if __name__ == "__main__":
    main()
