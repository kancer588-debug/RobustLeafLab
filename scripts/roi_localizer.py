"""Train a class-agnostic lesion localizer on frozen DINOv2 patch tokens."""

import re
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from run_full_frame_trial import FULL_TRANSFORM
from run_subtype_trial import SubtypeProbe
from train_online_candidate import CLASSES, sha256


RESOLUTION = 336
GRID = RESOLUTION // 14
LOCAL_FRACTION = 0.50
LOCALIZER_EPOCHS = 10
LOCALIZER_SEED = 42
LOCALIZER_BATCH = 8
HEALTHY_TARGET = 72
WINDOW = torch.arange(GRID).float().add(0.5).div(GRID)
Y_COORD, X_COORD = torch.meshgrid(WINDOW, WINDOW, indexing="ij")


def fit_canvas(image):
    factor = RESOLUTION / max(image.size)
    width, height = (max(1, round(dimension * factor)) for dimension in image.size)
    left, top = (RESOLUTION - width) // 2, (RESOLUTION - height) // 2
    canvas = Image.new("RGB", (RESOLUTION, RESOLUTION), (124, 116, 104))
    canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR), (left, top))
    layout = torch.tensor([left, top, width, height], dtype=torch.float32) / RESOLUTION
    return FULL_TRANSFORM(canvas), layout


def area_mask(layout, boxes=None):
    left, top, width, height = layout
    x, y = (X_COORD - left) / width, (Y_COORD - top) / height
    valid = (x >= 0) & (x < 1) & (y >= 0) & (y < 1)
    if boxes is None:
        return valid
    positive = torch.zeros_like(valid)
    for x0, y0, x1, y1 in boxes:
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & valid
        if not inside.any() and valid.any():
            distance = (x - (x0 + x1) / 2).square() + (y - (y0 + y1) / 2).square()
            inside.view(-1)[distance.masked_fill(~valid, float("inf")).flatten().argmin()] = True
        positive |= inside
    return positive


class RegionDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row["path"]) as source:
            image, layout = fit_canvas(source.convert("RGB"))
        positive = area_mask(layout, row["target"])
        negative = area_mask(layout) & ~(positive | area_mask(layout, row["ignore"]))
        return image, positive, negative, CLASSES.index(row["label"])


def thin_late_frames(records):
    last = {}
    named_blocks = set()
    accepted = []
    for row in sorted(records, key=lambda item: (item["capture_group"], PurePosixPath(item["origin"]).stem)):
        group = row["capture_group"]
        stem = PurePosixPath(row["origin"]).stem
        timestamp = re.search(r"20\d{12}", stem)
        if timestamp:
            moment = datetime.strptime(timestamp[0], "%Y%m%d%H%M%S")
            if group in last and (moment - last[group]).total_seconds() < 30:
                continue
            last[group] = moment
        else:
            file_number = re.search(r"\((\d+)\)$", stem)
            key = (group, (int(file_number[1]) - 1) // 12) if file_number else (group, stem)
            if key in named_blocks:
                continue
            named_blocks.add(key)
        accepted.append(row)
    return accepted


def split_regions(screened, reviewed_early, train_rows):
    early = []
    for row in screened:
        box = reviewed_early.get(str(row["review_id"]))
        if row["label"] == "early_blight" and box:
            early.append({**row, "target": box, "ignore": row["target"]})
    late = thin_late_frames([row for row in screened if row["label"] == "late_blight"])
    healthy = [{**row, "target": [], "ignore": []} for row in sorted(train_rows, key=lambda row: row["sha256"])
               if row["label"] == "healthy"][:HEALTHY_TARGET]
    positives = early + late
    checks = [row for row in positives if row["capture_group"] == "Jodhpur_Land" or row["capture_group"].startswith("img20221006")]
    reserved = {row["sha256"] for row in checks}
    training = [row for row in positives if row["sha256"] not in reserved] + healthy
    for group in (training, checks):
        if len({row["sha256"] for row in group}) != len(group):
            raise ValueError("The localizer data contain repeated identities")
        for row in group:
            if sha256(Path(row["path"])) != row["sha256"]:
                raise ValueError(f"Changed localization input: {row['origin']}")
    return training, checks


@torch.no_grad()
def tokenize(encoder, records):
    buffer = []
    for images, positive, negative, labels in DataLoader(RegionDataset(records), batch_size=LOCALIZER_BATCH):
        patches = encoder.dino.forward_features(images.to(encoder.device))["x_norm_patchtokens"].cpu()
        buffer.append((patches, positive.flatten(1), negative.flatten(1), labels))
    return tuple(torch.cat([group[index] for group in buffer]) for index in range(len(buffer[0])))


def train_localizer(encoder, study, check):
    torch.manual_seed(LOCALIZER_SEED)
    features, positives, negatives, labels = tokenize(encoder, study)
    measured = features[positives | negatives]
    localizer = SubtypeProbe(measured.mean(0), measured.std(0, unbiased=False).clamp_min(0.02))
    optimizer = torch.optim.AdamW(localizer.parameters(), lr=0.003, weight_decay=0.05)
    counts = torch.bincount(labels, minlength=3)
    if not (counts > 0).all():
        raise ValueError("A disease or healthy control class is missing")
    weights = len(labels) / (3 * counts[labels])
    loss_history = []
    for epoch in range(LOCALIZER_EPOCHS):
        total = 0.
        for ix in torch.randperm(len(labels)).split(LOCALIZER_BATCH):
            prediction = localizer(features[ix]).reshape(len(ix), -1)
            positive_loss = (F.softplus(-prediction) * positives[ix]).sum(1) / positives[ix].sum(1).clamp_min(1)
            negative_loss = (F.softplus(prediction) * negatives[ix]).sum(1) / negatives[ix].sum(1).clamp_min(1)
            per_image = torch.where(labels[ix] == 0, negative_loss, (positive_loss + negative_loss) / 2)
            loss = (per_image * weights[ix]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ix)
        loss_history.append(total / len(labels))
        print(f"Localizer epoch {epoch + 1}/{LOCALIZER_EPOCHS}: {loss_history[-1]:.4f}", flush=True)
    localizer.eval().requires_grad_(False)
    features, positive, negative, labels = tokenize(encoder, check)
    with torch.no_grad():
        logits = localizer(features).reshape(len(labels), -1).masked_fill(~(positive | negative), -torch.inf)
        hits = positive.gather(1, logits.argmax(1, keepdim=True)).flatten()
    diagnostics = {CLASSES[label]: {"total": int((labels == label).sum()), "top_point_in_box": int(hits[labels == label].sum())}
                   for label in labels.unique().tolist()}
    return localizer, {"training_counts": dict(Counter(row["label"] for row in study)),
                       "checks": diagnostics, "loss": loss_history}


@torch.no_grad()
def locate(encoder, localizer, images, layouts):
    features = encoder.dino.forward_features(images.contiguous())["x_norm_patchtokens"]
    mask = torch.stack([area_mask(layout) for layout in layouts]).flatten(1).to(images.device)
    scores = localizer(features).reshape(len(images), -1).masked_fill(~mask, -torch.inf)
    peaks, indices = scores.max(1)
    xy = torch.stack((indices.remainder(GRID), indices // GRID), 1).float().add(0.5).div(GRID)
    return xy, peaks >= 0.


def crop(images, centers):
    transforms = images.new_zeros((len(images), 2, 3))
    transforms[:, 0, 0] = transforms[:, 1, 1] = LOCAL_FRACTION
    transforms[:, :, 2] = centers.clamp(LOCAL_FRACTION / 2, 1 - LOCAL_FRACTION / 2) * 2 - 1
    grid = F.affine_grid(transforms, (len(images), 3, 224, 224), align_corners=False)
    return F.grid_sample(images, grid, align_corners=False)
