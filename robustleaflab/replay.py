"""Replay a labeled, hash-verified image manifest with the exported model."""

import csv
from pathlib import Path
import re

from .backbone import verify
from .models import CLASSES, read_image


def evaluate(manifest, images_root, model, *, apply_exif=False, check_frozen=False):
    expected_column = f"expected_{model.name}"
    required = {"image", "sha256", "source", "true_label"}
    if check_frozen:
        required.add(expected_column)
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        missing = required - set(rows.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        samples = list(rows)
    if not samples:
        raise ValueError("The replay manifest has no images.")

    root = (images_root if images_root is not None else manifest.parent).resolve()
    class_index = {label: i for i, label in enumerate(CLASSES)}
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]
    predictions = []
    seen = set()
    mismatches = 0
    for row_number, row in enumerate(samples, start=2):
        image = row["image"]
        image_path = Path(image or "")
        digest = row["sha256"] or ""
        truth = row["true_label"]
        source = row["source"]
        expected = row.get(expected_column)
        if (not image or image_path.is_absolute() or not source or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or truth not in class_index or (check_frozen and expected not in class_index)):
            raise ValueError(f"Invalid replay record at CSV row {row_number}.")
        path = (root / image_path).resolve()
        if not path.is_relative_to(root) or path in seen:
            raise ValueError(f"Path outside image root or repeated image at CSV row {row_number}: {image}")
        seen.add(path)
        scores = model(read_image(verify(path, digest), apply_exif=apply_exif).unsqueeze(0))[0]
        label = CLASSES[int(scores.argmax())]
        matrix[class_index[truth]][class_index[label]] += 1
        result = {"image": image, "sha256": digest, "source": source, "true_label": truth, "prediction": label}
        if check_frozen:
            result["expected_label"] = expected
            mismatches += int(label != expected)
        predictions.append(result)

    correct = sum(matrix[i][i] for i in range(len(CLASSES)))
    return {
        "model": model.name,
        "model_version": model.model_version,
        "orientation": "exif" if apply_exif else "raw",
        "images": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions),
        "class_order": list(CLASSES),
        "confusion_matrix": matrix,
        "frozen_mismatches": mismatches if check_frozen else None,
        "predictions": predictions,
    }
