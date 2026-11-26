"""Command-line inference for the RobustLeafLab release models."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .backbone import prepare
from .models import CLASSES, MODEL_NAMES, ROOT, RobustLeafClassifier, read_image
from .replay import evaluate


IMAGE_ERRORS = (OSError, ValueError, SyntaxError, Image.DecompressionBombError)


def model_arguments(command, orientation):
    command.add_argument("--model", choices=MODEL_NAMES, default="current")
    command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    command.add_argument("--models-dir", type=Path, default=ROOT / "models")
    command.add_argument("--cache-dir", type=Path, default=ROOT / ".model-cache")
    command.add_argument("--orientation", choices=("exif", "raw"), default=orientation,
                         help=f"Image orientation policy (default: {orientation})")


def predict_images(model, paths, *, apply_exif):
    had_error = False
    for path in paths:
        result = {"image": str(path), "model": model.name, "model_version": model.model_version}
        try:
            image = read_image(path, apply_exif=apply_exif)
        except IMAGE_ERRORS as error:
            print(json.dumps({**result, "error": str(error)}, ensure_ascii=False))
            had_error = True
            continue
        scores = model(image.unsqueeze(0))[0].cpu()
        result["label"] = CLASSES[int(scores.argmax())]
        result["decision_scores"] = dict(zip(CLASSES, scores.tolist(), strict=True))
        print(json.dumps(result, ensure_ascii=False))
    return int(had_error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Predict the labels of leaf images")
    predict.add_argument("images", type=Path, nargs="+")
    model_arguments(predict, orientation="exif")
    replay = commands.add_parser("evaluate", help="Evaluate a hash-verified labeled CSV manifest")
    replay.add_argument("manifest", type=Path, help="CSV with image, sha256, source, and true_label")
    replay.add_argument("--images-root", type=Path, help="Image root (default: the manifest directory)")
    replay.add_argument("--check-frozen", action="store_true",
                        help="Compare against the expected_<model> column and exit nonzero on mismatch")
    model_arguments(replay, orientation="raw")
    fetch = commands.add_parser("fetch-backbone", help="Cache the official pinned DINOv2 files")
    fetch.add_argument("--cache-dir", type=Path, default=ROOT / ".model-cache")
    commands.add_parser("list-models", help="List runnable model names")
    args = parser.parse_args(argv)
    if args.command == "list-models":
        print("\n".join(MODEL_NAMES))
        return 0
    try:
        if args.command == "fetch-backbone":
            prepare(args.cache_dir)
            print("DINOv2 files verified.")
            return 0
        device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        model = RobustLeafClassifier(args.model, args.models_dir, args.cache_dir, device)
        apply_exif = args.orientation == "exif"
        if args.command == "predict":
            return predict_images(model, args.images, apply_exif=apply_exif)
        report = evaluate(args.manifest, args.images_root, model, apply_exif=apply_exif,
                          check_frozen=args.check_frozen)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return int(bool(report["frozen_mismatches"]))
    except IMAGE_ERRORS as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
