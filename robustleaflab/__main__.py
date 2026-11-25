"""Command-line inference for the RobustLeafLab release models."""

import argparse
import json
from pathlib import Path

import torch

from .backbone import prepare
from .models import CLASSES, MODEL_NAMES, ROOT, RobustLeafClassifier, read_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Predict the labels of leaf images")
    predict.add_argument("images", type=Path, nargs="+")
    predict.add_argument("--model", choices=MODEL_NAMES, default="current")
    predict.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    predict.add_argument("--models-dir", type=Path, default=ROOT / "models")
    predict.add_argument("--cache-dir", type=Path, default=ROOT / ".model-cache")
    fetch = commands.add_parser("fetch-backbone", help="Cache the official pinned DINOv2 files")
    fetch.add_argument("--cache-dir", type=Path, default=ROOT / ".model-cache")
    commands.add_parser("list-models", help="List runnable model names")
    args = parser.parse_args()
    if args.command == "list-models":
        print("\n".join(MODEL_NAMES))
        return
    try:
        if args.command == "fetch-backbone":
            prepare(args.cache_dir)
            print("DINOv2 files verified.")
            return
        device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        model = RobustLeafClassifier(args.model, args.models_dir, args.cache_dir, device)
        for path in args.images:
            scores = model(read_image(path).unsqueeze(0))[0].cpu()
            print(json.dumps({"image": str(path), "model": args.model, "label": CLASSES[int(scores.argmax())],
                              "decision_scores": dict(zip(CLASSES, scores.tolist()))}, ensure_ascii=False))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
