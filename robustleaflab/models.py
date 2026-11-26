"""Run the exported AC42 baselines and the locked DINOv2 candidate stack."""

import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

from .backbone import load_dino, verify
from .directional import DirectionalReview


ROOT = Path(__file__).resolve().parent.parent
CLASSES = ("healthy", "early_blight", "late_blight")
MODEL_NAMES = ("a42", "c42", "ac42", "ac42-finetuned", "dino-subtype", "dino-health", "v0.1.0", "current")
CORE_MODEL_VERSION = "v0.1.0"
PREPROCESS = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def read_image(path, *, apply_exif=False):
    with Image.open(path) as image:
        if apply_exif:
            image = ImageOps.exif_transpose(image)
        return PREPROCESS(image.convert("RGB"))


def checkpoint(weights_dir, name):
    registry_path = weights_dir / "manifest.json"
    if not registry_path.is_file():
        raise FileNotFoundError("Extract the core release ZIP into the repository or pass --models-dir.")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    item = registry["files"][name]
    path = verify(weights_dir / name, item["sha256"])
    return torch.load(path, map_location="cpu", weights_only=True)


class LinearHead(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.register_buffer("mean", state["mean"].clone())
        self.register_buffer("scale", state["scale"].clone())
        self.linear = nn.Linear(self.mean.numel(), 1)
        self.load_state_dict(state, strict=True)

    def forward(self, features):
        return self.linear(((features - self.mean) / self.scale).clamp(-6, 6)).flatten()


def fuse_subtype(prior, evidence, fraction):
    early, late = prior[:, 1], prior[:, 2]
    maximum = torch.maximum(early, late)
    margin = torch.lerp(early - late, evidence, fraction)
    return torch.stack((prior[:, 0], maximum + margin.clamp(max=0), maximum + (-margin).clamp(max=0)), 1)


def apply_health(scores, evidence, strength):
    diseases = scores[:, 1:]
    maximum = diseases.amax(1, keepdim=True)
    margin = maximum[:, 0] - scores[:, 0] + strength * (0.5 * evidence).tanh()
    return torch.cat((torch.zeros_like(margin[:, None]), diseases - maximum + margin[:, None]), 1)


def apply_early(scores, evidence, strength):
    eligible = (scores.argmax(1) == 0) & (scores[:, 1] >= scores[:, 2])
    correction = (strength * (0.5 * evidence).tanh()).clamp_min(0)
    healthy = scores[:, 0] - torch.where(eligible, correction, 0)
    return torch.cat((healthy[:, None], scores[:, 1:]), 1)


class RobustLeafClassifier(nn.Module):
    """Load a fixed baseline or a selected stage of the current inference stack."""

    def __init__(self, name, models_dir=ROOT / "models", cache=ROOT / ".model-cache", device="cpu"):
        super().__init__()
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {name}")
        self.name = name
        self.model_version = CORE_MODEL_VERSION
        self.device = torch.device(device)
        baseline = name if name in ("a42", "c42", "ac42") else "ac42-finetuned"
        saved = checkpoint(models_dir, f"{baseline}.pt")
        if saved["model_name"] != "resnet18" or tuple(saved["classes"]) != CLASSES:
            raise ValueError(f"Incompatible baseline checkpoint: {baseline}")
        self.resnet = models.resnet18(weights=None, num_classes=len(CLASSES))
        self.resnet.load_state_dict(saved["model_state"], strict=True)
        self.dino = None
        if name.startswith("dino-") or name in ("v0.1.0", "current"):
            subtype = checkpoint(models_dir, "dino-subtype-head.pt")
            if subtype["feature"] != "dino":
                raise ValueError("Incompatible DINOv2 subtype feature format")
            self.dino = load_dino(cache, self.device)
            self.subtype = LinearHead(subtype["state"])
            self.fraction = subtype["fraction"]
            if name != "dino-subtype":
                health = checkpoint(models_dir, "dino-health-head.pt")
                self.health = LinearHead(health["state"])
                self.health_strength = health["strength"]
            if name in ("v0.1.0", "current"):
                early = checkpoint(models_dir, "early-review-head.pt")
                if early["view"] != "standard" or early["mode"] != "rescue_only":
                    raise ValueError("Incompatible early-review checkpoint")
                self.early = LinearHead(early["state"])
                self.early_strength = early["strength"]
        if name == "current":
            self.directional_review = DirectionalReview()
            self.model_version = self.directional_review.settings["version"]
        self.resnet.to(self.device, memory_format=torch.channels_last)
        self.requires_grad_(False).to(self.device).eval()

    @torch.inference_mode()
    def forward(self, images):
        images = images.to(self.device)
        scores = self.resnet(images.contiguous(memory_format=torch.channels_last))
        if self.dino is None:
            return scores
        tokens = self.dino.forward_features(images.contiguous())
        features = torch.cat((tokens["x_norm_clstoken"], tokens["x_norm_patchtokens"].mean(1)), 1)
        scores = fuse_subtype(scores, self.subtype(features), self.fraction)
        if self.name != "dino-subtype":
            scores = apply_health(scores, self.health(features), self.health_strength)
        if self.name in ("v0.1.0", "current"):
            scores = apply_early(scores, self.early(features), self.early_strength)
        if self.name == "current":
            scores = self.directional_review(scores, tokens["x_norm_clstoken"], tokens["x_norm_patchtokens"], features)
        return scores
