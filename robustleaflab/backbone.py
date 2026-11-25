"""Fetch pinned DINOv2 source code and pretrained tensors from their publishers."""

import hashlib
import os
import sys
import tempfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from zipfile import ZipFile

import torch


REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
SOURCE_URL = f"https://codeload.github.com/facebookresearch/dinov2/zip/{REVISION}"
SOURCE_HASH = "04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
WEIGHTS_HASH = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
WEIGHTS_NAME = "dinov2_vits14_pretrain.pth"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return path


def fetch(path, url, expected):
    if path.is_file():
        return verify(path, expected)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            pending = Path(stream.name)
            request = Request(url, headers={"User-Agent": "RobustLeafLab course model setup"})
            print(f"Downloading {path.name}", file=sys.stderr)
            with urlopen(request, timeout=45) as response:
                while block := response.read(1024 * 1024):
                    stream.write(block)
        verify(pending, expected).replace(path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    return path


def prepare(cache):
    source = fetch(cache / "dinov2-source.zip", SOURCE_URL, SOURCE_HASH)
    weights = fetch(cache / WEIGHTS_NAME, WEIGHTS_URL, WEIGHTS_HASH)
    return source, weights


def load_dino(cache, device):
    source, weights = prepare(cache)
    root = cache / f"dinov2-{REVISION}"
    with ZipFile(source) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            if len(relative.parts) < 3 or relative.parts[:2] != (root.name, "dinov2") or relative.suffix != ".py":
                continue
            if any(part in (".", "..") or "\\" in part or ":" in part for part in relative.parts):
                raise ValueError("Unexpected archive member path")
            destination = cache.joinpath(*relative.parts)
            data = archive.read(member)
            if not destination.is_file() or destination.read_bytes() != data:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    sys.path.insert(0, str(root))
    try:
        from dinov2.hub.backbones import dinov2_vits14
    finally:
        sys.path.remove(str(root))
    model = dinov2_vits14(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model.requires_grad_(False).to(device).eval()
