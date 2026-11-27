"""Shared models, image transforms and dataset code for RobustLeafLab.

The experiment tests *background sensitivity* rather than pretending that a
three-class leaf classifier is field-ready. PlantDoc is never used to train.
"""
import csv
import hashlib
import os
import random
from math import isfinite, log
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms

CLASSES = ('healthy', 'early_blight', 'late_blight')
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TRAIN_CROP_SCALE = (0.80, 1.0)
TRAIN_CROP_RATIO_MIN = 3.0 / 4.0
TRAIN_CROP_RATIO_MAX = 4.0 / 3.0
TRAIN_PAD_ASPECT_MIN = 0.75
TRAIN_PAD_ASPECT_MAX = 2.5


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RandomPadAspect:
    """Fit the full image in a random aspect canvas, then resize to a square."""

    size = 224

    def __init__(self, ratio_min, ratio_max):
        if not (isfinite(ratio_min) and isfinite(ratio_max) and 0 < ratio_min <= ratio_max):
            raise ValueError('Training pad aspect ratios must be finite, positive, and ordered')
        self.log_min, self.log_max = log(ratio_min), log(ratio_max)
        self.fill = tuple(round(value * 255) for value in MEAN)

    def sample_ratio(self):
        return torch.empty((), dtype=torch.float64).uniform_(self.log_min, self.log_max).exp().item()

    def layout(self, image_size, canvas_ratio):
        width, height = image_size
        if canvas_ratio >= 1:
            canvas_width, canvas_height = self.size, max(1, round(self.size / canvas_ratio))
        else:
            canvas_width, canvas_height = max(1, round(self.size * canvas_ratio)), self.size
        scale = min(canvas_width / width, canvas_height / height)
        fit_width = min(canvas_width, max(1, round(width * scale)))
        fit_height = min(canvas_height, max(1, round(height * scale)))
        return (canvas_width, canvas_height), (fit_width, fit_height)

    def __call__(self, image):
        canvas_size, fit_size = self.layout(image.size, self.sample_ratio())
        fit_image = image.resize(fit_size, Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', canvas_size, self.fill)
        left = (canvas_size[0] - fit_size[0]) // 2
        top = (canvas_size[1] - fit_size[1]) // 2
        canvas.paste(fit_image, (left, top))
        return canvas.resize((self.size, self.size), Image.Resampling.BILINEAR)


def transform(train=False, *, train_crop_ratio_min=TRAIN_CROP_RATIO_MIN,
              train_crop_ratio_max=TRAIN_CROP_RATIO_MAX, train_pad_aspect_probability=0.0,
              train_pad_aspect_min=TRAIN_PAD_ASPECT_MIN, train_pad_aspect_max=TRAIN_PAD_ASPECT_MAX):
    """Build the training augmentation or the fixed evaluation transform."""
    if train:
        if not (isfinite(train_crop_ratio_min) and isfinite(train_crop_ratio_max)
                and 0 < train_crop_ratio_min <= train_crop_ratio_max):
            raise ValueError('Training crop aspect ratios must be finite, positive, and ordered')
        if not 0 <= train_pad_aspect_probability <= 1:
            raise ValueError('Training pad aspect probability must be between 0 and 1')
        crop_ratios = (train_crop_ratio_min, train_crop_ratio_max)
        geometry = transforms.RandomResizedCrop(224, scale=TRAIN_CROP_SCALE, ratio=crop_ratios)
        if train_pad_aspect_probability:
            if crop_ratios != (TRAIN_CROP_RATIO_MIN, TRAIN_CROP_RATIO_MAX):
                raise ValueError('Training pad aspect requires the default crop aspect ratios')
            padding = RandomPadAspect(train_pad_aspect_min, train_pad_aspect_max)
            geometry = transforms.RandomChoice([geometry, padding], p=[
                1 - train_pad_aspect_probability, train_pad_aspect_probability])
        ops = [geometry,
               transforms.RandomHorizontalFlip(),
               transforms.RandomRotation(10),
               transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                      saturation=0.15)]
    else:
        ops = [transforms.Resize(256), transforms.CenterCrop(224)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def mask_from_segmented(image):
    """Return soft leaf foreground alpha from PlantVillage segmented RGB image.

    Segmented PlantVillage images have a near-black removed background. The
    threshold is intentionally conservative; imperfect edges are a limitation.
    """
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    alpha = (rgb.max(axis=2) > 22).astype('uint8') * 255
    return Image.fromarray(alpha, mode='L').filter(ImageFilter.GaussianBlur(radius=1.5))


def random_background(image, segmented, rng=None, mode="standard"):
    """mode=holdout uses unseen texture/palette for stress testing."""
    """Keep leaf pixels, replace nonleaf area with randomized synthetic texture.

    A mask is REQUIRED; absence fails loudly instead of silently turning the
    background-randomization treatment into ordinary augmentation.
    """
    rng = rng or np.random.default_rng()
    image = image.convert('RGB')
    segmented = segmented.convert('RGB').resize(image.size, Image.Resampling.NEAREST)
    alpha = np.asarray(mask_from_segmented(segmented), dtype=np.float32)[:, :, None] / 255.0
    h, w = image.height, image.width
    palette = np.array([[60, 100, 60], [145, 123, 84], [75, 83, 66],
                        [173, 162, 128], [112, 132, 153], [55, 72, 46]], dtype=float)
    if mode == 'holdout':
        # Not used in training: alternate brighter/darker colors and broad
        # spatial structures. Still synthetic, NOT a field generalization test.
        palette = np.array([[32, 53, 36], [211, 181, 94], [97, 128, 184],
                            [192, 87, 57], [47, 134, 112], [189, 197, 203]], dtype=float)
    elif mode != 'standard':
        raise ValueError(f'Unknown background mode: {mode}')
    base = palette[int(rng.integers(len(palette)))].reshape(1, 1, 3)
    # Low-frequency texture, not random pixel snow. This is a controlled shift
    # test, not a claim to generate realistic agricultural scenes.
    small = rng.normal(0, 30, (max(2, h // 24), max(2, w // 24), 3))
    noise = Image.fromarray(np.clip(small + 128, 0, 255).astype('uint8'))
    smooth = np.asarray(noise.resize((w, h), Image.Resampling.BICUBIC), dtype=float) - 128
    if mode == 'holdout':
        # Broad contrasting bands unseen in standard background generation.
        yy, xx = np.mgrid[:h, :w]
        band = 30 * np.sin(xx / max(1, w) * rng.uniform(3, 8) * np.pi + rng.uniform(0, 6.28))
        smooth += band[:, :, None]
    yy, xx = np.mgrid[:h, :w]
    gradient = ((xx / max(w-1, 1) - .5) * rng.uniform(-35, 35)
                + (yy / max(h-1, 1) - .5) * rng.uniform(-35, 35))[:, :, None]
    bg = np.clip(base + smooth + gradient, 0, 255)
    orig = np.asarray(image, dtype=float)
    return Image.fromarray(np.clip(orig * alpha + bg * (1 - alpha), 0, 255).astype('uint8'))


def prepare_natural_background(field_photo, size):
    background = ImageOps.fit(field_photo.convert('RGB'), size,
                              method=Image.Resampling.BICUBIC)
    return background.filter(ImageFilter.GaussianBlur(radius=max(2, min(size) / 70)))


def composite_natural_background(image, segmented, background):
    image = image.convert('RGB')
    segmented = segmented.convert('RGB').resize(image.size, Image.Resampling.NEAREST)
    alpha = np.asarray(mask_from_segmented(segmented), dtype=np.float32)[:, :, None] / 255.0
    front = np.asarray(image, dtype=np.float32)
    behind = np.asarray(background, dtype=np.float32)
    return Image.fromarray(np.clip(front * alpha + behind * (1 - alpha), 0, 255).astype(np.uint8))


def natural_background(image, segmented, field_photo):
    """Composite a PlantVillage leaf onto a blurred field training photo."""
    return composite_natural_background(
        image, segmented, prepare_natural_background(field_photo, image.size))


def load_natural_background(source, size, cache_dir):
    if cache_dir is None:
        with Image.open(source) as field:
            return prepare_natural_background(field, size)

    stat = Path(source).stat()
    # Bump the recipe version when preparation changes. Include Pillow so a
    # library upgrade cannot reuse backgrounds generated by a different build.
    identity = ('natural-v1', Image.__version__, source, stat.st_size, stat.st_mtime_ns, size)
    key = hashlib.sha256(repr(identity).encode('utf-8')).hexdigest()
    cached = cache_dir / f'{key}.png'
    try:
        with Image.open(cached) as background:
            if background.mode == 'RGB' and background.size == size:
                return background.copy()
    except OSError:
        # A missing or unreadable derived asset can be rebuilt from the source.
        pass

    with Image.open(source) as field:
        background = prepare_natural_background(field, size)
    with NamedTemporaryFile(dir=cache_dir, suffix='.png', delete=False) as temp:
        staged = Path(temp.name)
    try:
        background.save(staged, format='PNG', compress_level=1)
        try:
            os.replace(staged, cached)
        except PermissionError:
            # On Windows another worker can have the completed entry open.
            if not cached.is_file():
                raise
    finally:
        staged.unlink(missing_ok=True)
    return background


class ManifestDataset(Dataset):
    def __init__(self, csv_path, train=False, background_prob=0.0,
                 background_stress=False, seed=2026, stress_mode="standard",
                 allow_unmasked_field=False, background_style="synthetic",
                 background_cache_dir=None, *, train_crop_ratio_min=TRAIN_CROP_RATIO_MIN,
                 train_crop_ratio_max=TRAIN_CROP_RATIO_MAX, train_pad_aspect_probability=0.0,
                 train_pad_aspect_min=TRAIN_PAD_ASPECT_MIN, train_pad_aspect_max=TRAIN_PAD_ASPECT_MAX):
        with open(csv_path, encoding='utf-8', newline='') as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f'Empty manifest: {csv_path}')
        self.tf = transform(train, train_crop_ratio_min=train_crop_ratio_min,
                            train_crop_ratio_max=train_crop_ratio_max,
                            train_pad_aspect_probability=train_pad_aspect_probability,
                            train_pad_aspect_min=train_pad_aspect_min, train_pad_aspect_max=train_pad_aspect_max)
        self.train, self.background_prob = train, background_prob
        self.background_stress, self.seed = background_stress, seed
        self.stress_mode = stress_mode
        self.allow_unmasked_field = allow_unmasked_field
        self.background_style = background_style
        if background_style not in ('synthetic', 'natural'):
            raise ValueError(f'Unsupported background training style: {background_style}')
        self.natural_pool = [r['path'] for r in self.rows if r.get('source') == 'field_extra']
        if train and background_prob and background_style == 'natural' and not self.natural_pool:
            raise ValueError('Natural background arm requires field_extra images in TRAIN manifest')
        if background_prob or background_stress:
            missing = [r for r in self.rows if not r.get('mask_path') or
                       not Path(r['mask_path']).is_file()]
            # The v3 field photographs have no segmentation masks. They still
            # receive ordinary geometric/color augmentation, but NEVER a
            # fabricated leaf segmentation. All PlantVillage masks remain required.
            invalid = [r for r in missing if background_stress or
                       not (allow_unmasked_field and r.get('source') == 'field_extra')]
            if invalid:
                raise ValueError(f'Missing {len(invalid)} required segmented masks; '
                                 f"first: {invalid[0]['path']}")
        self.background_cache_dir = None
        if background_cache_dir is not None:
            if not (train and background_prob and background_style == 'natural' and not background_stress):
                raise ValueError('Background cache requires natural background training')
            self.background_cache_dir = Path(background_cache_dir).resolve()
            self.background_cache_dir.mkdir(parents=True, exist_ok=True)

    def prepare_background_cache(self):
        """Populate the optional cache for the masked image sizes in this manifest."""
        if self.background_cache_dir is None:
            raise ValueError('Set a background cache directory before preparing assets')
        sizes = set()
        for row in self.rows:
            if row.get('mask_path') and Path(row['mask_path']).is_file():
                with Image.open(row['path']) as image:
                    sizes.add(image.size)
        for size in sorted(sizes):
            for source in self.natural_pool:
                load_natural_background(source, size, self.background_cache_dir)
        return len(sizes) * len(self.natural_pool)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(row['path']) as im:
            img = im.convert('RGB')
        apply_bg = self.background_stress or (self.background_prob and random.random() < self.background_prob)
        if apply_bg and row.get('mask_path') and Path(row['mask_path']).is_file():
            with Image.open(row['mask_path']) as seg:
                if self.background_stress:
                    # Exactly the same perturbation, independent of run/model.
                    rng = np.random.default_rng(self.seed + idx * 104729)
                else:
                    rng = np.random.default_rng(random.getrandbits(64))
                if self.train and not self.background_stress and self.background_style == 'natural':
                    background = load_natural_background(random.choice(self.natural_pool), img.size,
                                                         self.background_cache_dir)
                    img = composite_natural_background(img, seg, background)
                else:
                    img = random_background(img, seg, rng,
                                            mode=self.stress_mode if self.background_stress else 'standard')
        return self.tf(img), CLASSES.index(row['label'])


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*block(3, 32), *block(32, 64), *block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128, len(CLASSES)))

    def forward(self, x):
        return self.head(self.features(x))


def create_model(name, pretrained=False):
    if name == 'cnn':
        if pretrained:
            raise ValueError('CNN scratch model cannot load ImageNet weights')
        return SmallCNN()
    if name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, len(CLASSES))
        return net
    raise ValueError(f'Unknown model: {name}')


def predict(model, loader, device):
    model.eval()
    targets, preds, probs, losses = [], [], [], []
    ce = nn.CrossEntropyLoss(reduction='none')
    non_blocking = getattr(loader, 'pin_memory', False) and torch.device(device).type == 'cuda'
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
            logits = model(x)
            prob = logits.softmax(1).cpu()
            preds.extend(prob.argmax(1).tolist())
            probs.extend(prob.tolist())
            losses.extend(ce(logits, y).cpu().tolist())
            targets.extend(y.cpu().tolist())
    return (np.array(targets), np.array(preds), np.array(probs),
            float(np.mean(losses)))
