"""Optional inference views for evaluating a frozen leaf classifier."""
from PIL import Image, ImageOps
import torch
from torchvision import transforms

from common import MEAN, STD


VIEW_MODES = ('center', 'full-frame', 'five-crop')


class FullFrame:
    """Preserve the full image and pad with the rounded ImageNet mean."""

    def __call__(self, image):
        return ImageOps.pad(image, (224, 224), method=Image.Resampling.BILINEAR,
                            color=tuple(round(value * 255) for value in MEAN))


class FiveCrop:
    """Return a batch of normalized views from a single image."""

    def __init__(self):
        self.resize = transforms.Resize(256)
        self.crop = transforms.FiveCrop(224)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __call__(self, image):
        return torch.stack([self.normalize(crop) for crop in self.crop(self.resize(image))])


class AverageCropLogits(torch.nn.Module):
    """Average the classifier's logits over all views of each image."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        batch, crops = images.shape[:2]
        logits = self.model(images.flatten(0, 1))
        return logits.reshape(batch, crops, -1).mean(1)


def apply_view(mode, dataset, model):
    if mode not in VIEW_MODES:
        raise ValueError(f'Unknown inference view: {mode}')
    if mode == 'five-crop':
        dataset.tf = FiveCrop()
        return AverageCropLogits(model)
    if mode == 'full-frame':
        dataset.tf = transforms.Compose([
            FullFrame(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return model
