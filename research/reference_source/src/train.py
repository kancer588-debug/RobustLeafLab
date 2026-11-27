"""Train a leaf classifier and select its checkpoint on validation data."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Sampler

from common import (CLASSES, TRAIN_CROP_RATIO_MAX, TRAIN_CROP_RATIO_MIN,
                    TRAIN_PAD_ASPECT_MAX, TRAIN_PAD_ASPECT_MIN,
                    ManifestDataset, create_model, predict, seed_everything)


class SourceClassSampler(Sampler[int]):
    """Draw a fixed-size epoch with bounded repeats and source-class weighting."""

    max_appearances = 2

    def __init__(self, rows, seed):
        groups = [(row.get('dataset') or row.get('source'), row['label']) for row in rows]
        if not groups or any(not source for source, _ in groups):
            raise ValueError('Source-class sampling requires image provenance in the train manifest')
        self.group_counts = Counter(groups)
        self.num_images = len(groups)
        weights = torch.tensor([self.group_counts[group] ** -0.5 for group in groups], dtype=torch.float64)
        # Each image has two slots. Each slot can be drawn at most once per epoch.
        self.weights = weights.repeat(self.max_appearances)
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        slots = torch.multinomial(self.weights, self.num_images, replacement=False, generator=self.generator)
        return iter(slots.remainder(self.num_images).tolist())

    def __len__(self):
        return self.num_images


class ClassBalancedSourceSampler(SourceClassSampler):
    """Allocate equal class quotas and weight capture sources within each class."""

    def __init__(self, rows, seed):
        super().__init__(rows, seed)
        label_indices = defaultdict(list)
        for index, row in enumerate(rows):
            label_indices[row['label']].append(index)
        if set(label_indices) != set(CLASSES):
            raise ValueError('Class-balanced sampling requires all configured classes')
        quota, remainder = divmod(self.num_images, len(CLASSES))
        self.class_draw_counts = {label: quota + int(i < remainder) for i, label in enumerate(CLASSES)}
        self.class_pools = []
        for label, draws in self.class_draw_counts.items():
            indices = torch.tensor(label_indices[label], dtype=torch.long)
            if draws > len(indices) * self.max_appearances:
                raise ValueError(f'Class {label} has too few images for quota {draws} and the two-draw limit')
            weights = self.weights[indices].repeat(self.max_appearances)
            self.class_pools.append((indices, weights, draws))

    def __iter__(self):
        selected = []
        for indices, weights, draws in self.class_pools:
            slots = torch.multinomial(weights, draws, replacement=False, generator=self.generator)
            selected.append(indices[slots.remainder(len(indices))])
        joined = torch.cat(selected)
        order = torch.randperm(self.num_images, generator=self.generator)
        return iter(joined[order].tolist())


def add_loader_arguments(parser):
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--pin-memory', action='store_true', help='Pin batches for CUDA transfers')
    parser.add_argument('--persistent-workers', action='store_true',
                        help='Reuse training workers between epochs')


def make_loader(dataset, args, device, *, train, sampler=None):
    workers = args.num_workers
    if not train and args.val_num_workers is not None:
        workers = args.val_num_workers
    if workers < 0:
        raise ValueError('Worker counts must be non-negative')
    if args.persistent_workers and args.num_workers == 0:
        raise ValueError('Use at least one worker with --persistent-workers')
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train and sampler is None,
                      sampler=sampler,
                      num_workers=workers,
                      pin_memory=args.pin_memory and device.type == 'cuda',
                      persistent_workers=train and args.persistent_workers)


def make_optimizer(model, pretrained, lr_scale=1.0):
    if pretrained:
        return torch.optim.AdamW([
            {'params': model.fc.parameters(), 'lr': 1e-3 * lr_scale},
            {'params': (p for n, p in model.named_parameters() if not n.startswith('fc.')),
             'lr': 1e-4 * lr_scale}], weight_decay=1e-4)
    return torch.optim.AdamW(model.parameters(), lr=1e-3 * lr_scale, weight_decay=1e-4)


def train_batch(model, optimizer, criterion, x, y):
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    return loss


def validation_metrics(model, loader, device):
    true, predicted, _, loss = predict(model, loader, device)
    accuracy = float(accuracy_score(true, predicted))
    f1 = float(f1_score(true, predicted, labels=range(len(CLASSES)), average='macro', zero_division=0))
    return loss, accuracy, f1


def image_identities(dataset):
    return {row.get('sha256') or Path(row['path']).resolve() for row in dataset.rows}


def train(args):
    if args.pretrained and args.model != 'resnet18':
        raise ValueError('Only ResNet18 supports pretrained weights')
    if args.bg_prob and not args.pretrained:
        raise ValueError('Background augmentation arm must use pretrained ResNet18')
    if not 0 <= args.bg_prob <= 1:
        raise ValueError('bg-prob must be between 0 and 1')
    if not 1 <= args.epochs or not 1 <= args.patience:
        raise ValueError('epochs and patience must be >= 1')
    lr_scale = getattr(args, 'lr_scale', 1.0)
    if not math.isfinite(lr_scale) or lr_scale <= 0:
        raise ValueError('--lr-scale must be finite and positive')
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_manifest = Path(args.train_manifest or (Path(args.data_dir) / 'train.csv'))
    val_manifest = Path(args.val_manifest or (Path(args.data_dir) / 'val.csv'))
    crop_ratio_min = getattr(args, 'train_crop_ratio_min', TRAIN_CROP_RATIO_MIN)
    crop_ratio_max = getattr(args, 'train_crop_ratio_max', TRAIN_CROP_RATIO_MAX)
    pad_probability = getattr(args, 'train_pad_aspect_probability', 0.0)
    pad_ratio_min = getattr(args, 'train_pad_aspect_min', TRAIN_PAD_ASPECT_MIN)
    pad_ratio_max = getattr(args, 'train_pad_aspect_max', TRAIN_PAD_ASPECT_MAX)
    train_ds = ManifestDataset(train_manifest, train=True,
                               background_prob=args.bg_prob, seed=args.seed,
                               allow_unmasked_field=args.allow_unmasked_field,
                               background_style=args.bg_style,
                               background_cache_dir=args.background_cache_dir,
                               train_crop_ratio_min=crop_ratio_min,
                               train_crop_ratio_max=crop_ratio_max,
                               train_pad_aspect_probability=pad_probability,
                               train_pad_aspect_min=pad_ratio_min,
                               train_pad_aspect_max=pad_ratio_max)
    val_ds = ManifestDataset(val_manifest)
    field_val_dl = None
    if field_val_path := getattr(args, 'field_val_manifest', None):
        field_val_ds = ManifestDataset(field_val_path)
        if {row['label'] for row in field_val_ds.rows} != set(CLASSES):
            raise ValueError('The field validation manifest must contain all configured classes')
        if image_identities(field_val_ds) & (image_identities(train_ds) | image_identities(val_ds)):
            raise ValueError('Field validation must be disjoint from training and the first validation split')
        field_val_dl = make_loader(field_val_ds, args, device, train=False)
    train_manifest_hash = hashlib.sha256(train_manifest.read_bytes()).hexdigest()
    val_manifest_hash = hashlib.sha256(val_manifest.read_bytes()).hexdigest()
    sampling = getattr(args, 'sampling', 'uniform')
    sampler_class = {'source-class': SourceClassSampler, 'class-source': ClassBalancedSourceSampler}.get(sampling)
    sampler = sampler_class(train_ds.rows, args.seed) if sampler_class is not None else None
    train_dl = make_loader(train_ds, args, device, train=True, sampler=sampler)
    val_dl = make_loader(val_ds, args, device, train=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ('best.pt', 'history.csv', 'config.json')):
        raise FileExistsError(f'{out} already has training artifacts. Choose a new --out directory.')
    init_checkpoint = getattr(args, 'init_checkpoint', None)
    model = create_model(args.model, args.pretrained and init_checkpoint is None)
    if init_checkpoint is not None:
        with Path(init_checkpoint).open('rb') as checkpoint_file:
            init_checkpoint_hash = hashlib.file_digest(checkpoint_file, 'sha256').hexdigest()
            checkpoint_file.seek(0)
            checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
        if checkpoint['model_name'] != args.model or tuple(checkpoint['classes']) != CLASSES:
            raise ValueError('Initialization checkpoint has a different architecture or class order')
        model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    opt = make_optimizer(model, args.pretrained, lr_scale)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    # Reserve the destination before either process can write a checkpoint.
    history_file = out / 'history.csv'
    history_file.touch(exist_ok=False)
    best, wait, history = -1., 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, n_samples = 0., 0
        for x, y in train_dl:
            x = x.to(device, non_blocking=train_dl.pin_memory)
            y = y.to(device, non_blocking=train_dl.pin_memory)
            loss = train_batch(model, opt, criterion, x, y)
            n_samples += y.size(0)
            running_loss += loss.item() * y.size(0)
        val_loss, val_accuracy, f1 = validation_metrics(model, val_dl, device)
        result = {'epoch': epoch, 'train_loss': running_loss/n_samples,
                  'val_loss': val_loss, 'val_accuracy': val_accuracy, 'val_macro_f1': f1}
        selection_score = f1
        if field_val_dl is not None:
            field_loss, field_accuracy, field_f1 = validation_metrics(model, field_val_dl, device)
            selection_score = (f1 + field_f1) / 2
            result.update(field_val_loss=field_loss, field_val_accuracy=field_accuracy,
                          field_val_macro_f1=field_f1, selection_macro_f1=selection_score)
        history.append(result)
        print(json.dumps(result), flush=True)
        if selection_score > best + 1e-6:
            best, wait = selection_score, 0
            selected = result
            torch.save({'model_state': model.state_dict(), 'model_name': args.model,
                        'pretrained': args.pretrained, 'bg_prob': args.bg_prob,
                        'seed': args.seed, 'classes': CLASSES, 'bg_style': args.bg_style,
                        'sampling': sampling,
                        'transform': 'resize256_center224_imagenet'}, out / 'best.pt')
        else:
            wait += 1
        scheduler.step()
        # Persist partial histories in case the GPU run is interrupted.
        with history_file.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        if wait >= args.patience:
            break
    config = vars(args) | {'device': str(device), 'sampling': sampling,
                            'best_val_macro_f1': selected['val_macro_f1'],
                            'early_stopped_at': history[-1]['epoch'],
                            'train_manifest_sha256': train_manifest_hash,
                            'val_manifest_sha256': val_manifest_hash,
                            'train_images': len(train_ds), 'val_images': len(val_ds)}
    if sampler is not None:
        config['sampling_details'] = {
            'weight_exponent': -0.5,
            'samples_per_epoch': len(sampler),
            'max_appearances_per_image': sampler.max_appearances,
            'group_counts': [{'source': source, 'label': label, 'images': count}
                             for (source, label), count in sorted(sampler.group_counts.items())],
        }
        if isinstance(sampler, ClassBalancedSourceSampler):
            config['sampling_details']['class_draw_counts'] = sampler.class_draw_counts
    if field_val_dl is not None:
        config.update(checkpoint_selection='mean_validation_macro_f1', best_selection_macro_f1=best,
                      best_field_val_macro_f1=selected['field_val_macro_f1'], best_epoch=selected['epoch'],
                      field_val_images=len(field_val_dl.dataset),
                      field_val_manifest_sha256=hashlib.sha256(Path(field_val_path).read_bytes()).hexdigest())
    if init_checkpoint is not None:
        config['init_checkpoint_sha256'] = init_checkpoint_hash
    if train_ds.background_cache_dir is not None:
        config['background_cache_dir'] = str(train_ds.background_cache_dir)
    config['pin_memory_effective'] = train_dl.pin_memory
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    return config


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['cnn', 'resnet18'], required=True)
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--bg-prob', type=float, default=0.,
                   help='Probability of replacing masked training background; D=0.75')
    p.add_argument('--bg-style', choices=['synthetic','natural'], default='synthetic',
                   help='Training background replacement recipe (not stress test)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--train-manifest', help='Optional explicit train CSV (v3)')
    p.add_argument('--val-manifest', help='Optional explicit validation CSV (v3)')
    p.add_argument('--allow-unmasked-field', action='store_true',
                   help='For v3 only: leave unmasked source=field_extra images unmodified')
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--sampling', choices=['uniform', 'source-class', 'class-source'], default='uniform',
                   help='Training sampler. Class-source sets equal class quotas before weighting sources.')
    p.add_argument('--init-checkpoint', help='Initialize from an existing compatible checkpoint')
    p.add_argument('--lr-scale', type=float, default=1.0, help='Multiply both starting learning rates')
    p.add_argument('--train-crop-ratio-min', type=float, default=TRAIN_CROP_RATIO_MIN,
                   help='Minimum width-to-height ratio for the training crop')
    p.add_argument('--train-crop-ratio-max', type=float, default=TRAIN_CROP_RATIO_MAX,
                   help='Maximum width-to-height ratio for the training crop')
    p.add_argument('--train-pad-aspect-probability', type=float, default=0.0,
                   help='Probability of using padded geometry in place of the training crop')
    p.add_argument('--train-pad-aspect-min', type=float, default=TRAIN_PAD_ASPECT_MIN,
                   help='Minimum width-to-height ratio for the padded training canvas')
    p.add_argument('--train-pad-aspect-max', type=float, default=TRAIN_PAD_ASPECT_MAX,
                   help='Maximum width-to-height ratio for the padded training canvas')
    p.add_argument('--field-val-manifest', help='Average its macro-F1 with the first validation split')
    p.add_argument('--background-cache-dir', help='Cache prepared natural backgrounds as lossless PNGs')
    p.add_argument('--val-num-workers', type=int, help='Validation workers (defaults to --num-workers)')
    add_loader_arguments(p)
    return p


if __name__ == '__main__':
    train(get_parser().parse_args())
