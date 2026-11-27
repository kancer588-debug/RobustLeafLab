"""Prepare v3 multi-source manifests WITHOUT changing v2 data or PlantDoc.

Input: an extracted folder or .zip downloaded from Mendeley 10.17632/swtt5wy2pr.1.
Only exact tomato leaf class folders are accepted. Images are copied into a
controlled destination after byte/near-duplicate audits and fixed splitting.

Reusing held-out data for model selection is prohibited. No PlantDoc image is
added to training or validation, and original PlantVillage test is unchanged.
"""
import argparse
import csv
import hashlib
import io
import json
import random
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

CLASSES = ('healthy', 'early_blight', 'late_blight')
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
# Explicit exact class aliases; NEVER match by substring (e.g. Healthy Fruits).
ALIASES = {
    'healthy leaf': 'healthy', 'tomato healthy leaf': 'healthy',
    'early blight leaf': 'early_blight', 'tomato early blight leaf': 'early_blight',
    'late blight leaf': 'late_blight', 'tomato late blight leaf': 'late_blight',
}
FIELDS = ('path', 'mask_path', 'label', 'sha256', 'source')


def normalized(name):
    return re.sub(r'[^a-z0-9]+', ' ', str(name).lower()).strip()


def detect_label(parts):
    """Require exactly one recognized CLASS DIRECTORY in a path."""
    found = {ALIASES[normalized(part)] for part in parts[:-1]
             if normalized(part) in ALIASES}
    return next(iter(found)) if len(found) == 1 else None


def dhash(image):
    """64-bit difference hash, useful for near-duplicate *screening* only."""
    gray = image.convert('L').resize((9, 8), Image.Resampling.BILINEAR)
    px = list(gray.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(px[y * 9 + x] > px[y * 9 + x + 1])
    return bits


def load_base(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Base manifest is empty: {path}')
    for row in rows:
        if row.get('label') not in CLASSES or not Path(row.get('path', '')).is_file():
            raise ValueError(f'Invalid base manifest row: {row}')
    return rows


def source_images(source):
    """Yield (origin, bytes) from a directory or top-level ZIP; no archive extraction."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f'Field dataset missing: {source}')
    if source.is_file():
        if source.suffix.lower() != '.zip':
            raise ValueError('Field source must be an extracted directory or .zip archive')
        with zipfile.ZipFile(source) as archive:
            for name in sorted(archive.namelist()):
                parts = PurePosixPath(name.replace('\\', '/')).parts
                if not parts or PurePosixPath(name).suffix.lower() not in IMAGE_EXT:
                    continue
                label = detect_label(parts)
                if label:
                    with archive.open(name) as f:
                        yield name, label, f.read()
        return
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXT:
            continue
        parts = path.relative_to(source).parts
        label = detect_label(parts)
        if label:
            yield str(path), label, path.read_bytes()


def audit_candidates(source):
    good, corrupt = [], []
    for origin, label, raw in source_images(source):
        try:
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                if im.width < 32 or im.height < 32:
                    raise ValueError('image dimensions below 32 pixels')
                phash = dhash(im)
                source_format = (im.format or 'PNG').upper()
        except (UnidentifiedImageError, OSError, ValueError) as e:
            corrupt.append({'origin': origin, 'reason': str(e)[:160]})
            continue
        good.append({'origin': origin, 'label': label,
                     'sha256': hashlib.sha256(raw).hexdigest(),
                     'dhash': phash, 'format': source_format})
    return good, corrupt


def existing_hashes_and_phashes(manifest_paths):
    items = []
    for manifest in manifest_paths:
        for row in load_base(manifest):
            with Image.open(row['path']) as im:
                phash = dhash(im)
            items.append((row['sha256'], phash, str(manifest), row['label']))
    return items


def prepare(args):
    base = Path(args.base_data)
    # Never use source data / existing models as a write destination.
    out = Path(args.out)
    imported = Path(args.import_dir)
    originals = {name: base / f'{name}.csv' for name in ('train', 'val', 'test')}
    originals['plantdoc'] = base / 'external_plantdoc.csv'
    for name, path in originals.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing original {name} manifest: {path}')
    if args.near_threshold < 0 or args.near_threshold > 10:
        raise ValueError('--near-threshold must be 0–10; a dHash is only a screening heuristic')
    if out.resolve() == base.resolve():
        raise ValueError('v3 out must differ from v2 data directory')
    if (out / 'train.csv').exists():
        raise FileExistsError(f'{out}/train.csv exists. Use a NEW output folder to preserve reproducibility.')
    if imported.exists() and any(imported.iterdir()):
        raise FileExistsError(f'Import folder not empty: {imported}; choose a NEW folder')

    print('[1/4] Reading original manifests and screening existing images...', flush=True)
    reference = existing_hashes_and_phashes(originals.values())
    existing_exact = {item[0] for item in reference}
    existing_phashes = [item[1] for item in reference]
    print('[2/4] Reading field images from source...', flush=True)
    raw_candidates, corrupt = audit_candidates(args.extra_root)
    counts_raw = Counter(c['label'] for c in raw_candidates)
    if not raw_candidates:
        raise ValueError('No matching Healthy Leaf / Early Blight Leaf / Late Blight Leaf image folders. '
                         'Extract any nested archives, then check folder names. '
                         'Other classes are intentionally ignored.')
    rejected = Counter()
    near_examples = []
    accepted = []
    seen = {}
    accepted_phashes = []
    for item in sorted(raw_candidates, key=lambda d: (d['label'], d['sha256'], d['origin'])):
        sha = item['sha256']
        if sha in existing_exact:
            rejected['exact_overlap_with_originals'] += 1
            continue
        if sha in seen:
            if seen[sha] != item['label']:
                rejected['conflicting_label_duplicate'] += 1
            else:
                rejected['duplicate_within_field_source'] += 1
            continue
        # Near matches are conservatively EXCLUDED pending manual review.
        # dHash is not proof of identity, and screening may exclude distinct leaves.
        closest_existing = min(((p ^ item['dhash']).bit_count() for p in existing_phashes), default=64)
        if closest_existing <= args.near_threshold:
            rejected['near_original_quarantined'] += 1
            if len(near_examples) < 30:
                near_examples.append({'origin': item['origin'], 'label': item['label'],
                                      'min_dhash_distance': closest_existing})
            continue
        closest_new = min(((p ^ item['dhash']).bit_count() for p in accepted_phashes), default=64)
        if closest_new <= args.near_threshold:
            rejected['near_extra_quarantined'] += 1
            continue
        seen[sha] = item['label']
        accepted_phashes.append(item['dhash'])
        accepted.append(item)
    field_counts = Counter(c['label'] for c in accepted)
    absent = [c for c in CLASSES if field_counts[c] < 7]
    if absent:
        raise ValueError('Too few clean field images after duplicate screening for classes '
                         f'{absent}. Counts: {dict(field_counts)}. No output was written.')

    print('[3/4] Building reproducible per-class field splits...', flush=True)
    rng = random.Random(args.split_seed)
    subsets = {'train': [], 'val': [], 'test': []}
    for label in CLASSES:
        group = sorted((c for c in accepted if c['label'] == label), key=lambda c: c['sha256'])
        rng.shuffle(group)
        n = len(group)
        n_train, n_val = int(.70 * n), int(.15 * n)
        if not n_train or not n_val or n-n_train-n_val < 1:
            raise ValueError(f'Cannot split class {label}: {n}')
        subsets['train'] += group[:n_train]
        subsets['val'] += group[n_train:n_train+n_val]
        subsets['test'] += group[n_train+n_val:]

    print('[4/4] Copying approved field images and writing NEW manifests...', flush=True)
    imported.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    extra_rows = {}
    # Streaming: do not keep hundreds of megabytes of original field photos
    # in Python RAM. For an input ZIP, keep one archive handle open.
    source = Path(args.extra_root)
    archive = zipfile.ZipFile(source) if source.is_file() else None
    try:
        for split, items in subsets.items():
            rows = []
            for item in items:
                # The extension is normalized; no unsafe upstream filename is reused.
                fmt = item['format']
                ext = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'BMP': '.bmp'}.get(fmt)
                if ext is None:
                    raise ValueError(f'Unsupported image encoding: {fmt}')
                file = imported / item['label'] / f"{item['sha256']}{ext}"
                file.parent.mkdir(parents=True, exist_ok=True)
                raw = archive.read(item['origin']) if archive else Path(item['origin']).read_bytes()
                if hashlib.sha256(raw).hexdigest() != item['sha256']:
                    raise RuntimeError(f'Source image changed during import: {item["origin"]}')
                file.write_bytes(raw)
                rows.append({'path': str(file.resolve()), 'mask_path': '',
                             'label': item['label'], 'sha256': item['sha256'],
                             'source': 'field_extra'})
            extra_rows[split] = sorted(rows, key=lambda d: d['path'])
    finally:
        if archive is not None:
            archive.close()

    def write(file, rows):
        with file.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    original_rows = {k: load_base(path) for k, path in originals.items()}
    for split in ('train', 'val'):
        combined = [{**row, 'source': 'plantvillage'} for row in original_rows[split]] + extra_rows[split]
        write(out / f'{split}.csv', combined)
        write(out / f'field_{split}.csv', extra_rows[split])
    write(out / 'field_test.csv', extra_rows['test'])

    all_groups = [original_rows['train'], original_rows['val'], original_rows['test'],
                  original_rows['plantdoc'], extra_rows['train'], extra_rows['val'], extra_rows['test']]
    hashes = [r['sha256'] for group in all_groups for r in group]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError('Cross-split exact image duplication found after import')

    audit = {
        'source': 'Mendeley Data: Tomato Leaf and Fruit Disease Image Dataset, V1',
        'source_doi': '10.17632/swtt5wy2pr.1', 'source_license': 'CC BY 4.0',
        'selected_classes': list(CLASSES), 'split_seed': args.split_seed,
        'near_duplicate_screening': '64-bit dHash; similarity is a heuristic; quarantined rather than labeled duplicate',
        'near_threshold': args.near_threshold,
        'field_candidates_by_class': dict(counts_raw),
        'field_accepted_by_class': dict(field_counts),
        'field_train_count': len(extra_rows['train']),
        'field_val_count': len(extra_rows['val']),
        'field_test_count': len(extra_rows['test']),
        'new_combined_train_count': len(original_rows['train']) + len(extra_rows['train']),
        'new_combined_val_count': len(original_rows['val']) + len(extra_rows['val']),
        'original_pv_test_count': len(original_rows['test']),
        'original_plantdoc_count': len(original_rows['plantdoc']),
        'rejected': dict(rejected), 'corrupt_count': len(corrupt),
        'near_quarantine_examples': near_examples, 'corrupt_examples': corrupt[:20],
        'cautions': [
            'Do not use PlantDoc to tune the new model: it has already been inspected in v2.',
            'dHash may exclude distinct leaves; audit near matches manually before publishing.',
            'Image-level splits do not ensure distinct plants across splits.',
            'Verify selected field labels and license with the original downloaded release.',
        ],
    }
    (out / 'audit_v3.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k not in ('near_quarantine_examples', 'corrupt_examples')}, indent=2))
    return audit


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extra-root', required=True,
                   help='Folder or zip downloaded from Mendeley (extract nested ZIPs first)')
    p.add_argument('--base-data', default='data', help='Existing v2 manifests (read only)')
    p.add_argument('--out', default='data_v3', help='NEW manifest folder')
    p.add_argument('--import-dir', default='datasets/field_imported_v3', help='NEW image folder')
    p.add_argument('--split-seed', type=int, default=2026)
    p.add_argument('--near-threshold', type=int, default=2,
                   help='Quarantine images with dHash Hamming distance <= this (default 2)')
    return p


if __name__ == '__main__':
    prepare(parser().parse_args())
