"""Audit controlled/field data and create *fixed* stratified source manifests.

Never change image membership between seeds. This script combines PlantDoc's
upstream train/test folders ONLY for a completely external evaluation dataset.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

PV = {'Tomato___healthy': 'healthy', 'Tomato___Early_blight': 'early_blight',
      'Tomato___Late_blight': 'late_blight'}
PD = {'Tomato leaf': 'healthy', 'Tomato Early blight leaf': 'early_blight',
      'Tomato leaf late blight': 'late_blight'}
EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def collect(root, mapping, mask_root=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset directory not found: {root}')
    masks = {}
    if mask_root:
        for label_folder in mapping:
            # In PlantVillage raw/segmented, images are typically named the
            # same as the matching raw/color image.
            for path in Path(mask_root).resolve().rglob('*'):
                if (path.is_file() and path.suffix.lower() in EXT and
                        path.parent.name == label_folder):
                    stem = path.stem
                    if stem.lower().endswith('_final_masked'):
                        stem = stem[:-len('_final_masked')]
                    key = (label_folder.lower(), stem.lower())
                    if key in masks:
                        raise ValueError(f'Ambiguous segmentation pairing: {key}')
                    masks[key] = str(path.resolve())
    records, seen = [], {}
    duplicates = []
    for dirname, label in mapping.items():
        folders = sorted(p for p in root.rglob(dirname) if p.is_dir())
        if root.name == dirname:
            folders.insert(0, root)
        if not folders:
            raise ValueError(f'Missing folder "{dirname}" under {root}')
        for folder in folders:
            for path in sorted(folder.rglob('*')):
                if path.suffix.lower() not in EXT or not path.is_file():
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen:
                    if seen[digest] != label:
                        raise ValueError(f'Identical image has conflicting labels: {path}')
                    duplicates.append(str(path))
                    continue
                seen[digest] = label
                mask = masks.get((dirname.lower(), path.stem.lower()), '')
                records.append({'path': str(path.resolve()), 'mask_path': mask,
                                'label': label, 'sha256': digest})
    return records, duplicates


def write_manifest(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path','mask_path','label','sha256'])
        writer.writeheader()
        writer.writerows(rows)


def prepare(pv_root, pd_root, out, mask_root=None, split_seed=2026):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pv, pv_dupes = collect(pv_root, PV, mask_root)
    pd, pd_dupes = collect(pd_root, PD)
    pv_hashes = {r['sha256'] for r in pv}
    overlap = [r['path'] for r in pd if r['sha256'] in pv_hashes]
    if overlap:
        raise ValueError(f'Cross-dataset byte-level overlap, remove before testing: {overlap[:3]}')
    rng = random.Random(split_seed)
    splits = {'train': [], 'val': [], 'test': []}
    for label in PV.values():
        group = sorted((r for r in pv if r['label'] == label), key=lambda r:r['path'])
        rng.shuffle(group)
        n = len(group)
        n_train, n_val = int(n * .70), int(n * .15)
        splits['train'] += group[:n_train]
        splits['val'] += group[n_train:n_train + n_val]
        splits['test'] += group[n_train + n_val:]
    for key, rows in splits.items():
        write_manifest(out / f'{key}.csv', sorted(rows, key=lambda r:r['path']))
    write_manifest(out / 'external_plantdoc.csv', sorted(pd, key=lambda r:r['path']))
    counts = {k: {'total': len(v), 'by_class': dict(Counter(r['label'] for r in v)),
                  'masks': sum(bool(r['mask_path']) for r in v)} for k,v in splits.items()}
    counts['external_plantdoc'] = {'total': len(pd),
        'by_class': dict(Counter(r['label'] for r in pd))}
    audit = {'split_seed': split_seed, 'dataset_counts': counts,
             'pv_exact_duplicates_removed': len(pv_dupes),
             'pd_exact_duplicates_removed': len(pd_dupes),
             'cross_dataset_exact_duplicates': 0,
             'group_split_caution': 'Image-level split; possible different images of the same leaf across splits. Prefer author leaf-group metadata when available.',
             'pd_usage': 'External test ONLY. Train and test upstream folders combined; no PD samples selected or tuned on.'}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--pv-root', required=True,
                   help='PlantVillage raw/color ONLY, not raw parent')
    a.add_argument('--pd-root', required=True,
                   help='Cropped PlantDoc repository containing train and test')
    a.add_argument('--mask-root', help='PlantVillage raw/segmented root; required for background treatment')
    a.add_argument('--out', default='data')
    a.add_argument('--split-seed', type=int, default=2026)
    args = a.parse_args()
    prepare(**vars(args))
