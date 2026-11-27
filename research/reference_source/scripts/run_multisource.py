"""Train E/F/G multi-source models; preserve every v2 output and checkpoint.

E: ResNet18 ImageNet pretraining on PV train + new source field train.
F: E plus PV-only masked synthetic background intervention.
G: E plus PV-only masked natural background composites using FIELD TRAIN only.
Both select the best checkpoint on the ORIGINAL PV validation split, just as
v2 did. This isolates adding training images better; PlantDoc is an audit.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ARMS = {
    'E_multisource': ['--model', 'resnet18', '--pretrained'],
    'F_multisource_bg': ['--model', 'resnet18', '--pretrained', '--bg-prob', '0.75',
                         '--allow-unmasked-field'],
    'G_multisource_natural': ['--model', 'resnet18', '--pretrained', '--bg-prob', '0.75',
                              '--bg-style', 'natural', '--allow-unmasked-field'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data_v3')
    p.add_argument('--v2-data', default='data')
    p.add_argument('--out', default='outputs/v3')
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--sampling', choices=['uniform', 'source-class'], default='uniform')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--val-num-workers', type=int)
    p.add_argument('--pin-memory', action='store_true')
    p.add_argument('--persistent-workers', action='store_true')
    p.add_argument('--background-cache-dir', help='Lossless background cache used by arm G only')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force-eval', action='store_true')
    args = p.parse_args(argv)
    if args.num_workers < 0 or (args.val_num_workers is not None and args.val_num_workers < 0):
        p.error('Worker counts must be non-negative')
    if args.persistent_workers and args.num_workers == 0:
        p.error('Use at least one worker with --persistent-workers')

    loader_flags = ['--num-workers', str(args.num_workers)]
    if args.pin_memory:
        loader_flags.append('--pin-memory')
    if args.persistent_workers:
        loader_flags.append('--persistent-workers')
    if args.val_num_workers is not None:
        loader_flags += ['--val-num-workers', str(args.val_num_workers)]
    if args.sampling != 'uniform':
        loader_flags += ['--sampling', args.sampling]

    data, v2, root = Path(args.data), Path(args.v2_data), Path(args.out)
    if root.resolve() == Path('outputs').resolve():
        raise ValueError('Use a separate v3 folder, not original outputs/')
    needed = [data/'train.csv', data/'field_val.csv', data/'field_test.csv',
              v2/'val.csv', v2/'test.csv', v2/'external_plantdoc.csv']
    for path in needed:
        if not path.is_file():
            raise FileNotFoundError(f'Prepare v3 manifests first: {path}')
    train_hash, val_hash = sha(data/'train.csv'), sha(v2/'val.csv')

    def execute(cmd):
        print('+ ' + ' '.join(map(str, cmd)), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    for arm in args.arms:
        for seed in args.seeds:
            run = root/f'{arm}_s{seed}'
            ck, config = run/'best.pt', run/'config.json'
            cache_dir = (str(Path(args.background_cache_dir).resolve())
                         if arm == 'G_multisource_natural' and args.background_cache_dir else None)
            cache_flags = ['--background-cache-dir', cache_dir] if cache_dir else []
            if ck.is_file() and config.is_file():
                meta = json.loads(config.read_text(encoding='utf-8'))
                if (meta.get('train_manifest_sha256') != train_hash or
                        meta.get('val_manifest_sha256') != val_hash or
                        meta.get('seed') != seed):
                    raise RuntimeError(f'{run} trained on different data/seed. '
                                       'Never overwrite prior weights; choose a new --out directory.')
                if meta.get('sampling', 'uniform') != args.sampling:
                    raise RuntimeError(f'{run} used a different sampler. Use a new --out directory.')
                requested_loader = (args.num_workers, args.pin_memory, args.persistent_workers,
                                    cache_dir, args.val_num_workers)
                saved_loader = (meta.get('num_workers', 0), meta.get('pin_memory', False),
                                meta.get('persistent_workers', False), meta.get('background_cache_dir'),
                                meta.get('val_num_workers'))
                tuning_requested = any(requested_loader[:4]) or args.val_num_workers is not None
                if tuning_requested and saved_loader != requested_loader:
                    raise RuntimeError(f'{run} has different loader settings. Use a new --out directory.')
                print(f'[SKIP trained] {run}', flush=True)
            else:
                if ck.is_file() or config.is_file() or (run/'history.csv').is_file():
                    raise RuntimeError(f'Incomplete previous training in {run}. '
                                       'Back it up and use a new output directory to prevent overwriting.')
                execute([sys.executable, 'src/train.py', *ARMS[arm],
                         '--seed', str(seed), '--train-manifest', str(data/'train.csv'),
                         '--val-manifest', str(v2/'val.csv'),
                         '--out', str(run), '--epochs', str(args.epochs),
                         '--batch-size', str(args.batch_size), *loader_flags, *cache_flags])
            evals = [
                ('pv_original', v2/'test.csv', []),
                ('pv_stress', v2/'test.csv', ['--background-stress']),
                ('pv_stress_holdout', v2/'test.csv', ['--background-stress', '--stress-mode', 'holdout']),
                ('field_validation', data/'field_val.csv', []),
                ('field_holdout', data/'field_test.csv', []),
                ('plantdoc_audit', v2/'external_plantdoc.csv', []),
            ]
            for name, manifest, options in evals:
                target = run/f'eval_{name}'
                if (target/'metrics.json').is_file() and not args.force_eval:
                    print(f'[SKIP eval] {target}', flush=True)
                    continue
                execute([sys.executable, 'src/evaluate.py', '--checkpoint', str(ck),
                         '--manifest', str(manifest), '--out', str(target),
                         '--batch-size', str(args.batch_size), *options])
    if not args.dry_run:
        execute([sys.executable, 'scripts/summarize_multisource.py',
                 '--v2-out', 'outputs', '--v3-out', str(root)])


if __name__ == '__main__':
    main()
