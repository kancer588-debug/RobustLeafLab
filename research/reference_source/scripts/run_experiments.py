"""Run selected experiment arms and seeds; preserve completed results.

PlantDoc is an external audit, NOT a model-selection or tuning signal.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ARMS = {
    'A_cnn': ['--model', 'cnn'],
    'B_scratch': ['--model', 'resnet18'],
    'C_transfer': ['--model', 'resnet18', '--pretrained'],
    'D_bgaug': ['--model', 'resnet18', '--pretrained', '--bg-prob', '0.75'],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--out', default='outputs')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force-eval', action='store_true', help='Recompute completed metrics')
    p.add_argument('--holdout-stress', action='store_true',
                   help='Also run a different, unseen synthetic background recipe')
    a = p.parse_args()

    def execute(cmd):
        print(' '.join(map(str, cmd)), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=True)

    for arm in a.arms:
        for seed in a.seeds:
            run = Path(a.out) / f'{arm}_s{seed}'
            ready = (run / 'best.pt').is_file() and (run / 'config.json').is_file()
            if not ready:
                # A best.pt from an interrupted run is NOT a complete checkpoint.
                execute([sys.executable, 'src/train.py', *ARMS[arm], '--seed', str(seed),
                         '--data-dir', a.data, '--out', str(run),
                         '--epochs', str(a.epochs), '--batch-size', str(a.batch_size)])
            else:
                print(f'[SKIP training] {run}: best.pt + config.json exist', flush=True)
            evaluations = [('pv', 'test.csv', []),
                           ('pv_stress', 'test.csv', ['--background-stress']),
                           ('plantdoc', 'external_plantdoc.csv', [])]
            if a.holdout_stress:
                evaluations.append(('pv_stress_holdout', 'test.csv',
                                    ['--background-stress', '--stress-mode', 'holdout']))
            for name, manifest, opts in evaluations:
                target = run / f'eval_{name}'
                if (target / 'metrics.json').is_file() and not a.force_eval:
                    print(f'[SKIP evaluation] {target}: metrics.json exists', flush=True)
                    continue
                execute([sys.executable, 'src/evaluate.py', '--checkpoint', str(run / 'best.pt'),
                         '--manifest', str(Path(a.data) / manifest), '--out', str(target),
                         '--batch-size', str(a.batch_size), *opts])

    if not a.dry_run:
        execute([sys.executable, 'src/aggregate.py', '--runs', a.out,
                 '--out', str(Path(a.out) / 'summary')])
        for seed in a.seeds:
            for domain in ['pv', 'pv_stress', 'plantdoc'] + (['pv_stress_holdout'] if a.holdout_stress else []):
                file_a = Path(a.out) / f'C_transfer_s{seed}' / f'eval_{domain}' / 'predictions.csv'
                file_b = Path(a.out) / f'D_bgaug_s{seed}' / f'eval_{domain}' / 'predictions.csv'
                if not (file_a.is_file() and file_b.is_file()):
                    print(f'[SKIP comparison] seed {seed} {domain}: missing C/D predictions')
                    continue
                target = Path(a.out) / 'summary' / f'C_vs_D_{domain}_s{seed}.json'
                if target.is_file() and not a.force_eval:
                    print(f'[SKIP comparison] {target}')
                    continue
                execute([sys.executable, 'src/compare.py', '--a', str(file_a),
                         '--b', str(file_b), '--out', str(target)])


if __name__ == '__main__':
    main()
