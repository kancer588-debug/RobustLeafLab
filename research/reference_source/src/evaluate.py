"""Frozen checkpoint evaluation, domain shift stress test and audit artifacts."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score)
from torch.utils.data import DataLoader

from common import CLASSES, ManifestDataset, create_model, predict
from inference_views import VIEW_MODES, apply_view


def wilson_interval(correct, n, z=1.959963984540054):
    if n < 1:
        raise ValueError('n must be >= 1')
    p = correct/n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n)) / den
    hw = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [max(0, mid-hw), min(1, mid+hw)]


def evaluate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if tuple(ck['classes']) != tuple(CLASSES):
        raise ValueError('Checkpoint class ordering does not match current code')
    model = create_model(ck['model_name'], False)
    model.load_state_dict(ck['model_state'])
    model = model.to(device)
    ds = ManifestDataset(args.manifest, background_stress=args.background_stress,
                         seed=args.stress_seed, stress_mode=args.stress_mode)
    view = getattr(args, 'view', 'center')
    model = apply_view(view, ds, model)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
    truth, predicted, proba, loss = predict(model, dl, device)
    acc = float(accuracy_score(truth, predicted))
    cm = confusion_matrix(truth, predicted, labels=[0,1,2])
    experiment = {'model': ck['model_name'], 'pretrained': ck['pretrained']}
    if 'checkpoint_merge' in ck:
        experiment['checkpoint_merge'] = ck['checkpoint_merge']
        experiment['background_probability'] = None
    else:
        experiment['background_probability'] = float(ck['bg_prob'])
    metrics = {
        'n': len(truth), 'accuracy': acc,
        'accuracy_95pct_wilson_ci': wilson_interval(int((truth == predicted).sum()), len(truth)),
        'macro_f1': float(f1_score(truth, predicted, labels=[0,1,2],
                                   average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(truth, predicted)),
        'cross_entropy': loss, 'view': view, 'background_stress': args.background_stress,
        'stress_seed': args.stress_seed if args.background_stress else None,
        'stress_mode': args.stress_mode if args.background_stress else None,
        'class_report': classification_report(truth, predicted, labels=[0,1,2],
                                               target_names=CLASSES, output_dict=True,
                                               zero_division=0),
        'model_training_seed': int(ck['seed']),
        'experiment': experiment}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    fields = ['path','sha256','true_label','pred_label','confidence','correct'] + [f'prob_{c}' for c in CLASSES]
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(ds.rows):
            writer.writerow({'path': row['path'], 'sha256': row.get('sha256',''),
                 'true_label': CLASSES[truth[i]], 'pred_label': CLASSES[predicted[i]],
                 'confidence': float(proba[i, predicted[i]]),
                 'correct': int(truth[i] == predicted[i]),
                 **{f'prob_{c}': float(proba[i,j]) for j,c in enumerate(CLASSES)}})
    with (out / 'confusion_matrix.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true\\pred'] + list(CLASSES))
        for c,row in zip(CLASSES,cm): writer.writerow([c]+row.tolist())
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, cmap='Blues', xticks_rotation=25, colorbar=False, values_format='d')
    ax.set_title('Confusion matrix (rows = true labels)')
    fig.tight_layout()
    fig.savefig(out/'confusion_matrix.png', dpi=180)
    plt.close(fig)
    print(json.dumps({k: metrics[k] for k in ('n','accuracy','macro_f1','balanced_accuracy')},indent=2))
    return metrics


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--view', choices=VIEW_MODES, default='center',
                   help='Image framing at inference. Five-crop averages logits across views.')
    p.add_argument('--background-stress',action='store_true',
                   help='Swap only held-out PV leaf backgrounds; requires mask_path')
    p.add_argument('--stress-seed',type=int,default=2026)
    p.add_argument('--stress-mode',choices=['standard','holdout'],default='standard',
                   help='holdout background recipe is intentionally different from training')
    return p


if __name__=='__main__':
    evaluate(parser().parse_args())
