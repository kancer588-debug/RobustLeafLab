"""Make one fixed A42/C42 checkpoint and run the public in-domain regression check."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader

from common import CLASSES, ManifestDataset, create_model, predict
from scene_routing import FROZEN_CHECKPOINTS


def checked_load(path, expected):
    import hashlib
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError(f'Unexpected checkpoint: {path}')
        stream.seek(0)
        return torch.load(stream, map_location='cpu', weights_only=True)


def measure(net, loader, device, state):
    net.load_state_dict(state, strict=True)
    truth, predictions, _, _ = predict(net, loader, device)
    matrix = confusion_matrix(truth, predictions, labels=range(len(CLASSES)))
    healthy, early, late = matrix.sum(axis=1)
    return {
        'n': len(truth),
        'accuracy': float((truth == predictions).mean()),
        'macro_f1': float(f1_score(truth, predictions, average='macro')),
        'late_recall': float(matrix[2, 2] / late),
        'early_recall': float(matrix[1, 1] / early),
        'healthy_false_alarm_rate': float(matrix[0, 1:].sum() / healthy),
        'late_to_early': int(matrix[2, 1]),
        'late_to_healthy': int(matrix[2, 0]),
        'confusion': matrix.tolist(),
    }


def main(out):
    a, c = (checked_load(*FROZEN_CHECKPOINTS[name]) for name in ('A', 'C'))
    for checkpoint in (a, c):
        if (checkpoint['model_name'], tuple(checkpoint['classes']),
                checkpoint['seed'], checkpoint['pretrained']) != ('resnet18', CLASSES, 42, True):
            raise ValueError('The fixed merge requires the matching pretrained seed-42 checkpoints')
    a_state, c_state = a['model_state'], c['model_state']
    if a_state.keys() != c_state.keys():
        raise ValueError('Checkpoint layouts differ')

    torch.set_num_threads(4)
    merged = {}
    unequal_integer_buffers = []
    for key, first in a_state.items():
        second = c_state[key]
        if first.shape != second.shape or first.dtype != second.dtype:
            raise ValueError(f'Shape or type mismatch for {key}')
        if first.is_floating_point():
            weight = torch.lerp(first, second, .5)
            if not bool(torch.isfinite(weight).all()):
                raise ValueError(f'Non-finite merged value for {key}')
            merged[key] = weight
        else:
            if not torch.equal(first, second):
                if not key.endswith('.num_batches_tracked'):
                    raise ValueError(f'Unexpected integer buffer mismatch for {key}')
                unequal_integer_buffers.append(key)
            merged[key] = second
    provenance = {
        'method': 'fixed_parameter_average',
        'fractions': {'A42': .5, 'C42': .5},
        'parent_sha256': {f'{key}42': entry[1] for key, entry in FROZEN_CHECKPOINTS.items()},
        'integer_buffer_source': 'C42',
        'unequal_integer_buffers': unequal_integer_buffers,
    }
    checkpoint = {
        'model_name': 'resnet18', 'model_state': merged, 'classes': CLASSES, 'seed': 42,
        'pretrained': True, 'bg_prob': None, 'checkpoint_merge': provenance,
    }
    # Freeze the single candidate before opening any evaluation manifest.
    path = out / 'best.pt'
    with path.open('xb') as stream:
        torch.save(checkpoint, stream)
    frozen = torch.load(path, map_location='cpu', weights_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = create_model('resnet18', False).to(device)
    loader = DataLoader(ManifestDataset('data/val.csv'), batch_size=32, shuffle=False)
    print(f'Fixed single checkpoint saved. Verifying three checkpoints on {device}.', flush=True)
    report = {
        'manifest': 'data/val.csv',
        'view': 'center',
        'classes': CLASSES,
        'checkpoint_merge': provenance,
        'metrics': {name: measure(net, loader, device, state)
                    for name, state in [('A42', a_state), ('C42', c_state), ('AC42', frozen['model_state'])]},
    }
    with (out / 'plantvillage_validation.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
