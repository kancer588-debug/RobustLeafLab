"""Write one comparison table for v2 A-D and v3 E-F; no invented values."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v2-out', default='outputs')
    p.add_argument('--v3-out', default='outputs/v3')
    a = p.parse_args()
    rows = []
    domain_map_v2 = {'pv': 'PV_test', 'pv_stress': 'PV_synthetic_stress',
                     'pv_stress_holdout': 'PV_holdout_stress', 'plantdoc': 'PlantDoc_audit'}
    domain_map_v3 = {'pv_original': 'PV_test', 'pv_stress': 'PV_synthetic_stress',
                     'pv_stress_holdout': 'PV_holdout_stress',
                     'field_validation': 'New_field_val', 'field_holdout': 'New_field_test', 'plantdoc_audit': 'PlantDoc_audit'}
    for root, version, domain_map in [(Path(a.v2_out), 'v2', domain_map_v2),
                                      (Path(a.v3_out), 'v3', domain_map_v3)]:
        for path in sorted(root.glob('*/eval_*/metrics.json')):
            stem = path.parent.name.removeprefix('eval_')
            if stem not in domain_map:
                continue
            metric = json.loads(path.read_text(encoding='utf-8'))
            run = path.parent.parent.name
            if '_s' not in run:
                continue
            arm, seed = run.rsplit('_s', 1)
            rows.append({'version': version, 'arm': arm, 'seed': seed,
                         'evaluation': domain_map[stem], 'n': metric['n'],
                         'accuracy': round(float(metric['accuracy']), 6),
                         'macro_f1': round(float(metric['macro_f1']), 6),
                         'balanced_accuracy': round(float(metric['balanced_accuracy']), 6)})
    output = Path(a.v3_out)/'comparison_v2_v3.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['version','arm','seed','evaluation','n',
                                               'accuracy','macro_f1','balanced_accuracy'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} real evaluations to {output}')
    for row in rows:
        print(f"{row['version']} {row['arm']:<18} seed={row['seed']:>3} "
              f"{row['evaluation']:<20} Acc={100*row['accuracy']:.2f}% "
              f"MacroF1={row['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
