"""Aggregate real recorded metrics; never fill missing runs with placeholders."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',default='outputs')
    p.add_argument('--out',default='outputs/summary')
    a=p.parse_args()
    root,out=Path(a.runs),Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in sorted(root.glob('*/eval_*/metrics.json')):
        run=path.parent.parent.name
        model=run.rsplit('_s',1)[0]
        seed=run.rsplit('_s',1)[-1]
        domain=path.parent.name.replace('eval_','')
        metric=json.loads(path.read_text())
        rows.append({'experiment':model,'seed':seed,'domain':domain,
                     'n':metric['n'],'accuracy':metric['accuracy'],
                     'macro_f1':metric['macro_f1'],
                     'balanced_accuracy':metric['balanced_accuracy']})
    if not rows:
        raise SystemExit('No completed metrics.json found; run real evaluation first')
    with (out/'all_runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    grouped=defaultdict(list)
    for r in rows:
        grouped[(r['experiment'],r['domain'])].append(r)
    summary=[]
    for (name,domain),recs in sorted(grouped.items()):
        item={'experiment':name,'domain':domain,'n_seeds':len(recs)}
        for metric in ['accuracy','macro_f1','balanced_accuracy']:
            vals=[float(r[metric]) for r in recs]
            item[f'{metric}_mean']=float(np.mean(vals))
            item[f'{metric}_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        summary.append(item)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    # Plot only when the same domain exists for multiple experiments.
    groups=defaultdict(list)
    for r in summary: groups[r['domain']].append(r)
    for domain,data in groups.items():
        fig,ax=plt.subplots(figsize=(8,4))
        labels=[r['experiment'] for r in data]
        vals=[r['accuracy_mean']*100 for r in data]
        errors=[(r['accuracy_sd'] or 0)*100 for r in data]
        ax.bar(labels,vals,yerr=errors,capsize=3)
        ax.set(ylabel='Accuracy (%)',ylim=(0,100),title=f'{domain}: accuracy across finished runs')
        ax.tick_params(axis='x',rotation=22)
        fig.tight_layout();fig.savefig(out/f'{domain}_accuracy.png',dpi=180)
        plt.close(fig)
    print('Wrote',out/'summary.json','with',len(rows),'real completed evaluations')


if __name__=='__main__':
    main()
