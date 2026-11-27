"""Compare two frozen models on identical image IDs with paired bootstrap CIs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from common import CLASSES


def read(path):
    with open(path,newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    by_key={r.get('sha256') or r['path']:r for r in rows}
    if len(by_key)!=len(rows):
        raise ValueError('Duplicate prediction image identifiers')
    return by_key


def compare(a,b,repeats=5000,seed=2026):
    one,two=read(a),read(b)
    if set(one)!=set(two):
        raise ValueError('Paired comparison requires identical evaluated images')
    keys=sorted(one)
    true=np.array([CLASSES.index(one[k]['true_label']) for k in keys])
    true_b=[two[k]['true_label'] for k in keys]
    if [CLASSES[i] for i in true]!=true_b:
        raise ValueError('Different ground-truth labels for same images')
    p=np.array([CLASSES.index(one[k]['pred_label']) for k in keys])
    q=np.array([CLASSES.index(two[k]['pred_label']) for k in keys])
    rng=np.random.default_rng(seed)
    deltas_acc=np.empty(repeats)
    deltas_f1=np.empty(repeats)
    for i in range(repeats):
        ix=rng.integers(len(true),size=len(true))
        deltas_acc[i]=accuracy_score(true[ix],q[ix])-accuracy_score(true[ix],p[ix])
        deltas_f1[i]=f1_score(true[ix],q[ix],labels=[0,1,2],average='macro',zero_division=0)-f1_score(true[ix],p[ix],labels=[0,1,2],average='macro',zero_division=0)
    gain_acc=float(accuracy_score(true,q)-accuracy_score(true,p))
    gain_f1=float(f1_score(true,q,labels=[0,1,2],average='macro')-f1_score(true,p,labels=[0,1,2],average='macro'))
    return {'n_paired':len(true),'delta_B_minus_A_accuracy':gain_acc,
            'delta_accuracy_95pct_paired_bootstrap_ci':np.quantile(deltas_acc,[.025,.975]).tolist(),
            'delta_B_minus_A_macro_f1':gain_f1,
            'delta_macro_f1_95pct_paired_bootstrap_ci':np.quantile(deltas_f1,[.025,.975]).tolist(),
            'a_correct_b_wrong':int(((true==p)&(true!=q)).sum()),
            'a_wrong_b_correct':int(((true!=p)&(true==q)).sum()),
            'bootstrap_repeats':repeats,'bootstrap_seed':seed,
            'note':'Paired image bootstrap is conditional on this evaluation set; a non-overlap rule for separate marginal confidence intervals is not used.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--a',required=True,help='CSV of model A predictions')
    p.add_argument('--b',required=True,help='CSV of model B predictions')
    p.add_argument('--out',required=True)
    a=p.parse_args(); result=compare(a.a,a.b)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
