"""Optional public-source downloader. Requires git and network connectivity.

Default sparse PV acquisition avoids unrelated crop categories. Upstream
repository contents/availability can change; verify resulting audit.json.
"""
import argparse
import subprocess
from pathlib import Path

from sys import executable

PV_FOLDERS=['Tomato___healthy','Tomato___Early_blight','Tomato___Late_blight']


def run(args):
    print('+',' '.join(map(str,args)),flush=True)
    subprocess.run(args,check=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='datasets')
    p.add_argument('--pv-only',action='store_true')
    p.add_argument('--pd-only',action='store_true')
    p.add_argument('--prepare',action='store_true',help='Create fixed manifests if both datasets exist')
    a=p.parse_args()
    folder=Path(a.out).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    pv=folder/'PlantVillage-Dataset'; pd=folder/'PlantDoc-Dataset'
    if not a.pd_only and not pv.exists():
        run(['git','clone','--depth','1','--filter=blob:none','--sparse',
             'https://github.com/spMohanty/PlantVillage-Dataset.git',str(pv)])
        subdirs=[f'raw/{kind}/{name}' for kind in ('color','segmented') for name in PV_FOLDERS]
        run(['git','-C',str(pv),'sparse-checkout','set',*subdirs])
    if not a.pv_only and not pd.exists():
        run(['git','clone','--depth','1',
             'https://github.com/pratikkayal/PlantDoc-Dataset.git',str(pd)])
    for key,root in [('PlantVillage color',pv/'raw'/'color'),
                     ('PlantVillage segmented',pv/'raw'/'segmented'),
                     ('PlantDoc',pd)]:
        if root.exists():
            n=sum(path.suffix.lower() in {'.jpg','.jpeg','.png'} for path in root.rglob('*'))
            print(f'{key}: {n} image files at {root}')
    if a.prepare:
        if not all(x.is_dir() for x in [pv/'raw'/'color',pv/'raw'/'segmented',pd]):
            raise FileNotFoundError('Need both PV and PD to prepare manifests')
        run([executable,'src/prepare_data.py','--pv-root',str(pv/'raw'/'color'),
             '--mask-root',str(pv/'raw'/'segmented'),
             '--pd-root',str(pd),'--out','data','--split-seed','2026'])


if __name__=='__main__':
    main()
