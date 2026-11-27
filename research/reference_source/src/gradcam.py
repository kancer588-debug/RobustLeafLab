"""Grad-CAM for ResNet18, plus optional foreground saliency mass audit.

Attention maps are qualitative and not proof of biological causality.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import CLASSES, create_model, mask_from_segmented, transform
from torchvision import transforms as T


class GradCAM:
    def __init__(self, net):
        if not hasattr(net, 'layer4'):
            raise ValueError('Grad-CAM visualization currently supports ResNet18')
        self.act, self.grad = None, None
        self.hook = net.layer4[-1].conv2.register_forward_hook(self._capture)
        self.net = net

    def _capture(self, module, inp, out):
        self.act = out
        out.register_hook(lambda g: setattr(self, 'grad', g))

    def map(self, x, class_idx=None):
        self.net.eval()
        self.net.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.net(x)
            if class_idx is None:
                class_idx = int(logits.argmax(1)[0])
            logits[0,class_idx].backward()
            weights = self.grad.mean(dim=(2,3),keepdim=True)
            heat = F.relu((weights*self.act).sum(dim=1,keepdim=True))
            heat = F.interpolate(heat, size=x.shape[-2:],mode='bilinear',align_corners=False)
            heat = heat[0,0].detach().cpu().numpy()
            heat /= max(float(heat.max()), 1e-12)
            return heat, class_idx, logits.softmax(1)[0].detach().cpu().numpy()

    def close(self):
        self.hook.remove()


def heatmap_overlay(pil_image, heat, opacity=.45):
    # Match model's resize256_center224 preprocessing for correct overlay.
    display = np.asarray(T.CenterCrop(224)(T.Resize(256)(pil_image.convert('RGB'))),dtype=float)/255
    cmap = plt.get_cmap('jet')(heat)[:,:,:3]
    blended = np.clip((1-opacity)*display + opacity*cmap,0,1)
    return Image.fromarray((blended*255).astype('uint8'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--out',default='outputs/gradcam')
    p.add_argument('--limit',type=int,default=16)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=create_model(ck['model_name'],False)
    net.load_state_dict(ck['model_state'])
    cam=GradCAM(net)
    with Path(a.manifest).open(newline='',encoding='utf8') as f:
        records=list(csv.DictReader(f))[:a.limit]
    outputs=[]
    try:
        for i,row in enumerate(records):
            original=Image.open(row['path']).convert('RGB')
            x=transform()(original).unsqueeze(0)
            heat,idx,prob=cam.map(x)
            result=heatmap_overlay(original,heat)
            path=out/f'{i:03d}_gradcam.png'
            result.save(path)
            attention=None
            if row.get('mask_path') and Path(row['mask_path']).is_file():
                with Image.open(row['mask_path']) as seg:
                    foreground=T.CenterCrop(224)(T.Resize(256)(mask_from_segmented(seg)))
                leaf=np.asarray(foreground,dtype=float)/255.
                attention=float((heat*leaf).sum()/max(heat.sum(),1e-10))
            outputs.append({'path':row['path'],'true':row['label'],
                            'pred':CLASSES[idx],'confidence':float(prob[idx]),
                            'foreground_attention_mass': attention,'overlay':str(path)})
            print(i, row['label'], CLASSES[idx], f'{prob[idx]:.3f}')
    finally:
        cam.close()
    with (out/'gradcam_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=outputs[0].keys());w.writeheader();w.writerows(outputs)
    print('Grad-CAM attention is explanatory, NOT causal evidence.')


if __name__=='__main__':
    main()
