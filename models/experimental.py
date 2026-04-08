# YOLOv5 experimental modules

import math
import numpy as np
import torch
import torch.nn as nn

from models.common import Conv
from utils.downloads import attempt_download


class Sum(nn.Module):
    def __init__(self, n, weight=False):
        super().__init__()
        self.weight = weight
        self.iter = range(n - 1)
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)

    def forward(self, x):
        y = x[0]
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super().__init__()
        n = len(k)
        if equal_ch:
            i = torch.linspace(0, n - 1e-6, c2).floor()
            c_ = [(i == g).sum() for g in range(n)]
        else:
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(n)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 1)
        return y, None


def attempt_load(weights, device=None, inplace=True, fuse=True):
    from models.yolo import Detect, Model

    model = Ensemble()
    weight_list = weights if isinstance(weights, list) else [weights]
    for w in weight_list:
        ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)
        ckpt_model = (ckpt.get('ema') or ckpt['model']).to(device).float()

        if not hasattr(ckpt_model, 'stride'):
            ckpt_model.stride = torch.tensor([32.0])
        if hasattr(ckpt_model, 'names') and isinstance(ckpt_model.names, (list, tuple)):
            ckpt_model.names = dict(enumerate(ckpt_model.names))

        model.append(ckpt_model.fuse().eval() if fuse and hasattr(ckpt_model, 'fuse') else ckpt_model.eval())

    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is Conv:
            m._non_persistent_buffers_set = set()

    if len(model) == 1:
        return model[-1]

    print(f'Ensemble created with {weight_list}\n')
    for k in ('names', 'nc', 'yaml'):
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride
    return model
