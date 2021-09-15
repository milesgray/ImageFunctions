
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import piq

from argparse import Namespace
from .registry import register


@register("ms_ssim_loss")
class MultiScaleSSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = piq.MultiScaleSSIMLoss()

    def forward(self, x, y):
        return self.loss_fn(x, y)
        
@register("gmsd_loss")
class GMSDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = piq.GMSDLoss()

    def forward(self, x, y):
        return self.loss_fn(x, y)

@register("ms_gmsd_loss")
class MultiScaleGMSDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = piq.MultiScaleGMSDLoss()

    def forward(self, x, y):
        return self.loss_fn(x, y)

@register("haar_psi_loss")
class HaarPSILoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = piq.HaarPSILoss()

    def forward(self, x, y):
        return self.loss_fn(x, y)

@register("fsim_loss")
class FSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = piq.FSIMLoss()

    def forward(self, x, y):
        return self.loss_fn(x, y)

@register("vif_loss")
class VIFLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = piq.VIFLoss()

    def forward(self, x, y):
        return self.loss_fn(x, y)

@register("vsi_loss")
class VSILoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = piq.VSILoss()

    def forward(self, x, y):
        return self.loss_fn(x, y)
