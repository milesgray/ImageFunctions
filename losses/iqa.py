
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

@register("vsi_loss")
class DSSLoss(nn.Module):
    def __init__(self, reduction: str = 'mean',
                 data_range: Union[int, float] = 1.0, dct_size: int = 8,
                 sigma_weight: float = 1.55, kernel_size: int = 3,
                 sigma_similarity: float = 1.5, percentile: float = 0.05):
        super().__init__()
        self.loss_fn = piq.DSSLoss(reduction=reduction, data_range=data_range, dct_size=dct_size
                                   sigma_weight=sigma_weight, sigma_similarity=sigma_similarity)

    def forward(self, x, y):
        """Computation of DSS as a loss function.
        Args:
            x: Tensor of prediction of the network.
            y: Reference tensor.
        Returns:
            Value of DSS loss to be minimized. 0 <= DSS <= 1.
        """
        return self.loss_fn(x, y)