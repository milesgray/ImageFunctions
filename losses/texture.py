# https://github.com/hhb072/WaveletSRNet/blob/master/networks.py#L15
  
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register

@register("texture")
class TexturesLoss(nn.Module):
    def __init__(self, nc=3, alpha=1.2, margin=0):
        super().__init__()
        self.nc = nc
        self.alpha = alpha
        self.margin = margin

    def forward(self, x, y):
        xi = x.contiguous().view(x.size(0), -1, self.nc, x.size(2), x.size(3))
        yi = y.contiguous().view(y.size(0), -1, self.nc, y.size(2), y.size(3))

        xi2 = torch.sum(xi * xi, dim=2)
        yi2 = torch.sum(yi * yi, dim=2)

        loss = F.relu(yi2.mul(self.alpha) - xi2 + self.margin)

        return loss.mean()