import torch
import torch.nn as nn

from .registry import register

@register("charbonnier")
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6, mode=None):
        super().__init__()
        self.eps = eps
        self.mode = mode

    def forward(self, x, y, mask=None):
        N = x.size(1)
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)
        if mask is not None:
            loss = loss * mask
        if self.mode == 'sum':
            loss = torch.sum(loss) / N
        else:
            loss = loss.mean()
        return loss
    
@register("mse_super_res")
class MSESuperResLoss(nn.Module):
    def __init__(self, size_average=False):
        super().__init__()
        self.size_average = size_average
            
    def forward(self, x, y):
        z = x - y 
        z2 = z * z
        if size_average:
            return z2.mean()
        else:
            return z2.sum().div(x.size(0)*2)