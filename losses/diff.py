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
    
@register("multiscale_l1")
class MultiscaleL1Loss(nn.Module):
    def __init__(self, scale=5):
        super(MultiscaleL1Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        self.weights = [1, 0.5, 0.25, 0.125, 0.125]
        self.weights = self.weights[:scale]

    def forward(self, input, target, mask=None):
        loss = 0
        for i in range(len(self.weights)):
            if mask is not None:
                loss += self.weights[i] * self.criterion(input * mask, target * mask)
            else:
                loss += self.weights[i] * self.criterion(input, target)
            if i != len(self.weights) - 1:
                input = self.downsample(input)
                target = self.downsample(target)
                if mask is not None:
                    mask = self.downsample(mask)
        return loss
    
@register("mse_super_res")
class MSESuperResLoss(nn.Module):
    def __init__(self, size_average=False):
        super().__init__()
        self.size_average = size_average
            
    def forward(self, x, y):
        z = x - y 
        z2 = z * z
        if self.size_average:
            return z2.mean()
        else:
            return z2.sum().div(x.size(0)*2)