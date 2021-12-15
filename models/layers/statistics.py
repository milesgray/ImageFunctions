import torch
import torch.nn as nn
import torch.nn.functional as F

def mean_channels(x):
    assert(x.dim() == 4)
    spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (x.size(2) * x.size(3))
def stdv_channels(x):
    assert(x.dim() == 4)
    mean = mean_channels(x)
    variance = (x - mean).sum(3, keepdim=True).sum(2, keepdim=True) / (x.size(2) * x.size(3))
    return variance

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range,
                rgb_mean=(0.40005, 0.42270, 0.45802), 
                rgb_std=(0.28514, 0.31383, 0.28289), 
                sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
