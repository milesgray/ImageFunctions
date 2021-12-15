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
def get_mean_std(x):
    r = x[:, 0, :, :]
    g = x[:, 1, :, :]
    b = x[:, 2, :, :]
    r_std, r_mean = torch.std_mean(r)
    g_std, g_mean = torch.std_mean(g)
    b_std, b_mean = torch.std_mean(b)
    return (r_mean, g_mean, b_mean), (r_std, g_std, b_std)
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

class SubMeanStd(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class AddMeanStd(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) * std.view(3, 1, 1, 1)
        self.bias.data = sign * torch.Tensor(rgb_mean)
        for p in self.parameters():
            p.requires_grad = False