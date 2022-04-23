import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from .registry import register

def mean_channels(x: torch.Tensor) -> torch.Tensor:
    assert(x.dim() == 4)
    spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (x.size(2) * x.size(3))
def stdv_channels(x: torch.Tensor) -> torch.Tensor:
    assert(x.dim() == 4)
    mean = mean_channels(x)
    variance = (x - mean).sum(3, keepdim=True).sum(2, keepdim=True) / (x.size(2) * x.size(3))
    return variance
def get_mean_std_rgb(x: torch.Tensor):
    r = x[:, 0, :, :]
    g = x[:, 1, :, :]
    b = x[:, 2, :, :]
    r_std, r_mean = torch.std_mean(r)
    g_std, g_mean = torch.std_mean(g)
    b_std, b_mean = torch.std_mean(b)
    return (r_mean, g_mean, b_mean), (r_std, g_std, b_std)
def get_mean_std_nd(feat: torch.Tensor, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

@register("mean_shift")
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

@register("sub_mean_std")
class SubMeanStd(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

@register("add_mean_std")
class AddMeanStd(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) * std.view(3, 1, 1, 1)
        self.bias.data = sign * torch.Tensor(rgb_mean)
        for p in self.parameters():
            p.requires_grad = False
            
# ------------------------------------------------------------------------------------------------------------------
# Minibatch standard deviation layer 
# reference: https://github.com/akanimax/pro_gan_pytorch/blob/master/pro_gan_pytorch/CustomLayers.py#L300
# ------------------------------------------------------------------------------------------------------------------
@register("minibatch_std_dev")
class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer
    """

    def __init__(self):
        """
        derived class constructor
        """
        super().__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y
    
@register("minibatch_std")
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x