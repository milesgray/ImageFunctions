import torch
import torch.nn as nn
import torch.nn.functional as F

from .channel_attn import ChannelAttention, MixPoolChannelAttention
from .spatial_attn import SpatialAttention, CatPoolSpatialAttention
from .registry import register

@register("balanced_attn")
class BalancedAttention(nn.Module):
    def __init__(self, in_planes, reduction=16, use_pool=True):
        super().__init__()

        if use_pool:
            self.ca = MixPoolChannelAttention(in_planes, reduction)
            self.sa = CatPoolSpatialAttention()
        else:
            self.ca = ChannelAttention(in_planes, reduction)
            self.sa = SpatialAttention()

    def forward(self, x):
        ca_ch = self.ca(x)
        sa_ch = self.sa(x)
        out = ca_ch.mul(sa_ch).mul(x)
        return out