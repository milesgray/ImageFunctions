import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .learnable import Balance
from .registry import register

@register("channel_attn")
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.attn = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.PReLU(channel // reduction),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.attn(y)
        return y

@register("mix_pool_channel_attn")
class MixPoolChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.net = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, 
                          padding=0, 
                          bias=bias),
                nn.PReLU(channel // reduction),
                nn.Conv2d(channel // reduction, channel, 1, 
                          padding=0, 
                          bias=bias)
        )
        self.balance = Balance(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.net(self.avg_pool(x))
        max_out = self.net(self.max_pool(x))
        out = self.balance(avg_out, max_out)
        return self.sigmoid(out)
    
# contrast-aware channel attention module
@register("contrast_aware_attn")
class CCALayer(nn.Module):
    def __init__(self, channel: int, reduction: int=16):
        super().__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y    