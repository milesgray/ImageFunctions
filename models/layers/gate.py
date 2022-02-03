
from functools import partial
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BasicConv, Flatten
from .pool import ZPool, ChannelPool
from .registry import register

@register("gate_conv")
class GateConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel,
                 conv_fn=nn.Conv2d, 
                 act_fn=nn.ELU,
                 gate_fn=nn.Sigmoid,
                 conv_args={}):
        super().__init__()
        self.conv = conv_fn(in_channel, out_channel * 2, kernel, 
                         **conv_args)
        self.act = act_fn()
        self.gate = gate_fn()

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.split(x, x.size(1) // 2, dim=1)
        x = self.act(x)
        gate = self.gate(gate)
        x = x * gate
        return nn.Hardsigmoid()(x)

@register("pool_gate")
class PoolGate(nn.Module):
    def __init__(self, pool_channels, 
                       out_channels,
                       kernel=7, 
                       pool_dim=1,
                       activation_fn=nn.ReLU, 
                       gate_fn=nn.Sigmoid,
                       norm_fn=nn.BatchNorm2d):
        super().__init__()
        self.pool = ZPool(dim=pool_dim)
        self.conv = nn.Conv2d(pool_channels, out_channels, kernel, 
                              padding=kernel//2 + 1,
                              bias=False)
        self.norm = nn.Identity() if norm_fn is None else norm_fn(out_planes,
                            eps=1e-5,
                            momentum=0.01,
                            affine=True)
        self.act = nn.Identity() if activation_fn is None else activation_fn()
        self.gate = nn.Identity() if gate_fn is None else gate_fn()

    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y)
        y = self.act(y)
        y = self.gate(y)
        return y

@register("channel_gate")
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

@register('spatial_gate')
class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

@register("dual_gate")
class DualGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out