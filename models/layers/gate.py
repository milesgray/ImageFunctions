
from functools import partial
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pool import ZPool
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