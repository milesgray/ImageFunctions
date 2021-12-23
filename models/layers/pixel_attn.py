from functools import partial
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from .learnable import Scale, Balance
from .softmax import SpatialSoftmax2d, ChannelSoftmax2d
from .pool import ZPool, SpatialMaxPool, SpatialMeanPool
from .gate import GateConv, PoolGate
from .statistics import stdv_channels

class PixelAttention(nn.Module):
    def __init__(self, f_in, 
                 f_out=None, 
                 resize="same", 
                 scale=2, 
                 softmax=True, 
                 use_pool=True,
                 use_gate=False,
                 gate_params=None,
                 add_contrast=False,
                 learn_weight=True, 
                 channel_wise=True, 
                 spatial_wise=True):
        super().__init__()
        if f_out is None:
            f_out = f_in
        
        self.add_contrast = add_contrast
        self.contrast = stdv_channels

        self.sigmoid = nn.Sigmoid()
        # layers for defined resizing of input so that it matches output
        if resize == "up":
            self.resize = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        elif resize == "down":
            self.resize = nn.AvgPool2d(scale, stride=scale)
        else:
            self.resize = nn.Identity()
        # automatic resizing to ensure input and output sizes match for attention/residual layer
        if f_in != f_out:
            self.resize = nn.Sequential(*[self.resize, nn.Conv2d(f_in, f_out, 1, bias=False)])

        if gate_params is not None:
            self.gate = partial(PoolGate, **gate_params)
        else:
            self.gate = nn.Identity()
        self.use_gate = use_gate
        self.use_pool = use_pool
        # layers for optional channel-wise and/or spacial attention
        self.channel_wise = channel_wise
        self.spatial_wise = spatial_wise
        if self.channel_wise:
            self.channel_conv = nn.Conv2d(f_out, f_out, 1, groups=f_out, bias=False)
            if self.use_pool:
                self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
                self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
                self.channel_theta = self.conv(
                    f_out, f_out // 8, kernel_size=1, padding=0, bias=False)
                self.channel_phi = self.conv(
                    f_out, f_out // 8, kernel_size=1, padding=0, bias=False)
                self.channel_g = self.conv(
                    f_out, f_out // 2, kernel_size=1, padding=0, bias=False)
                self.channel_o = self.conv(
                    f_out // 2, f_out, kernel_size=1, padding=0, bias=False)
                
            if self.use_gate:
                self.channel_gate = GateConv(f_out, f_out, 1, conv_args={"bias":False})
        if self.spatial_wise:
            self.spatial_conv = nn.Conv2d(f_out, f_out, 1, bias=False)
            if self.use_pool:
                self.spatial_avg_pool = SpatialMeanPool()
                self.spatial_max_pool = SpatialMaxPool()
            if self.use_gate:
                gate = []
                gate.append(nn.Conv2d(2, f_out, 7, 
                                      padding=3,
                                      bias=False))
                gate.append(nn.Hardsigmoid())   
                gate.append(ZPool())
                self.spatial_gate = nn.Sequential(*gate)
        if not self.channel_wise and not self.spatial_wise:
            self.conv = GateConv(f_out, f_out, 1, conv_args={"bias":False})

        # optional softmax operations for channel-wise and spatial attention layers
        self.use_softmax = softmax
        if self.use_softmax:
            self.spatial_softmax = SpatialSoftmax2d(temp=1.0, requires_grad=learn_weight)
            self.channel_softmax = ChannelSoftmax2d(temp=1.0, requires_grad=learn_weight)

        # optional learnable scaling layer that is applied after attention
        self.learn_weight = learn_weight
        if self.learn_weight:
            self.global_scale = Scale(1.0)
            self.channel_scale = Scale(1.0)
            self.spatial_scale = Scale(1.0)
            self.global_balance = Balance()

    def forward(self, x):
        """Creates an attention mask in the same shape as input.
        Supports spatial softmax that operates on an entire 2d tensor,
        channel softmax is the traditional 1d softmax that is applied to each channel

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """        
        # make x same shape as y
        x = self.resize(x)
        if self.add_contrast:
            x = x + self.contrast(x)
        if self.spatial_wise:
            spatial_y = self.spatial_conv(x)
            if self.use_pool:
                phi = self.channel_avg_pool(self.phi(spatial_y))
                g = self.channel_max_pool(self.g(spatial_y))
                # Perform reshapes
                theta = spatial_y.view(-1, self.channels // 8, x.shape[2] * x.shape[3])
                phi = phi.view(-1, self.channels // 8, x.shape[2] * x.shape[3] // 4)
                g = g.view(-1, self.channels // 2, x.shape[2] * x.shape[3] // 4)
                # Matmul and softmax to get attention maps
                beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
                # Attention map times g path
                o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1,
                                                                self.channels // 2,
                                                                x.shape[2],
                                                                x.shape[3]))
                spatial_y = spatial_y + o
            if self.use_gate:
                spatial_gate = self.spatial_gate(spatial_y)
            spatial_y = self.sigmoid(spatial_y)
            if self.learn_weight:
                spatial_y = self.spatial_scale(spatial_y)
            if self.use_softmax:
                spatial_y = self.spatial_softmax(spatial_y)
            if self.use_gate:
                spatial_y = spatial_y * spatial_gate
            spatial_out = torch.mul(x, spatial_y)
        if self.channel_wise:
            channel_y = self.channel_conv(x)
            if self.use_gate:
                channel_gate = self.channel_gate(channel_y)
            channel_y = self.sigmoid(channel_y)
            if self.learn_weight:
                channel_y = self.channel_scale(channel_y)
            if self.use_softmax:
                channel_y = self.channel_softmax(channel_y)
            if self.use_gate:
                channel_y = channel_y * channel_gate
            channel_out = torch.mul(x, channel_y)
        if self.channel_wise and self.spatial_wise:
            out = self.global_balance(spatial_out, channel_out)
        elif self.channel_wise:
            out = channel_out
        elif self.spatial_wise:
            out = spatial_out
        else:
            y = self.conv(x)
            y = self.sigmoid(y)
            if self.use_softmax:
                y = self.spatial_softmax(y)
            out = torch.mul(x, y)
        if self.learn_weight:
            out = self.global_scale(out)
        return out