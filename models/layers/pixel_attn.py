from functools import partial
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from .learnable import Scale, Balance
from .softmax import SpatialSoftmax2d, ChannelSoftmax2d
from .pool import ZPool

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))
def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance

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

class PoolGate(nn.Module):
    def __init__(self, pool_channels, 
                       out_channels,
                       kernel=7, 
                       pool_dim=1,
                       activation=nn.ReLU, 
                       gate=nn.Sigmoid,
                       norm=nn.BatchNorm2d):
        super().__init__()
        self.hard = hard
        self.pool = ZPool(dim=pool_dim)
        self.conv = nn.Conv2d(pool_channels, out_channels, kernel, 
                              padding=kernel//2 + 1,
                              bias=False)
        if norm is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm(out_planes,
                            eps=1e-5,
                            momentum=0.01,
                            affine=True)
        if activation is None:
            self.act = nn.Identity()
        else:
            self.act = activation()
        if gate is None:
            self.gate = nn.Identity()
        else:
            self.gate = gate()

    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y)
        y = self.act(y)
        y = self.gate(y)
        return y

class PixelAttention(nn.Module):
    '''Pixel Attention Layer'''
    def __init__(self, f_in, 
                 f_out=None, 
                 resize="same", 
                 scale=2, 
                 softmax=True, 
                 gate=False,
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
        # layers for optional channel-wise and/or spacial attention
        self.channel_wise = channel_wise
        self.spatial_wise = spatial_wise
        if self.channel_wise:
            self.channel_conv = nn.Conv2d(f_out, f_out, 1, groups=f_out, bias=False)
            if self.gate:
                self.channel_gate = GateConv(f_out, f_out, 1, conv_args={"bias":False})
        if self.spatial_wise:
            self.spatial_conv = nn.Conv2d(f_out, f_out, 1, bias=False)
            if self.gate:
                self.spatial_gate_conv = nn.Conv2d(2, f_out, 7, 
                                                    padding=3,
                                                    bias=False)
                self.spatial_gate_act = nn.Hardsigmoid()           
                self.spatial_gate_pool = ZPool()
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
            if self.gate:
                spatial_gate = self.spatial_gate_pool(spatial_y)
                spatial_gate = self.spatial_gate_conv(spatial_gate)                
                spatial_gate = self.spatial_gate_act(spatial_gate)
            spatial_y = self.sigmoid(spatial_y)
            if self.learn_weight:
                spatial_y = self.spatial_scale(spatial_y)
            if self.use_softmax:
                spatial_y = self.spatial_softmax(spatial_y)
            if self.gate:
                spatial_y = spatial_y * spatial_gate
            spatial_out = torch.mul(x, spatial_y)
        if self.channel_wise:
            channel_y = self.channel_conv(x)
            if self.gate:
                #channel_gate = channel_y.permute(0,2,1,3).contiguous()
                #channel_gate = self.channel_gate_pool(channel_gate)  
                channel_gate = self.channel_gate(channel_y)
                #channel_gate = self.channel_gate_norm(channel_gate)
                #channel_gate = self.channel_gate_act(channel_gate)
                #channel_gate = channel_gate.permute(0,2,1,3).contiguous()
            channel_y = self.sigmoid(channel_y)
            if self.learn_weight:
                channel_y = self.channel_scale(channel_y)
            if self.use_softmax:
                channel_y = self.channel_softmax(channel_y)
            if self.gate:
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