from functools import partial
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register
from .learnable import Scale, Balance
from .softmax import SpatialSoftmax2d, ChannelSoftmax2d
from .gate import GateConv, PoolGate
from .statistics import stdv_channels
from .common import get_valid_padding

@register("pixel_attn")
class PixelAttention(nn.Module):
    def __init__(self, f_in: int, 
                 f_out: int=None, 
                 kernel: int=1,
                 spatial_k: int=7,
                 resize: str="same", 
                 interpolation: str="bilinear",
                 scale: int=2, 
                 softmax: bool=True, 
                 add_contrast: bool=False,
                 learn_weight: bool=True, 
                 channel_wise: bool=True, 
                 channel_out_fn: callable=torch.mul,
                 spatial_wise: bool=True,
                 spatial_out_fn: callable=torch.mul,
                 dropout: float=0.0):
        """Highly configurable novel attention mask that
        attends to both channel-wise and spatial-wise views in different ways
        of the image-based feature with shape :math:`(B, C, W, H)`.
        Attention masks are calculated using sigmoid as a base and then
        optionally have a softmax operation applied.
        Uses a 2D version of softmax for spatial attention calculation.
        Can learn scaling weights for all important fusion operations.

        Args:
            f_in (int): Number of channels for input tensor. Assumed to be
                equal to the output channels if no `f_out` is specified.
            f_out (int, optional): Number of output channels to reshape the
                input features to. Adds an additional 1x1 Conv2d to reshape 
                before any attention calculations are performed. 
                Defaults to None.
            resize (str, optional): Directive for either downsample or upsample,
                if resizing is being applied. "same", "down", "up". 
                Defaults to "same".
            scale (int, optional): Resizing factor if `resize` is not "same". 
                Defaults to 2.
            softmax (bool, optional): Apply softmax after sigmoid when creating
                attention masks. Defaults to True.
            add_contrast (bool, optional): [description]. Defaults to False.
            learn_weight (bool, optional): [description]. Defaults to True.
            channel_wise (bool, optional): [description]. Defaults to True.
            channel_out_fn (callable, optional): [description]. Defaults to torch.mul.
            spatial_wise (bool, optional): [description]. Defaults to True.
            spatial_out_fn (callable, optional): [description]. Defaults to torch.mul.
            dropout (float, optional): [description]. Defaults to 0.0.
        """
        super().__init__()
        if f_out is None:
            f_out = f_in
            
        self.channels = f_out
        
        self.interpolation = interpolation
        
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
            self.resize = nn.Sequential(*[self.resize, nn.Conv2d(f_in, f_out, 1, 
                                                                 bias=False)])

        # layers for optional channel-wise and/or spacial attention
        self.channel_wise = channel_wise
        self.spatial_wise = spatial_wise
        if self.channel_wise:
            self.channel_conv = nn.Conv2d(f_out, f_out, kernel, 
                                          padding=get_valid_padding(kernel, 0), 
                                          groups=f_out, 
                                          bias=False)
            self.channel_out_fn = channel_out_fn
        if self.spatial_wise:
            self.spatial_conv = nn.Conv2d(f_out, f_out, spatial_k, 
                                          padding=get_valid_padding(spatial_k, 0), 
                                          bias=False)
            self.spatial_out_fn = spatial_out_fn                                          
        if not self.channel_wise and not self.spatial_wise:
            self.conv = GateConv(f_out, f_out, 1, 
                                 conv_args={"bias":False})

        # optional softmax operations for channel-wise and spatial attention layers
        self.use_softmax = softmax
        if self.use_softmax:
            self.spatial_softmax = SpatialSoftmax2d(temp=1.0, 
                                                    requires_grad=learn_weight)
            self.channel_softmax = ChannelSoftmax2d(temp=1.0, 
                                                    requires_grad=learn_weight)

        # optional learnable scaling layer that is applied after attention
        self.learn_weight = learn_weight
        if self.learn_weight:
            self.global_scale = Scale(1.0)
            self.channel_scale = Scale(1.0)
            self.spatial_scale = Scale(1.0)
            self.global_balance = Balance()
        
        self.use_dropout = dropout > 0
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout)

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
            spatial_y = self.sigmoid(spatial_y)
            if self.learn_weight:
                spatial_y = self.spatial_scale(spatial_y)
            if self.use_softmax:
                spatial_y = self.spatial_softmax(spatial_y)
            if self.use_dropout:
                spatial_y = self.dropout(spatial_y)
            spatial_y = F.interpolate(spatial_y, 
                            size=x.shape[-2:], 
                            mode=self.interpolation, 
                            align_corners=True)
            spatial_out = self.spatial_out_fn(x, spatial_y)
        if self.channel_wise:
            channel_y = self.channel_conv(x)
            channel_y = self.sigmoid(channel_y)
            if self.learn_weight:
                channel_y = self.channel_scale(channel_y)
            if self.use_softmax:
                channel_y = self.channel_softmax(channel_y)
            if self.use_dropout:
                channel_y = self.dropout(channel_y)
            channel_out = self.channel_out_fn(x, channel_y)
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