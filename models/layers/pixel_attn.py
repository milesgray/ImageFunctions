import torch
import torch.nn as nn
import torch.nn.functional as F
from .learnable import Scale, Balance
from .softmax import SpatialSoftmax2d, ChannelSoftmax2d

class PixelAttention(nn.Module):
    '''Pixel Attention Layer'''
    def __init__(self, f_in, 
                 f_out=None, 
                 resize="same", 
                 scale=2, 
                 softmax=True, 
                 learn_weight=True, 
                 channel_wise=True, 
                 spatial_wise=True):
        super().__init__()
        if f_out is None:
            f_out = f_in

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
            self.resize = nn.Sequential(*[self.resize, nn.Conv2d(f_in, f_out, 1)])

        # layers for optional channel-wise and/or spacial attention
        self.channel_wise = channel_wise
        self.spatial_wise = spatial_wise
        if self.channel_wise:
            self.channel_conv = nn.Conv2d(f_out, f_out, 1, groups=f_out)
        if self.spatial_wise:
            self.spatial_conv = nn.Conv2d(f_out, f_out, 1)
        if not self.channel_wise and not self.spatial_wise:
            self.conv = nn.Conv2d(f_out, f_out, 1)

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
        if self.spatial_wise:
            spatial_y = self.spatial_conv(x)
            if self.learn_weight:
                spatial_y = self.spatial_scale(spatial_y)
            spatial_y = self.sigmoid(spatial_y)
            if self.use_softmax:
                spatial_y = self.spatial_softmax(spatial_y)
            spatial_out = torch.mul(x, spatial_y)
        if self.channel_wise:
            channel_y = self.channel_conv(x)
            if self.learn_weight:
                channel_y = self.channel_scale(channel_y)
            channel_y = self.sigmoid(channel_y)
            if self.use_softmax:
                channel_y = self.channel_softmax(channel_y)
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