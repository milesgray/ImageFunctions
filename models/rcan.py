import math
from argparse import Namespace

import torch
import torch.nn as nn
from kornia.geometry.subpix import spatial_softmax2d

from models import register


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class SpatialSoftmax2d(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.temp = temp

    def forward(self, x):
        x = spatial_softmax2d(x, temperature=self.temp)
        return x

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class PA(nn.Module):
    '''Pixel Attention Layer'''
    def __init__(self, f_in, f_out=None, resize="same", scale=2, softmax=True, learn_weight=True, channel_wise=True, spatial_wise=True):
        super().__init__()
        if f_out is None:
            f_out = f_in
        
        self.sigmoid = nn.Sigmoid()
        if resize == "up":
            self.resize = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        elif resize == "down":
            self.resize = nn.AvgPool2d(scale, stride=scale)
        else:
            self.resize = nn.Identity()
        if f_in != f_out:
            self.resize = nn.Sequential(*[self.resize, nn.Conv2d(f_in, f_out, 1)])
        self.channel_wise = channel_wise
        self.spatial_wise = spatial_wise
        if channel_wise:
            self.channel_conv = nn.Conv2d(f_out, f_out, 1, groups=f_out)
        if spatial_wise:
            self.spatial_conv = nn.Conv2d(f_out, f_out, 1)
        if not channel_wise and not spatial_wise:
            self.conv = nn.Conv2d(f_out, f_out, 1)
        
        self.use_softmax = softmax
        if self.use_softmax:
            self.spatial_softmax = SpatialSoftmax2d()
            self.channel_softmax = nn.Softmax2d()
        self.learn_weight = learn_weight
        if self.learn_weight:
            self.weight_scale = Scale(1.0)

    def forward(self, x):
        x = self.resize(x)
        if self.spatial_wise:
            spatial_y = self.spatial_conv(x)
            spatial_y = self.sigmoid(spatial_y)
            if self.use_softmax:
                spatial_y = self.softmax(spatial_y)
            spatial_out = torch.mul(x, spatial_y)
        if self.channel_wise:
            channel_y = self.channel_conv(x)
            channel_y = self.sigmoid(channel_y)
            if self.use_softmax:
                channel_y = self.softmax(channel_y)
            channel_out = torch.mul(x, channel_y)
        if self.channel_wise and self.spatial_wise:
            out = spatial_out + channel_out
        elif self.channel_wise:
            out = channel_wise
        elif self.spatial_wise:
            out = spatial_wise
        else:
            y = self.conv(x)
            y = self.sigmoid(y)
            if self.use_softmax:
                y = self.spatial_softmax(y)
            out = torch.mul(x, y)
        if self.learn_weight:
            out = self.weight_scale(out)
        return out

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return torch.mul(x, y)

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale
        self.pa = PA(n_feat)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res = torch.sub(res, self.pa(x))
        return torch.add(res, x)

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        
        self.body = nn.Sequential(*modules_body)

        self.pa = PA(n_feat)

    def forward(self, x):
        res = self.body(x)
        res = torch.mul(res, self.pa(res))
        res = torch.add(res, x)
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(RCAN, self).__init__()
        self.args = args

        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        self.use_mean = args.use_mean_shift
        if self.use_mean:
            self.sub_mean = MeanShift(args.rgb_range, args.rgb_mean, args.rgb_std)
            self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            modules_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)]
            self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        if self.use_mean: 
            x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x).add(x)
        #res += x

        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        if self.use_mean: 
            x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


@register('rcan')
def make_rcan(n_resgroups=10, n_resblocks=20, n_feats=64, reduction=16,
              scale=2, no_upsampling=False, 
              use_mean_shift=False, rgb_range=1, rgb_mean=(0.39884, 0.42088, 0.45812),
              rgb_std=(0.28514, 0.31383, 0.28289)):
    args = Namespace()
    args.n_resgroups = n_resgroups
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.reduction = reduction

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.use_mean_shift = use_mean_shift
    args.rgb_range = rgb_range
    # RGB mean for movie 11 fractal set # RGB mean for DIV2K
    args.rgb_mean = rgb_mean#(0.4488, 0.4371, 0.4040)
    # RGB STD mean for movie 11 fractal set
    args.rgb_std = rgb_std
    args.res_scale = 1
    args.n_colors = 3
    return RCAN(args)
