# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PixelAttention, MeanShift, Scale, Balance
from models import register
from .layers.activations import create as create_act

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, pa=False, act=nn.ReLU(True), res_scale=1):

        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = Scale(init_value=res_scale)
        self.use_pa = pa
        if pa:
            self.pa_add = PixelAttention(n_feats)
            self.pa_sub = PixelAttention(n_feats)
            self.balance_add = Balance()

    def forward(self, x):
        res = self.body(x)
        if self.use_pa:
            y = res.sub(self.pa_sub(x))
            y = self.res_scale(y)
            res = self.balance_add(y, self.pa_add(res))
        else:
            res = self.res_scale(res)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super().__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        use_bn = args.use_bn
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True) if args.act == "relu" else create_act(args.act)
        self.url = None
        self.use_mean_shift = args.use_mean_shift
        if self.use_mean_shift:
            self.sub_mean = MeanShift(args.rgb_range, rgb_mean=args.rgb_mean, rgb_std=args.rgb_std)
            self.add_mean = MeanShift(args.rgb_range, rgb_mean=args.rgb_mean, rgb_std=args.rgb_std, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, bn=use_bn, act=act, pa=args.use_pa, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        if self.use_mean_shift:
            x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        if self.use_mean_shift:
            x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


@register('edsr-baseline')
def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1, act="relu",
                       scale=2, no_upsampling=False, use_bn=False, use_mean_shift=False, use_pa=False,
                       rgb_mean=(0.40005, 0.42270, 0.45802), rgb_std=(0.28514, 0.31383, 0.28289),
                       rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    args.act = act
    args.use_bn = use_bn

    args.use_pa = use_pa

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    
    args.use_mean_shift = use_mean_shift
    args.rgb_mean = rgb_mean
    args.rgb_std = rgb_std
    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDSR(args)


@register('edsr')
def make_edsr(n_resblocks=32, n_feats=256, res_scale=0, act="relu",
              scale=2, no_upsampling=False, use_bn=False, use_mean_shift=False, use_pa=False,
              rgb_mean=(0.40005, 0.42270, 0.45802), rgb_std=(0.28514, 0.31383, 0.28289), 
              rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    args.act = act
    args.use_bn = use_bn

    args.use_pa = use_pa

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.use_mean_shift = use_mean_shift
    args.rgb_mean = rgb_mean
    args.rgb_std = rgb_std
    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDSR(args)
