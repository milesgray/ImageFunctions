import math
from argparse import Namespace

import torch
import torch.nn as nn

from ..layers import ChannelAttention, PixelAttention, BalancedAttention
from ..layers import Balance, Scale
from ..layers import MeanShift
from ..layers import ScaleAwareAdapt

from models import register


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


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

        super().__init__(*m)

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super().__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(ChannelAttention(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale
        self.pa = PixelAttention(n_feat)
        self.balance = Balance()
        self.attn_scale = Scale()
        self.body_scale = Scale()

    def forward(self, x):
        res = self.body(x)
        res = self.body_scale(x)
        res = torch.sub(res, self.attn_scale(self.pa(x)))
        return self.balance(res, x)

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super().__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        
        self.body = nn.Sequential(*modules_body)

        self.pa = PixelAttention(f_in, f_out=n_feat)
        self.balance = Balance()

    def forward(self, x):
        res = self.body(x)
        res = torch.mul(res, self.pa(res))
        res = self.balance(res, x)
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=default_conv):
        super().__init__()
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
        
        self.head_attn = BalancedAttention(n_feats)
        
        self.body_balance = Balance()

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

        res = self.body_balance(self.body(x), self.head_attn(x))

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
