# Deep Back-Projection Networks For Super-Resolution
# https://arxiv.org/abs/1803.02735

from argparse import Namespace

import torch
import torch.nn as nn

from ..layers import PixelAttention, MeanShift, HessianAttention
from ..layers import MSHF, DiEnDec, DAC
from ..layers import get_mean_std_rgb

from models import register

class FreqSplit(nn.Module):
    def __init__(self, fft_size=32, 
                 hop_length=2, 
                 win_length=8, 
                 window="hann_window"):
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)

    def forward(self, x):
        x_stft = torch.stft(x, self.fft_size, self.hop_length, self.win_length, self.window.cuda())
        real = x_stft[..., 0]
        imag = x_stft[..., 1]

        return real, imag


def projection_conv(in_channels, out_channels, scale, up=True, shuffle=False):
    kernel_size, stride, padding = {
        2: (6, 2, 2),
        4: (8, 4, 2),
        8: (12, 8, 2)
    }[scale]
    if up:
        if shuffle:
            resize = nn.PixelShuffle(scale)
            in_channels = in_channels // (scale * scale)
            conv_f = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
            return nn.Sequential(*[resize, conv_f])
        else:
            return nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding
            )
    else:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )

class DenseProjection(nn.Module):
    def __init__(self, in_channels, nr, scale, 
                 up=True, 
                 bottleneck=True, 
                 use_pa=True, 
                 use_shuffle=False,
                 use_pa_learn_scale=False):
        super().__init__()
        if bottleneck:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)
            ])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels

        layers_1 = [
            projection_conv(inter_channels, nr, scale, up, shuffle=not use_shuffle),
            nn.PReLU(nr)
        ]
        layers_2 = [
            projection_conv(nr, inter_channels, scale, not up, shuffle=use_shuffle),
            nn.PReLU(inter_channels)
        ]
        layers_3 = [
            projection_conv(inter_channels, nr, scale, up, shuffle=use_shuffle),
            nn.PReLU(nr)
        ]
        self.use_pa = use_pa
        if self.use_pa:
            layers_1.append(PixelAttention(nr, learn_scale=use_pa_learn_scale))
            layers_2.append(PixelAttention(inter_channels, learn_scale=use_pa_learn_scale))

        self.conv_1 = nn.Sequential(*layers_1)
        self.conv_2 = nn.Sequential(*layers_2)
        self.conv_3 = nn.Sequential(*layers_3)

        if self.use_pa:
            self.pa_x = PixelAttention(inter_channels, f_out=nr, resize="up" if up else "down", scale=scale, learn_scale=use_pa_learn_scale)
            self.pa_out = PixelAttention(nr, learn_scale=use_pa_learn_scale)

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        a_0 = self.conv_1(x)
        b_0 = self.conv_2(a_0)
        e = b_0.sub(x)
        a_1 = self.conv_3(e)

        out = a_0.add(a_1)

        if self.use_pa:
            out = out * (self.pa_out(out) + self.pa_x(x))

        return out

class FDDBPN(nn.Module):
    def __init__(self, args):
        super().__init__()
        scale = args.scale[0]

        self.depth = args.depth
        self.use_pa = args.use_pa
        self.use_pa_learn_scale = args.use_pa_learn_scale
        self.use_pa_bridge = args.use_pa_bridge
        self.use_hessian = args.use_hessian_attn
        self.use_hessian_out = args.use_hessian_out_attn

        initial = [
            nn.Conv2d(args.n_colors, args.n_feats_in, 3, padding=1),
            nn.PReLU(args.n_feats_in),
            nn.Conv2d(args.n_feats_in, args.n_feats, 1),
            nn.PReLU(args.n_feats)
        ]
        self.initial = nn.Sequential(*initial)

        self.split = FreqSplit()

        channels = args.n_feats
        if self.use_hessian:
            self.hessian_attn_in = HessianAttention(channels)
            
        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        if self.use_pa:
            self.attnmodules = nn.ModuleList()

        for i in range(self.depth):
            self.upmodules.append(
                DenseProjection(channels, args.n_feats, scale, up=True, bottleneck=i > 1,
                                use_pa=args.use_pa, use_shuffle=i%2==1,
                                use_pa_learn_scale=self.use_pa_learn_scale)
            )
            if i != 0:
                channels += args.n_feats
        self.no_upsampling = args.no_upsampling
        if self.no_upsampling:
            self.total_depth = self.depth
        else:
            self.total_depth = self.depth - 1

        self.out_dim = args.n_feats_out

        channels = args.n_feats
        for i in range(self.total_depth):
            self.downmodules.append(
                DenseProjection(channels, args.n_feats, scale, up=False, bottleneck=i != 0,
                                use_pa=args.use_pa, use_pa_learn_scale=self.use_pa_learn_scale)
            )
            channels += args.n_feats

        if self.use_pa_bridge:
            channels = args.n_feats
            for i in range(self.total_depth):
                self.attnmodules.append(
                    PixelAttention(channels, learn_scale=self.use_pa_learn_scale)
                )
                channels += args.n_feats

        reconstruction = [
            nn.Conv2d(self.depth * args.n_feats, self.out_dim, 3, padding=1)
        ]
        self.reconstruction = nn.Sequential(*reconstruction)

        if self.use_hessian_out:
            self.hessian_attn_out = HessianAttention(self.out_dim)

        self.use_mean_shift = args.use_mean_shift
        if self.use_mean_shift:
            self.sub_mean = MeanShift(args.rgb_range, args.rgb_mean, args.rgb_std)
            self.add_mean = MeanShift(args.rgb_range, args.rgb_mean, args.rgb_std, 1)

    def forward(self, x):
        if self.use_mean_shift:
            x = self.sub_mean(x)
        x = self.initial(x)
        x_real, x_imag = self.split(x)
        if self.use_hessian:
            x_real = self.hessian_attn_in(x_real)

        h_r_list = []
        h_i_list = []
        l_i_list = []
        l_r_list = []
        for i in range(self.total_depth):
            if i == 0:
                l_i = x_imag
                l_r = x_real
            else:
                l_r = torch.cat(l_r_list, dim=1)
                l_i = torch.cat(l_i_list, dim=1)
            l_r_list.append(self.upmodules[i](l_r))
            l_i_list.append(self.upmodules[i](l_i))
            if self.use_pa_bridge:
                h_r = self.attnmodules[i](torch.cat(h_r_list, dim=1))
                h_i = self.attnmodules[i](torch.cat(h_i_list, dim=1))
            else:
                h_r = torch.cat(h_r_list, dim=1)
                h_i = torch.cat(h_i_list, dim=1)
            l_r_list.append(self.downmodules[i](h_r))
            l_i_list.append(self.downmodules[i](h_i))
        if self.no_upsampling:
            if self.use_pa_bridge:
                h_r = self.attnmodules[-1](torch.cat(h_r_list, dim=1))
                h_i = self.attnmodules[-1](torch.cat(h_i_list, dim=1))
            else:
                h_r = torch.cat(h_r_list, dim=1)
                h_i = torch.cat(h_i_list, dim=1)
            l_r_list.append(self.downmodules[-1](h_r))
            l_i_list.append(self.downmodules[-1](h_i))
        h_r_list.append(self.upmodules[-1](torch.cat(l_r_list, dim=1)))
        h_i_list.append(self.upmodules[-1](torch.cat(l_i_list, dim=1)))
        h_list = h_r_list + h_i_list
        out = self.reconstruction(torch.cat(h_list, dim=1))
        if self.use_hessian_out:
            out = self.hessian_attn_out(out)
        if self.use_mean_shift:
            out = self.add_mean(out)

        return out

@register('fddbpn')
def make_fddbpn(n_feats_in=64, n_feats=32, n_feats_out=64, depth=5,
               use_pa=True, use_pa_learn_scale=False, use_pa_bridge=False,
               use_hessian_attn=True, use_hessian_out_attn=True,
               scale=2, no_upsampling=False,
               rgb_range=1, use_mean_shift=False,
               rgb_mean=(0.39884, 0.42088, 0.45812),
               rgb_std=(0.28514, 0.31383, 0.28289)):
    args = Namespace()
    args.n_feats_in = n_feats_in
    args.n_feats = n_feats
    args.n_feats_out = n_feats_out
    args.depth = depth

    args.scale = [scale]
    args.use_pa = use_pa
    args.use_pa_learn_scale = use_pa_learn_scale
    args.use_pa_bridge = use_pa_bridge
    args.no_upsampling = no_upsampling
    args.use_hessian_attn = use_hessian_attn
    args.use_hessian_out_attn = use_hessian_out_attn

    args.use_mean_shift = use_mean_shift
    args.rgb_range = rgb_range
    args.rgb_mean = rgb_mean
    args.rgb_std = rgb_std
    args.n_colors = 3
    return FDDBPN(args)
