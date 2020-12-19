# Deep Back-Projection Networks For Super-Resolution
# https://arxiv.org/abs/1803.02735

from argparse import Namespace

import torch
import torch.nn as nn

from models import register

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.40005, 0.42270, 0.45802), rgb_std=(0.28514, 0.31383, 0.28289), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class PA(nn.Module):
    '''Pixel Attention Layer'''
    def __init__(self, nf, resize="same"):
        super().__init__()
        
        self.sigmoid = nn.Sigmoid()
        if resize == "up":
            self.resize = nn.Upsample(scale_factor=2)
        elif resize == "down":
            self.resize = nn.AvgPool2d(2, stride=2)
        else:
            self.resize = nn.Identity()
        self.conv = nn.Conv2d(nf, nf, 1)

    def forward(self, x):
        x = self.resize(x)
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out

def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {
        2: (6, 2, 2),
        4: (8, 4, 2),
        8: (12, 8, 2)
    }[scale]
    if up:
        conv_f = nn.ConvTranspose2d
    else:
        conv_f = nn.Conv2d

    return conv_f(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding
    )

class DenseProjection(nn.Module):
    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True, use_pa=True):
        super(DenseProjection, self).__init__()
        if bottleneck:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)
            ])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels

        self.conv_1 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])
        self.conv_2 = nn.Sequential(*[
            projection_conv(nr, inter_channels, scale, not up),
            nn.PReLU(inter_channels)
        ])
        self.conv_3 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])
        self.use_pa = use_pa
        if self.use_pa:
            self.pa_x = PA(nr, resize="up" if up else "down")
            self.pa_out = PA(nr)

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        a_0 = self.conv_1(x)
        b_0 = self.conv_2(a_0)
        e = b_0.sub(x)
        a_1 = self.conv_3(e)

        out = a_0.add(a_1)

        if self.use_pa:
            out = self.pa_x(x).add(self.pa_out(out))

        return out

class DDBPN(nn.Module):
    def __init__(self, args):
        super().__init__()
        scale = args.scale[0]

        n0 = args.n_feats0
        nr = args.n_feats
        self.depth = args.depth

        self.sub_mean = MeanShift(args.rgb_range, args.rgb_mean, args.rgb_std)
        initial = [
            nn.Conv2d(args.n_colors, args.n_feats0, 3, padding=1),
            nn.PReLU(args.n_feats0),
            nn.Conv2d(args.n_feats0, args.n_feats, 1),
            nn.PReLU(args.n_feats)
        ]
        self.initial = nn.Sequential(*initial)

        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        channels = args.n_feats
        for i in range(self.depth):
            self.upmodules.append(
                DenseProjection(channels, args.n_feats, scale, True, i > 1)
            )
            if i != 0:
                channels += args.n_feats
        
        channels = args.n_feats
        for i in range(self.depth - 1):
            self.downmodules.append(
                DenseProjection(channels, args.n_feats, scale, False, i != 0)
            )
            channels += args.n_feats

        reconstruction = [
            nn.Conv2d(self.depth * args.n_feats, args.n_colors, 3, padding=1) 
        ]
        self.reconstruction = nn.Sequential(*reconstruction)

        self.add_mean = MeanShift(args.rgb_range, args.rgb_mean, args.rgb_std, 1)

        self.out_dim = args.n_colors

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.initial(x)

        h_list = []
        l_list = []
        for i in range(self.depth - 1):
            if i == 0:
                l = x
            else:
                l = torch.cat(l_list, dim=1)
            h_list.append(self.upmodules[i](l))
            l_list.append(self.downmodules[i](torch.cat(h_list, dim=1)))
        
        h_list.append(self.upmodules[-1](torch.cat(l_list, dim=1)))
        out = self.reconstruction(torch.cat(h_list, dim=1))
        out = self.add_mean(out)

        return out

@register('ddbpn')
def make_ddbpn(n_feats0=128, n_feats=32, depth=5, use_pa=True,
             scale=2, no_upsampling=False, rgb_range=1,
             rgb_mean=None, rgb_std=None):
    args = Namespace()
    args.n_feats0 = n_feats0
    args.n_feats = n_feats
    args.depth = depth
    
    args.scale = [scale]
    args.use_pa = use_pa
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    # RGB mean for movie 11 fractal set # RGB mean for DIV2K
    args.rgb_mean = (0.39884, 0.42088, 0.45812) if rgb_mean is None else rgb_mean#(0.4488, 0.4371, 0.4040)
    # RGB STD mean for movie 11 fractal set
    args.rgb_std = (0.28514, 0.31383, 0.28289) if rgb_std is None else rgb_std
    args.n_colors = 3
    return DDBPN(args)
