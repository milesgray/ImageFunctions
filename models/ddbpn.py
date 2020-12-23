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
    def __init__(self, f_in, f_out=None, resize="same", scale=2):
        super().__init__()
        if f_out is None:
            f_out = f_in
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax2d()
        if resize == "up":
            self.resize = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        elif resize == "down":
            self.resize = nn.AvgPool2d(scale, stride=scale)
        else:
            self.resize = nn.Identity()
        if f_in != f_out:
            self.resize = nn.Sequential(*[self.resize, nn.Conv2d(f_in, f_out, 1)])
        self.conv = nn.Conv2d(f_out, f_out, 1)

    def forward(self, x):
        x = self.resize(x)
        y = self.conv(x)
        y = self.softmax(self.sigmoid(y))
        out = torch.mul(x, y)

        return out

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
    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True, use_pa=True, use_shuffle=False):
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
            layers_1.append(PA(nr))
            layers_2.append(PA(inter_channels))
        
        self.conv_1 = nn.Sequential(*layers_1)
        self.conv_2 = nn.Sequential(*layers_2)
        self.conv_3 = nn.Sequential(*layers_3)

        if self.use_pa:
            self.pa_x = PA(inter_channels, f_out=nr, resize="up" if up else "down", scale=scale)
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
            out = out * (self.pa_out(out) + self.pa_x(x))

        return out

class DDBPN(nn.Module):
    def __init__(self, args):
        super().__init__()
        scale = args.scale[0]

        self.depth = args.depth

        initial = [
            nn.Conv2d(args.n_colors, args.n_feats_in, 3, padding=1),
            nn.PReLU(args.n_feats_in),
            nn.Conv2d(args.n_feats_in, args.n_feats, 1),
            nn.PReLU(args.n_feats)
        ]
        self.initial = nn.Sequential(*initial)

        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        channels = args.n_feats
        for i in range(self.depth):
            self.upmodules.append(
                DenseProjection(channels, args.n_feats, scale, up=True, bottleneck=i > 1, use_pa=args.use_pa, use_shuffle=i%2==1)
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
                DenseProjection(channels, args.n_feats, scale, up=False, bottleneck=i != 0, use_pa=args.use_pa)
            )
            channels += args.n_feats

        reconstruction = [
            nn.Conv2d(self.depth * args.n_feats, self.out_dim, 3, padding=1) 
        ]
        self.reconstruction = nn.Sequential(*reconstruction)

        self.use_mean_shift = args.use_mean_shift
        if self.use_mean_shift:
            self.sub_mean = MeanShift(args.rgb_range, args.rgb_mean, args.rgb_std)
            self.add_mean = MeanShift(args.rgb_range, args.rgb_mean, args.rgb_std, 1)

    def forward(self, x):
        if self.use_mean_shift:
            x = self.sub_mean(x)
        x = self.initial(x)

        h_list = []
        l_list = []
        for i in range(self.total_depth):
            if i == 0:
                l = x
            else:
                l = torch.cat(l_list, dim=1)
            h_list.append(self.upmodules[i](l))
            l_list.append(self.downmodules[i](torch.cat(h_list, dim=1)))
        if self.no_upsampling:
            l_list.append(self.downmodules[-1](torch.cat(h_list, dim=1)))
        h_list.append(self.upmodules[-1](torch.cat(l_list, dim=1)))
        out = self.reconstruction(torch.cat(h_list, dim=1))
        if self.use_mean_shift:
            out = self.add_mean(out)

        return out

@register('ddbpn')
def make_ddbpn(n_feats_in=64, n_feats=32, n_feats_out=64, depth=5, use_pa=True,
               scale=2, no_upsampling=False, rgb_range=1,
               use_mean_shift=False, 
               rgb_mean=(0.39884, 0.42088, 0.45812), 
               rgb_std=(0.28514, 0.31383, 0.28289)):
    args = Namespace()
    args.n_feats_in = n_feats_in
    args.n_feats = n_feats
    args.n_feats_out = n_feats_out
    args.depth = depth
    
    args.scale = [scale]
    args.use_pa = use_pa
    args.no_upsampling = no_upsampling

    args.use_mean_shift = use_mean_shift
    args.rgb_range = rgb_range
    args.rgb_mean = rgb_mean
    args.rgb_std = rgb_std
    args.n_colors = 3
    return DDBPN(args)
