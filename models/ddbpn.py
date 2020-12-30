# Deep Back-Projection Networks For Super-Resolution
# https://arxiv.org/abs/1803.02735

from argparse import Namespace

import torch
import torch.nn as nn

from models import register
from layers import FourierINR

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class MSHF(nn.Module):
    def __init__(self, n_channels, kernel=3):
        super(MSHF, self).__init__()

        pad = int((kernel - 1) / 2)

        self.grad_xx = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_yy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_xy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)

        for m in self.modules():
            if m == self.grad_xx:
                m.weight.data.zero_()
                m.weight.data[:, :, 1, 0] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, 1, -1] = 1
            elif m == self.grad_yy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 1] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, -1, 1] = 1
            elif m == self.grad_xy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 0] = 1
                m.weight.data[:, :, 0, -1] = -1
                m.weight.data[:, :, -1, 0] = -1
                m.weight.data[:, :, -1, -1] = 1

    def forward(self, x):
        fxx = self.grad_xx(x)
        fyy = self.grad_yy(x)
        fxy = self.grad_xy(x)
        hessian = ((fxx + fyy) + ((fxx - fyy) ** 2 + 4 * (fxy ** 2)) ** 0.5) / 2
        return hessian

class DiEnDec(nn.Module):
    def __init__(self, n_channels, act=nn.ReLU(inplace=True)):
        super(DiEnDec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
            nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
        )
        self.gate = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        output = self.gate(self.decoder(self.encoder(x)))
        return output

class DAC(nn.Module):
    def __init__(self, n_channels):
        super(DAC, self).__init__()

        self.mean = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
        )
        self.std = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
        )

    def forward(self, observed_feat, referred_feat):
        assert (observed_feat.size()[:2] == referred_feat.size()[:2])
        size = observed_feat.size()
        referred_mean, referred_std = calc_mean_std(referred_feat)
        observed_mean, observed_std = calc_mean_std(observed_feat)

        normalized_feat = (observed_feat - observed_mean.expand(
            size)) / observed_std.expand(size)
        referred_mean = self.mean(referred_mean)
        referred_std = self.std(referred_std)
        output = normalized_feat * referred_std.expand(size) + referred_mean.expand(size)
        return output

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

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class PA(nn.Module):
    '''Pixel Attention Layer'''
    def __init__(self, f_in, f_out=None, resize="same", scale=2, softmax=True, learn_scale=True):
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
        self.conv = nn.Conv2d(f_out, f_out, 1)
        self.use_softmax = softmax
        if self.use_softmax:
            self.softmax = nn.Softmax2d()
        self.learn_scale = learn_scale
        if self.learn_scale:
            self.scale = Scale(1.0)

    def forward(self, x):
        x = self.resize(x)
        y = self.conv(x)
        y = self.sigmoid(y)
        if self.use_softmax:
            y = self.softmax(y)
        out = torch.mul(x, y)
        if self.learn_scale:
            out = self.scale(out)
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
    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True, use_pa=True, use_shuffle=False,
                 use_pa_learn_scale=False):
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
            layers_1.append(PA(nr, learn_scale=use_pa_learn_scale))
            layers_2.append(PA(inter_channels, learn_scale=use_pa_learn_scale))
        
        self.conv_1 = nn.Sequential(*layers_1)
        self.conv_2 = nn.Sequential(*layers_2)
        self.conv_3 = nn.Sequential(*layers_3)

        if self.use_pa:
            self.pa_x = PA(inter_channels, f_out=nr, resize="up" if up else "down", scale=scale, learn_scale=use_pa_learn_scale)
            self.pa_out = PA(nr, learn_scale=use_pa_learn_scale)

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
        self.use_pa = args.use_pa
        self.use_pa_learn_scale = args.use_pa_learn_scale
        self.use_pa_bridge = args.use_pa_bridge
        self.use_hessian = args.use_hessian_attn

        initial = [
            nn.Conv2d(args.n_colors, args.n_feats_in, 3, padding=1),
            nn.PReLU(args.n_feats_in),
            nn.Conv2d(args.n_feats_in, args.n_feats, 1),
            nn.PReLU(args.n_feats)
        ]
        self.initial = nn.Sequential(*initial)

        channels = args.n_feats
        if self.use_hessian:
            self.coder = nn.Sequential(DiEnDec(3, nn.ReLU(True)))
            self.dac = nn.Sequential(DAC(channels))
            self.hessian3 = nn.Sequential(MSHF(channels, kernel=3))
            self.hessian5 = nn.Sequential(MSHF(channels, kernel=5))
            self.hessian7 = nn.Sequential(MSHF(channels, kernel=7))

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
                    PA(channels, learn_scale=self.use_pa_learn_scale)
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

    def forward(self, x, coords=None):
        if self.use_mean_shift:
            x = self.sub_mean(x)
        x = self.initial(x)
        if self.use_hessian:
            hessian3 = self.hessian3(x)
            hessian5 = self.hessian5(x)
            hessian7 = self.hessian7(x)
            hessian = torch.cat((torch.mean(hessian3, dim=1, keepdim=True),
                                    torch.mean(hessian5, dim=1, keepdim=True),
                                    torch.mean(hessian7, dim=1, keepdim=True))
                                , 1)
            hessian = self.coder(hessian)
            attention = torch.sigmoid(self.dac[0](hessian.expand(x.size()), x))
            x = x * attention

        h_list = []
        l_list = []
        for i in range(self.total_depth):
            if i == 0:
                l = x
            else:
                l = torch.cat(l_list, dim=1)                
            h_list.append(self.upmodules[i](l))
            if self.use_pa_bridge:
                h = self.attnmodules[i](torch.cat(h_list, dim=1))
            else:
                h = torch.cat(h_list, dim=1)
            l_list.append(self.downmodules[i](h))
        if self.no_upsampling:
            if self.use_pa_bridge:
                h = self.attnmodules[-1](torch.cat(h_list, dim=1))
            else:
                h = torch.cat(h_list, dim=1)
            l_list.append(self.downmodules[-1](h))
        h_list.append(self.upmodules[-1](torch.cat(l_list, dim=1)))
        out = self.reconstruction(torch.cat(h_list, dim=1))
        if self.use_mean_shift:
            out = self.add_mean(out)

        return out

@register('ddbpn')
def make_ddbpn(n_feats_in=64, n_feats=32, n_feats_out=64, depth=5, 
               use_pa=True, use_pa_learn_scale=False, use_pa_bridge=False,
               use_hessian_attn=True, scale=2, no_upsampling=False, 
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

    args.use_mean_shift = use_mean_shift
    args.rgb_range = rgb_range
    args.rgb_mean = rgb_mean
    args.rgb_std = rgb_std
    args.n_colors = 3
    return DDBPN(args)
