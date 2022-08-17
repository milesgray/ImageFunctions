# Deep Back-Projection Networks For Super-Resolution
# https://arxiv.org/abs/1803.02735
from typing import Tuple, Callable
from argparse import Namespace
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models import register
from ImageFunctions.layers import MeanShift, HessianAttention, PixelAttention
#from layers import FourierINR

#######################################
class LinearResidual(nn.Module):
    def __init__(self, args: Namespace, transform: Callable):
        super().__init__()

        self.args = args
        self.transform = transform
        self.weight = nn.Parameter(
            torch.tensor(args.weight).float(), requires_grad=args.learnable_weight)

    def forward(self, x: Tensor) -> Tensor:
        if self.args.weighting_type == 'shortcut':
            return self.transform(x) + self.weight * x
        elif self.args.weighting_type == 'residual':
            return self.weight * self.transform(x) + x
        else:
            raise ValueError

def create_activation(activation_type: str, *args, **kwargs) -> nn.Module:
    if activation_type == 'leaky_relu':
        return nn.LeakyReLU(*args, **kwargs)

class FourierINR(nn.Module):
    """
    INR with Fourier features as specified in https://people.eecs.berkeley.edu/~bmild/fourfeat/
    """
    def __init__(self, in_features, args: Namespace, num_fourier_feats=64, layer_sizes=[64,64,64], out_features=64,
                 has_bias=True, activation="leaky_relu",
                 learnable_basis=True,):
        super().__init__()

        layers = [
            nn.Linear(num_fourier_feats * 2, layer_sizes[0], bias=has_bias),
            create_activation(activation)
        ]

        for index in range(len(layer_sizes) - 1):
            transform = nn.Sequential(
                nn.Linear(layer_sizes[index], layer_sizes[index + 1], bias=has_bias),
                create_activation(activation)
            )

            if args.residual.enabled:
                layers.append(LinearResidual(args.residual, transform))
            else:
                layers.append(transform)

        layers.append(nn.Linear(layer_sizes[-1], out_features, bias=has_bias))

        self.model = nn.Sequential(*layers)

        # Initializing the basis
        basis_matrix = args.scale * torch.randn(num_fourier_feats, in_features)
        self.basis_matrix = nn.Parameter(basis_matrix, requires_grad=learnable_basis)

    def compute_fourier_feats(self, coords: Tensor) -> Tensor:
        sines = (2 * np.pi * coords @ self.basis_matrix.t()).sin() # [batch_size, num_fourier_feats]
        cosines = (2 * np.pi * coords @ self.basis_matrix.t()).cos() # [batch_size, num_fourier_feats]

        return torch.cat([sines, cosines], dim=1) # [batch_size, num_fourier_feats * 2]

    def forward(self, coords: Tensor) -> Tensor:
        return self.model(self.compute_fourier_feats(coords))

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
            conv_f = nn.Conv2d(in_channels, out_channels, 1, groups=in_channels)
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
            layers_1.append(PixelAttention(nr, learn_weight=use_pa_learn_scale))
            layers_2.append(PixelAttention(inter_channels, learn_weight=use_pa_learn_scale))

        self.conv_1 = nn.Sequential(*layers_1)
        self.conv_2 = nn.Sequential(*layers_2)
        self.conv_3 = nn.Sequential(*layers_3)

        if self.use_pa:
            self.pa_x = PixelAttention(inter_channels, f_out=nr, resize="up" if up else "down", scale=scale, learn_weight=use_pa_learn_scale)
            self.pa_out = PixelAttention(nr, learn_weight=use_pa_learn_scale)

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
        
        self.scale = args.scale[0]
        self.depth = args.depth
        self.use_pa = args.use_pa
        self.use_pa_learn_scale = args.use_pa_learn_scale
        self.use_pa_bridge = args.use_pa_bridge
        self.use_hessian = args.use_hessian_attn
        self.no_upsampling = args.no_upsampling
        self.num_feats = args.n_feats

        fourier_args = Namespace()
        fourier_args.scale = 1.0
        fourier_args.residual = Namespace()
        fourier_args.residual.weight = 1.0
        fourier_args.residual.weighting_type = 'residual'
        fourier_args.residual.learnable_weight = True
        fourier_args.residual.enabled = True

        self.fourier = FourierINR(2, fourier_args,
                                  num_fourier_feats=args.fourier_features,
                                  layer_sizes=args.fourier_layer_sizes,
                                  out_features=args.fourier_out)

        initial = [
            nn.Conv2d(args.n_colors, args.n_feats_in, 3, padding=1),
            nn.PReLU(args.n_feats_in),
            nn.Conv2d(args.n_feats_in, self.num_feats, 1),
            nn.PReLU(self.num_feats)
        ]
        self.initial = nn.Sequential(*initial)

        if self.use_hessian:
            self.hessian = HessianAttention(self.num_feats)

        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        if self.use_pa:
            self.attnmodules = nn.ModuleList()

        channels = self.num_feats
        for i in range(self.depth):
            self.upmodules.append(
                DenseProjection(channels, self.num_feats, self.scale, 
                                up=True, bottleneck=i > 1,
                                use_pa=self.use_pa, 
                                use_shuffle=i%2==1,
                                use_pa_learn_scale=self.use_pa_learn_scale)
            )
            if i != 0:
                channels += self.num_feats
                
        
        if self.no_upsampling:
            self.total_depth = self.depth
        else:
            self.total_depth = self.depth - 1

        self.out_dim = args.n_feats_out

        channels = self.num_feats
        for i in range(self.total_depth):
            self.downmodules.append(
                DenseProjection(channels, self.num_feats, self.scale, 
                                up=False, 
                                bottleneck=i != 0,
                                use_pa=self.use_pa, 
                                use_pa_learn_scale=self.use_pa_learn_scale)
            )
            channels += self.num_feats

        if self.use_pa_bridge:
            channels = self.num_feats
            for i in range(self.total_depth):
                self.attnmodules.append(
                    PixelAttention(channels, 
                                   learn_weight=self.use_pa_learn_scale)
                )
                channels += self.num_feats
                
        reconstruction = [
            nn.Conv2d(self.depth * self.num_feats, self.out_dim, 3,
                      padding=1)
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
            x = self.hessian(x)

        if coords:
            freq_coord = self.fourier(coords.view(-1, 2))
        else:
            freq_coord = False

        h_list = []
        l_list = []
        for i in range(self.total_depth):
            if i == 0:
                l = x
            else:
                l = torch.cat(l_list, dim=1)
            if freq_coord:
                h_list.append(self.upmodules[i](torch.cat(l, freq_coord.view(l.shape[0], -1, l.shape[2], l.shape[3]), dim=1)))
            else:
                h_list.append(self.upmodules[i](l))
            if self.use_pa_bridge:
                h = self.attnmodules[i](torch.cat(h_list, dim=1))
            else:
                h = torch.cat(h_list, dim=1)
            if freq_coord:
                l_list.append(self.downmodules[i](torch.cat(h, freq_coord.view(h.shape[0], -1, h.shape[2], h.shape[3]), dim=1)))
            else:
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
def make_ddbpn(n_feats_in=64, n_feats=32,
               n_feats_out=64, depth=5,
               use_pa=True,
               use_pa_learn_scale=False,
               use_pa_bridge=False,
               use_hessian_attn=True,
               scale=2,
               no_upsampling=False,
               fourier_out=32, fourier_features=32, fourier_layer_sizes=[32,32],
               rgb_range=1,
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
    args.use_pa_learn_scale = use_pa_learn_scale
    args.use_pa_bridge = use_pa_bridge
    args.no_upsampling = no_upsampling
    args.use_hessian_attn = use_hessian_attn

    args.fourier_out = fourier_out
    args.fourier_features = fourier_features
    args.fourier_layer_sizes = fourier_layer_sizes

    args.use_mean_shift = use_mean_shift
    args.rgb_range = rgb_range
    args.rgb_mean = rgb_mean
    args.rgb_std = rgb_std
    args.n_colors = 3
    return DDBPN(args)
