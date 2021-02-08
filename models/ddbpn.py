# Deep Back-Projection Networks For Super-Resolution
# https://arxiv.org/abs/1803.02735
from typing import Tuple, Callable
from argparse import Namespace
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from kornia.geometry.subpix import spatial_softmax2d

from models import register
#from layers import FourierINR

##################################################################################################
# https://github.com/zongyi-li/fourier_neural_operator/blob/master/scripts/fourier_on_images.py #
#################################################################################################

def compl_mul2d(a, b):
    op = partial(torch.einsum, "bctq,dctq->bdtq")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, mode):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = mode #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = mode

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 2, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels,  x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=( x.size(-2), x.size(-1)))
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes):
        super(SimpleBlock2d, self).__init__()

        self.conv1 = SpectralConv2d(1, 16, modes=modes)
        self.conv2 = SpectralConv2d(16, 32, modes=modes)
        self.conv3 = SpectralConv2d(32, 64, modes=modes)

        self.pool = nn.MaxPool2d(2, 2)


        self.fc1 = nn.Linear(64 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Net2d(nn.Module):
    def __init__(self, modes, width):
        """
        A wrapper function
        """
        super(Net2d, self).__init__()
        
        self.conv1 = SimpleBlock2d(modes, modes,  width)

    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

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
        super(FourierINR, self).__init__()

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

class HessianAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.coder = nn.Sequential(DiEnDec(3, nn.ReLU(True)))
        self.dac = nn.Sequential(DAC(channels))
        self.hessian3 = nn.Sequential(MSHF(channels, kernel=3))
        self.hessian5 = nn.Sequential(MSHF(channels, kernel=5))
        self.hessian7 = nn.Sequential(MSHF(channels, kernel=7))

    def forward(self, x):
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
        return x


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
                spatial_y = self.spatial_softmax(spatial_y)
            spatial_out = torch.mul(x, spatial_y)
        if self.channel_wise:
            channel_y = self.channel_conv(x)
            channel_y = self.sigmoid(channel_y)
            if self.use_softmax:
                channel_y = self.channel_softmax(channel_y)
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
            layers_1.append(PA(nr, learn_weight=use_pa_learn_scale))
            layers_2.append(PA(inter_channels, learn_weight=use_pa_learn_scale))
        
        self.conv_1 = nn.Sequential(*layers_1)
        self.conv_2 = nn.Sequential(*layers_2)
        self.conv_3 = nn.Sequential(*layers_3)

        if self.use_pa:
            self.pa_x = PA(inter_channels, f_out=nr, resize="up" if up else "down", scale=scale, learn_weight=use_pa_learn_scale)
            self.pa_out = PA(nr, learn_weight=use_pa_learn_scale)

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
            nn.Conv2d(args.n_feats_in, args.n_feats, 1),
            nn.PReLU(args.n_feats)
        ]
        self.initial = nn.Sequential(*initial)

        channels = args.n_feats
        if self.use_hessian:
            self.hessian = HessianAttention(channels)

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
                    PA(channels, learn_weight=self.use_pa_learn_scale)
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
def make_ddbpn(n_feats_in=64, n_feats=32, n_feats_out=64, depth=5, 
               use_pa=True, use_pa_learn_scale=False, use_pa_bridge=False,
               use_hessian_attn=True, scale=2, no_upsampling=False, 
               fourier_out=32, fourier_features=32, fourier_layer_sizes=[32,32],
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

    args.fourier_out = fourier_out
    args.fourier_features = fourier_features
    args.fourier_layer_sizes = fourier_layer_sizes

    args.use_mean_shift = use_mean_shift
    args.rgb_range = rgb_range
    args.rgb_mean = rgb_mean
    args.rgb_std = rgb_std
    args.n_colors = 3
    return DDBPN(args)
