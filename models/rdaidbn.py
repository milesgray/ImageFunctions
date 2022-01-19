# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

from argparse import Namespace
from functools import partial

import torch
import torch.nn as nn
from .layers import PixelAttention, NonLocalAttention
from .layers import Balance, Scale, stdv_channels
from .layers import SpectralConv2d
from .layers import create as create_layer
from .layers.activations import create as create_act

from models import register

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                     padding=padding, 
                     bias=bias, 
                     dilation=dilation,
                     groups=groups)

class SpectralIMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.distilled_channels = int(self.in_channels * distillation_rate)
        self.remaining_channels = int(self.in_channels - self.distilled_channels)
        self.c1 = SpectralConv2d(in_channels=self.in_channels, out_channels=self.in_channels, modes1=12, modes2=12)
        self.c2 = SpectralConv2d(in_channels=self.remaining_channels, out_channels=self.in_channels, modes1=12, modes2=12)
        self.c3 = SpectralConv2d(in_channels=self.remaining_channels, out_channels=self.in_channels, modes1=12, modes2=12)
        self.c4 = SpectralConv2d(in_channels=self.remaining_channels, out_channels=self.distilled_channels, modes1=12, modes2=12)        
        self.act = create_act('leakyrelu', negative_slope=0.05)
        self.c5 = conv_layer(self.distilled_channels * 4, self.in_channels, 1)
        
        self.balance = Balance()
        self.attn = NonLocalAttention(self.in_channels)

    def forward(self, x):
        out_c1 = self.act(self.c1(x))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.balance(self.attn(self.c5(out)), x)
        return out_fused

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, norm=nn.LayerNorm, pa=False, act=nn.ReLU(True), res_scale=1):

        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if norm:
                m.append(norm(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = Scale(init_value=res_scale)
        self.use_pa = pa
        if pa:
            self.pa_add = PixelAttention(n_feats, n_feats)
            self.pa_sub = PixelAttention(n_feats)
            self.balance_add = Balance()
        self.residual_balance = Balance(0.0)

    def forward(self, x):
        res = self.body(x)
        if self.use_pa:
            y = res.sub(self.pa_sub(x))
            y = self.res_scale(y)
            res = self.self.balance_add(y, self.pa_add(res))
        else:
            res = self.res_scale(res)
        res = self.residual_balance(res, x)

        return res

class RDAB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, 
                 kernel=3,
                 act="gelu",
                 attn_fn=PixelAttention):
        super().__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            conv_layer(Cin, G, kernel, stride=1),
            create_act(act)
        ])
        self.attn = attn_fn(G)

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, self.attn(out)), 1)

class RDAB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, 
                 kernel=3,
                 act="gelu",
                 attn_fn=PixelAttention):
        super().__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDAB_Conv(G0 + c*G, G,
                                  kernel=kernel,
                                  act=act,
                                  attn_fn=attn_fn))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)
        self.res_balance = Balance()

    def forward(self, x):
        return self.res_balance(self.LFF(self.convs(x)), x)

class RDAIDBN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        r = args.scale[0]
        nColors = args.n_colors # number of color channels expected as input
        G0 = args.G0    # baseline channel amount, also the output channel amount
        kernel = args.RDANkSize    # kernel size for SFE and GFF conv layers
        RDABkSize = args.RDABkSize    # kernel size of RDB conv layers
        act = args.act  # activation function for layers
        D = args.D  # number of RDB blocks
        C = args.C  # conv layers per RDB block
        G = args.G  # output channels of each conv layer within RDB block, which are all concat together
        
        # configure attention layer function from string input
        attn_fn = args.attn_fn  if hasattr(args, "attn_fn") else PixelAttention        
        attn_fn = eval(attn_fn) if isinstance(attn_fn, str) else attn_fn
        
        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(nColors, G0, kernel, padding=(kernel-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kernel, padding=(kernel-1)//2, stride=1)
        
        self.SFE_attn = attn_fn(G0)
        self.SFE_res_attn = attn_fn(G0)
        self.SFE_balance = Balance()

        # Redidual dense blocks and dense feature fusion
        self.rda_branch = nn.ModuleList()
        for i in range(D):
            self.rda_branch.append(
                RDAB(growRate0=G0, 
                    growRate=G, 
                    nConvLayers=C,
                    kernel=RDABkSize,
                    act=act,
                    attn_fn=attn_fn)
            )
            
        # IMD nonlocal branch
        self.imd_branch = nn.ModuleList()
        self.imd_rda_balancers = nn.ModuleList()
        for i in range(D//2):
            self.imd_branch.append(SpectralIMDModule(G0))
            self.imd_rda_balancers.append(Balance(0.0))
        self.imd_branch_balance = Balance()
        
        # residual branch
        self.branch = ResBlock(conv_layer, G0, 5, pa=True)
        self.branch_balance = Balance()

        # Global Feature Fusion 
        #   - This takes all of the RDB block outputs concatenated together and
        #     combines all of the channels down to `G0`, which is the expected output
        #     channel dimension size
        self.GFF = nn.Sequential(*[
            nn.Conv2d(D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kernel, padding=(kernel-1)//2, stride=1)
        ])
        
        # learns a weighted sum of GFF and SFE_res_attn
        # - final output if not upsampled
        self.GFF_balance = Balance()

        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            # Up-sampling net
            if r == 2 or r == 3:
                self.UpNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(G, nColors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            elif r == 4:
                self.UpNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, nColors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFE_balance(self.SFENet2(f__1), self.SFE_attn(f__1))
        
        n = self.branch(x)
        residual = self.branch_balance(n, self.SFE_res_attn(f__1))

        out = []
        for i, block in enumerate(self.rda_branch):
            x = block(x)
            out.append(x)            
            if i % 2 == 0:
                y = self.imd_branch[i//2](x)
                x = self.imd_rda_balancers[i//2](x, y)

        x = self.GFF(torch.cat(out, dim=1))
        x = self.GFF_balance(x, residual)

        if self.args.no_upsampling:
            return x
        else:
            return self.UpNet(x)


@register('rdaidbn')
def make_rdaidbn(blocks=20, layers=6, filters=32, attn_fn='PixelAttention', act="gelu",
             out_filters=64, RDANkSize=3, RDABkSize=3, 
             scale=2, no_upsampling=True):
    args = Namespace()
    args.D = blocks
    args.C = layers
    args.G = filters
    args.G0 = out_filters
    args.act = act
    args.attn_fn = attn_fn
    
    args.RDANkSize = RDANkSize
    args.RDABkSize = RDABkSize
    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.n_colors = 3
    return RDAIDBN(args)
