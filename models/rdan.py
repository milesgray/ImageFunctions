# Residual Dense Attention Network for Image Super-Resolution
# Uses Residual Dense Network (https://arxiv.org/abs/1802.08797) as base architecture

from argparse import Namespace
from functools import partial

import torch
import torch.nn as nn
from .layers import PixelAttention, LocalMultiHeadChannelAttention

from models import register

class RDAB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, k=3,
                 attn_fn=PixelAttention):
        super().__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, k, padding=(k-1)//2, stride=1),
            nn.GLU()
        ])
        self.out_attn = attn_fn(G)
        self.in_attn = attn_fn(Cin)

    def forward(self, x):
        out = self.conv(x)
        in_attn = self.in_attn(x)
        out_attn = self.out_attn(out)
        return torch.cat((x + in_attn, out + out_attn), 1)

class RDAB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers,
                 k=3,
                 attn_fn=PixelAttention):
        super().__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDAB_Conv(G0 + c*G, G, attn_fn=attn_fn))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        r = args.scale[0]
        G0 = args.G0
        k = args.k
        attn_fn = args.attn_fn  if hasattr(args, "attn_fn") else PixelAttention
        attn_fn = eval(attn_fn) if isinstance(attn_fn, str) else attn_fn

        # number of RDB blocks, conv layers, out channels
        self.D = args.D
        C = args.C
        G = args.G

        # Shallow feature extraction net
        self.SFE1 = nn.Conv2d(args.n_colors, G0, k, padding=(k-1)//2, stride=1)        
        self.SFE2 = nn.Conv2d(G0, G0, k, padding=(k-1)//2, stride=1)
        self.SFE_attn = attn_fn(G0)

        # Redidual dense blocks and dense feature fusion
        self.RDABs = nn.ModuleList()
        for i in range(self.D):
            self.RDABs.append(
                RDAB(growRate0 = G0,
                    growRate = G,
                    nConvLayers = C,
                    attn_fn = attn_fn)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, k, padding=(k-1)//2, stride=1)
        ])

        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            # Up-sampling net
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * r * r, k, padding=(k-1)//2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(G, args.n_colors, k, padding=(k-1)//2, stride=1)
                ])
            elif r == 4:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * 4, k, padding=(k-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, G * 4, k, padding=(k-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, args.n_colors, k, padding=(k-1)//2, stride=1)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFE1(x)
        x  = self.SFE2(f__1)

        RDABs_out = []
        for i in range(self.D):
            x = self.RDABs[i](x)
            RDABs_out.append(x)

        x = self.GFF(torch.cat(RDABs_out, 1))
        x += f__1 - self.SFE_attn(f__1)

        if self.args.no_upsampling:
            return x
        else:
            return self.UPNet(x)


@register('rdan')
def make_rdan(D=20, C=6, G=32, attn_fn='PixelAttention',
             G0=64, k=3, RDNconfig=None,
             scale=2, no_upsampling=False):
    args = Namespace()
    args.D = D
    args.C = C
    args.G = G
    args.attn_fn = attn_fn
    args.G0 = G0
    args.k = k
    # preset architecture params from original RDN
    RDNstaticConfig = {
        'A': (20, 6, 32),
        'B': (16, 8, 64),
    }
    if RDNconfig in RDNstaticConfig:
        args.D, args.C, args.G = RDNstaticConfig[RDNconfig]

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.n_colors = 3
    return RDAN(args)
