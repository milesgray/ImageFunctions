# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

from argparse import Namespace
from functools import partial

import torch
import torch.nn as nn

from .layers import PixelAttention, NonLocalAttention
from .layers import Balance, stdv_channels
from .layers.activations import create as create_act

from models import register

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, 
                 kernel=3,
                 act="gelu",
                 attn_fn=PixelAttention):
        super().__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kernel, padding=(kernel-1)//2, stride=1),
            create_act(act)
        ])
        self.attn = attn_fn(G)

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, self.attn(out)), 1)

class RDB(nn.Module):
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
            convs.append(RDB_Conv(G0 + c*G, G,
                                  kernel=kernel,
                                  act=act,
                                  attn_fn=attn_fn))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)
        self.res_balance = Balance()

    def forward(self, x):
        return self.res_balance(self.LFF(self.convs(x)), x)

class RDN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        r = args.scale[0]
        nColors = args.n_colors # number of color channels expected as input
        G0 = args.G0    # baseline channel amount, also the output channel amount
        kernel = args.RDNkSize    # kernel size for SFE and GFF conv layers
        RDBkSize = args.RDBkSize    # kernel size of RDB conv layers
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
        self.RDBs = nn.ModuleList()
        for i in range(D):
            self.RDBs.append(
                RDB(growRate0=G0, 
                    growRate=G, 
                    nConvLayers=C,
                    kernel=RDBkSize,
                    act=act,
                    attn_fn=attn_fn)
            )

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

        RDBs_out = []
        for rdb in self.RDBs:
            x = rdb(x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, dim=1))
        x = self.GFF_balance(x, self.SFE_res_attn(f__1))

        if self.args.no_upsampling:
            return x
        else:
            return self.UpNet(x)


@register('rdn')
def make_rdn(D=20, C=6, G=32, attn_fn='PixelAttention', act="gelu",
             G0=64, RDNkSize=3, RDBkSize=3, RDNconfig=None,
             scale=2, no_upsampling=True):
    args = Namespace()
    args.D = D
    args.C = C
    args.G = G
    args.act = act
    args.attn_fn = attn_fn
    args.G0 = G0
    args.RDNkSize = RDNkSize
    args.RDBkSize = RDBkSize
    # original RDN suggested parameters
    RDNstaticConfig = {
        'A': (20, 6, 32),
        'B': (16, 8, 64),
    }
    if RDNconfig in RDNstaticConfig:
        args.D, args.C, args.G = RDNstaticConfig[RDNconfig]

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.n_colors = 3
    return RDN(args)
